// SceneRenderer: the per-frame GPU uploads.
//
// This is the MAIN-THREAD side of the renderer's Prepare-stage tier
// boundary, and it is a separate translation unit for exactly that reason:
// everything here takes an SDL_GPUCopyPass* and touches the device, while
// everything ScenePrep does (scene/scene_prep.hpp) is device-free and
// worker-safe. Making the tier a file boundary as well as a type boundary
// means a reviewer sees a gpu/ include appear in the wrong TU.
//
// All of it runs inside the single consolidated copy pass render_frame opens
// before any render pass — copy passes cannot begin inside a render pass.

#include "scene_renderer.hpp"
#include "scene_renderer_internal.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/render_context.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/scene_prep.hpp"

#include <SDL3/SDL_gpu.h>
#include <algorithm>
#include <cstring>
#include <glm/glm.hpp>

namespace bracken::scene {

namespace gpu = bracken::gpu;

void SceneRenderer::upload_instance_buffers(SDL_GPUCopyPass* copy_pass) {
    if (!copy_pass) {
        return;
    }

    // Upload main instance buffer
    size_t total_instances = 0;
    for (ModelHandle h : instance_batches_touched_) {
        total_instances += instance_batches_[h].size();
    }
    if (collect_stats_) {
        render_stats_.instances_drawn = static_cast<uint32_t>(total_instances);
    }

    if (total_instances > 0) {
        size_t required_size = total_instances * sizeof(gpu::InstanceData);

        if (!instance_storage_buffer_ || instance_storage_capacity_ < required_size) {
            instance_storage_capacity_ = required_size * 2; // over-allocate
            instance_storage_buffer_ = gpu::GPUBuffer::create_dynamic(context_->device(), gpu::GPUBuffer::Type::Storage,
                                                                      instance_storage_capacity_);
            if (!instance_storage_buffer_) {
                BRACKEN_LOG_ERROR(LOG_TAG, "Failed to create instance storage buffer");
                instance_storage_capacity_ = 0;
            }
        }

        if (instance_storage_buffer_) {
            // Pack in ascending handle order — must match the iteration order in
            // render_instanced_models so base_instance offsets line up. Bins are
            // written straight into the mapped transfer buffer (one pass over
            // the data instead of bin -> pack scratch -> transfer buffer).
            if (gpu::GPUBuffer::Mapping mapping = instance_storage_buffer_->map()) {
                auto* dst = static_cast<gpu::InstanceData*>(mapping.data());
                (void)for_each_instance_batch(
                    instance_batches_, 0,
                    [&](ModelHandle, const std::vector<gpu::InstanceData>& instances, uint32_t) {
                        std::memcpy(dst, instances.data(), instances.size() * sizeof(gpu::InstanceData));
                        dst += instances.size();
                    });
                mapping.upload(copy_pass, required_size);
            }
        }
    }

    // Upload shadow instance buffer (per-cascade sections packed contiguously).
    // Layout: [cascade 0 instances][cascade 1 instances]... each cascade in ascending handle order.
    size_t total_shadow = 0;
    for (int c = 0; c < MAX_SHADOW_CASCADES; ++c) {
        for (ModelHandle h : shadow_batches_touched_[c]) {
            total_shadow += shadow_instance_batches_[c][h].size();
        }
    }

    if (total_shadow > 0) {
        size_t required_size = total_shadow * sizeof(gpu::ShadowInstanceData);

        if (!shadow_instance_storage_buffer_ || shadow_instance_storage_capacity_ < required_size) {
            shadow_instance_storage_capacity_ = required_size * 2;
            shadow_instance_storage_buffer_ = gpu::GPUBuffer::create_dynamic(
                context_->device(), gpu::GPUBuffer::Type::Storage, shadow_instance_storage_capacity_);
            if (!shadow_instance_storage_buffer_) {
                BRACKEN_LOG_ERROR(LOG_TAG, "Failed to create shadow instance storage buffer");
                shadow_instance_storage_capacity_ = 0;
                shadow_cascade_base_instance_ = {0, 0, 0, 0};
                return;
            }
        }

        // Cascade sections packed contiguously, straight into the mapped
        // transfer buffer (same single-pass packing as the main instances).
        if (gpu::GPUBuffer::Mapping mapping = shadow_instance_storage_buffer_->map()) {
            auto* dst = static_cast<gpu::ShadowInstanceData*>(mapping.data());
            uint32_t packed = 0;
            for (int c = 0; c < MAX_SHADOW_CASCADES; ++c) {
                shadow_cascade_base_instance_[c] = packed;
                packed = for_each_instance_batch(
                    shadow_instance_batches_[c], packed,
                    [&](ModelHandle, const std::vector<gpu::ShadowInstanceData>& instances, uint32_t) {
                        std::memcpy(dst, instances.data(), instances.size() * sizeof(gpu::ShadowInstanceData));
                        dst += instances.size();
                    });
            }
            mapping.upload(copy_pass, required_size);
        }
    } else {
        shadow_cascade_base_instance_ = {0, 0, 0, 0};
    }
}

void SceneRenderer::upload_bone_palettes(SDL_GPUCopyPass* copy_pass, const FramePacket& frame) {
    if (!copy_pass || bone_palette_matrix_count_ == 0) {
        return;
    }

    // Upload into the CURRENT slot of the double buffer; the other slot still
    // holds last frame's palettes for the skinned prepass' prev-pose reads.
    auto& buffer = bone_palette_buffers_[bone_palette_current_];
    auto& capacity = bone_palette_capacities_[bone_palette_current_];

    const size_t required_size = bone_palette_matrix_count_ * sizeof(glm::mat4);
    if (!buffer || capacity < required_size) {
        // Grow geometrically with a floor of 64 KB (~16 palettes of 64 joints).
        capacity = std::max(required_size * 2, static_cast<size_t>(64 * 1024));
        buffer = gpu::GPUBuffer::create_dynamic(context_->device(), gpu::GPUBuffer::Type::Storage, capacity);
        if (!buffer) {
            BRACKEN_LOG_ERROR(LOG_TAG, "Failed to create bone palette storage buffer");
            capacity = 0;
            return;
        }
    }

    // Copy each palette straight from packet memory into the mapped transfer
    // buffer at its prefix-summed matrix_offset — the offsets were assigned
    // serially at seal in palettes() order, so destination offsets line up
    // with every draw's palette offset by construction (and the copies
    // parallelize over disjoint ranges — see prep_copy_palettes).
    gpu::GPUBuffer::Mapping mapping = buffer->map();
    if (!mapping) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Failed to map bone palette transfer buffer");
        return;
    }
    ScenePrep::prep_copy_palettes(core::jobs::JobSystem::instance(), frame.palettes(),
                                  static_cast<glm::mat4*>(mapping.data()));
    mapping.upload(copy_pass, required_size);
}

} // namespace bracken::scene
