#include "bracken/render/gi/voxel_scene.hpp"

#include "bracken/core/hash.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/model_loader.hpp"

#include <algorithm>

namespace bracken::render::gi {
namespace {
constexpr const char* LOG_TAG = "gi";

/// Linear [0,1] -> the RGBA8 the voxelizer unpacks. The alpha lane is written
/// opaque and currently ignored on the way back out; RG4's emissive support
/// claims it.
[[nodiscard]] std::uint32_t pack_rgba8(const glm::vec3& c) {
    const auto q = [](float v) {
        return static_cast<std::uint32_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    return (0xFFu << 24) | (q(c.b) << 16) | (q(c.g) << 8) | q(c.r);
}

} // namespace

void VoxelScene::build_models(const ModelManager& models) {
    BRACKEN_PROFILE_ZONE("VoxelScene::build_models");

    triangles_.clear();
    model_ranges_.assign(models.model_count(), ModelRange{});
    stats_.models_with_geometry = 0;

    for (ModelHandle h = 0; h < static_cast<ModelHandle>(models.model_count()); ++h) {
        const Model* model = models.get_model(h);
        if (model == nullptr) {
            continue;
        }
        const auto first = static_cast<std::uint32_t>(triangles_.size());

        for (const Mesh& mesh : model->meshes) {
            // SHADOW-ONLY PROXIES ARE SKIPPED, for the same reason `rt_scene`'s
            // gather skips them: they are stand-ins the main pass never draws,
            // and the real geometry they stand in for is in this same buffer.
            // Voxelizing both would double a tree's occlusion.
            if (mesh.shadow_only) {
                continue;
            }
            const std::size_t vertex_count =
                mesh.vertices.empty() ? mesh.skinned_vertices.size() : mesh.vertices.size();
            if (mesh.indices.empty() || vertex_count == 0) {
                continue; // uploaded without retaining geometry
            }

            // Positions come from whichever array this mesh actually filled —
            // a skinned mesh keeps its BIND POSE in `skinned_vertices`, and the
            // bind pose is the right thing to voxelize: the clipmap represents
            // the world's static occlusion, and a character's own contribution
            // to bounced light at a 25-unit voxel is not worth re-voxelizing
            // every frame for.
            const auto position_at = [&mesh](std::uint32_t i) -> glm::vec3 {
                if (!mesh.vertices.empty()) {
                    return mesh.vertices[i].position;
                }
                return mesh.skinned_vertices[i].position;
            };

            const std::size_t triangle_count = mesh.indices.size() / 3;
            for (std::size_t tri = 0; tri < triangle_count; ++tri) {
                const std::uint32_t i0 = mesh.indices[tri * 3 + 0];
                const std::uint32_t i1 = mesh.indices[tri * 3 + 1];
                const std::uint32_t i2 = mesh.indices[tri * 3 + 2];
                if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
                    continue;
                }

                // The per-triangle base colour the loader already baked for the
                // RT path. Reused rather than re-derived: the two paths must
                // agree about what a surface's albedo is, or the same wall
                // bounces a different colour depending on which GI mode is on.
                glm::vec3 albedo{mesh.material.base_color_factor};
                if (tri < mesh.rt_triangle_base_colors.size()) {
                    albedo = glm::vec3(mesh.rt_triangle_base_colors[tri]);
                }

                const glm::vec3 p0 = position_at(i0);
                VoxelTriangle t;
                t.v0 = p0;
                t.e1 = position_at(i1) - p0;
                t.e2 = position_at(i2) - p0;
                t.packed_albedo = pack_rgba8(albedo);
                triangles_.push_back(t);
            }
        }

        const auto count = static_cast<std::uint32_t>(triangles_.size()) - first;
        model_ranges_[h] = {first, count};
        if (count > 0) {
            ++stats_.models_with_geometry;
        }
    }

    stats_.triangles = static_cast<std::uint32_t>(triangles_.size());
    stats_.triangle_bytes = triangles_.size() * sizeof(VoxelTriangle);
    triangles_dirty_ = true;
    // The instance list indexes into what just moved, so it has to be rebuilt
    // against the new ranges before anything dispatches.
    has_instance_hash_ = false;

    BRACKEN_LOG_INFO(LOG_TAG, "voxel scene: {} triangles over {} models ({:.1f} MB)",
                     stats_.triangles, stats_.models_with_geometry,
                     static_cast<double>(stats_.triangle_bytes) / (1024.0 * 1024.0));
}

void VoxelScene::set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    BRACKEN_PROFILE_ZONE("VoxelScene::set_instances");

    // Hash the handle SEQUENCE and every transform. Unlike the TLAS there is no
    // refit path to protect, so one hash is enough: either the list is what was
    // uploaded last frame or it is not.
    std::uint64_t hash = core::k_fnv1a_offset_basis_64;
    const auto mix = [&hash](std::uint32_t word) {
        hash ^= static_cast<std::uint64_t>(word);
        hash *= core::k_fnv1a_prime_64;
    };
    mix(static_cast<std::uint32_t>(instances.size()));
    for (const auto& [handle, transform] : instances) {
        mix(static_cast<std::uint32_t>(handle));
        const auto* words = reinterpret_cast<const std::uint32_t*>(&transform[0][0]);
        for (int i = 0; i < 16; ++i) {
            mix(words[i]);
        }
    }
    if (has_instance_hash_ && hash == instance_hash_) {
        return;
    }
    instance_hash_ = hash;
    has_instance_hash_ = true;

    instances_.clear();
    instances_.reserve(instances.size());
    stats_.instances_without_geometry = 0;
    std::uint64_t work = 0;

    for (const auto& [handle, transform] : instances) {
        if (handle < 0 || static_cast<std::size_t>(handle) >= model_ranges_.size()) {
            ++stats_.instances_without_geometry;
            continue;
        }
        const ModelRange& range = model_ranges_[static_cast<std::size_t>(handle)];
        if (range.count == 0) {
            // Counted, never silently skipped: an unvoxelized building is a
            // hole in the GI that reads as a bug in the cone tracer.
            ++stats_.instances_without_geometry;
            continue;
        }
        VoxelInstance inst;
        inst.model_to_world = transform;
        inst.first_triangle = range.first;
        inst.triangle_count = range.count;
        inst.work_offset = static_cast<std::uint32_t>(work);
        instances_.push_back(inst);
        work += range.count;
    }

    stats_.instances = static_cast<std::uint32_t>(instances_.size());
    stats_.work_items = work;
    stats_.instance_bytes = instances_.size() * sizeof(VoxelInstance);
    instances_dirty_ = true;
}

void VoxelScene::upload(gpu::GPUDevice& device) {
    BRACKEN_PROFILE_ZONE("VoxelScene::upload");

    if (triangles_dirty_ && !triangles_.empty()) {
        triangle_buffer_ = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Storage,
                                                         triangles_.data(),
                                                         triangles_.size() * sizeof(VoxelTriangle));
        triangles_dirty_ = false;
    }
    if (instances_dirty_ && !instances_.empty()) {
        // Recreated rather than mapped: the list changes only when the scene's
        // placement set does, which for a village is once at load.
        instance_buffer_ = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Storage,
                                                         instances_.data(),
                                                         instances_.size() * sizeof(VoxelInstance));
        instances_dirty_ = false;
    }
}

void VoxelScene::shutdown() {
    triangle_buffer_.reset();
    instance_buffer_.reset();
    triangles_.clear();
    model_ranges_.clear();
    instances_.clear();
    triangles_dirty_ = false;
    instances_dirty_ = false;
    has_instance_hash_ = false;
    stats_ = VoxelSceneStats{};
}

} // namespace bracken::render::gi
