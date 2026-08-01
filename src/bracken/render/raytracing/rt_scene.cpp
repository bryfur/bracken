#include "bracken/render/raytracing/rt_scene.hpp"

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

#include <chrono>

namespace bracken::render::rt {
namespace {

/** Gather one model's triangles in model space, plus attributes if wanted. */
struct ModelGeometry {
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> colors;
    bool has_cpu_geometry = false;
};

ModelGeometry gather(const Model& model, bool want_attributes) {
    ModelGeometry g;
    for (const Mesh& mesh : model.meshes) {
        if (mesh.indices.empty() || mesh.vertices.empty()) {
            continue; // uploaded models have had their CPU geometry released
        }
        g.has_cpu_geometry = true;
        const auto base = static_cast<uint32_t>(g.positions.size());
        for (const gpu::Vertex3D& v : mesh.vertices) {
            g.positions.push_back(v.position);
            if (want_attributes) {
                g.normals.push_back(v.normal);
                g.colors.push_back(v.color * mesh.material.base_color_factor);
            }
        }
        for (uint32_t i : mesh.indices) {
            g.indices.push_back(base + i);
        }
    }
    return g;
}

} // namespace

RtScene::~RtScene() = default;

void RtScene::build_blas_for_models(const ModelManager& models, bool want_attributes) {
    BRACKEN_PROFILE_ZONE("RtScene::build_blas");
    const auto start = std::chrono::steady_clock::now();

    entries_.clear();
    blas_nodes_.clear();
    blas_triangles_.clear();
    attributes_.clear();
    stats_ = {};

    // Collect the models worth building for, so the parallel build below has a
    // dense index space.
    struct Pending {
        ModelHandle handle{};
        ModelGeometry geometry;
        Blas blas;
    };
    std::vector<Pending> pending;

    for (ModelHandle h = 0; h < static_cast<ModelHandle>(models.model_count()); ++h) {
        const Model* m = models.get_model(h);
        if (m == nullptr || m->meshes.empty()) {
            continue;
        }
        ModelGeometry g = gather(*m, want_attributes);
        if (!g.has_cpu_geometry || g.indices.size() < 3) {
            continue;
        }
        pending.push_back({h, std::move(g), {}});
    }

    if (pending.empty()) {
        BRACKEN_LOG_WARN("rt",
                         "RtScene: no model has CPU-side geometry. Models release it at GPU "
                         "upload, so the ray-tracing scene must be built from models loaded with "
                         "geometry retained.");
        return;
    }

    // One BLAS per model, built in parallel. This is the whole reason the
    // build is affordable at load: ~80 ms of single-threaded work over 52
    // models becomes a fraction of that across the pool.
    core::jobs::parallel_for(core::jobs::JobSystem::instance(), std::size_t{0}, pending.size(),
                             [&pending](std::size_t begin, std::size_t end) {
                                 for (std::size_t i = begin; i < end; ++i) {
                                     pending[i].blas = build_blas(pending[i].geometry.positions,
                                                                  pending[i].geometry.indices);
                                 }
                             });

    // Concatenate serially, in a deterministic order, so offsets are stable
    // between runs and the structure is reproducible.
    for (Pending& p : pending) {
        if (p.blas.empty()) {
            continue;
        }
        ModelEntry entry;
        entry.node_offset = static_cast<uint32_t>(blas_nodes_.size());
        entry.triangle_offset = static_cast<uint32_t>(blas_triangles_.size());
        entry.node_count = static_cast<uint32_t>(p.blas.nodes.size());
        entry.bounds_min = p.blas.bounds_min;
        entry.bounds_max = p.blas.bounds_max;
        entry.valid = true;

        blas_nodes_.insert(blas_nodes_.end(), p.blas.nodes.begin(), p.blas.nodes.end());
        blas_triangles_.insert(blas_triangles_.end(), p.blas.triangles.begin(),
                               p.blas.triangles.end());

        if (want_attributes) {
            // Attributes are indexed in lockstep with triangles, so the
            // reflection shader needs no second lookup table.
            attributes_.reserve(blas_triangles_.size());
            for (const BvhTriangle& tri : p.blas.triangles) {
                TriangleAttributes a;
                const std::size_t i0 = static_cast<std::size_t>(tri.prim_id) * 3;
                if (i0 + 2 < p.geometry.indices.size()) {
                    const uint32_t v0 = p.geometry.indices[i0 + 0];
                    const uint32_t v1 = p.geometry.indices[i0 + 1];
                    const uint32_t v2 = p.geometry.indices[i0 + 2];
                    if (v0 < p.geometry.normals.size() && v1 < p.geometry.normals.size() &&
                        v2 < p.geometry.normals.size()) {
                        a.n0 = p.geometry.normals[v0];
                        a.n1 = p.geometry.normals[v1];
                        a.n2 = p.geometry.normals[v2];
                        a.base_color = p.geometry.colors[v0];
                    }
                }
                attributes_.push_back(a);
            }
        }

        entries_[p.handle] = entry;
        stats_.models++;
    }

    stats_.unique_triangles = static_cast<uint32_t>(blas_triangles_.size());
    stats_.blas_nodes = static_cast<uint32_t>(blas_nodes_.size());
    blas_dirty_ = true;

    const auto end = std::chrono::steady_clock::now();
    stats_.build_ms =
        std::chrono::duration<float, std::milli>(end - start).count();

    BRACKEN_LOG_INFO("rt",
                     "RtScene: {} models, {} unique triangles, {} BLAS nodes, {:.1f} MB, {:.1f} ms",
                     stats_.models, stats_.unique_triangles, stats_.blas_nodes,
                     static_cast<double>(blas_nodes_.size() * sizeof(BvhNode) +
                                         blas_triangles_.size() * sizeof(BvhTriangle) +
                                         attributes_.size() * sizeof(TriangleAttributes)) /
                         (1024.0 * 1024.0),
                     stats_.build_ms);
}

void RtScene::set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    BRACKEN_PROFILE_ZONE("RtScene::set_instances");
    if (entries_.empty()) {
        return;
    }

    std::vector<TlasInstance> list;
    list.reserve(instances.size());
    uint32_t drawn = 0;

    for (const auto& [handle, transform] : instances) {
        auto it = entries_.find(handle);
        if (it == entries_.end() || !it->second.valid) {
            continue; // no BLAS for this model: it simply casts no rays
        }
        const ModelEntry& e = it->second;

        // World AABB of the transformed model bounds. Transforming all eight
        // corners rather than the min/max pair: a rotated box's extents are
        // not the transformed extents, and getting that wrong makes the TLAS
        // cull geometry that is actually visible.
        glm::vec3 lo(std::numeric_limits<float>::max());
        glm::vec3 hi(std::numeric_limits<float>::lowest());
        for (int c = 0; c < 8; ++c) {
            const glm::vec3 corner{(c & 1) ? e.bounds_max.x : e.bounds_min.x,
                                   (c & 2) ? e.bounds_max.y : e.bounds_min.y,
                                   (c & 4) ? e.bounds_max.z : e.bounds_min.z};
            const glm::vec3 w = glm::vec3(transform * glm::vec4(corner, 1.0f));
            lo = glm::min(lo, w);
            hi = glm::max(hi, w);
        }

        TlasInstance inst;
        inst.world_to_model = glm::inverse(transform);
        inst.bounds_min = lo;
        inst.bounds_max = hi;
        inst.blas_node_offset = e.node_offset;
        inst.blas_triangle_offset = e.triangle_offset;
        list.push_back(inst);
        drawn += e.node_count;
    }

    tlas_ = build_tlas(list);
    stats_.instances = static_cast<uint32_t>(list.size());
    stats_.tlas_nodes = static_cast<uint32_t>(tlas_.nodes.size());
    stats_.drawn_triangles = drawn;
    tlas_dirty_ = true;
}

void RtScene::upload(gpu::GPUDevice& device) {
    BRACKEN_PROFILE_ZONE("RtScene::upload");

    if (blas_dirty_ && !blas_nodes_.empty()) {
        blas_node_buffer_ = gpu::GPUBuffer::create_static(
            device, gpu::GPUBuffer::Type::Storage, blas_nodes_.data(),
            blas_nodes_.size() * sizeof(BvhNode));
        blas_triangle_buffer_ = gpu::GPUBuffer::create_static(
            device, gpu::GPUBuffer::Type::Storage, blas_triangles_.data(),
            blas_triangles_.size() * sizeof(BvhTriangle));
        if (!attributes_.empty()) {
            attribute_buffer_ = gpu::GPUBuffer::create_static(
                device, gpu::GPUBuffer::Type::Storage, attributes_.data(),
                attributes_.size() * sizeof(TriangleAttributes));
        }
        blas_dirty_ = false;
    }

    if (tlas_dirty_ && !tlas_.nodes.empty()) {
        // Recreated rather than updated: the TLAS changes size whenever the
        // instance set does, and a static scene uploads it exactly once.
        tlas_node_buffer_ = gpu::GPUBuffer::create_static(
            device, gpu::GPUBuffer::Type::Storage, tlas_.nodes.data(),
            tlas_.nodes.size() * sizeof(BvhNode));
        tlas_instance_buffer_ = gpu::GPUBuffer::create_static(
            device, gpu::GPUBuffer::Type::Storage, tlas_.instances.data(),
            tlas_.instances.size() * sizeof(TlasInstance));
        tlas_dirty_ = false;
    }

    stats_.gpu_bytes = blas_nodes_.size() * sizeof(BvhNode) +
                       blas_triangles_.size() * sizeof(BvhTriangle) +
                       attributes_.size() * sizeof(TriangleAttributes) +
                       tlas_.nodes.size() * sizeof(BvhNode) +
                       tlas_.instances.size() * sizeof(TlasInstance);
}

SDL_GPUBuffer* RtScene::tlas_nodes() const {
    return tlas_node_buffer_ ? tlas_node_buffer_->handle() : nullptr;
}
SDL_GPUBuffer* RtScene::tlas_instances() const {
    return tlas_instance_buffer_ ? tlas_instance_buffer_->handle() : nullptr;
}
SDL_GPUBuffer* RtScene::blas_nodes() const {
    return blas_node_buffer_ ? blas_node_buffer_->handle() : nullptr;
}
SDL_GPUBuffer* RtScene::blas_triangles() const {
    return blas_triangle_buffer_ ? blas_triangle_buffer_->handle() : nullptr;
}
SDL_GPUBuffer* RtScene::triangle_attributes() const {
    return attribute_buffer_ ? attribute_buffer_->handle() : nullptr;
}

} // namespace bracken::render::rt
