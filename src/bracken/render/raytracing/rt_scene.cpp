#include "bracken/render/raytracing/rt_scene.hpp"

#include "bracken/core/hash.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <chrono>
#include <cstring>

namespace bracken::render::rt {
namespace {

/** Gather one model's triangles in model space, plus attributes if wanted. */
struct ModelGeometry {
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> triangle_colors;
    bool has_cpu_geometry = false;
};

ModelGeometry gather(const Model& model, bool want_attributes) {
    ModelGeometry g;
    for (const Mesh& mesh : model.meshes) {
        // A skinned mesh keeps its geometry in `skinned_vertices`; `vertices`
        // is empty for it. Reading only the latter meant every skinned model —
        // i.e. every animated character — silently got no BLAS, so it cast no
        // ray-traced shadow at all. The instance list is not the place that
        // dropped it; it never had anything to reference.
        //
        // The positions taken here are the BIND POSE. A skinned instance
        // therefore casts a bind-pose shadow that follows it around the world
        // without animating. Doing better means re-skinning and refitting the
        // BVH per frame; this at least gives the character a shadow.
        const std::size_t vertex_count =
            mesh.vertices.empty() ? mesh.skinned_vertices.size() : mesh.vertices.size();
        if (mesh.indices.empty() || vertex_count == 0) {
            continue; // uploaded models have had their CPU geometry released
        }
        g.has_cpu_geometry = true;
        const auto base = static_cast<uint32_t>(g.positions.size());
        for (const gpu::Vertex3D& v : mesh.vertices) {
            g.positions.push_back(v.position);
            if (want_attributes) {
                g.normals.push_back(v.normal);
            }
        }
        for (const gpu::SkinnedVertex& v : mesh.skinned_vertices) {
            g.positions.push_back(v.position);
            if (want_attributes) {
                g.normals.push_back(v.normal);
            }
        }
        const std::size_t triangle_count = mesh.indices.size() / 3;
        for (std::size_t tri = 0; tri < triangle_count; ++tri) {
            const uint32_t i0 = mesh.indices[tri * 3 + 0];
            const uint32_t i1 = mesh.indices[tri * 3 + 1];
            const uint32_t i2 = mesh.indices[tri * 3 + 2];
            g.indices.push_back(base + i0);
            g.indices.push_back(base + i1);
            g.indices.push_back(base + i2);
            if (want_attributes) {
                if (tri < mesh.rt_triangle_base_colors.size()) {
                    g.triangle_colors.push_back(mesh.rt_triangle_base_colors[tri]);
                } else if (i0 < mesh.vertices.size() && i1 < mesh.vertices.size() &&
                           i2 < mesh.vertices.size()) {
                    g.triangle_colors.push_back(
                        (mesh.vertices[i0].color + mesh.vertices[i1].color +
                         mesh.vertices[i2].color) /
                        3.0f * mesh.material.base_color_factor);
                } else {
                    g.triangle_colors.push_back(mesh.material.base_color_factor);
                }
            }
        }
    }
    return g;
}

/// The two hashes `set_instances` decides on, both from one pass over the list.
struct InstanceHashes {
    uint64_t content = 0;  ///< count + handles + transforms: "did anything move"
    uint64_t topology = 0; ///< count + handles only: "is this the same tree's worth of instances"
};

/**
 * Content and topology hashes of an instance list.
 *
 * Fields are hashed explicitly rather than as a byte blob over the pair: the
 * pair has padding between the handle and the matrix, and hashing
 * uninitialized padding makes the answer depend on allocator history rather
 * than on the scene.
 *
 * FNV-1a over 4-byte words, not bytes. At 2569 instances this walks ~44k
 * words; a byte-at-a-time walk of the same data is the difference between
 * ~10 us and ~40 us, and the whole point of this hash is to be much cheaper
 * than the ~490 us build it guards.
 *
 * The topology hash is a SEPARATE accumulator over the same walk rather than a
 * second pass: it costs one multiply-xor per instance and saves re-reading the
 * whole list. It deliberately hashes the handle SEQUENCE, not a set — a
 * reordered list has the same instances but a different slot-to-input mapping,
 * which the refit path relies on, so reordering must read as a topology change.
 */
InstanceHashes hash_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    InstanceHashes out{core::k_fnv1a_offset_basis_64, core::k_fnv1a_offset_basis_64};
    const auto mix = [](uint64_t& h, uint32_t word) {
        h ^= static_cast<uint64_t>(word);
        h *= core::k_fnv1a_prime_64;
    };
    // The count participates so that a truncated list cannot collide with the
    // longer one it is a prefix of.
    mix(out.content, static_cast<uint32_t>(instances.size()));
    mix(out.topology, static_cast<uint32_t>(instances.size()));
    for (const auto& [handle, transform] : instances) {
        mix(out.content, static_cast<uint32_t>(handle));
        mix(out.topology, static_cast<uint32_t>(handle));
        const auto* m = reinterpret_cast<const uint32_t*>(glm::value_ptr(transform));
        for (int i = 0; i < 16; ++i) {
            mix(out.content, m[i]);
        }
    }
    return out;
}

/**
 * World-space AABB and inverse transform of one instance.
 *
 * All eight corners are transformed rather than the min/max pair: a rotated
 * box's extents are not the transformed extents, and getting that wrong makes
 * the TLAS cull geometry that is actually visible.
 */
void fill_instance_placement(const glm::vec3& bounds_min, const glm::vec3& bounds_max,
                             const glm::mat4& transform, TlasInstance& out) {
    glm::vec3 lo(std::numeric_limits<float>::max());
    glm::vec3 hi(std::numeric_limits<float>::lowest());
    for (int c = 0; c < 8; ++c) {
        const glm::vec3 corner{(c & 1) ? bounds_max.x : bounds_min.x,
                               (c & 2) ? bounds_max.y : bounds_min.y,
                               (c & 4) ? bounds_max.z : bounds_min.z};
        const glm::vec3 w = glm::vec3(transform * glm::vec4(corner, 1.0f));
        lo = glm::min(lo, w);
        hi = glm::max(hi, w);
    }
    out.world_to_model = glm::inverse(transform);
    out.bounds_min = lo;
    out.bounds_max = hi;
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

    // The BLAS set is about to be renumbered, so every offset in the current
    // TLAS is stale and neither the "nothing moved" skip nor the refit path may
    // be allowed to keep it. Dropping the remembered hashes forces the next
    // set_instances through a full rebuild even if the scene has not moved a
    // millimetre — which is what happens when a host enables reflections
    // mid-run and every BLAS gains attributes.
    has_instance_hash_ = false;
    last_transforms_.clear();
    input_index_of_slot_.clear();
    refits_since_build_ = 0;
    last_build_sah_ = 0.0f;

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
    // between runs and the structure is reproducible. Each BLAS is converted
    // to the 64-byte traversal layout on the way in — the GPU never sees a
    // build node, and a BLAS's node links are relative to its own root, so the
    // conversion is per-model and the offsets stay a simple running total.
    std::vector<BvhNodeGpu> converted;
    for (Pending& p : pending) {
        if (p.blas.empty()) {
            continue;
        }
        convert_to_gpu_nodes(p.blas.nodes, converted, convert_scratch_);
        if (converted.empty()) {
            continue;
        }
        ModelEntry entry;
        entry.node_offset = static_cast<uint32_t>(blas_nodes_.size());
        entry.triangle_offset = static_cast<uint32_t>(blas_triangles_.size());
        entry.node_count = static_cast<uint32_t>(converted.size());
        entry.triangle_count = static_cast<uint32_t>(p.blas.triangles.size());
        entry.bounds_min = p.blas.bounds_min;
        entry.bounds_max = p.blas.bounds_max;
        entry.valid = true;

        blas_nodes_.insert(blas_nodes_.end(), converted.begin(), converted.end());
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
                        const std::size_t tri_index = static_cast<std::size_t>(tri.prim_id);
                        if (tri_index < p.geometry.triangle_colors.size()) {
                            a.base_color = p.geometry.triangle_colors[tri_index];
                        }
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
                     static_cast<double>(blas_nodes_.size() * sizeof(BvhNodeGpu) +
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

    const InstanceHashes hashes = hash_instances(instances);

    // (a) Nothing moved: keep the TLAS, keep the buffers, do no work.
    if (has_instance_hash_ && hashes.content == instance_hash_) {
        return;
    }

    // (b) Same models in the same order, moved. The size check is not
    // redundant with the topology hash — it is what makes the *indexing* below
    // safe rather than merely probable, since `last_transforms_` and
    // `input_index_of_slot_` are indexed by input position.
    const bool same_topology = has_instance_hash_ && hashes.topology == topology_hash_ &&
                               instances.size() == last_transforms_.size() && !tlas_.nodes.empty();
    const bool refit_budget_left = refits_since_build_ < k_tlas_max_refits_between_builds;

    bool refitted = false;
    if (same_topology && refit_budget_left) {
        refitted = refit_tlas_in_place(instances);
    }
    if (!refitted) {
        rebuild_tlas(instances); // (c), and the fallback when a refit degraded
    }

    instance_hash_ = hashes.content;
    topology_hash_ = hashes.topology;
    has_instance_hash_ = true;
    last_transforms_.resize(instances.size());
    for (std::size_t i = 0; i < instances.size(); ++i) {
        last_transforms_[i] = instances[i].second;
    }

    stats_.tlas_nodes = static_cast<uint32_t>(tlas_gpu_nodes_.size());
    tlas_dirty_ = true;
}

void RtScene::rebuild_tlas(std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    BRACKEN_PROFILE_ZONE("RtScene::rebuild_tlas");

    tlas_build_scratch_.clear();
    tlas_build_scratch_.reserve(instances.size());
    filtered_to_input_scratch_.clear();
    filtered_to_input_scratch_.reserve(instances.size());
    uint32_t drawn = 0;

    for (std::size_t i = 0; i < instances.size(); ++i) {
        const auto& [handle, transform] = instances[i];
        auto it = entries_.find(handle);
        if (it == entries_.end() || !it->second.valid) {
            continue; // no BLAS for this model: it simply casts no rays
        }
        const ModelEntry& e = it->second;

        TlasInstance inst;
        fill_instance_placement(e.bounds_min, e.bounds_max, transform, inst);
        inst.blas_node_offset = e.node_offset;
        inst.blas_triangle_offset = e.triangle_offset;
        tlas_build_scratch_.push_back(inst);
        filtered_to_input_scratch_.push_back(static_cast<uint32_t>(i));
        drawn += e.triangle_count;
    }

    tlas_ = build_tlas(tlas_build_scratch_);

    // Compose slot -> input from the builder's slot -> filtered permutation and
    // the filter's filtered -> input mapping, once, here. The refit path then
    // walks slots directly and never needs an inverse permutation.
    input_index_of_slot_.resize(tlas_.instances.size());
    for (std::size_t slot = 0; slot < tlas_.instances.size(); ++slot) {
        input_index_of_slot_[slot] = filtered_to_input_scratch_[tlas_.source_index[slot]];
    }

    // Convert to the traversal layout. Two linear passes over ~2N nodes for N
    // instances, against a binned-SAH build of the same list — a few percent
    // of a path that already runs only when the instance SET changed.
    convert_to_gpu_nodes(tlas_.nodes, tlas_gpu_nodes_, convert_scratch_);

    last_build_sah_ = sah_cost(tlas_.nodes);
    refits_since_build_ = 0;
    stats_.tlas_builds++;
    stats_.instances = static_cast<uint32_t>(tlas_build_scratch_.size());
    stats_.drawn_triangles = drawn;
}

bool RtScene::refit_tlas_in_place(std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    BRACKEN_PROFILE_ZONE("RtScene::refit_tlas");

    // Only the instances whose transform actually differs pay for an inverse
    // and eight corner transforms. That is the bulk of the saving in practice:
    // the playground moves one skinned character out of 2570 instances, so the
    // refit touches one placement and then does a pure bounds sweep.
    for (std::size_t slot = 0; slot < tlas_.instances.size(); ++slot) {
        const uint32_t input = input_index_of_slot_[slot];
        const glm::mat4& transform = instances[input].second;
        if (transform == last_transforms_[input]) {
            continue;
        }
        auto it = entries_.find(instances[input].first);
        if (it == entries_.end() || !it->second.valid) {
            // The handle sequence is unchanged by construction, so this is
            // unreachable unless the BLAS set was rebuilt underneath us — in
            // which case the offsets are stale too and a rebuild is the answer.
            return false;
        }
        fill_instance_placement(it->second.bounds_min, it->second.bounds_max, transform,
                                tlas_.instances[slot]);
    }

    refit_tlas(tlas_);

    // QUALITY GATE, measured rather than assumed. A refit keeps the topology
    // the instances had when they were last built, so its bounds only grow;
    // `sah_cost` is exactly the "how much more will a ray walk through" number
    // and it is one linear pass over the nodes we just touched, i.e. cache-hot
    // and far cheaper than the build it decides against. The counter backstop
    // in set_instances covers the case this metric cannot see: motion that
    // keeps the cost flat while the tree stops matching the scene.
    const float cost = sah_cost(tlas_.nodes);
    if (last_build_sah_ > 0.0f && cost > last_build_sah_ * k_tlas_refit_sah_rebuild_ratio) {
        return false; // caller rebuilds; the half-refit tree is simply replaced
    }

    // Reconvert and re-upload. THE 32-BYTE TREE STAYS AUTHORITATIVE and the
    // 64-byte form is re-derived, rather than refitting the uploaded form
    // directly: it keeps `convert_to_gpu_nodes` the single producer of every
    // byte the GPU sees (so `bvh_gpu_layout.test.cpp` still covers the refit
    // path for free), and it keeps `sah_cost` above reading the same structure
    // the builder produced. The price is two extra linear passes over ~2N
    // nodes, which is noise against the binned-SAH build being skipped.
    convert_to_gpu_nodes(tlas_.nodes, tlas_gpu_nodes_, convert_scratch_);

    ++refits_since_build_;
    stats_.tlas_refits++;
    return true;
}

void RtScene::upload(gpu::GPUDevice& device, SDL_GPUCommandBuffer* cmd) {
    BRACKEN_PROFILE_ZONE("RtScene::upload");

    if (blas_dirty_ && !blas_nodes_.empty()) {
        blas_node_buffer_ = gpu::GPUBuffer::create_static(
            device, gpu::GPUBuffer::Type::Storage, blas_nodes_.data(),
            blas_nodes_.size() * sizeof(BvhNodeGpu));
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

    if (tlas_dirty_ && !tlas_gpu_nodes_.empty()) {
        const std::size_t node_bytes = tlas_gpu_nodes_.size() * sizeof(BvhNodeGpu);
        const std::size_t instance_bytes = tlas_.instances.size() * sizeof(TlasInstance);

        if (cmd != nullptr) {
            // Persistent buffers, grown only when the instance count outgrows
            // them and otherwise written in place on the frame's command
            // buffer. The previous code created two buffers and submitted its
            // OWN command buffer here, in the middle of frame recording, every
            // time any instance moved.
            if (tlas_node_buffer_ == nullptr || tlas_node_capacity_ < node_bytes) {
                tlas_node_capacity_ = node_bytes + node_bytes / 2; // headroom
                tlas_node_buffer_ = gpu::GPUBuffer::create_dynamic(
                    device, gpu::GPUBuffer::Type::Storage, tlas_node_capacity_);
            }
            if (tlas_instance_buffer_ == nullptr || tlas_instance_capacity_ < instance_bytes) {
                tlas_instance_capacity_ = instance_bytes + instance_bytes / 2;
                tlas_instance_buffer_ = gpu::GPUBuffer::create_dynamic(
                    device, gpu::GPUBuffer::Type::Storage, tlas_instance_capacity_);
            }
            if (tlas_node_buffer_ && tlas_instance_buffer_) {
                tlas_node_buffer_->update(cmd, tlas_gpu_nodes_.data(), node_bytes);
                tlas_instance_buffer_->update(cmd, tlas_.instances.data(), instance_bytes);
                tlas_dirty_ = false;
            }
        } else {
            // No frame command buffer (a host calling set_instances outside a
            // frame). Immediate create-and-submit, as before.
            tlas_node_buffer_ = gpu::GPUBuffer::create_static(
                device, gpu::GPUBuffer::Type::Storage, tlas_gpu_nodes_.data(), node_bytes);
            tlas_instance_buffer_ = gpu::GPUBuffer::create_static(
                device, gpu::GPUBuffer::Type::Storage, tlas_.instances.data(), instance_bytes);
            tlas_node_capacity_ = node_bytes;
            tlas_instance_capacity_ = instance_bytes;
            tlas_dirty_ = false;
        }
    }

    stats_.blas_bytes = blas_nodes_.size() * sizeof(BvhNodeGpu) +
                        blas_triangles_.size() * sizeof(BvhTriangle);
    stats_.attribute_bytes = attributes_.size() * sizeof(TriangleAttributes);
    stats_.tlas_bytes = tlas_gpu_nodes_.size() * sizeof(BvhNodeGpu) +
                        tlas_.instances.size() * sizeof(TlasInstance);
    stats_.gpu_bytes = stats_.blas_bytes + stats_.attribute_bytes + stats_.tlas_bytes;
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
