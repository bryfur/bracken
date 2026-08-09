#include "bracken/render/raytracing/rt_scene.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/hash.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/render/lighting/alias_table.hpp"
#include "bracken/render/lighting/light_target.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>

namespace bracken::render::rt {
namespace {

/** Gather one model's triangles in model space, plus attributes if wanted. */
struct ModelGeometry {
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> triangle_colors;
    /// Per-triangle emitted radiance, in lockstep with `indices / 3`. Gathered
    /// UNCONDITIONALLY, like `triangle_opacity` below and for the same reason:
    /// an emitting triangle is a LIGHT, not a shading attribute, and the
    /// configuration that wants those lights most — local shadows on,
    /// reflections off — is exactly the one with no attribute buffer.
    ///
    /// `any_emissive` false means the model emits nothing anywhere, and the
    /// vector is RELEASED at the end of the gather rather than the walk being
    /// skipped: every `Pending` holds its geometry simultaneously, so 12 bytes
    /// per triangle across the whole model set is megabytes of transient peak
    /// for data that is all zero. The record walk's bounds check on this
    /// vector's size then skips a released model for free.
    std::vector<glm::vec3> triangle_emissive;
    /// Per-triangle opacity in [0, 1], in lockstep with `indices / 3`. Unlike
    /// the COLOURS above it is gathered UNCONDITIONALLY, not under
    /// `want_attributes`: it rides `BvhTriangle::opacity` rather than the
    /// attribute buffer, and the configuration that needs it most — shadows on,
    /// reflections off — is exactly the one where no attribute buffer exists.
    /// `any_cutout` false means every triangle is opaque and the fill is skipped.
    std::vector<float> triangle_opacity;
    bool any_emissive = false;
    bool any_cutout = false;
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
        // SHADOW-ONLY PROXIES DO NOT BELONG IN THE ACCELERATION STRUCTURE.
        //
        // `shadow_only` marks a mesh the main pass never draws — for procedural
        // trees, a low-poly ellipsoid standing in for thousands of leaf quads so
        // the RASTER shadow pass avoids alpha-test overdraw (tree_generator.cpp).
        // That trade does not exist here: the traced walk has the real leaf
        // cards in the same BLAS, so including the proxy meant a tree cast its
        // shadow TWICE, from two different silhouettes — a doubling the raster
        // path never has, because there the two are mutually exclusive by
        // construction (`cast_shadows=false` removes the leaves from exactly the
        // pass the proxy is drawn in).
        //
        // It is worse than a double shadow, because a BLAS is shared by both
        // walk types: closest-hit rays saw the proxy too, so reflections and GI
        // probes were shading an invisible grey ellipsoid floating in a canopy.
        //
        // Dropping it fixes both, and is the whole of the fix here — the
        // converse flag is deliberately NOT acted on. `cast_shadows=false` leaf
        // cards must stay: they are the real, visible geometry, so a closest-hit
        // ray has to find them, and suppressing them for any-hit walks alone
        // needs a PER-TRIANGLE flag rather than a gather-time skip (one model's
        // meshes share one BLAS). That flag is the same `BvhTriangle` byte the
        // alpha-cutout work wants, and both should land together — until then a
        // leaf card occludes as a solid quad, which is the reported
        // tree-shadow-is-too-dark behaviour and is tracked with it.
        if (mesh.shadow_only) {
            continue;
        }
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
        // `cast_shadows == false` is NOT honoured here, and that is the one
        // remaining half of the raster contract this gather does not implement.
        // Those meshes are the real, visible leaf cards: a closest-hit ray must
        // find them or reflections and GI lose the canopy, so suppressing them
        // for ANY-HIT walks alone needs a per-triangle flag rather than a
        // gather-time skip. The opacity below is the better answer anyway — it
        // makes a leaf card occlude in proportion to how much of it is actually
        // leaf, which is what the raster flag was approximating by turning the
        // whole mesh off and substituting a proxy.
        const bool mesh_cutout = !mesh.rt_triangle_opacity.empty();
        for (std::size_t tri = 0; tri < triangle_count; ++tri) {
            const uint32_t i0 = mesh.indices[tri * 3 + 0];
            const uint32_t i1 = mesh.indices[tri * 3 + 1];
            const uint32_t i2 = mesh.indices[tri * 3 + 2];
            g.indices.push_back(base + i0);
            g.indices.push_back(base + i1);
            g.indices.push_back(base + i2);

            // Opacity is gathered for EVERY model, opaque or not, so the
            // per-triangle array stays indexable by `prim_id` without a second
            // mapping. `any_cutout` is what lets an all-opaque model skip the
            // fill entirely, and it is the common case.
            const float opacity =
                mesh_cutout && tri < mesh.rt_triangle_opacity.size()
                    ? mesh.rt_triangle_opacity[tri]
                    : 1.0f;
            g.triangle_opacity.push_back(opacity);
            g.any_cutout = g.any_cutout || opacity < 1.0f;

            // Emission, from the same per-triangle bake and with the same
            // fallback shape as the colour below: the mesh-wide factor when the
            // proxy is absent (an untextured emissive mesh bakes the factor for
            // every triangle, so absence here means the model was loaded
            // without retention and only the factor survives).
            //
            // Outside `want_attributes` because the RECORDS derived from it are
            // outside it — see the record walk in build_blas_for_models.
            glm::vec3 emissive(0.0f);
            if (tri < mesh.rt_triangle_emissive.size()) {
                emissive = mesh.rt_triangle_emissive[tri];
            } else {
                emissive = mesh.material.emissive_factor;
            }
            g.triangle_emissive.push_back(emissive);
            g.any_emissive =
                g.any_emissive || emissive.r > 0.0f || emissive.g > 0.0f || emissive.b > 0.0f;

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
    // A model that emits nowhere carries an all-zero array, and `pending` holds
    // every model's geometry at once — so releasing it here is what keeps the
    // now-unconditional fill above proportional to the emitters that actually
    // exist instead of to the whole model set. Swap rather than `clear()`:
    // clear() keeps the capacity, which is the memory this is about.
    if (!g.any_emissive) {
        std::vector<glm::vec3>().swap(g.triangle_emissive);
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
InstanceHashes hash_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances,
                              uint32_t skinned_first) {
    InstanceHashes out{core::k_fnv1a_offset_basis_64, core::k_fnv1a_offset_basis_64};
    const auto mix = [](uint64_t& h, uint32_t word) {
        h ^= static_cast<uint64_t>(word);
        h *= core::k_fnv1a_prime_64;
    };
    // The count participates so that a truncated list cannot collide with the
    // longer one it is a prefix of.
    mix(out.content, static_cast<uint32_t>(instances.size()));
    mix(out.topology, static_cast<uint32_t>(instances.size()));
    // The skinned partition point goes into the TOPOLOGY hash and deliberately
    // NOT into the content hash. It decides a bit baked into every built
    // instance record (k_tlas_skinned_flag), so a change to it must force a
    // rebuild rather than a refit — a refit updates bounds and leaves the flags
    // as the last build wrote them. It says nothing about where anything IS,
    // which is what the content hash is for: mixing it there would be a second
    // way to answer "did something move", and the movement answer must stay
    // exactly what it was before capsules existed.
    mix(out.topology, skinned_first);
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
    emissive_records_.clear();
    emissive_total_power_ = 0.0f;
    emissive_dirty_ = true;
    ++emissive_generation_;
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
                                     Pending& p = pending[i];
                                     p.blas = build_blas(p.geometry.positions, p.geometry.indices);
                                     // Alpha-cutout coverage, keyed by prim_id
                                     // — the SOURCE triangle ordinal, so this
                                     // survives whatever order the builder
                                     // emitted leaves in. Skipped entirely for
                                     // an all-opaque model, which is most of
                                     // them; `build_blas` already left 1.0.
                                     if (!p.geometry.any_cutout) {
                                         continue;
                                     }
                                     const std::vector<float>& op =
                                         p.geometry.triangle_opacity;
                                     for (BvhTriangle& t : p.blas.triangles) {
                                         if (t.prim_id < op.size()) {
                                             t.opacity = op[t.prim_id];
                                         }
                                     }
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
                        if (tri_index < p.geometry.triangle_emissive.size()) {
                            a.emissive = p.geometry.triangle_emissive[tri_index];
                            a.emissive_luminance = lighting::target_luminance(a.emissive);
                        }
                    }
                }
                attributes_.push_back(a);
            }
        }

        // R6: an emitting triangle is also a LIGHT. OUTSIDE the attribute block
        // above, and that placement is load-bearing rather than tidy. A record
        // carries its own radiance/area/power and the Full-DI pass never binds
        // the attribute buffer at all, so gating this walk on `want_attributes`
        // made the one configuration that wants emissive lights most — local
        // shadows on, reflections off — the one configuration that silently had
        // none of them. Exactly the reasoning that put `BvhTriangle::opacity`
        // outside it too.
        //
        // The vertices come from the BVH triangle rather than from the index
        // buffer because that is where they are already in origin+edge form,
        // and because it guarantees the record names a triangle the shadow ray
        // can actually hit.
        //
        // The test is a scale-FREE radiance floor, not a power floor: power is
        // radiance x AREA, and area is the one factor of it an instance
        // transform changes. `k_emissive_min_power` is applied in world space,
        // where the world triangle exists — see rebuild_emissive_records.
        //
        // A LATENT HAZARD, inert today and worth naming before it is not: this
        // walks REFERENCES, not source triangles. The binned-SAH builder emits
        // each source triangle exactly once, so `prim_id` is unique across
        // `p.blas.triangles` and each emitter contributes one record. A builder
        // that SPATIALLY SPLITS — duplicating a reference across a plane — would
        // emit one `prim_id` k times, and this loop would push the whole
        // UNCLIPPED triangle at full emission k times over. That is a k-fold
        // brightening of precisely the largest emitters, which are the ones a
        // spatial split targets. Anyone adding one has to de-duplicate here.
        for (const BvhTriangle& tri : p.blas.triangles) {
            const std::size_t ti = tri.prim_id;
            if (ti >= p.geometry.triangle_emissive.size()) {
                continue; // a model that emits nowhere released the array
            }
            const glm::vec3 e = p.geometry.triangle_emissive[ti];
            if (!(lighting::target_luminance(e) > k_emissive_min_luminance)) {
                continue;
            }
            entry.emissive.push_back(ModelEmissiveTriangle{tri.v0, tri.e1, tri.e2, e});
        }

        stats_.emissive_triangles += static_cast<uint32_t>(entry.emissive.size());
        entries_[p.handle] = std::move(entry);
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

void RtScene::set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances,
                            uint32_t skinned_first) {
    BRACKEN_PROFILE_ZONE("RtScene::set_instances");
    if (entries_.empty()) {
        return;
    }
    skinned_first_ = std::min<uint32_t>(skinned_first, static_cast<uint32_t>(instances.size()));

    const InstanceHashes hashes = hash_instances(instances, skinned_first_);

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

    // R6. The emissive records are WORLD SPACE, so they are stale whenever any
    // instance moved — the refit case included, which is why this is not gated
    // on `refitted`. What the topology decides is whether record INDICES move:
    // a refit keeps the instance sequence, so record k is the same triangle it
    // was last frame and a reservoir may carry it; a rebuild renumbers, so the
    // epoch advances and emissive history is dropped. Dropping it is CORRECT
    // rather than a limitation — the alternative is a reservoir pointing at
    // whatever triangle now occupies index k, which is the exact failure the
    // light remap exists to prevent for analytic lights.
    rebuild_emissive_records(instances, !refitted);

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
        // The skinned flag rides the offset's high bit — see k_tlas_skinned_flag
        // in bvh.hpp for why it is a bit and not a field, and for the ceiling
        // that makes it free. The caller partitions the list, so this is a
        // comparison against the partition point rather than a per-instance
        // lookup. Asserted rather than masked: an offset that reached the sign
        // bit would silently mark real geometry invisible to every shadow ray.
        BRACKEN_ASSERT(e.node_offset <= k_tlas_max_blas_nodes,
                       "BLAS node offset {} has reached the skinned-flag bit", e.node_offset);
        inst.blas_node_offset = e.node_offset;
        if (i >= skinned_first_) {
            inst.blas_node_offset |= k_tlas_skinned_flag;
        }
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
        // Placement only — `fill_instance_placement` writes the bounds and the
        // inverse transform and never touches `blas_node_offset`, so the
        // skinned flag the last BUILD stamped survives every refit. That is the
        // invariant that makes the flag safe to carry in the offset at all: a
        // refit path that rewrote the offset would strip it, and the character
        // would silently return to casting its bind pose the frame after the
        // first refit. Flag changes force a rebuild through the topology hash.
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

namespace {

/// Place one model-space emissive triangle into world space through `transform`.
/// One definition, called by both the rebuild and the refresh path, because the
/// refresh's whole correctness claim is that it produces the record the rebuild
/// would have.
void emissive_place(const ModelEmissiveTriangle& src, const glm::mat4& transform,
                    EmissiveTriangle& out) {
    out.v0 = glm::vec3(transform * glm::vec4(src.v0, 1.0f));
    // Edges are DIRECTIONS: transformed with w = 0, so translation does not
    // apply and a scaled instance's area still comes out right below.
    out.e1 = glm::vec3(transform * glm::vec4(src.e1, 0.0f));
    out.e2 = glm::vec3(transform * glm::vec4(src.e2, 0.0f));
    out.radiance = src.radiance;
    out.area = 0.5f * glm::length(glm::cross(out.e1, out.e2));
    out.power = lighting::target_luminance(out.radiance) * out.area;
}

} // namespace

void RtScene::rebuild_emissive_records(
    std::span<const std::pair<ModelHandle, glm::mat4>> instances, bool topology_changed) {
    BRACKEN_PROFILE_ZONE("RtScene::rebuild_emissive_records");

    if (stats_.emissive_triangles == 0) {
        // No model emits. Keep the fast path a single integer compare: the
        // overwhelming majority of scenes never allocate a record.
        if (!emissive_records_.empty()) {
            emissive_records_.clear();
            emissive_sources_.clear();
            emissive_instances_.clear();
            emissive_total_power_ = 0.0f;
            emissive_dirty_ = true;
            ++emissive_generation_;
            stats_.emissive_epoch++;
        }
        stats_.emissive_records = 0;
        return;
    }

    // THE REFIT ANALOGUE, and it is not an optimization so much as the
    // difference between the feature being affordable and not. Every emitter's
    // world triangles are stale whenever anything moves, and "anything moves"
    // is every frame in a scene with one animated character. Rebuilding all
    // ~8.7 k playground records each time cost 0.38 ms of MAIN-THREAD time in
    // `RTInstances`, a pass that was already the most expensive CPU pass in the
    // frame. Refreshing only the records whose own instance moved costs a
    // fraction of that, and the village's 8.6 k static window triangles — which
    // are the bulk — are simply not touched.
    if (!topology_changed && refresh_emissive_records(instances)) {
        stats_.emissive_records = static_cast<uint32_t>(emissive_records_.size());
        return;
    }

    emissive_records_.clear();
    emissive_sources_.clear();
    emissive_instances_.clear();
    stats_.emissive_records_below_power = 0;

    // Walk the INPUT instance order rather than the TLAS slot order: the slot
    // permutation is the builder's and changes with the tree, while the input
    // order is the frame packet's and is what the topology hash certifies. Tying
    // record indices to the packet is what makes "same topology => same record
    // index" true.
    for (std::size_t i = 0; i < instances.size(); ++i) {
        const auto& [handle, transform] = instances[i];
        auto it = entries_.find(handle);
        if (it == entries_.end() || !it->second.valid || it->second.emissive.empty()) {
            continue;
        }
        EmissiveInstance slice;
        slice.input_index = static_cast<uint32_t>(i);
        slice.record_begin = static_cast<uint32_t>(emissive_records_.size());
        for (const ModelEmissiveTriangle& src : it->second.emissive) {
            EmissiveTriangle rec;
            emissive_place(src, transform, rec);
            // THE CALIBRATED THRESHOLD, IN THE UNITS IT IS DOCUMENTED IN.
            // `k_emissive_min_power` is linear radiance x WORLD units squared,
            // and area is the one factor of power an instance transform
            // changes: a model authored in metres and instanced at 100x has a
            // world area ten thousand times its model area. Testing MODEL-space
            // power — which is what this did until the units were reconciled —
            // made the effective world floor `k_emissive_min_power * s^2`, an
            // invisible scale-dependent cut that rejected exactly the small lit
            // windows this phase set out to light. The prefilter at BLAS build
            // is a scale-free radiance floor for that reason, and the power
            // decision belongs here, where the world triangle exists. It
            // subsumes the old degenerate-transform check: a collapsed instance
            // has zero area and therefore zero power.
            if (!(rec.power >= k_emissive_min_power)) {
                ++stats_.emissive_records_below_power;
                continue;
            }
            emissive_records_.push_back(rec);
            emissive_sources_.push_back(src);
        }
        slice.record_count = static_cast<uint32_t>(emissive_records_.size()) - slice.record_begin;
        if (slice.record_count > 0) {
            emissive_instances_.push_back(slice);
        }
    }

    // OVER BUDGET: keep the brightest, and — load-bearingly — keep them IN
    // ORDER. Sorting the survivors by power is the obvious reading of "keep the
    // brightest" and it would destroy the property the refresh path above
    // stands on: that each instance owns a CONTIGUOUS slice. So the cut is a
    // power THRESHOLD (nth_element over a copy of the powers) applied as a
    // stable filter — NOT a sort.
    //
    // TIES AT THE THRESHOLD ARE NOT A CORNER CASE HERE. `power` is
    // luminance(radiance) * area, and `emissive_place` transforms the edges
    // with w = 0, so two instances of one model differing only in TRANSLATION
    // produce BIT-IDENTICAL powers. A pack of same-orientation lampposts is
    // therefore a tie group hundreds of records wide. A strict `>` filter alone
    // drops the whole group the cut lands in — keeping far fewer than the
    // budget allows while the arithmetic still claims `size - cap` were
    // dropped. So the tied group is admitted in record order until the budget
    // is full, which fills the cap exactly and keeps every slice contiguous
    // because both decisions are made inside the one in-order walk.
    stats_.emissive_records_dropped = 0;
    if (emissive_records_.size() > k_max_emissive_records) {
        const std::size_t before_size = emissive_records_.size();

        emissive_weight_scratch_.resize(before_size);
        for (std::size_t i = 0; i < before_size; ++i) {
            emissive_weight_scratch_[i] = emissive_records_[i].power;
        }
        const auto cut = emissive_weight_scratch_.begin() +
                         static_cast<std::ptrdiff_t>(k_max_emissive_records);
        std::nth_element(emissive_weight_scratch_.begin(), cut, emissive_weight_scratch_.end(),
                         std::greater<float>());
        const float threshold = *cut;

        // Everything strictly above the nth element sits in the first
        // k_max_emissive_records slots of the partition, so this is <= the cap
        // by construction and the quota below never underflows. Every power
        // here is finite and >= k_emissive_min_power (filtered at the push
        // above), so there is no NaN to reason about in the ordering.
        const auto strictly_greater = static_cast<std::size_t>(
            std::count_if(emissive_weight_scratch_.begin(), emissive_weight_scratch_.end(),
                          [threshold](float w) { return w > threshold; }));
        BRACKEN_ASSERT(strictly_greater <= k_max_emissive_records);
        std::size_t tie_quota = k_max_emissive_records - strictly_greater;

        std::size_t out = 0;
        for (EmissiveInstance& slice : emissive_instances_) {
            const uint32_t begin = slice.record_begin;
            const uint32_t count = slice.record_count;
            slice.record_begin = static_cast<uint32_t>(out);
            uint32_t kept = 0;
            for (uint32_t k = 0; k < count; ++k) {
                const std::size_t from = static_cast<std::size_t>(begin) + k;
                const float power = emissive_records_[from].power;
                if (!(power > threshold)) {
                    if (power < threshold || tie_quota == 0) {
                        continue;
                    }
                    --tie_quota; // exactly at the threshold, budget still open
                }
                emissive_records_[out] = emissive_records_[from];
                emissive_sources_[out] = emissive_sources_[from];
                ++out;
                ++kept;
            }
            slice.record_count = kept;
        }
        BRACKEN_ASSERT(out <= k_max_emissive_records);
        emissive_records_.resize(out);
        emissive_sources_.resize(out);
        std::erase_if(emissive_instances_,
                      [](const EmissiveInstance& s) { return s.record_count == 0; });

        // Report what was actually KEPT rather than what the budget arithmetic
        // predicted. With the tie admission above the two agree, and the
        // recount is what keeps them agreeing if this cut is ever changed again
        // — the no-silent-drops rule is about the real number.
        const std::size_t dropped = before_size - out;
        stats_.emissive_records_dropped = static_cast<uint32_t>(dropped);

        if (!emissive_over_budget_logged_) {
            emissive_over_budget_logged_ = true;
            BRACKEN_LOG_WARN("rt",
                             "RtScene: {} emissive light records over the {} budget; keeping the "
                             "brightest. Logged once; RtStats::emissive_records_dropped tracks it "
                             "per rebuild.",
                             dropped, k_max_emissive_records);
        }
    }

    rebuild_emissive_alias();
    stats_.emissive_records = static_cast<uint32_t>(emissive_records_.size());
    emissive_dirty_ = true;
    ++emissive_generation_;
    if (topology_changed) {
        stats_.emissive_epoch++;
    }
}

bool RtScene::refresh_emissive_records(
    std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    BRACKEN_PROFILE_ZONE("RtScene::refresh_emissive_records");
    if (emissive_instances_.empty() || emissive_records_.size() != emissive_sources_.size()) {
        return false; // nothing built yet, or the two arrays disagree: rebuild
    }
    if (last_transforms_.size() != instances.size()) {
        return false; // the caller's own refit precondition; do not guess
    }

    bool moved = false;
    // Rigid motion preserves area, hence power, hence the alias table — so the
    // table is only rebuilt when a power actually changes, which in practice
    // means a SCALED emitter. A relative epsilon rather than an exact compare:
    // re-transforming through a slightly different matrix perturbs the cross
    // product in the last bits, and rebuilding an 8 k alias table because a
    // float moved by 1 ULP would give the whole optimization back.
    bool powers_changed = false;
    for (const EmissiveInstance& slice : emissive_instances_) {
        const uint32_t input = slice.input_index;
        if (input >= instances.size()) {
            return false;
        }
        const glm::mat4& transform = instances[input].second;
        if (transform == last_transforms_[input]) {
            continue;
        }
        moved = true;
        for (uint32_t k = 0; k < slice.record_count; ++k) {
            const std::size_t idx = static_cast<std::size_t>(slice.record_begin) + k;
            const float before = emissive_records_[idx].power;
            // From the MODEL-SPACE source, never from the previous world-space
            // record: composing transforms frame after frame would accumulate
            // drift for as long as the scene runs.
            emissive_place(emissive_sources_[idx], transform, emissive_records_[idx]);
            const float after = emissive_records_[idx].power;
            // MEMBERSHIP, not just weight. The power floor is applied in world
            // space, so it depends on the instance transform and a scaled
            // emitter can cross it without the record set changing shape. This
            // path cannot express that: its whole contract is that record
            // indices are preserved, so it can neither add the record back nor
            // remove it. Hand the question to the rebuild instead — nothing
            // else would, because the topology hash is handles and count only,
            // and the drop would otherwise have hysteresis: an instance that
            // scaled back up would never recover its records.
            if ((before >= k_emissive_min_power) != (after >= k_emissive_min_power)) {
                return false; // caller rebuilds
            }
            if (std::abs(after - before) > 1e-3f * std::max(before, 1e-6f)) {
                powers_changed = true;
            }
        }
    }

    if (!moved) {
        return true; // nothing emissive moved; the uploaded buffer is still right
    }
    if (powers_changed) {
        rebuild_emissive_alias();
    }
    emissive_dirty_ = true;
    ++emissive_generation_;
    return true;
}

void RtScene::rebuild_emissive_alias() {
    // Power-weighted Vose, built INTO the records — one bucket per record, so
    // there is no second array and no second shader binding.
    emissive_weight_scratch_.resize(emissive_records_.size());
    for (std::size_t i = 0; i < emissive_records_.size(); ++i) {
        emissive_weight_scratch_[i] = emissive_records_[i].power;
    }
    emissive_total_power_ =
        lighting::build_alias_table(emissive_weight_scratch_, emissive_alias_scratch_);
    for (std::size_t i = 0; i < emissive_records_.size(); ++i) {
        emissive_records_[i].alias_prob = emissive_alias_scratch_[i].prob;
        emissive_records_[i].alias_index = emissive_alias_scratch_[i].alias;
    }
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

    // Emissive records ride the same persistent-grow-only pattern the TLAS
    // does, and for the same reason: they change whenever an emitter moves,
    // which in a scene with a hovering orb is every frame.
    if (emissive_dirty_ && !emissive_records_.empty()) {
        const std::size_t bytes = emissive_records_.size() * sizeof(EmissiveTriangle);
        if (cmd != nullptr) {
            if (emissive_buffer_ == nullptr || emissive_capacity_ < bytes) {
                emissive_capacity_ = bytes + bytes / 2;
                // ComputeStorage, not Storage: rt_local_shadow.comp.hlsl reads
                // this through a READ-WRITE slot (its eight read-only ones were
                // spent before R6 existed), and SDL3 refuses a read-write
                // binding on a buffer without COMPUTE_STORAGE_WRITE. Nothing
                // ever writes it.
                emissive_buffer_ = gpu::GPUBuffer::create_dynamic(
                    device, gpu::GPUBuffer::Type::ComputeStorage, emissive_capacity_);
            }
            if (emissive_buffer_) {
                emissive_buffer_->update(cmd, emissive_records_.data(), bytes);
                emissive_dirty_ = false;
            }
        } else {
            emissive_buffer_ = gpu::GPUBuffer::create_static(
                device, gpu::GPUBuffer::Type::ComputeStorage, emissive_records_.data(), bytes);
            emissive_capacity_ = bytes;
            emissive_dirty_ = false;
        }
    }

    stats_.emissive_bytes = emissive_records_.size() * sizeof(EmissiveTriangle);
    stats_.blas_bytes = blas_nodes_.size() * sizeof(BvhNodeGpu) +
                        blas_triangles_.size() * sizeof(BvhTriangle);
    stats_.attribute_bytes = attributes_.size() * sizeof(TriangleAttributes);
    stats_.tlas_bytes = tlas_gpu_nodes_.size() * sizeof(BvhNodeGpu) +
                        tlas_.instances.size() * sizeof(TlasInstance);
    stats_.gpu_bytes =
        stats_.blas_bytes + stats_.attribute_bytes + stats_.tlas_bytes + stats_.emissive_bytes;
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
SDL_GPUBuffer* RtScene::emissive_records() const {
    return (emissive_buffer_ && !emissive_records_.empty()) ? emissive_buffer_->handle() : nullptr;
}

} // namespace bracken::render::rt
