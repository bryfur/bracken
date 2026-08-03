#pragma once

// Emissive triangles as LIGHTS — the scene-level record set the froxel-RIS
// local-shadow pass draws its second candidate stream from.
// R6, docs/rt_lighting_plan.md §5.7.
//
// TWO LEVELS, and the split is what makes this affordable.
//
//  * PER MODEL, once at BLAS build time: which of a model's triangles emit
//    enough to be worth sampling, in MODEL space (`ModelEmissiveTriangle`).
//    This is the expensive walk — every triangle of every loaded model — and it
//    runs exactly as often as `RtScene::build_blas_for_models` does.
//  * PER SCENE, when the instance list changes: those triangles pushed through
//    each instance's transform into world space, power-sorted under a cap, and
//    given a Vose alias row each (`EmissiveTriangle`). This is O(emissive
//    triangles x instances of their models), not O(scene triangles).
//
// THE THRESHOLD, and why it is on POWER rather than on radiance. A record costs
// 64 bytes, one alias bucket, and a share of every candidate draw in the frame;
// what earns that is the light the triangle actually delivers, which is
// radiance x area. A dim wall-sized panel matters and a brilliant speck does
// not. `k_emissive_min_power` is stated in the engine's own units (linear
// radiance x world units squared) and documented at its definition.
//
// WHAT THIS IS NOT. It is not a second scene representation (§8's standing
// rule): the records are derived from the BLAS triangles that already exist,
// they are rebuilt from them rather than maintained, and nothing traces against
// them — they are sampled, and the ray goes into the same TLAS as every other.

#include "bracken/render/lighting/light_target.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

namespace bracken::render::rt {

/**
 * One emitting triangle of ONE MODEL, in model space, kept next to the model's
 * BLAS entry.
 *
 * Stored as origin + two edges rather than three vertices because that is the
 * form both consumers want: the world-space flatten transforms one point and
 * two vectors (an edge needs no translation), and the shader's uniform-area
 * sample is `v0 + e1*a + e2*b` directly.
 */
struct ModelEmissiveTriangle {
    glm::vec3 v0{0.0f};
    glm::vec3 e1{0.0f};
    glm::vec3 e2{0.0f};
    /// Linear RGB emitted radiance, from the material proxy's per-triangle bake.
    glm::vec3 radiance{0.0f};
};

/**
 * One emitting triangle of the SCENE, in world space — the 64-byte GPU record,
 * mirrored field for field by `EmissiveRecord` in `shaders/rt_emissive.hlsli`.
 *
 * THE ALIAS ROW RIDES IN THE RECORD. Vose's table has exactly as many buckets
 * as it has entries, so a separate table buffer would be a second array indexed
 * by the same integer — and the pass that reads it had already spent all eight
 * of SDL3 GPU's read-only storage-buffer slots before this phase existed. One
 * buffer, two loads, no header trick.
 */
struct EmissiveTriangle {
    glm::vec3 v0{0.0f};
    /// World-space area. The sampling pdf is per-triangle-uniform, so the
    /// shader needs this to convert between area and discrete measure.
    float area = 0.0f;
    glm::vec3 e1{0.0f};
    /// Vose bucket: probability of keeping this index when landed on.
    float alias_prob = 1.0f;
    glm::vec3 e2{0.0f};
    /// Vose bucket: where the rest of this bucket's mass goes.
    uint32_t alias_index = 0;
    glm::vec3 radiance{0.0f};
    /// `target_luminance(radiance) * area` — the alias weight, kept so the
    /// shader can recover this record's pdf (`power / total_power`) with one
    /// load instead of a luminance dot and a cross product.
    float power = 0.0f;
};

static_assert(sizeof(EmissiveTriangle) == 64,
              "EmissiveTriangle is the GPU record; keep it 16-byte aligned and in step with "
              "shaders/rt_emissive.hlsli");

/**
 * Minimum `luminance(radiance) * area` for a triangle to become a light.
 *
 * DELIBERATELY LOW, and it took a measurement to find out why it has to be. The
 * tempting formulation — "deliver at least X at a reference distance" — reads
 * well and is wrong, because a per-TRIANGLE threshold cannot tell a small
 * bright emitter from a large dim one once tessellation is involved. The
 * playground's orbs emit 2.4/1.7/0.55 (luminance 1.77) over a radius-8 sphere
 * cut into 140 facets: total power ~1400, per-facet power ~5. Any threshold
 * high enough to reject a dim 1 m^2 panel also rejects every facet of the
 * brightest object in the scene.
 *
 * So this constant does one job only: keep the flatten off the ~600 k triangles
 * that emit EXACTLY NOTHING. Deciding what is worth sampling is the CAP's job
 * below, where the power sort can compare records against each other instead of
 * against an absolute number that has no scene-independent meaning.
 */
inline constexpr float k_emissive_min_power = 1.0f;

/**
 * Hard cap on scene-level records.
 *
 * 8192 x 64 B = 512 KB, noise next to a 30 MB acceleration structure. The
 * figure is sized from measurement rather than aesthetics: the playground
 * village flattens to ~8.7 k records (4962 emitting model triangles across 52
 * models, instanced 2570 times — mostly lit windows on the building pack), so a
 * 4096 cap would throw away half of a scene this plan explicitly set out to
 * light ("lit windows at dusk bouncing onto the street").
 *
 * Over-budget is NOT silent: the flatten keeps the BRIGHTEST records by power
 * and reports the drop in `RtSceneStats::emissive_records_dropped`, per the
 * engine's no-silent-drops rule. Dropping the dimmest is also the RIGHT
 * degradation for a power-weighted alias table — those records were the ones
 * the sampler would almost never have drawn.
 */
inline constexpr uint32_t k_max_emissive_records = 8192;

// ---------------------------------------------------------------------------
// The CPU mirror of shaders/rt_emissive.hlsli.
//
// Same rule as light_target.hpp, reservoir.hpp and cluster_index.hpp: the
// claims these functions make — that the sample is uniform over the triangle,
// that p̂ is the same physical quantity the analytic path's p̂ is — are
// MATHEMATICAL and invisible in a screenshot, so they are pinned by a test that
// needs no device, and transliterated into HLSL exactly once.
// ---------------------------------------------------------------------------

/**
 * A point distributed uniformly over the triangle. Mirror of
 * `rtEmissiveSamplePoint`: the square-root parametrization, (u, v) mapped to
 * barycentrics (1 - sqrt(u), v*sqrt(u)) on the two edges from `v0`.
 */
[[nodiscard]] inline glm::vec3 emissive_sample_point(const EmissiveTriangle& r, glm::vec2 xi) {
    const float su = std::sqrt(std::clamp(xi.x, 0.0f, 1.0f));
    const float b1 = 1.0f - su;
    const float b2 = std::clamp(xi.y, 0.0f, 1.0f) * su;
    return r.v0 + r.e1 * b1 + r.e2 * b2;
}

/// Mirror of `EmissiveSample`.
struct EmissiveShading {
    glm::vec3 l{0.0f, 1.0f, 0.0f};
    glm::vec3 radiance{0.0f};
    float dist = 0.0f;
    bool valid = false;
};

/// Mirror of `rtEmissiveShading`, including the two-sided cosine and the
/// `max(d^2, area)` solid-angle clamp — both documented at that function.
[[nodiscard]] inline EmissiveShading emissive_shading(const EmissiveTriangle& r,
                                                      const glm::vec3& p) {
    EmissiveShading s;
    const glm::vec3 c = r.v0 + (r.e1 + r.e2) * (1.0f / 3.0f);
    const glm::vec3 to_light = c - p;
    const float d2 = glm::dot(to_light, to_light);
    if (d2 < 1e-8f || r.area <= 0.0f) {
        return s;
    }
    const float d = std::sqrt(d2);
    s.l = to_light / d;
    s.dist = d;

    const glm::vec3 n_raw = glm::cross(r.e1, r.e2);
    const float n_len = glm::length(n_raw);
    if (n_len < 1e-8f) {
        return s;
    }
    const float cos_emitter = std::abs(glm::dot(n_raw / n_len, -s.l));
    if (cos_emitter <= 0.0f) {
        return s;
    }
    s.radiance = r.radiance * (cos_emitter * r.area / std::max(d2, r.area));
    s.valid = true;
    return s;
}

/// Mirror of `rtEmissiveTarget`.
[[nodiscard]] inline float emissive_target(const EmissiveTriangle& r, const glm::vec3& p,
                                           const glm::vec3& n) {
    const EmissiveShading s = emissive_shading(r, p);
    if (!s.valid) {
        return 0.0f;
    }
    const float ndotl = std::max(glm::dot(n, s.l), 0.0f);
    if (ndotl <= 0.0f) {
        return 0.0f;
    }
    return lighting::target_luminance(s.radiance) * ndotl;
}

} // namespace bracken::render::rt
