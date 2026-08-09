#pragma once

// The RIS target function for local-light shadow sampling, and the attenuation
// it is built out of — the CPU mirror of `shaders/cluster_index.hlsli`'s
// `pointFalloff`/`spotConeFactor` and `shaders/cluster_light_load.hlsli`'s
// `clusterLightTarget*`.
//
// WHY THIS HAS TO BE ONE COPY. The RIS estimator is unbiased only if the target
// function it importance-samples against agrees with the integrand the forward
// pass actually evaluates. It does not have to be the SAME function — a target
// function is allowed to be an approximation, that is the point of RIS — but it
// does have to be zero exactly where the forward loop contributes zero. The
// forward loop's zeros are structural: `dist > radius` is a `continue`, the
// range window `(1 - (d/R)^4)^2` reaches zero at the radius, and a spot's
// smoothstep cone reaches zero at `outer_cos`. If the sampler's copy of any of
// those drifted, the pass would pick lights that contribute nothing (wasting
// the frame's one ray) or, worse, never pick lights that do.
//
// So the shader shares one arithmetic copy across the fragment and compute
// paths — the cluster_index/cluster_light_load move R2a made for the froxel
// index — and this header is the CPU mirror `light_target.test.cpp` pins
// against the binner's own radius cull.
//
// The APPROXIMATION, stated: the target function is the luminance of the
// UNSHADOWED DIFFUSE contribution with the albedo left out. Albedo is the same
// for every candidate at a pixel, so it factors out of the reservoir's
// normalized weights entirely; the specular lobe is left out because the
// compute pass has no view-dependent material term worth the divergence and its
// omission only shifts which light a glancing highlight prefers, never which
// lights are candidates at all.
//
// GPU-free on purpose: no `gpu/`, no SDL, nothing but glm — same rule as
// cluster_index.hpp.

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

namespace bracken::render::lighting {

/// Smooth range window x inverse-square. Mirror of `pointFalloff` in
/// shaders/cluster_index.hlsli, operation for operation — including the
/// `max(radius, 1e-4)` and `max(d*d, 1e-4)` guards, which are what keep a
/// zero-radius or coincident light finite rather than infinite.
[[nodiscard]] inline float point_falloff(float dist, float radius) {
    const float r = std::clamp(dist / std::max(radius, 1e-4f), 0.0f, 1.0f);
    const float r2 = r * r;
    const float win = std::clamp(1.0f - r2 * r2, 0.0f, 1.0f);
    return (win * win) / std::max(dist * dist, 1e-4f);
}

/// Spot cone falloff. Mirror of `spotConeFactor`. `l_to_surface` is the
/// SURFACE->LIGHT unit vector (the forward loop's `L`), `spot_dir` the
/// direction the light points; the negation inside is what makes those two
/// conventions meet.
[[nodiscard]] inline float spot_cone_factor(const glm::vec3& l_to_surface, const glm::vec3& spot_dir,
                                            float inner_cos, float outer_cos) {
    const float cd = glm::dot(spot_dir, -l_to_surface);
    // HLSL smoothstep(edge0, edge1, x). Callers reach this with
    // inner_cos >= outer_cos — `FrameBuilder::Lane::add_spot_light`
    // (scene/frame_builder.cpp) normalizes a transposed pair, because the
    // shader's smoothstep is UNDEFINED for edge0 > edge1 — so the denominator
    // is non-negative; the guard below only covers the degenerate equal-angle
    // cone, which the shader also leaves undefined and this mirror resolves as
    // a hard edge.
    const float denom = inner_cos - outer_cos;
    if (std::abs(denom) < 1e-8f) {
        return cd >= inner_cos ? 1.0f : 0.0f;
    }
    const float t = std::clamp((cd - outer_cos) / denom, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/// Rec. 709 luma, the same weights the shader's `clusterTargetLuminance` uses.
[[nodiscard]] inline float target_luminance(const glm::vec3& c) {
    return glm::dot(c, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

/**
 * p̂ for a point light at surface (`P`, `N`). Zero when the light is out of
 * range or behind the surface — both of which are `continue`s in
 * accumulateClusterLights, which is the property that makes the estimator
 * bias-free.
 */
[[nodiscard]] inline float point_light_target(const glm::vec3& light_pos, float radius,
                                              const glm::vec3& color, float intensity,
                                              const glm::vec3& p, const glm::vec3& n) {
    const glm::vec3 to_light = light_pos - p;
    const float dist = glm::length(to_light);
    if (dist > radius) {
        return 0.0f;
    }
    const glm::vec3 l = to_light / std::max(dist, 1e-4f);
    const float ndotl = std::max(glm::dot(n, l), 0.0f);
    if (ndotl <= 0.0f) {
        return 0.0f;
    }
    return target_luminance(color * intensity) * point_falloff(dist, radius) * ndotl;
}

/** p̂ for a spot light — the point target times the cone factor. */
[[nodiscard]] inline float spot_light_target(const glm::vec3& light_pos, float radius,
                                             const glm::vec3& direction, float inner_cos, float outer_cos,
                                             const glm::vec3& color, float intensity,
                                             const glm::vec3& p, const glm::vec3& n) {
    const glm::vec3 to_light = light_pos - p;
    const float dist = glm::length(to_light);
    if (dist > radius) {
        return 0.0f;
    }
    const glm::vec3 l = to_light / std::max(dist, 1e-4f);
    const float cone = spot_cone_factor(l, direction, inner_cos, outer_cos);
    if (cone <= 0.0f) {
        return 0.0f;
    }
    const float ndotl = std::max(glm::dot(n, l), 0.0f);
    if (ndotl <= 0.0f) {
        return 0.0f;
    }
    return target_luminance(color * intensity) * point_falloff(dist, radius) * cone * ndotl;
}

} // namespace bracken::render::lighting
