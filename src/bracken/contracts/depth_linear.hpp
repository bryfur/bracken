#pragma once

/**
 * @file depth_linear.hpp
 * @brief THE engine's depth linearization: device depth -> positive view depth.
 *
 * This header is the SINGLE SOURCE for the formula. `shaders/depth_linear.hlsli`
 * is its transliteration and every shader that linearizes depth includes that
 * file rather than open-coding the expression; `depth_linear.test.cpp` pins this
 * half against the inverse-projection reference. Change one, change both.
 *
 * ## Derivation
 *
 * `glm::perspectiveRH_ZO(fovy, aspect, zn, zf)` (what
 * `GLM_FORCE_DEPTH_ZERO_TO_ONE` makes `glm::perspective` resolve to) writes
 *
 *     m[2][2] = zf / (zn - zf)        =: a
 *     m[3][2] = -(zf * zn) / (zf - zn) =: b
 *     m[2][3] = -1
 *
 * so for a view-space point at `z_v` (negative in front of a right-handed
 * camera),
 *
 *     clip.z = a * z_v + b
 *     clip.w = -z_v
 *     d      = clip.z / clip.w
 *
 * Substituting the positive view depth `t = -z_v`:
 *
 *     d = (-a * t + b) / t = b / t - a
 *     => t = b / (d + a)                                    ... (*)
 *
 * (*) is exact for ANY `zn`/`zf` pair, which is why the same two terms serve the
 * engine's reversed-Z projection without a second formula. The engine passes
 * far as `zn` and near as `zf` (CLAUDE.md: near -> depth 1, far -> depth 0), so
 *
 *     a = near / (far - near) > 0
 *     b = near * far / (far - near) > 0
 *     b / a = far
 *
 * and (*) checks out at both ends: `t(1) = b / (1 + a) = near`, `t(0) = b / a =
 * far`. `scene_renderer.cpp` recovers the camera near plane with exactly that
 * `d = 1` case.
 *
 * ## The constant that was wrong (fixed 2026-08-02)
 *
 * Four traced passes (the sun-visibility temporal denoiser and its spatial
 * filter, the gradient pass, and the local-DI spatial filter) carried a copy
 * that SUBTRACTED `a` instead of adding it: `b / max(d - a, 1e-6)`. Feeding a
 * true depth `t` through it gives, using `a / b = 1 / far`,
 *
 *     t_wrong = b / (b / t - 2a) = t / (1 - 2 * t / far)
 *
 * i.e. a distance-dependent overestimate, not a constant factor. At the default
 * `far = 4000` (DrawDistance::D4000):
 *
 *     t = 10    -> x1.0050  (+0.5%)
 *     t = 100   -> x1.0526  (+5.3%)
 *     t = 500   -> x1.3333  (+33.3%)   <- the ~33% R2b measured
 *     t = 1000  -> x2.0000  (+100%)
 *     t >= 2000 -> the denominator turns non-positive and the `max(.., 1e-6)`
 *                  clamp pins EVERY pixel past far/2 to the same constant
 *                  b / 1e-6, so depth comparisons out there were comparing two
 *                  identical numbers and had stopped discriminating at all.
 *
 * Only the sign was wrong; the `a`/`b` fills (`projection[2][2]` /
 * `projection[3][2]`) were correct everywhere.
 */

#include <cmath>

#include <glm/glm.hpp>

namespace bracken::scene {

/// The two projection terms depth linearization needs, so a pass can carry two
/// floats instead of a matrix. `a` is `projection[2][2]`, `b` is
/// `projection[3][2]` — of the SAME (jittered, if TAA is on) projection the
/// depth buffer was rendered with.
struct DepthLinearTerms {
    float a = 0.0f;
    float b = 0.0f;
};

/// Pull the linearization terms out of a projection matrix. The one place the
/// index pair is spelled out.
[[nodiscard]] inline DepthLinearTerms depth_linear_terms(const glm::mat4& projection) {
    return DepthLinearTerms{projection[2][2], projection[3][2]};
}

/**
 * Device depth -> positive linear view depth, `|b / (d + a)|`.
 *
 * Exact for standard and reversed Z alike as long as `a`/`b` come from the live
 * projection (see the derivation above). The `abs` is a no-op for a real
 * projection — both terms and the denominator are positive — and is kept
 * because it is what makes the expression safe to state without a sign
 * convention attached.
 *
 * Degenerate `a == b == 0` (a pass that never filled its uniforms) returns 0
 * rather than a NaN, which is what lets the depth-aware upsamples degrade to
 * plain bilinear instead of poisoning their weight sums.
 */
[[nodiscard]] inline float linear_view_depth(float device_depth, float a, float b) {
    const float denom = device_depth + a;
    return std::abs(denom) > 1e-9f ? std::abs(b / denom) : 0.0f;
}

[[nodiscard]] inline float linear_view_depth(float device_depth, DepthLinearTerms terms) {
    return linear_view_depth(device_depth, terms.a, terms.b);
}

} // namespace bracken::scene
