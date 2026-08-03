#pragma once

#include <cstddef>
#include <glm/glm.hpp>

namespace bracken::render::rt {

/**
 * ShadowCapsule — one analytic occluder the traced VISIBILITY passes test
 * before they walk the BVH.
 *
 * WHY THIS TYPE EXISTS. A skinned mesh's BLAS holds the model's BIND POSE:
 * the acceleration structure is built once from the CPU geometry, and a
 * skinned instance carries only the character's root transform. Because the
 * traced sun mask REPLACES the cascade result rather than combining with it,
 * an animated character used to drag a rigid bind-pose statue's shadow around
 * the world — blocky, and not animating (docs/rt_lighting_plan.md §1 limit 3).
 *
 * §5.8 of that plan is the answer, and it is deliberately not "re-skin and
 * refit every frame": for the two VISIBILITY signals (sun shadow, local-light
 * DI) the question is a boolean, so a dozen analytic capsules driven by the
 * live skeleton answer it at a fraction of the cost and with none of the
 * staleness. The bind-pose triangles STAY in the closest-hit paths
 * (reflections, GI probe rays), where a character has to be shadeable and the
 * pose error is invisible at probe scale — removing them there would delete
 * the character from every mirror in the scene.
 *
 * WHAT IT COSTS, stated so nobody reads it as free fidelity: a capsule figure
 * is a SIMPLIFICATION. A hat brim, a cloak, a weapon silhouette and every
 * fold of cloth are gone from the shadow — what remains is a smooth, correctly
 * animating body. That trade is the point.
 *
 * GEOMETRY. A capsule is the set of points within `radius` of the SEGMENT
 * p0..p1 — i.e. a cylinder with a hemisphere at each end. `p0 == p1` is legal
 * and degenerates to a sphere, which is what makes a head or a fist
 * expressible without a special case. World space, same units as everything
 * else in the frame packet.
 *
 * LAYOUT. 32 bytes, two float4 registers, so the GPU-side `ShadowCapsule` in
 * `shaders/rt_trace.hlsli` is a byte-for-byte mirror and a `StructuredBuffer`
 * of these needs no repacking on upload. The radius rides in p0's w and the
 * pad in p1's w rather than the two vec3s being adjacent, because two
 * consecutive `float3`s do NOT pack in HLSL.
 *
 * Leaf layer: glm and `<std>` only, like every other contracts header — the
 * frame packet carries these, so this must not drag the renderer in.
 */
struct ShadowCapsule {
    glm::vec3 p0{0.0f};
    float radius = 0.0f;
    glm::vec3 p1{0.0f};
    /// Explicit, zeroed: the packet's lane merge is a memcpy and
    /// frame_builder.test.cpp compares packets byte for byte, so an
    /// uninitialized pad byte is a nondeterministic byte in a gated frame.
    float _pad0 = 0.0f;
};

static_assert(sizeof(ShadowCapsule) == 32, "ShadowCapsule must stay two float4 registers");
static_assert(alignof(ShadowCapsule) == 4, "ShadowCapsule must not gain alignment padding");
static_assert(offsetof(ShadowCapsule, p0) == 0);
static_assert(offsetof(ShadowCapsule, radius) == 12);
static_assert(offsetof(ShadowCapsule, p1) == 16);
static_assert(offsetof(ShadowCapsule, _pad0) == 28);

} // namespace bracken::render::rt
