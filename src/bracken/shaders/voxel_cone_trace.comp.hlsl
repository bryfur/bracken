// voxel_cone_trace.comp.hlsl - fill the probe field's ray-results buffer by
// CONE TRACING the voxel radiance clipmap. Phase RG3 of the raster GI path.
//
// THIS IS THE DROP-IN REPLACEMENT FOR gi_probe_trace.comp.hlsl. It writes the
// identical contract into the identical buffer:
//
//     giRayResults[probe * rayCount + ray] = float4(radiance, distance)
//
// with `distance` = rayMaxDistance on a miss and NEGATIVE when the sample
// starts inside geometry (which is what the classify pass counts as a backface
// and uses to decide a probe is buried). Everything downstream is untouched and
// shared with the ray-traced path: the blend, the Chebyshev visibility moments,
// classification, relocation, the cascade blend, the fog in-scatter, and every
// fragment shader that samples the field.
//
// WHY A CONE PER "RAY". The probe field asks for `rayCount` directions and gets
// one answer each. A ray into a voxel volume would have to step at voxel
// resolution for its whole length and would still alias badly, because a voxel
// is a low-frequency summary and a ray is an infinitely thin query. A CONE is
// the query the representation was built for: it widens with distance, so the
// mip whose voxels match its width answers the whole cross-section in ONE
// filtered tap. That substitution is what makes this affordable, and it is why
// the volume is mipmapped and sampled trilinearly rather than loaded.
//
// The cone's half-angle comes from the probe's ray budget: `rayCount`
// directions covering the sphere means each owns a solid angle of
// 4*pi/rayCount, and a cone of that solid angle has
// cos(theta) = 1 - 2/rayCount. Deriving it rather than tuning it is what keeps
// the field's total gathered energy independent of the ray budget, so changing
// the quality tier changes noise and cost but not brightness.

#include "voxel_common.hlsli"
#include "gi_probe.hlsli"

[[vk::binding(0, 2)]]
cbuffer VoxelConeTraceUniforms : register(b0, space2) {
    /// Probes scheduled this dispatch, and directions per probe. Same meaning
    /// and same values as the ray-traced path's, because the consumer is the
    /// same buffer.
    uint voxelProbeCount;
    uint voxelRayCount;
    /// Rotates the direction set per update, exactly as the traced path does —
    /// a fixed set converges to a `rayCount`-direction quadrature of the scene
    /// rather than to the scene.
    uint voxelFrameIndex;
    /// Distance a cone reports when it never accumulates enough occlusion to
    /// count as a hit. Must be the ray-traced path's `rayMaxDistance` or the
    /// visibility moments would change meaning between GI modes.
    float voxelRayMaxDistance;

    /// Camera position, for cascade selection.
    float3 voxelCameraPosition;
    /// How far a cone may march before it is called a miss. Bounded by the
    /// coarsest cascade — past that there is no data, only the sky.
    float voxelMaxDistance;

    /// Sky radiance a cone that escapes the clipmap returns, so an outdoor
    /// probe is lit rather than black. The ray-traced path samples the IBL
    /// capture for this; a constant is the honest V1 stand-in and is the single
    /// biggest quality gap between the two paths.
    float3 voxelSkyColor;
    /// Global multiplier on gathered radiance. 1 is energy-preserving; the
    /// field's own bounce accumulation is what carries multi-bounce.
    float voxelBounceScale;

    /// The three cascade base corners, xyz used.
    int4 voxelCascadeBases[VOXEL_CASCADE_COUNT];
};

/// The probes this dispatch is updating — the SAME record type and the same
/// buffer the ray-traced trace consumes, so the schedule is shared.
struct GiProbeUpdate {
    float3 worldPosition;
    uint slot;
    uint flags;
    float3 _pad;
};

// SET 0 ORDER: samplers FIRST, then read-only storage buffers. Not a style
// choice — SDL3 GPU binds them as two packed arrays in that order and the
// engine's shaders/README.md spells it out.
//
// The three cascades' radiance volumes, sampled trilinearly with a REPEATING
// address mode — storage is a torus, so the sampler's wrap is exactly the
// addressing's wrap.
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture3D<float4> voxelRadiance0 : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState voxelRadianceSampler0 : register(s0, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture3D<float4> voxelRadiance1 : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState voxelRadianceSampler1 : register(s1, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture3D<float4> voxelRadiance2 : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState voxelRadianceSampler2 : register(s2, space0);

[[vk::binding(3, 0)]] StructuredBuffer<GiProbeUpdate> giProbeUpdates : register(t3, space0);

[[vk::binding(0, 1)]] RWStructuredBuffer<float4> giRayResults : register(u0, space1);

/// One filtered tap. Split out because HLSL cannot index an array of textures
/// without a bindless path, so the cascade choice is a branch — and a branch
/// per step is far cheaper than the alternative of tracing every cascade.
float4 voxelSampleCascade(int cascade, float3 world, float mip) {
    const float3 uvw = voxelSampleUvw(cascade, world);
    if (cascade == 0) {
        return voxelRadiance0.SampleLevel(voxelRadianceSampler0, uvw, mip);
    }
    if (cascade == 1) {
        return voxelRadiance1.SampleLevel(voxelRadianceSampler1, uvw, mip);
    }
    return voxelRadiance2.SampleLevel(voxelRadianceSampler2, uvw, mip);
}

/// Is this world point inside `cascade`'s window? The clipmap's containment
/// test, against the base this dispatch was told — torus addressing wraps ANY
/// coordinate to a valid texel, so without this a cone leaving the window reads
/// the opposite side of the world with full confidence.
bool voxelWorldInWindow(int cascade, float3 world) {
    const int3 coord = voxelCoordAt(cascade, world);
    return voxelInWindow(coord, voxelCascadeBases[cascade].xyz);
}

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint ray = tid.x;
    const uint probe = tid.y;
    if (ray >= voxelRayCount || probe >= voxelProbeCount) {
        return;
    }

    const GiProbeUpdate update = giProbeUpdates[probe];
    const float3 origin = update.worldPosition;

    // The SAME direction set and the SAME per-frame rotation as the traced
    // path, so the two modes' probes integrate over identical quadratures and a
    // scene lit either way is comparable.
    float3 c0, c1, c2;
    giRayRotation(voxelFrameIndex, c0, c1, c2);
    const float3 d = giSphericalFibonacci(ray, voxelRayCount);
    const float3 dir = normalize(c0 * d.x + c1 * d.y + c2 * d.z);

    // Solid angle per direction -> cone half-angle. Derived, not tuned: it is
    // what keeps total gathered energy independent of the ray budget.
    const float cosTheta = 1.0 - 2.0 / max(float(voxelRayCount), 1.0);
    const float tanHalf = sqrt(max(1.0 - cosTheta * cosTheta, 0.0)) / max(cosTheta, 1e-4);

    // START OFF THE SURFACE. A probe frequently sits inside or against
    // geometry, and a cone beginning in an opaque voxel would immediately
    // saturate and report black. One cascade-0 voxel of offset is the smallest
    // step that cannot re-sample the cell the probe is in.
    float t = VOXEL_BASE_SIZE;

    // Is the probe itself buried? The classify pass reads a NEGATIVE distance
    // as "this direction started inside geometry" and counts those to decide a
    // probe is inside a wall and must be relocated or written off. Answering it
    // from the volume directly is both cheaper and more reliable than the
    // traced path's backface test.
    bool startedInside = false;
    {
        const int c = voxelSelectCascade(origin, voxelCameraPosition);
        if (c >= 0 && voxelWorldInWindow(c, origin)) {
            startedInside = voxelSampleCascade(c, origin, 0.0).a > 0.5;
        }
    }

    float4 accum = float4(0.0, 0.0, 0.0, 0.0);
    float hitDistance = -1.0;

    [loop] while (t < voxelMaxDistance && accum.a < VOXEL_CONE_TERMINATE_ALPHA) {
        const float3 world = origin + dir * t;

        // Finest cascade that still contains the sample with filtering margin.
        // -1 means the cone has left the clipmap entirely, and there is nothing
        // further out to find.
        const int cascade = voxelSelectCascade(world, voxelCameraPosition);
        if (cascade < 0 || !voxelWorldInWindow(cascade, world)) {
            break;
        }

        const float diameter = voxelConeDiameter(t, tanHalf);
        const float mip = voxelConeMip(diameter, cascade);
        float4 s = voxelSampleCascade(cascade, world, mip);

        // OPACITY SCALES WITH STEP LENGTH. The stored alpha is a coverage
        // fraction for a voxel-sized step; a step that crosses several voxels
        // must occlude proportionally more or a widening cone would see through
        // walls simply because it was taking longer strides.
        const float step = voxelConeStep(t, tanHalf);
        s.a = saturate(s.a * (step / voxelSize(cascade)));

        accum = voxelConeAccumulate(accum, s);

        if (hitDistance < 0.0 && accum.a >= VOXEL_CONE_HIT_ALPHA) {
            hitDistance = t;
        }
        t += step;
    }

    // The probe field's contract, exactly. A cone that never reached the hit
    // threshold is a miss and reports the far distance, which is what makes the
    // Chebyshev moments read "nothing in the way" rather than "a wall at zero".
    float3 radiance = accum.rgb * voxelBounceScale;
    float distance = hitDistance >= 0.0 ? hitDistance : voxelRayMaxDistance;

    // What the cone did not block is sky. Without this an outdoor probe gathers
    // only bounced light and the whole field reads far too dark — the traced
    // path gets this from the IBL capture, and matching it properly is RG4.
    radiance += voxelSkyColor * (1.0 - accum.a);

    if (startedInside) {
        // Negative distance is the classify pass's "inside geometry" signal.
        // The radiance is zeroed for the same reason the traced path zeroes a
        // backface hit: a probe inside a wall must not pick up the wall's own
        // lit exterior and read as brightly lit.
        radiance = float3(0.0, 0.0, 0.0);
        distance = -max(hitDistance, VOXEL_BASE_SIZE);
    }

    giRayResults[probe * voxelRayCount + ray] = float4(radiance, distance);
}
