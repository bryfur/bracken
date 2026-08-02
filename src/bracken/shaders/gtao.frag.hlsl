/**
 * GTAO Fragment Shader with an SSILVB VISIBILITY BITMASK and a bent cone.
 *
 * Horizon-based ground-truth ambient occlusion (Jimenez et al. 2016) whose
 * per-slice horizon PAIR was replaced in R5a by a 32-sector visibility bitmask
 * (Therrien et al. 2023, "Screen Space Indirect Lighting with Visibility
 * Bitmask", The Visual Computer 39). Same slices, same steps, same jitter,
 * same cosine weighting — three answers instead of one.
 *
 * Runs at the AO working resolution (half or full res per AOMode) from the
 * depth-prepass depth alone, BEFORE the forward pass.
 *
 * Per pixel: reconstruct the view-space position from depth and an edge-aware
 * normal from 4 extra depth taps, then march SLICE_COUNT screen-space slices.
 * Per slice, the hemicircle around the projected normal — the angular range
 * [n - pi/2, n + pi/2], angles measured from the view vector and signed toward
 * the +omega march — is cut into SECTOR_COUNT equal sectors. Each of the
 * STEPS_PER_SLICE x 2 depth samples sets the bits of the sectors its occluder
 * covers, treating that occluder as a slab of OCCLUDER_THICKNESS world units
 * along the view ray. What is left is integrated with the SAME cosine weight
 * the analytic arc integral used, so an unoccluded pixel produces bit-for-bit
 * the value the arc integral produced (render/gtao.hpp is the CPU reference
 * and the tests pin that identity).
 *
 * Why a bitmask at all: a horizon pair can only say "everything past this
 * angle is blocked". A fence plank, a grass blade or a railing blocks a BAND
 * and nothing beyond it, and the old `thinOccluderCompensation` knob was a
 * unitless guess at that with no relation to how thick the occluder was — see
 * `gtao_slab_span` in render/gtao.hpp for the full comparison.
 *
 * Slice rotation / step offset are jittered per pixel (IGN) and per frame
 * (golden-ratio sequence) — the temporal pass integrates the rotating pattern
 * over time.
 *
 * OUTPUT (RGBA8, shaders/bent_cone.hlsli owns the encoding):
 *   R  AO, 1 = unoccluded
 *   GB octahedral WORLD-space bent normal (mean unoccluded direction)
 *   A  cone aperture as sin^2(half-angle); >= BENT_CONE_OPEN means "fully
 *      open, no directional data", which is what sky and far-field pixels and
 *      the white dummy texture all report.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer GTAOUniforms : register(b0, space3) {
    float4x4 projection;
    float4x4 invProjection;
    float4x4 invView;  // view -> world, for publishing the bent normal
    float2 screenSize;
    float2 invScreenSize;
    float radius;      // world/view-unit sample radius
    float radiusScale; // GraphicsSettings::ao_radius multiplier
    float falloffRange; // fraction of radius over which occlusion fades out
    float occluderThickness; // slab depth along the view ray, world units
    float temporalRotation; // per-frame slice-angle offset in [0,1)
    float temporalOffset;   // per-frame step offset in [0,1)
    float maxDistance; // view-space far cutoff for AO (fades over the last 10%)
    float _padding0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float> depthTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState depthSampler : register(s0, space2);

// viewPosFromDepth + interleavedGradientNoise
#include "lighting.hlsli"
#include "bent_cone.hlsli"

#define SLICE_COUNT 3
#define STEPS_PER_SLICE 4
// Sectors per slice hemicircle. 32 because the mask is a uint and countbits is
// plain ALU — no wave intrinsics anywhere in this pass. render/gtao.hpp's
// detail::GTAO_SECTOR_COUNT is the same number.
#define SECTOR_COUNT 32

static const float SECTOR_DELTA = PI_ / float(SECTOR_COUNT);
// Angle-addition recurrence constants: the sector integration walks boundary
// angles (step SECTOR_DELTA, and 2*SECTOR_DELTA for the doubled angle inside
// the antiderivative) with multiply-adds instead of 32 sin/cos pairs.
static const float SECTOR_COS_1 = cos(SECTOR_DELTA);
static const float SECTOR_SIN_1 = sin(SECTOR_DELTA);
static const float SECTOR_COS_2 = cos(2.0 * SECTOR_DELTA);
static const float SECTOR_SIN_2 = sin(2.0 * SECTOR_DELTA);
static const float SECTOR_COS_HALF = cos(0.5 * SECTOR_DELTA);
static const float SECTOR_SIN_HALF = sin(0.5 * SECTOR_DELTA);

float3 viewPosAt(float2 uv) {
    float d = depthTexture.SampleLevel(depthSampler, uv, 0);
    return viewPosFromDepth(uv, d, invProjection);
}

// Edge-aware normal reconstruction: per axis pick the neighbor whose depth is
// closer to the center so the derivative never crosses a silhouette (plain
// ddx/ddy smears normals across depth discontinuities -> AO halos on edges).
// Chosen over a prepass normal MRT: 4 extra half-res depth taps are cheaper
// than a full-res normal write+sample, and AO only needs geometric normals.
float3 reconstructNormal(float2 uv, float3 centerPos) {
    float3 l = viewPosAt(uv - float2(invScreenSize.x, 0.0));
    float3 r = viewPosAt(uv + float2(invScreenSize.x, 0.0));
    float3 u = viewPosAt(uv - float2(0.0, invScreenSize.y));
    float3 d = viewPosAt(uv + float2(0.0, invScreenSize.y));
    // Derivatives along +screen-x and +screen-y (screen y is down).
    float3 dx = (abs(l.z - centerPos.z) < abs(r.z - centerPos.z)) ? (centerPos - l) : (r - centerPos);
    float3 dy = (abs(u.z - centerPos.z) < abs(d.z - centerPos.z)) ? (centerPos - u) : (d - centerPos);
    // Same orientation as cross(ddy, ddx) with implicit derivatives.
    return normalize(cross(dy, dx));
}

// Sector bits covered by a normalised angular span [lo, hi] over the
// hemicircle. Rounds the width outward and the start inward, so a sub-sector
// occluder costs a whole sector instead of vanishing. CPU reference:
// render/gtao.hpp detail::gtao_mark_sectors.
uint markSectors(float lo, float hi) {
    float span = (hi - lo) * float(SECTOR_COUNT);
    if (!(span > 0.0)) {
        return 0u;
    }
    uint count = uint(min(float(SECTOR_COUNT), ceil(span)));
    uint start = min(uint(lo * float(SECTOR_COUNT)), uint(SECTOR_COUNT - 1));
    // 0xFFFFFFFF >> (32 - count), never (1 << count) - 1: count == 32 happens
    // and a 32-bit shift is undefined.
    return (0xFFFFFFFFu >> (uint(SECTOR_COUNT) - count)) << start;
}

float4 PSMain(PSInput input) : SV_Target {
    float2 uv = input.texCoord;
    float depth = depthTexture.SampleLevel(depthSampler, uv, 0);

    // "Unoccluded, no directional data" — R = 1, aperture = 1, and GB is then
    // ignored by every consumer (bent_cone.hlsli's BENT_CONE_OPEN escape).
    const float4 FULLY_OPEN = float4(1.0, 1.0, 1.0, 1.0);

    // Sky never gets AO: reversed-Z cleared/skybox depth is exactly 0.0 (same
    // test as volumetric_fog.frag). The explicit maxDistance cutoff below is
    // the real far-field fade; this only skips true sky pixels.
    if (depth <= 0.0) {
        return FULLY_OPEN;
    }

    float3 viewPos = viewPosFromDepth(uv, depth, invProjection);

    float3 V = normalize(-viewPos);
    float3 N = reconstructNormal(uv, viewPos);
    // The reconstruction can flip on grazing silhouettes; AO needs a
    // camera-facing normal.
    if (dot(N, V) < 0.0) {
        N = -N;
    }
    float3 worldN = normalize(mul(invView, float4(N, 0.0)).xyz);

    // Explicit far-field cutoff in linear view distance: past maxDistance the
    // AO footprint is subpixel at half resolution anyway. Fade over the last
    // 10% so the boundary doesn't pop. The normal is reconstructed BEFORE this
    // (unlike pre-R5a) because a faded pixel still has to publish a usable
    // bent normal, and the honest one is its own normal.
    float distanceFade = saturate((maxDistance - (-viewPos.z)) / max(0.1 * maxDistance, 1e-3));
    if (distanceFade <= 0.0) {
        return float4(1.0, bentOctEncode(worldN), 1.0);
    }

    float effRadius = radius * radiusScale;
    // The slab scales with the sampling radius: ao_radius asks for AO at a
    // larger world scale, and a slab that stayed fixed would then be a
    // vanishing fraction of every occluder the wider radius reaches.
    float effThickness = occluderThickness * radiusScale;

    // Screen-space (UV-unit) radius of the world-space sampling sphere at
    // this depth. Clamped so the footprint is never subpixel nor a whole
    // screen march.
    float screenRadiusUV = effRadius * projection[0][0] * 0.5 / max(-viewPos.z, 0.1);
    screenRadiusUV = clamp(screenRadiusUV, 2.0 * invScreenSize.x, 0.2);

    // Spatial (IGN) + temporal (golden ratio) jitter for slice rotation and
    // step offset. Two decorrelated IGN lattices via the transposed pixel.
    float noiseSlice = frac(interleavedGradientNoise(input.position.xy) + temporalRotation);
    float noiseStep = frac(interleavedGradientNoise(input.position.yx + 17.317) + temporalOffset);

    // Linear falloff over the outer falloffRange fraction of the radius:
    // weight = saturate(dist * falloffMul + falloffAdd) is 1 inside, 0 at the
    // radius edge.
    float falloffDist = max(effRadius * falloffRange, 1e-4);
    float falloffMul = -1.0 / falloffDist;
    float falloffAdd = effRadius / falloffDist;

    float visibility = 0.0;
    // First moment of the visibility lobe, view space: sum of w_k * direction
    // over the unoccluded sectors of every slice. Its direction is the bent
    // normal, its length relative to `visibility` is the cone's aperture.
    float3 bentSum = float3(0.0, 0.0, 0.0);

    [unroll]
    for (int slice = 0; slice < SLICE_COUNT; ++slice) {
        // Slices span pi (each covers both directions).
        float phi = PI_ * frac((float(slice) + noiseSlice) / float(SLICE_COUNT));
        float cosPhi = cos(phi);
        float sinPhi = sin(phi);

        // Screen y is down, view y is up.
        float2 omegaUV = float2(cosPhi, -sinPhi) * screenRadiusUV;
        float3 directionVec = float3(cosPhi, sinPhi, 0.0);

        // Project the normal into the slice plane (plane containing V and
        // the slice direction) — GTAO integrates visibility around it.
        float3 orthoDirectionVec = directionVec - dot(directionVec, V) * V;
        float3 axisVec = normalize(cross(orthoDirectionVec, V));
        float3 projectedNormalVec = N - axisVec * dot(N, axisVec);
        float projectedNormalVecLength = length(projectedNormalVec);
        float signNorm = sign(dot(orthoDirectionVec, projectedNormalVec));
        float cosNorm = saturate(dot(projectedNormalVec, V) / max(projectedNormalVecLength, 1e-5));
        float n = signNorm * acos(cosNorm);
        float sinN = sin(n);
        // In-plane basis: a direction at signed angle theta from V is
        // cos(theta) * V + sin(theta) * tangentVec, and theta == n is the
        // projected normal itself.
        float3 tangentVec = orthoDirectionVec / max(length(orthoDirectionVec), 1e-5);

        // Hemicircle base angle: sector k spans [base + k*DELTA, base + (k+1)*DELTA].
        float base = n - PI_ * 0.5;

        uint occluded = 0u;

        [unroll]
        for (int step = 0; step < STEPS_PER_SLICE; ++step) {
            float s = (float(step) + noiseStep) / float(STEPS_PER_SLICE);
            s *= s; // denser sampling near the shading point
            float2 offset = s * omegaUV;

            [unroll]
            for (int side = 0; side < 2; ++side) {
                float sideSign = (side == 0) ? -1.0 : 1.0;
                float3 sPos = viewPosAt(uv + sideSign * offset);
                float3 delta = sPos - viewPos;
                float dist = length(delta);
                // A sample landing on our own texel carries no direction.
                if (dist < 1e-4) {
                    continue;
                }

                // Front face of the occluder, and its back face one slab
                // thickness further along the view ray (V points at the
                // camera, so -V is "further away").
                float3 backDelta = delta - V * effThickness;
                float frontAngle = acos(clamp(dot(delta, V) / dist, -1.0, 1.0));
                float backAngle =
                    acos(clamp(dot(backDelta, V) / max(length(backDelta), 1e-5), -1.0, 1.0));

                // Radius falloff SHRINKS the occluder's angular extent to zero
                // instead of attenuating a horizon: at the edge of the
                // sampling sphere the span is sub-sector and marks nothing, so
                // occlusion fades out continuously exactly as it did before.
                float weight = saturate(dist * falloffMul + falloffAdd);
                backAngle = lerp(frontAngle, backAngle, weight);

                // Sign by march side, then normalise onto the hemicircle.
                // CPU reference: detail::gtao_slab_span.
                float loAngle = (sideSign > 0.0) ? frontAngle : -backAngle;
                float hiAngle = (sideSign > 0.0) ? backAngle : -frontAngle;
                float lo = saturate((loAngle - base) / PI_);
                float hi = saturate((hiAngle - base) / PI_);
                occluded |= markSectors(lo, hi);
            }
        }

        // ---- Cosine-weighted integration over the CLEAR sectors ----
        //
        // Sector k's exact weight is C(theta_{k+1}) - C(theta_k), where
        // C(theta) = sign(theta) * A(theta) and
        // A(theta) = (1/4)(-cos(2*theta - n) + cos n + 2*theta*sin n)
        // is the antiderivative of |sin| * cos(theta - n) — the very integrand
        // the analytic arc integral used, so all 32 weights sum to the
        // unoccluded arc value. CPU reference: detail::gtao_sector_weight.
        //
        // Angle-addition recurrences keep this to multiply-adds: `psi` tracks
        // (cos, sin) of 2*theta_k - n (which starts at n - pi), `dirMid`
        // tracks (cos, sin) of the sector MIDPOINT for the bent-normal sum.
        float2 psi = float2(-cosNorm, -sinN);
        float theta = base;
        float a0 = 0.25 * (-psi.x + cosNorm + 2.0 * theta * sinN);
        float prevC = (theta < 0.0) ? -a0 : a0;
        float2 dirMid = float2(sinN * SECTOR_COS_HALF + cosNorm * SECTOR_SIN_HALF,
                               -cosNorm * SECTOR_COS_HALF + sinN * SECTOR_SIN_HALF);

        float sliceVis = 0.0;
        float3 sliceDir = float3(0.0, 0.0, 0.0);

        [unroll]
        for (int k = 0; k < SECTOR_COUNT; ++k) {
            float2 psiNext = float2(psi.x * SECTOR_COS_2 - psi.y * SECTOR_SIN_2,
                                    psi.x * SECTOR_SIN_2 + psi.y * SECTOR_COS_2);
            float thetaNext = theta + SECTOR_DELTA;
            float aNext = 0.25 * (-psiNext.x + cosNorm + 2.0 * thetaNext * sinN);
            float cNext = (thetaNext < 0.0) ? -aNext : aNext;

            float w = (cNext - prevC) * (((occluded >> uint(k)) & 1u) != 0u ? 0.0 : 1.0);
            sliceVis += w;
            sliceDir += w * (dirMid.x * V + dirMid.y * tangentVec);

            psi = psiNext;
            theta = thetaNext;
            prevC = cNext;
            dirMid = float2(dirMid.x * SECTOR_COS_1 - dirMid.y * SECTOR_SIN_1,
                            dirMid.x * SECTOR_SIN_1 + dirMid.y * SECTOR_COS_1);
        }

        visibility += projectedNormalVecLength * sliceVis;
        bentSum += projectedNormalVecLength * sliceDir;
    }

    // Mean resultant length of the visibility lobe -> cone half-angle ->
    // aperture as sin^2(alpha). CPU reference: detail::gtao_cone_aperture.
    float bentLen = length(bentSum);
    float resultant = saturate(bentLen / max(visibility, 1e-5));
    float m = max(1.5 * resultant - 1.0, 0.0);
    float coneCos = saturate(0.5 * (m + sqrt(m * m + 4.0 * m)));
    float aperture = saturate(1.0 - coneCos * coneCos);

    // A degenerate lobe (everything occluded, or the two march sides cancelling)
    // has no direction to report: fall back to the shading normal, which is
    // also what the aperture escape value tells consumers to do.
    float3 bentView = (bentLen > 1e-5) ? (bentSum / bentLen) : N;
    float3 bentWorld = normalize(mul(invView, float4(bentView, 0.0)).xyz);

    float ao = saturate(visibility / float(SLICE_COUNT));
    // Past the far cutoff both signals fade to "fully open", which is exactly
    // the escape value, so the boundary is continuous in every channel.
    ao = lerp(1.0, ao, distanceFade);
    aperture = lerp(1.0, aperture, distanceFade);

    return float4(ao, bentOctEncode(bentWorld), aperture);
}
