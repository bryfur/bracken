// rt_hit_shade.hlsli - what a ray hit is worth: the ONE hit-shading model every
// traced signal in this engine uses.
//
// Phase R3 of docs/rt_lighting_plan.md §5.4. Until R3 there were two answers to
// "a ray hit a triangle; how bright is it?" and only one of them was honest:
//
//   gi_probe_trace.comp.hlsl  albedo x (CSM-shadowed sun + local lights
//                             + pi x last frame's probe irradiance) / pi
//   rt_reflection.comp.hlsl   albedo x (sunColor x N.L + skyColor x 0.25)
//
// The second is wrong twice — the sun is UNSHADOWED, so a reflected facade
// standing in a building's shadow reflects as if it were in full sun, and the
// ambient is a FLAT CONSTANT, so every reflection is uniformly lifted by open
// sky regardless of what the reflected point can actually see. It also happened
// to be off by a factor of pi in the other direction (no Lambertian 1/pi, and
// `sunColor` arriving without `sun_intensity`), which is why it looked roughly
// right at the playground's sun_intensity of 3 and would not have at any other.
//
// This file is the first answer, extracted verbatim, so there is exactly one.
//
// THE RECIPE
//
//   L_out = albedo_flat * ( sunColor * max(0, N.L) * CSM(hitPos)
//                         + local point/spot lights, range-windowed, unshadowed
//                         + pi * probe-field irradiance at the hit ) / pi
//         | on a miss: the sky radiance in the ray direction
//
// Three of those terms are things this engine can answer cheaply and most
// software tracers cannot:
//
//  * `albedo_flat` is TriangleAttributes::baseColor, baked from the real albedo
//    texture's centroid-UV sample at load. SDL3 GPU has no bindless, so a ray
//    hit cannot sample a texture — the flat table is the answer.
//  * sun visibility is a CSM ATLAS FETCH, not a second ray, through the same
//    `calcShadow()` the forward pass calls. The cascades are already rendered
//    this frame, so every traced signal re-lights itself for free when the sun
//    moves.
//  * the ambient term is the probe field (render/gi/), which itself contains
//    sky (via probe-ray misses) and bounce (via its own recursion).
//
// UNITS. The probe atlas stores COSINE-WEIGHTED MEAN INCOMING RADIANCE
// (irradiance/pi), the same convention as the IBL irradiance cube — which is
// why the shading below multiplies the sampled bounce by pi once and divides
// the whole irradiance sum by pi once. Do not "simplify" that away: the two pis
// are on different terms.
//
// ---------------------------------------------------------------------------
// TWO-PHASE INCLUDE, the same contract as rt_trace.hlsli and for the same
// reason: the including shader needs `TriangleAttributes` to DECLARE its
// attribute buffer, and the functions need that buffer (and a dozen uniforms)
// to already exist.
//
//     #define RT_HIT_SHADE_TYPES
//     #include "rt_hit_shade.hlsli"      // TriangleAttributes
//     ... the shader's own cbuffer and binding block ...
//     #define RT_HIT_SHADE_FUNCTIONS
//     #include "rt_hit_shade.hlsli"      // the shading
//
// Phase 2 REQUIRES the including shader to have declared, at whatever bindings
// its own layout dictates:
//
//   sampled:   shadowAtlas + shadowCmpSampler (SamplerComparisonState)
//              shadowAtlasRaw + shadowRawSampler   (shadow_common.hlsli's pair)
//              giSkyCube + giSkySampler            (TextureCube, raw radiance)
//              giIrradiance + giIrradianceSampler  (banded atlas, LINEAR clamp)
//   ro storage texture:  giProbeState (Texture2D<uint>, banded)
//   ro storage buffer:   lightDataBuf (ByteAddressBuffer, ClusterGrid's layout)
//   ro storage buffer:   blasAttributes is NOT required here — the caller reads
//                        its own hit attributes and passes the albedo in.
//
// and in its cbuffer, by these exact names:
//
//   the ten ShadowDataUniforms members shadow_common.hlsli requires
//   float3 cameraPosition;  float3 cameraForward;   // calcShadow's view depth
//   float3 sunDirection;    // direction light TRAVELS
//   float3 sunColor;        // ALREADY multiplied by GraphicsSettings::sun_intensity
//   float  normalBias;      // offset before the probe tap, world units
//   float  bounceScale;     // scale on the probe term; 1 = true infinite bounce
//   float3 ambientSky;  float3 ambientGround;  float useSkyCube;
//   float  giEnabled;           // 0 = the probe atlases hold dummies
//   float  giAmbientFallback;   // see rtHitAmbient
//   int4   giCascadeBase[GI_CASCADE_COUNT];
//   uint   giLightCount;  uint giLightCapacity;

#if !defined(RT_HIT_SHADE_TYPES) && !defined(RT_HIT_SHADE_FUNCTIONS)
#error "Define RT_HIT_SHADE_TYPES or RT_HIT_SHADE_FUNCTIONS before including rt_hit_shade.hlsli"
#endif

// ---------------------------------------------------------------------------
// Phase 1 — types.
// ---------------------------------------------------------------------------
#if defined(RT_HIT_SHADE_TYPES) && !defined(BRACKEN_RT_HIT_SHADE_TYPES_INCLUDED)
#define BRACKEN_RT_HIT_SHADE_TYPES_INCLUDED

/// Per-triangle shading attributes, 80 B. Mirrors
/// render::rt::TriangleAttributes: three model-space vertex normals, one FLAT
/// base colour sampled at load from the albedo texture's centroid UV, and (R6)
/// the same treatment of the EMISSIVE channel
/// (internal/model_material_proxy.hpp). This is the whole material model a ray
/// hit gets, and the reason it is enough for stylized content is that the packs'
/// albedo maps are near-constant per triangle.
///
/// `emissiveLuminance` is a cache of the Rec.709 luma of `emissive`, computed
/// once on the CPU. The shading below does not read it — the BLAS-level light
/// gather does, and it lives here because the alternative was a parallel array
/// indexed the same way.
struct TriangleAttributes {
    float3 n0;
    float pad0;
    float3 n1;
    float pad1;
    float3 n2;
    float pad2;
    float4 baseColor;
    float3 emissive;
    float emissiveLuminance;
};

#endif // RT_HIT_SHADE_TYPES

// ---------------------------------------------------------------------------
// Phase 2 — the shading. Reads the resources and uniforms listed above by name.
// ---------------------------------------------------------------------------
#if defined(RT_HIT_SHADE_FUNCTIONS) && !defined(BRACKEN_RT_HIT_SHADE_FUNCTIONS_INCLUDED)
#define BRACKEN_RT_HIT_SHADE_FUNCTIONS_INCLUDED

#include "gi_probe.hlsli"

// calcShadow() + selectShadowCascade() over the cbuffer members and the atlas
// pair the including shader declared. Included HERE rather than by the entry
// shader so a consumer of this file cannot forget it and silently get an
// unshadowed sun back.
#include "shadow_common.hlsli"

// ---------------------------------------------------------------------------
// Miss: the sky
// ---------------------------------------------------------------------------

/// Radiance of a ray that escaped the scene.
///
/// The IBL sky-capture cube is the SAME radiance function the skybox draws
/// (sky_common.hlsli, shared by skybox.frag and sky_capture.frag), so a traced
/// miss agrees with the rasterized sky pixel next to it. When IBL is off the
/// cube slot holds the engine's cube dummy, so `useSkyCube` gates it and the
/// EnvironmentState hemisphere takes over.
float3 rtHitSkyRadiance(float3 dir) {
    if (useSkyCube > 0.5) {
        return giSkyCube.SampleLevel(giSkySampler, dir, 0).rgb;
    }
    return lerp(ambientGround, ambientSky, saturate(dir.y * 0.5 + 0.5));
}

// ---------------------------------------------------------------------------
// Ambient: the probe field
// ---------------------------------------------------------------------------

uint rtHitProbeState(int cascade, int slot) {
    return giProbeState.Load(int3(giProbeStateTexel(slot, cascade), 0)).x;
}

float3 rtHitProbeTap(int cascade, int slot, float3 dir) {
    // Payload texel centres live at origin + border + oct * payload, so oct = 0
    // lands exactly on the seam between the border texel and payload texel 0 —
    // which is the whole reason the border exists, and why this is a bilinear
    // sample rather than a Load.
    float2 texel = float2(giProbeAtlasOrigin(slot, GI_IRRADIANCE_TILE, cascade)) + float(GI_BORDER) +
                   giOctEncode(dir) * float(GI_IRRADIANCE_PAYLOAD);
    return giIrradiance.SampleLevel(giIrradianceSampler, texel / giAtlasSize(GI_IRRADIANCE_TILE), 0).rgb;
}

/// Cosine-weighted mean incoming radiance at `P` for a surface facing `N`,
/// from the FINEST cascade that covers `P` with a one-probe margin.
/// `covered` reports whether any cascade answered at all.
///
/// Trilinear over the 8 surrounding probes, each weighted by a smooth
/// backface term ((dot + 1)/2)^2 so a probe on the far side of the surface
/// contributes almost nothing. Deliberately does NOT apply the Chebyshev
/// visibility test the PRIMARY sampler (gi_sample.hlsli) applies: this is a
/// secondary term feeding a heavily filtered accumulator, the visibility
/// moments are themselves still converging while the field fills, and the leak
/// it prevents is one bounce deep.
float3 rtHitProbeIrradiance(float3 P, float3 N, out bool covered) {
    covered = false;
    for (int c = 0; c < GI_CASCADE_COUNT; ++c) {
        float sp = giCascadeSpacing(c);
        float3 local = P / sp - float3(giCascadeBase[c].xyz);
        // One-probe margin: the 8-tap cage must lie wholly inside the window,
        // or a tap wraps around the torus and reads the opposite side of the
        // world.
        if (any(local < 1.0) || any(local > float3(GI_GRID_X - 2, GI_GRID_Y - 2, GI_GRID_Z - 2))) {
            continue;
        }
        float3 baseF = floor(local);
        float3 f = local - baseF;
        int3 b = int3(baseF);

        float3 sum = float3(0, 0, 0);
        float wsum = 0.0;
        [unroll]
        for (int i = 0; i < 8; ++i) {
            int3 o = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            int3 absolute = giCascadeBase[c].xyz + b + o;
            int slot = giWrappedProbeIndex(absolute);
            if (rtHitProbeState(c, slot) == GI_PROBE_UNINITIALIZED) {
                continue;
            }
            float3 tri = lerp(1.0 - f, f, float3(o));
            float3 toProbe = float3(absolute) * sp - P;
            float len = length(toProbe);
            float wrapped = len > 1e-4 ? (dot(toProbe / len, N) + 1.0) * 0.5 : 1.0;
            // The +0.02 floor keeps a probe cage from collapsing to zero weight
            // on a surface that faces away from all eight, which would leave
            // the hit unlit rather than dimly lit.
            float w = tri.x * tri.y * tri.z * (wrapped * wrapped + 0.02);
            sum += w * rtHitProbeTap(c, slot, N);
            wsum += w;
        }
        covered = true;
        return wsum > 1e-6 ? sum / wsum : float3(0, 0, 0);
    }
    return float3(0, 0, 0);
}

/**
 * Mean incoming radiance to use as the hit's ambient, in the probe atlas's
 * units (irradiance/pi).
 *
 * `giAmbientFallback` is what separates the two callers, and it is not a knob:
 *
 *  - PROBE RAYS pass 0. A probe's own bounce term must not invent hemisphere
 *    ambient where the field has no answer — the probe already collects sky
 *    through its ray MISSES, and adding a constant on top of that would both
 *    double-count it and break the bounce series' convergence argument (the
 *    series sums albedo^k only because each term is the previous field, not the
 *    previous field plus a constant).
 *  - REFLECTION RAYS pass 1. A reflected point outside the field's coverage —
 *    a rooftop three cascades away — has to be lit by SOMETHING, and the
 *    engine's documented no-GI ambient tier is exactly this hemisphere. It is
 *    also what makes reflections improve with GI OFF: hemisphere ambient plus a
 *    shadowed sun plus local lights beats a flat `skyColor * 0.25`.
 */
float3 rtHitAmbient(float3 P, float3 N) {
    float3 hemisphere = lerp(ambientGround, ambientSky, saturate(N.y * 0.5 + 0.5));
    if (giEnabled <= 0.5) {
        return giAmbientFallback > 0.5 ? hemisphere : float3(0, 0, 0);
    }
    bool covered;
    float3 field = rtHitProbeIrradiance(P, N, covered);
    if (!covered && giAmbientFallback > 0.5) {
        return hemisphere;
    }
    return field;
}

// ---------------------------------------------------------------------------
// Local (point/spot) lights
// ---------------------------------------------------------------------------
//
// The byte layout is render::lighting's, and the offsets below are the same
// ones cluster_lighting.hlsli hardcodes:
//   [u32 headerCount][u32 pointCount][u32 spotCount][u32 pad]
//   [LightHeader * capacity][PointLight * capacity][SpotLight * capacity]
//
// Read WITHOUT the cluster grid: a ray hit is a world-space point and the
// froxel grid is screen-space, so the only correct traversal here is the flat
// list. That also keeps the pass off `cluster_compute.hlsli`'s four extra
// read-only storage buffers, which no ray-tracing pass can afford next to the
// four acceleration-structure ones (SDL3 GPU's ceiling is eight).

uint rtHitPointLightsBase() { return 16u + giLightCapacity * 8u; }
uint rtHitSpotLightsBase() { return 16u + giLightCapacity * (8u + 48u); }
// SpotLight's stride is 80 bytes since R6 (source_radius), PointLight's still 48.

/// Smooth windowed inverse-square falloff. Operation-identical to
/// `pointFalloff` in clusters.hlsli — a ray hit and a pixel must agree about
/// how far a lamp reaches, or a reflection brightens at a distance the direct
/// term has already cut off.
float rtHitPointFalloff(float dist, float radius) {
    float r = saturate(dist / max(radius, 1e-4));
    float r2 = r * r;
    float win = saturate(1.0 - r2 * r2);
    return (win * win) / max(dist * dist, 1e-4);
}

/// IRRADIANCE (not radiance) at `P` from every packet light in range.
/// No shadow test: ray-traced local-light shadows are the froxel-RIS pass's
/// (R2b), which is screen-space and cannot answer for a world-space hit. An
/// unshadowed lamp in a secondary bounce is a far smaller error than no lamp.
float3 rtHitLocalLights(float3 P, float3 N) {
    float3 acc = float3(0, 0, 0);
    uint count = min(giLightCount, 256u);
    for (uint i = 0; i < count; ++i) {
        uint2 header = lightDataBuf.Load2(16u + i * 8u);
        if (header.x == 0u) {
            uint base = rtHitPointLightsBase() + header.y * 48u;
            uint4 a = lightDataBuf.Load4(base);
            uint4 b = lightDataBuf.Load4(base + 16u);
            float3 position = asfloat(a.xyz);
            float radius = asfloat(a.w);
            float3 toLight = position - P;
            float dist = length(toLight);
            if (dist > radius || dist < 1e-4) {
                continue;
            }
            float3 L = toLight / dist;
            float ndotl = saturate(dot(N, L));
            if (ndotl <= 0.0) {
                continue;
            }
            acc += asfloat(b.xyz) * asfloat(b.w) * rtHitPointFalloff(dist, radius) * ndotl;
        } else {
            uint base = rtHitSpotLightsBase() + header.y * 80u;
            uint4 a = lightDataBuf.Load4(base);
            uint4 b = lightDataBuf.Load4(base + 16u);
            uint4 c = lightDataBuf.Load4(base + 32u);
            uint4 d = lightDataBuf.Load4(base + 48u);
            float3 position = asfloat(a.xyz);
            float radius = asfloat(a.w);
            float3 toLight = position - P;
            float dist = length(toLight);
            if (dist > radius || dist < 1e-4) {
                continue;
            }
            float3 L = toLight / dist;
            float ndotl = saturate(dot(N, L));
            if (ndotl <= 0.0) {
                continue;
            }
            float cone = smoothstep(asfloat(c.w), asfloat(b.w), dot(asfloat(b.xyz), -L));
            if (cone <= 0.0) {
                continue;
            }
            acc += asfloat(c.xyz) * asfloat(d.x) * rtHitPointFalloff(dist, radius) * ndotl * cone;
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// The hit
// ---------------------------------------------------------------------------

/**
 * Outgoing radiance of a Lambertian hit at `P` with world-space normal `N`,
 * flat albedo `albedo` and flat emitted radiance `emissive` (R6).
 *
 * EMISSION IS ADDED, NOT LIT. It is radiance leaving the surface on its own
 * account, so it does not go through the 1/pi Lambertian divide and it is not
 * modulated by albedo, shadow or ambient — a glowing window is exactly as bright
 * seen in a reflection as it is seen directly, which is the whole claim. That
 * also means a probe ray hitting an emitter deposits its light into the probe
 * field, so bounce FROM emissives arrives with no additional machinery: it is
 * the same recursion that already carries bounce from sunlit walls.
 *
 * The one thing this deliberately does NOT do is double-count against the
 * emissive-light sampler (R6's records): those are DIRECT lighting for the
 * froxel-RIS pass at PRIMARY surfaces, while this is the radiance of a
 * SECONDARY hit. A reflection ray that lands on a lamp sees the lamp; the
 * primary pixel under the lamp is lit by the sampler. They never describe the
 * same transport path.
 *
 * `noise` only rotates calcShadow's PCSS poisson disk; every caller pins
 * `shadowEnabled` to PCF, which never reaches it, so it exists to keep the
 * call deterministic rather than to matter. Pass something stable per
 * (pixel, frame) — a texel coordinate — not a random.
 *
 * The caller owns the BACKFACE decision: a probe ray must report a backface hit
 * as black-and-negative-distance so the classify pass can count it, while a
 * reflection ray should just shade the flipped normal. Nothing here inspects
 * the ray direction.
 */
float3 rtShadeHit(float3 P, float3 N, float3 albedo, float3 emissive, float2 noise) {
    float3 toSun = -sunDirection;
    float ndotl = saturate(dot(N, toSun));

    // calcShadow selects a cascade by VIEW depth, so the hit needs one.
    float viewDepth = dot(P - cameraPosition, cameraForward);
    float vis = calcShadow(P, N, toSun, viewDepth, noise);

    // IRRADIANCE at the hit. The ambient is stored as mean radiance, so it
    // becomes irradiance by multiplying by pi (see the units note at the top).
    float3 E = sunColor * ndotl * vis;
    E += rtHitLocalLights(P, N);
    E += GI_PI * bounceScale * rtHitAmbient(P + N * normalBias, N);

    // Lambertian outgoing radiance, plus what the surface emits on its own.
    return albedo * E / GI_PI + emissive;
}

#endif // RT_HIT_SHADE_FUNCTIONS
