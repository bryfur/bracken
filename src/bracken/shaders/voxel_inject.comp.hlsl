// voxel_inject.comp.hlsl - turn one cascade's ALBEDO+OPACITY volume into its
// RADIANCE volume by lighting every occupied voxel. Phase RG2 of the raster GI
// path.
//
// This is the pass that makes the voxels EMIT. The voxelizer stored what each
// voxel is made of; this stores what it sends back out, which is what a cone
// integrates.
//
// ONE THREAD PER VOXEL, not per light. The alternative — splat each light into
// the voxels it reaches — is what light propagation volumes do, and it scales
// with (lights x reach) rather than with the volume, which for a village of
// hundreds of lanterns is far worse than the 885k lanes here. It also cannot
// shadow, and shadowing is most of what makes GI look like GI.
//
// THE SUN IS SHADOWED BY THE CSM the raster path already rendered, read through
// the same shadow_common.hlsli every other consumer uses. That is the whole
// trick of raster GI: the expensive visibility query the ray-traced path spends
// a BVH walk on is already sitting in a texture, paid for by the direct
// lighting pass.
//
// NO NORMAL IS STORED, so N.L cannot be evaluated per voxel — and that is a
// property of the representation, not an omission. A voxel is a volume element:
// it holds whatever geometry passed through it, facing every which way. The
// isotropic answer (light it as though it scatters equally in all directions,
// with the cosine folded into an average) is the standard VXGI approximation
// and is why voxel GI is soft and slightly over-bright at grazing sun angles.
// VOXEL_LAMBERT_AVERAGE is that fold, and it is an exact mean rather than a fit.

#include "voxel_common.hlsli"

[[vk::binding(0, 2)]]
cbuffer VoxelInjectUniforms : register(b0, space2) {
    // ---- the shadow_common.hlsli block, embedded verbatim ------------------
    // Same shape and same order as GiProbeTraceUniforms carries it, for the
    // same reason: shadow_common.hlsli names these uniforms and the including
    // shader owns their slots, so every consumer repeats the block rather than
    // the file inventing a binding it cannot know.
    float4x4 shadowLightViewProjection[4];
    float4 cascadeSplits;
    float4 cascadeTexelWorld;
    float4 cascadeDepthRange;
    float shadowMapResolution;
    float lightSize;
    float shadowEnabled;
    float lastActiveSplit;
    float fadeStart;
    float fadeEnd;
    float _shadowPad0;
    float _shadowPad1;

    // ---- this pass's own ---------------------------------------------------
    int voxelCascade;
    int3 voxelBase;
    /// Camera position and forward, for the view depth `calcShadow` selects a
    /// cascade with. A voxel has no pixel and therefore no view depth of its
    /// own; giving it the same projection a fragment at that world point would
    /// get is what makes the two agree about which cascade shadows them.
    float3 voxelCameraPosition;
    float voxelSunIntensity;
    float3 voxelCameraForward;
    float voxelSkyFloor;
    /// Direction the sun's light TRAVELS (engine convention).
    float3 voxelSunDirection;
    float _voxelInjectPad0;
    float3 voxelSunColor;
    float _voxelInjectPad1;
    /// Ambient floor colour, so a voxel the sun cannot reach is not perfectly
    /// black — otherwise the first bounce in shadow is zero and interiors read
    /// as caves. The probe field applies the real sky term; this is only what
    /// stops the voxel volume from being the thing that zeroes it.
    float3 voxelSkyColor;
    /// Lights in `voxelLightData`, and the capacity its offsets were built with.
    /// The two differ: the buffer is laid out for the capacity, not the count.
    uint voxelLightCount;
    uint voxelLightCapacity;
    uint _voxelInjectPad2;
    float2 _voxelInjectPad3;
};

// ---- set 0: the CSM atlas, exactly as shadow_common.hlsli names it ---------
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D shadowAtlas : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerComparisonState shadowCmpSampler : register(s0, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D shadowAtlasRaw : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState shadowRawSampler : register(s1, space0);

/// READ-ONLY. This pass only reads the albedo volume, and binding it read-write
/// is not merely imprecise — a compute pass's read-write slots are a different
/// binding array from its read-only storage textures, and asking for a
/// read-write slot the shader never writes faults inside the driver with no
/// diagnostic at all.

/// The frame's clustered lights, in ClusterGrid's layout — the same buffer
/// rt_hit_shade.hlsli reads, so a lantern contributes the same irradiance to a
/// voxel that it does to a traced hit.
[[vk::binding(2, 0)]] ByteAddressBuffer voxelLightData : register(t2, space0);

/// READ-WRITE, even though this pass only reads it.
///
/// A 3D texture bound as a compute pass's READ-ONLY storage texture reads back
/// as ZERO on this backend — silently, with no error and no fault, which made
/// the whole clipmap look empty. It was bound read-write originally; it got
/// changed to read-only while chasing what turned out to be an unrelated crash
/// (SDL's 3D mipmap generator), and the change outlived the theory.
[[vk::image_format("r32ui")]]
[[vk::binding(0, 1)]] RWTexture3D<uint> voxelAlbedo : register(u0, space1);
[[vk::image_format("rgba16f")]]
[[vk::binding(1, 1)]] RWTexture3D<float4> voxelRadiance : register(u1, space1);

#include "shadow_common.hlsli"

/// Mean of `max(dot(n, l), 0)` over a uniform sphere of orientations — the
/// isotropic stand-in for the per-voxel cosine a volume element cannot have.
/// Using it rather than 1 is what stops voxel GI from being twice as bright as
/// the surface lighting it is supposed to be bouncing.
#define VOXEL_LAMBERT_AVERAGE 0.5

/// Radiance an emissive byte of 255 stands for. Emission is HDR and eight bits
/// are not, so the byte is a fraction of this ceiling — high enough that a
/// lantern reads as a light source in the bounce, low enough that quantization
/// is invisible against it.
#define VOXEL_EMISSIVE_SCALE 8.0
#define VOXEL_PI 3.141592653589793

/// Smooth windowed inverse-square falloff. Operation-identical to
/// `rtHitPointFalloff` and to `pointFalloff` in clusters.hlsli — a voxel and a
/// pixel must agree about how far a lamp reaches, or bounced light brightens at
/// a distance the direct term has already cut off.
float voxelPointFalloff(float dist, float radius) {
    float r = saturate(dist / max(radius, 1e-4));
    float r2 = r * r;
    float win = saturate(1.0 - r2 * r2);
    return (win * win) / max(dist * dist, 1e-4);
}

[numthreads(4, 4, 4)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= uint(VOXEL_RESOLUTION))) {
        return;
    }

    const uint packed = voxelAlbedo[tid];
    if (packed == 0u) {
        // Empty voxel, written explicitly rather than skipped: the radiance
        // volume is not cleared separately, so a voxel that emptied since the
        // last revoxelization has to be told.
        voxelRadiance[tid] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    const float3 albedo = float3(float((packed >> 0) & 0xFFu), float((packed >> 8) & 0xFFu),
                                 float((packed >> 16) & 0xFFu)) /
                          255.0;

    // STORAGE TEXEL -> WORLD POINT. Storage and window agree only modulo the
    // resolution, so the absolute lattice coordinate has to be recovered by
    // wrapping the base into the same torus and taking the offset from there.
    // Getting this backwards puts every voxel's lighting one window away from
    // the geometry it belongs to, which reads as GI that is simply wrong rather
    // than as anything recognizable.
    const int3 baseWrapped = voxelWrap(voxelBase);
    int3 local = int3(tid) - baseWrapped;
    local = int3(voxelWrapAxis(local.x, VOXEL_RESOLUTION), voxelWrapAxis(local.y, VOXEL_RESOLUTION),
                 voxelWrapAxis(local.z, VOXEL_RESOLUTION));
    const int3 absolute = voxelBase + local;
    const float3 world = voxelCenter(voxelCascade, absolute);

    // A voxel has no surface normal. Handing `calcShadow` the direction toward
    // the light for both N and L makes its normal-offset bias push the sample
    // toward the light, which is the right direction for a volume element: it
    // errs toward "lit" rather than toward self-shadowing an occupied cell.
    const float3 toLight = -voxelSunDirection;
    const float viewDepth = dot(world - voxelCameraPosition, voxelCameraForward);
    const float shadow = calcShadow(world, toLight, toLight, viewDepth, float2(0.0, 0.0));

    const float3 sun = voxelSunColor * (voxelSunIntensity * shadow * VOXEL_LAMBERT_AVERAGE);

    // ---- sky ---------------------------------------------------------------
    // A voxel sees the whole sphere, so the honest ambient is the sky's average
    // radiance rather than one authored constant. The capture's TOP MIP is that
    // average — sampling any direction at the coarsest level returns the whole
    // cube filtered down to one texel, which is exactly the isotropic term a
    // volume element wants and costs one tap.
    // NO SKY TERM HERE, and that is the fix for a real over-brightness rather
    // than a saving. A voxel has no visibility information, so injecting the
    // sky's mean radiance lights every voxel equally — including the ones deep
    // inside a building, which is exactly where the sky should not reach.
    //
    // The CONE already applies sky, weighted by `1 - accum.a`: the fraction of
    // the cone that reached open air. That term IS visibility-aware, because
    // the cone had to get past the geometry to earn it. Injecting sky as well
    // counts it twice and un-occluded, which is what put this mode roughly three
    // stops over the traced path (auto_ev -1.24 against 0.56).
    // ---- local lights ------------------------------------------------------
    // THIS IS WHAT MAKES NIGHT WORK. Without it a voxel's only illumination is
    // the sun, so after dusk the entire clipmap goes black and the GI term takes
    // the frame down with it — every lantern in a village is invisible to the
    // bounce. Unshadowed, deliberately: local shadows here would need a second
    // visibility structure, and an unshadowed lamp in a SECOND BOUNCE is a far
    // smaller error than no lamp at all (the same call rt_hit_shade.hlsli makes,
    // for the same reason).
    float3 localLight = float3(0, 0, 0);
    {
        const uint pointBase = 16u + voxelLightCapacity * 8u;
        const uint spotBase = 16u + voxelLightCapacity * (8u + 48u);
        const uint count = min(voxelLightCount, 256u);
        for (uint i = 0; i < count; ++i) {
            const uint2 header = voxelLightData.Load2(16u + i * 8u);
            if (header.x == 0u) {
                const uint b0 = pointBase + header.y * 48u;
                const uint4 a = voxelLightData.Load4(b0);
                const uint4 b = voxelLightData.Load4(b0 + 16u);
                const float3 toLight = asfloat(a.xyz) - world;
                const float dist = length(toLight);
                const float radius = asfloat(a.w);
                if (dist > radius || dist < 1e-4) {
                    continue;
                }
                localLight += asfloat(b.xyz) * asfloat(b.w) *
                              voxelPointFalloff(dist, radius) * VOXEL_LAMBERT_AVERAGE;
            } else {
                const uint b0 = spotBase + header.y * 80u;
                const uint4 a = voxelLightData.Load4(b0);
                const uint4 b = voxelLightData.Load4(b0 + 16u);
                const uint4 c = voxelLightData.Load4(b0 + 32u);
                const uint4 d = voxelLightData.Load4(b0 + 48u);
                const float3 toLight = asfloat(a.xyz) - world;
                const float dist = length(toLight);
                const float radius = asfloat(a.w);
                if (dist > radius || dist < 1e-4) {
                    continue;
                }
                const float3 L = toLight / dist;
                const float cone = smoothstep(asfloat(c.w), asfloat(b.w), dot(asfloat(b.xyz), -L));
                if (cone <= 0.0) {
                    continue;
                }
                localLight += asfloat(c.xyz) * asfloat(d.x) * voxelPointFalloff(dist, radius) *
                              cone * VOXEL_LAMBERT_AVERAGE;
            }
        }
    }

    // Outgoing radiance = what arrived, times what the surface reflects, PLUS
    // what the surface emits on its own. The 1/PI that turns irradiance into
    // radiance for a Lambertian surface is deliberately NOT applied: the probe
    // field this feeds carries "cosine-weighted mean incoming radiance"
    // (gi_sample.hlsli's stated unit) and the cone integral that fills it
    // normalizes by its own weight sum, so a 1/PI here would only have to be
    // undone there.
    // EMISSIVE rides the albedo volume's own alpha byte as an INTENSITY, tinted
    // by the albedo. A separate volume would cost another 3.5 MB per cascade to
    // carry a channel that is zero almost everywhere, and an emitter's colour is
    // its albedo in every asset this engine loads — a glowing lantern glass is
    // orange because it IS orange.
    //
    // That is why the packed alpha is an emissive intensity rather than a
    // coverage: coverage is already implied by the voxel existing at all, which
    // is what `opacity` below reports.
    const float emissiveIntensity = float((packed >> 24) & 0xFFu) / 255.0;
    const float3 emissive = albedo * (emissiveIntensity * VOXEL_EMISSIVE_SCALE);

    // LAMBERTIAN OUTGOING RADIANCE, operation-for-operation what
    // rt_hit_shade.hlsli's rtShadeHit returns: `albedo * E / pi + emissive`,
    // where E is IRRADIANCE. `sun` and `localLight` are irradiance (they carry
    // the isotropic cosine in place of N.L); `sky` is already radiance and its
    // pi cancels, so it sits outside the divide.
    //
    // The /pi is not optional and leaving it out is not a small error: it made
    // the night frame land 11 luma above the no-GI baseline, where the traced
    // path barely moves. The two GI sources feed ONE probe field, so they must
    // agree about units or the same scene reads at two brightnesses.
    // LAMBERTIAN OUTGOING RADIANCE, operation-for-operation what
    // rt_hit_shade.hlsli's rtShadeHit returns: `albedo * E / pi + emissive`,
    // where E is IRRADIANCE. `sun` and `localLight` carry the isotropic cosine
    // in place of N.L, so they are irradiance and the /pi converts them.
    //
    // The /pi is not optional: the two GI sources feed ONE probe field, so they
    // must agree about units or the same scene reads at two brightnesses.
    voxelRadiance[tid] = float4(albedo * ((sun + localLight) / VOXEL_PI) + emissive, 1.0);
}
