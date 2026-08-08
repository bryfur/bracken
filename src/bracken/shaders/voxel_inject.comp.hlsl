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
    float _voxelInjectPad2;
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
[[vk::binding(2, 0)]] Texture3D<uint> voxelAlbedo : register(t2, space0);

[[vk::image_format("rgba16f")]]
[[vk::binding(0, 1)]] RWTexture3D<float4> voxelRadiance : register(u0, space1);

#include "shadow_common.hlsli"

/// Mean of `max(dot(n, l), 0)` over a uniform sphere of orientations — the
/// isotropic stand-in for the per-voxel cosine a volume element cannot have.
/// Using it rather than 1 is what stops voxel GI from being twice as bright as
/// the surface lighting it is supposed to be bouncing.
#define VOXEL_LAMBERT_AVERAGE 0.5

[numthreads(4, 4, 4)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= uint(VOXEL_RESOLUTION))) {
        return;
    }

    const uint packed = voxelAlbedo.Load(int4(int3(tid), 0));
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
    const float opacity = float((packed >> 24) & 0xFFu) / 255.0;

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
    const float3 sky = voxelSkyColor * voxelSkyFloor;

    // Outgoing radiance = what arrived, times what the surface reflects. The
    // 1/PI that turns irradiance into radiance for a Lambertian surface is
    // deliberately NOT applied: the probe field this feeds carries
    // "cosine-weighted mean incoming radiance" (gi_sample.hlsli's stated unit)
    // and the cone integral that fills it normalizes by its own weight sum, so
    // a 1/PI here would only have to be undone there.
    voxelRadiance[tid] = float4(albedo * (sun + sky), opacity);
}
