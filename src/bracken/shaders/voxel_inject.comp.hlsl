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
// WITH ONE CAVEAT, and it is the reason `voxelSunVisibility` exists below: that
// texture covers the VIEW, and this volume is a camera-centred box that reaches
// well behind and past it. `calcShadow`'s out-of-frustum escapes both return
// FULL LIGHT, which is correct for a fragment and wrong for a voxel, so the
// cascade a voxel falls in is looked up directly rather than through the view
// depth. What no cascade covers at all is VOXEL_SUN_UNKNOWN_VISIBILITY.
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
    /// UNREAD, along with `voxelSkyFloor` above, and deliberately still here:
    /// this cbuffer is a byte-for-byte mirror of `gpu::VoxelInjectUniforms` and
    /// removing a field is a layout change. Injecting sky was tried and removed
    /// — see the "NO SKY TERM HERE" block below — because a voxel carries no
    /// visibility and the cone applies the same sky weighted by how much of
    /// itself reached open air.
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
/// Using it rather than 1 is what stops voxel GI from being four times as
/// bright as the surface lighting it is supposed to be bouncing.
///
/// (1/4pi) * integral over the sphere of max(n.l, 0) = (1/4pi) * pi = 1/4. The
/// tempting 0.5 is the mean of the cosine over the LIT HEMISPHERE only, which
/// is a different quantity and not the one a voxel full of geometry facing
/// every direction wants. The parity target is `rtShadeHit`, whose sun term is
/// `sunColor * saturate(dot(N, toSun)) * vis`, and the orientation-average of
/// that is a quarter.
#define VOXEL_LAMBERT_AVERAGE 0.25

/// Radiance an emissive byte of 255 stands for. Emission is HDR and eight bits
/// are not, so the byte is a fraction of this ceiling — high enough that a
/// lantern reads as a light source in the bounce, low enough that quantization
/// is invisible against it.
#define VOXEL_EMISSIVE_SCALE 8.0
#define VOXEL_PI 3.141592653589793

/// Sun visibility for a voxel no cascade covers at all — the far corners of the
/// coarsest cascade, outside every shadow map's footprint.
///
/// 1.0, which is what the code did for the whole out-of-domain volume before the
/// cascade scan below existed. It is the right answer for open terrain, which is
/// most of what is out there, and it is deliberately NOT the "conservative" 0.5:
/// a blanket half would darken the correct majority to fix the incorrect
/// minority and would put a hard seam at the boundary of a region the scan has
/// already emptied. If this is ever changed it must be measured on the two
/// GiMode::VoxelProbes pins, by auto_ev.
#define VOXEL_SUN_UNKNOWN_VISIBILITY 1.0

/// Does `cascade`'s shadow map actually cover this world point?
///
/// `calcShadow` is written for FRAGMENTS, which are inside the view frustum by
/// construction, and both of its escapes return FULL LIGHT: a view depth past
/// `lastActiveSplit`, and a light-space projection outside [0,1]. A voxel is not
/// a fragment. The clipmap is a camera-CENTRED box — cascade 0 spans +/-1200
/// world units in every direction and cascade 2 +/-4800 — so roughly half of it
/// sits BEHIND the camera at a NEGATIVE view depth, which `selectShadowCascade`
/// routes to cascade 0, whose ortho window is fitted around the nearest frustum
/// slice in FRONT of the camera and cannot contain it. Every such voxel was
/// injected as fully sunlit, which made the clipmap's contents depend on where
/// the camera happened to be pointing when that cascade was last relit.
///
/// The atlas usually does have the answer: shadow casters are culled against the
/// LIGHT frustum rather than the camera's, and the last cascade is fitted as a
/// bounding sphere of the whole far slice, so its quadrant genuinely contains
/// occluders behind the viewer. This is the test that lets injection go and find
/// the cascade that covers the point instead of taking calcShadow's escape.
///
/// Mirrors `sampleShadowCascade`'s projection exactly, normal-offset bias
/// included. The y flip does not change a bounds test and is kept only so the
/// two cannot drift apart.
bool voxelShadowCascadeContains(int cascade, float3 world, float3 N) {
    float4 lightSpacePos =
        mul(shadowLightViewProjection[cascade], float4(world + N * cascadeTexelWorld[cascade], 1.0));
    float3 sc = lightSpacePos.xyz / lightSpacePos.w;
    sc.x = sc.x * 0.5 + 0.5;
    sc.y = 1.0 - (sc.y * 0.5 + 0.5);
    return sc.x >= 0.0 && sc.x <= 1.0 && sc.y >= 0.0 && sc.y <= 1.0 && sc.z >= 0.0 && sc.z <= 1.0;
}

/// Sun visibility for a voxel, over the whole clipmap rather than over the view
/// frustum.
float voxelSunVisibility(float3 world, float3 toLight, float viewDepth) {
    if (shadowEnabled < 0.5) {
        return 1.0; // shadows off is shadows off
    }
    // IN DOMAIN: exactly what a fragment at this point receives, cascade blend
    // and distance fade included. Bit-identical to what this pass always did,
    // which is the point — the fix must not move the voxels that were right.
    const int cascade = selectShadowCascade(viewDepth);
    if (viewDepth >= 0.0 && viewDepth < lastActiveSplit &&
        voxelShadowCascadeContains(cascade, world, toLight)) {
        return calcShadow(world, toLight, toLight, viewDepth, float2(0.0, 0.0));
    }
    // OUT OF DOMAIN: take the first cascade that covers the point. Scanning all
    // four is safe even when fewer are active — get_shadow_uniforms clones the
    // last active cascade's matrix into the inactive slots, so a clone can only
    // match when the cascade it was cloned from already matched at a lower
    // index, and the ascending scan returns there first.
    //
    // N == L, so NdotL is 1, the slope term vanishes and the receiver bias is
    // the isotropic 0.5*texelWorld/depthRange — the right choice for a volume
    // element and the same direction of error the normal-offset argument above
    // makes.
    [unroll]
    for (int c = 0; c < 4; ++c) {
        if (voxelShadowCascadeContains(c, world, toLight)) {
            return sampleShadowCascade(c, world, toLight, 1.0, float2(0.0, 0.0), shadowEnabled);
        }
    }
    return VOXEL_SUN_UNKNOWN_VISIBILITY;
}

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

    // A voxel has no surface normal. Handing the shadow lookup the direction
    // toward the light for both N and L makes its normal-offset bias push the
    // sample toward the light, which is the right direction for a volume
    // element: it errs toward "lit" rather than toward self-shadowing an
    // occupied cell.
    const float3 toLight = -voxelSunDirection;
    const float viewDepth = dot(world - voxelCameraPosition, voxelCameraForward);
    const float shadow = voxelSunVisibility(world, toLight, viewDepth);

    const float3 sun = voxelSunColor * (voxelSunIntensity * shadow * VOXEL_LAMBERT_AVERAGE);

    // ---- sky ---------------------------------------------------------------
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
    // what the surface emits on its own.
    //
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

    // LAMBERTIAN OUTGOING RADIANCE, term for term what rt_hit_shade.hlsli's
    // rtShadeHit returns MINUS its third irradiance term: `albedo * E / pi +
    // emissive`, where E is IRRADIANCE. `sun` and `localLight` are irradiance
    // (they carry the isotropic cosine in place of N.L), so the /pi converts
    // them.
    //
    // The /pi is not optional and leaving it out is not a small error: it made
    // the night frame land 11 luma above the no-GI baseline, where the traced
    // path barely moves. The two GI sources feed ONE probe field, so they must
    // agree about units or the same scene reads at two brightnesses.
    //
    // THE MISSING TERM is rtShadeHit's `GI_PI * bounceScale * rtHitAmbient(...)`
    // — the probe field re-entering itself, which is what makes the traced field
    // multi-bounce. This pass binds no probe atlas, so the voxel field is
    // SINGLE-bounce: interiors and courtyards, where the second bounce is most
    // of the light, come out short. Closing it means feeding the previous
    // frame's irradiance atlas back in here.
    //
    // The alpha is 1 for any occupied voxel and 0 for an empty one, so mip-0
    // coverage is BINARY — every fractional alpha in the volume is produced by
    // the mip chain. The cone compositor reads this pair as PREMULTIPLIED
    // (voxel_common.hlsli), which is what the box-filtered chain makes of it.
    voxelRadiance[tid] = float4(albedo * ((sun + localLight) / VOXEL_PI) + emissive, 1.0);
}
