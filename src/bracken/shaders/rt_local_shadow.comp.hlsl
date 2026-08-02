// rt_local_shadow.comp.hlsl — froxel-RIS ray-traced shadows for LOCAL
// (point/spot) lights. Phase R2b of docs/rt_lighting_plan.md §5.3, Ratio tier.
//
// ONE SHADOW RAY PER PIXEL MAKES EVERY LOCAL LIGHT CAST A SOFT SHADOW. That is
// the whole claim, and it is not a trick of resolution — the pass genuinely
// only casts one ray, and the reason it is enough is that the ray is aimed by a
// reservoir rather than by a loop.
//
// The pipeline per pixel:
//
//  1. RECONSTRUCT the surface: world position from the depth prepass, normal
//     from the prepass G-buffer's octahedral pair — exactly as
//     rt_shadow.comp.hlsl does, down to the fallback when no G-buffer exists.
//  2. CANDIDATES from the froxel. The 16x9x24 cluster grid the frame already
//     binned IS a light-culling structure; the pixel's froxel is a short list of
//     the only lights that can possibly reach it, and drawing candidates from it
//     costs one indirection instead of a world-space acceleration structure of
//     its own. (Our synthesis, §3 of the plan — this is the piece nobody else's
//     ReSTIR gets for free.)
//  3. RIS over M candidates drawn uniformly from that list, weighted by the
//     light's own unshadowed contribution (clusterLightTarget, shared verbatim
//     with the fragment path's accumulator so "the forward loop skips it" and
//     "the sampler weights it zero" are the same statement). One survives.
//  4. TEMPORAL MERGE with the reprojected previous-frame reservoir, whose light
//     index is resolved through the frame's prev->current remap — a vanished
//     light drops the history rather than silently becoming a different light.
//     This is what makes ~1 ray/frame behave like dozens.
//  5. ONE any-hit ray at a point sampled on the winner's emitter sphere. Same
//     sampleCone the sun uses, with cosThetaMax from the emitter radius over the
//     distance instead of the solar disk — so the penumbra widens with distance
//     from the occluder for exactly the same reason it does for the sun.
//
// Output is one BIT per pixel packed 8x4 into a uint tile, byte-for-byte the
// same format rt_shadow.comp.hlsl produces, because it goes through the SAME
// three denoiser pipelines against a second set of targets. A pixel whose
// froxel is empty writes LIT: no local light reaches it, so there is nothing
// for the mask to darken.
//
// WHAT THE MASK MEANS DOWNSTREAM. It is a visibility RATIO (Heitz 2018), and
// the forward pass multiplies its whole point+spot subtotal by it. With one
// reservoir sample the ratio estimate collapses to the selected light's
// visibility bit — p̂(y) x W is the estimate of the candidate sum, which is
// also the denominator, so the two cancel — which is why the analytic lighting
// stays sharp and only the SHADOWING is stochastic.
//
// ---------------------------------------------------------------------------
// THE FULL-DI TIER (R5c, docs/rt_lighting_plan.md §5.3), `diMode != 0`
// ---------------------------------------------------------------------------
//
// Same reservoir, same ray, same everything above. What changes is what is done
// with the winner: instead of writing only its visibility BIT and letting the
// forward loop supply the lighting, the pass evaluates the winner's own BRDF
// and writes two DEMODULATED radiance estimates, and the forward loop then does
// not run at all for point and spot lights.
//
// `estimate = f(y) * W` is the ReSTIR estimator of the WHOLE candidate sum, not
// of one light — which is the same fact the ratio tier leans on, spent
// differently. Ratio keeps p̂(y)*W in the denominator so the two cancel and only
// V survives; Full keeps W and carries the radiance, which is what makes it
// correct for coloured lights and for specular highlights the aggregate ratio
// can only flatten. See render/lighting/local_di.hpp for the split itself.
//
// ONE SHADER, NOT A VARIANT, and the reason is the sibling: R5b's
// rt_local_shadow_gradient.comp.hlsl replays this pass's sampling decisions
// exactly, and a second trace variant would have to be mirrored by a second
// gradient variant to stay a replay of the thing that ran. The DI block is a
// uniform-branch tail on a shader whose cost is dominated by a BVH walk — it
// adds no traversal, no memory traffic on the Ratio path (the two extra
// storage-texture slots take 1x1 stand-ins there), and its register pressure is
// paid only where the branch is taken.

#include "rt_sampling.hlsli"

// For the BRDF halves the DI block evaluates: ndfGGX, visibilitySmithGGX and
// diffuseBurley. PBR_NO_DERIVATIVES drops specularAARoughness, the one function
// in that header that calls ddx/ddy and therefore cannot exist in a compute
// module.
#define PBR_NO_DERIVATIVES
#include "pbr.hlsli"
#include "local_di.hlsli"

#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtLocalShadowUniforms : register(b0, space2) {
    float4x4 invViewProj;    // clip -> world, for depth reconstruction
    uint2 outputSize;        // traced resolution
    uint tlasNodeCount;      // 0 = no acceleration structure this frame
    float normalBias;        // push along the normal against self-shadowing
    float depthBias;         // push toward the light
    /// Emitter radius in world units, for the sphere-cap ray. One value for
    /// every light until per-light authoring arrives in R6 (§5.7).
    float lightSourceRadius;
    /// History M is capped at this multiple of the current frame's candidate
    /// count, so accumulated confidence cannot outvote the present.
    float historyMaxMRatio;
    /// Advances the sample sequence. Must change every frame or the denoiser
    /// accumulates the same sample forever and never converges.
    uint frameIndex;
    uint hasGBuffer;         // 0 = reconstruct the normal from depth taps
    uint historyValid;       // 0 on the first frame, after a resize, after a gap
    uint candidateCount;     // M, before the froxel-length clamp
    /// LocalRtShadowMode: 0/1 = Ratio (bits only), 2 = Full (bits + the two
    /// demodulated DI targets). Ratio binds 1x1 stand-ins for those targets and
    /// this branch is what keeps them unwritten.
    uint diMode;
    /// The eye, for the specular half's view vector. The Ratio tier never
    /// needed it — a visibility bit has no view dependence — which is why it
    /// arrives with R5c rather than having been there all along.
    float3 cameraPosition;
    float _pad0;
};

// NOTE THE ABSENCE of the depthA/depthB pair every other depth-reading pass
// here carries. The froxel lookup needs the surface's VIEW DEPTH, and the
// reversed-Z linearization those two terms drive is only ever used elsewhere to
// compare one depth against another — a use that survives a constant factor.
// This one does not: an absolute view depth that disagrees with the CPU
// binner's lands the pixel in a different Z SLICE, and a wrong froxel is an
// empty light list, which reads as "the traced pass does nothing" rather than
// as a bug. So the depth comes from `clusterParams.view` below — the very
// matrix ClusterGrid binned against — and cannot disagree with it by
// construction.

// --- set 0: sampled textures first, then read-only storage buffers ----------
// The order is not a style choice — see shaders/README.md, "Binding order
// inside set 0 is not free".

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float> sceneDepth : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState sceneDepthSampler : register(s0, space0);

// Prepass G-buffer: RG = octahedral world normal, B = roughness, A = metallic.
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> sceneNormal : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState sceneNormalSampler : register(s1, space0);

// UV-space motion, current minus previous (contracts/velocity_math.hpp), for
// the reservoir's own reprojection. The denoiser reprojects the DENOISED
// signal; this reprojects the SAMPLER STATE, and they are different things —
// history here decides which light gets the ray, not how bright the pixel is.
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture2D<float2> velocityTexture : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState velocitySampler : register(s2, space0);

[[vk::binding(3, 0)]] StructuredBuffer<BvhNodeGpu> tlasNodes : register(t3, space0);
[[vk::binding(4, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t4, space0);
[[vk::binding(5, 0)]] StructuredBuffer<BvhNodeGpu> blasNodes : register(t5, space0);
[[vk::binding(6, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t6, space0);

// Froxel + light access on the COMPUTE binding sets (R2a). Claims read-only
// storage buffers 7..10 and set-2 uniform slot 1.
//
// WITHOUT the stable-id buffer, deliberately: SDL3 GPU caps a compute pipeline
// at EIGHT read-only storage buffers, the four above are non-negotiable, and a
// temporal reservoir's question is the REMAP's ("where did the light I picked
// last frame go?"), not the ids'. cluster_compute.hlsli spells the trade out.
#define CLUSTER_COMPUTE_SSBO_BINDING_BASE 7
#define CLUSTER_COMPUTE_PARAMS_BINDING 1
#include "cluster_compute.hlsli"

/// One pixel's reservoir. Mirrors render::lighting::Reservoir (16 B) field for
/// field — that header carries the rationale, including why p̂ is NOT stored.
struct RtReservoir {
    uint lightIndex;
    float weight;
    float sampleCount;
    float depth;
};

#define RT_RESERVOIR_NO_LIGHT 0xFFFFFFFFu

// --- set 1: read-write storage textures, then read-write storage buffers ----

/// One bit of local-light visibility per pixel, 8x4 pixels per uint. 1 = lit.
[[vk::binding(0, 1)]] RWTexture2D<uint> localShadowBits : register(u0, space1);
/// The Full-DI tier's two outputs (R5c), at this pass's traced resolution.
/// RGB is demodulated radiance — the material factors this pass does not know
/// divided out — and the alpha lanes carry what the reconstruction downstream
/// cannot recover: the texel's linear view depth, and the Schlick fraction that
/// lets the forward shader rebuild a metal's Fresnel exactly. local_di.hlsli
/// owns the layout and the derivation.
///
/// In Ratio mode these are 1x1 stand-ins and nothing writes them. A storage
/// image with no explicit format may only be written, never read, which is all
/// this pass does with them.
[[vk::image_format("rgba16f")]]
[[vk::binding(1, 1)]] RWTexture2D<float4> diDiffuseOut : register(u1, space1);
[[vk::image_format("rgba16f")]]
[[vk::binding(2, 1)]] RWTexture2D<float4> diSpecularOut : register(u2, space1);
[[vk::binding(3, 1)]] RWStructuredBuffer<RtReservoir> reservoirOut : register(u3, space1);
/// LAST FRAME'S reservoirs — READ ONLY, despite the read-write declaration.
/// The eight read-only storage-buffer slots SDL3 allows a compute pipeline are
/// exactly spent above (four acceleration buffers + four cluster buffers), so
/// this one lives in set 1 instead. Nothing writes it; the ping-pong swaps the
/// two bindings every frame, both with cycle = false so the contents survive
/// into the frame that reads them.
[[vk::binding(4, 1)]] RWStructuredBuffer<RtReservoir> prevReservoirs : register(u4, space1);

// The buffers and `tlasNodeCount` above are what the walks bind to.
#define RT_TRACE_TRAVERSAL
#include "rt_trace.hlsli"

// World-space position at a depth texel, for normal reconstruction.
float3 worldAt(int2 c, uint2 size) {
    float d = sceneDepth.Load(int3(c, 0));
    float2 ndc = (float2(c) + 0.5) / float2(size) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 w = mul(invViewProj, float4(ndc, d, 1.0));
    return w.xyz / w.w;
}

// Geometric normal from neighbouring depth taps — the no-G-buffer fallback,
// identical to rt_shadow.comp.hlsl's (closer neighbour per axis, so a depth
// discontinuity does not smear the normal across it).
float3 reconstructNormal(int2 c, uint2 size) {
    float3 p = worldAt(c, size);
    float3 px = worldAt(int2(min(c.x + 1, int(size.x) - 1), c.y), size);
    float3 nx = worldAt(int2(max(c.x - 1, 0), c.y), size);
    float3 py = worldAt(int2(c.x, min(c.y + 1, int(size.y) - 1)), size);
    float3 ny = worldAt(int2(c.x, max(c.y - 1, 0)), size);

    float3 dx = length(px - p) < length(p - nx) ? (px - p) : (p - nx);
    float3 dy = length(py - p) < length(p - ny) ? (py - p) : (p - ny);
    float3 n = cross(dy, dx);
    float len = length(n);
    return len > 1e-6 ? n / len : float3(0.0, 1.0, 0.0);
}

// One tile per group: 8x4 = 32 threads, each owning one bit of the output word.
// Assembled with an atomic OR into groupshared rather than WaveActiveBallot,
// for the reason spelled out in rt_shadow.comp.hlsl: the wave width is not a
// compile-time constant on this stack, and an LDS OR is correct at every width.
groupshared uint g_tileBits;

[numthreads(8, 4, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    if (gi == 0) {
        g_tileBits = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    const int2 coord = int2(tid.xy);
    const bool inBounds = tid.x < outputSize.x && tid.y < outputSize.y;

    // "Lit" is the default for everything the pass cannot or need not answer
    // for: out-of-bounds padding lanes, sky, and — importantly — any pixel no
    // local light reaches. The mask MULTIPLIES the local subtotal, so 1 is the
    // identity and any other default would darken geometry no light touches.
    bool lit = true;

    RtReservoir outRes;
    outRes.lightIndex = RT_RESERVOIR_NO_LIGHT;
    outRes.weight = 0.0;
    outRes.sampleCount = 0.0;
    outRes.depth = 0.0;

    // Full-DI accumulators. Alpha stays 0 for anything the pass cannot answer
    // for — sky, and out-of-bounds lanes — which is the validity gate the DI
    // denoiser and the forward reconstruction both read.
    float4 diDiffuse = float4(0.0, 0.0, 0.0, 0.0);
    float4 diSpecular = float4(0.0, 0.0, 0.0, 0.0);

    if (inBounds) {
        uint2 depthSize;
        sceneDepth.GetDimensions(depthSize.x, depthSize.y);
        const int2 depthCoord =
            int2((float2(coord) + 0.5) / float2(outputSize) * float2(depthSize));
        const float depth = sceneDepth.Load(int3(depthCoord, 0));

        // Reversed-Z: 0 is the far plane, so depth == 0 is sky. And with zero
        // binned lights every buffer below holds LAST frame's contents, which
        // is why clusterTotalLights() is checked before any of them is read.
        if (depth > 0.0 && clusterTotalLights() > 0u) {
            // The G-buffer's whole texel: RG = octahedral world normal,
            // B = roughness, A = metallic. Ratio only ever needed the normal;
            // the DI block needs the material half too, and it is the same
            // fetch.
            const float4 gbuf = sceneNormal.Load(int3(depthCoord, 0));
            float3 normal;
            if (hasGBuffer != 0u) {
                normal = octDecode(gbuf.xy);
            } else {
                normal = reconstructNormal(depthCoord, depthSize);
            }

            const float2 uv = (float2(coord) + 0.5) / float2(outputSize);
            float2 ndc = uv * 2.0 - 1.0;
            ndc.y = -ndc.y;
            const float4 world = mul(invViewProj, float4(ndc, depth, 1.0));
            const float3 worldPos = world.xyz / world.w;
            // Positive distance forward of the camera, through the binner's own
            // view matrix (render/lighting/cluster_index.hpp's contract). A
            // right-handed view matrix looks down -z, hence the negation.
            const float viewDepth = -mul(clusterParams.view, float4(worldPos, 1.0)).z;

            // The DI validity gate, set for every SHADED pixel and not only for
            // pixels a light reached: a froxel that has just emptied must
            // report "geometry here, no local light" (rgb 0, alpha > 0) so the
            // denoiser accumulates toward black, where "no answer" (alpha 0)
            // would freeze the previous frame's lighting in place.
            diDiffuse.a = viewDepth;

            const uint froxel = clusterFroxelIndex(uv, viewDepth);
            const uint listCount = clusterFroxelLightCount(froxel);

            if (listCount > 0u) {
                // ---- 3. RIS over M uniform draws from the froxel list -------
                // Source pdf is 1/listCount, so the RIS weight of a candidate
                // is p̂ * listCount. Drawing WITH replacement is deliberate: it
                // keeps the estimator's source pdf exactly uniform whatever the
                // relationship between M and the list length, where sampling
                // without replacement would need a different normalization for
                // short lists.
                uint rng = rtSeed(uint2(coord), frameIndex);
                const uint m = min(candidateCount, listCount);
                const float listCountF = float(listCount);

                uint y = RT_RESERVOIR_NO_LIGHT;
                float targetPdf = 0.0;
                float wSum = 0.0;
                float mCount = 0.0;

                for (uint i = 0; i < m; ++i) {
                    const uint slot = min((uint)(rtNextFloat(rng) * listCountF), listCount - 1u);
                    const uint packed = clusterFroxelLight(froxel, slot);
                    const float p = clusterLightTarget(packed, worldPos, normal);
                    mCount += 1.0;
                    const float w = p * listCountF;
                    if (w > 0.0) {
                        wSum += w;
                        if (rtNextFloat(rng) * wSum < w) {
                            y = packed;
                            targetPdf = p;
                        }
                    }
                }

                // ---- 4. temporal merge --------------------------------------
                if (historyValid != 0u) {
                    const float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0);
                    const float2 prevUv = uv - velocity; // velocity = curr - prev
                    if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
                        const int2 prevCoord =
                            clamp(int2(prevUv * float2(outputSize)), int2(0, 0), int2(outputSize) - 1);
                        RtReservoir hist =
                            prevReservoirs[uint(prevCoord.y) * outputSize.x + uint(prevCoord.x)];

                        // Depth disocclusion, relative rather than absolute for
                        // the same reason the shadow denoiser's is: a fixed
                        // epsilon that works at 2 m is meaningless at 400 m.
                        const bool depthOk =
                            hist.depth > 0.0 && abs(hist.depth - viewDepth) <= 0.05 * max(viewDepth, 1.0);

                        // The light index is LAST frame's packed index. Resolve
                        // it, or drop the history: a vanished light must not
                        // silently become whatever now occupies its slot.
                        uint mapped = RT_RESERVOIR_NO_LIGHT;
                        if (depthOk && hist.lightIndex != RT_RESERVOIR_NO_LIGHT &&
                            hist.sampleCount > 0.0) {
                            mapped = clusterRemapPrevLight(hist.lightIndex);
                        }

                        if (mapped != LIGHT_REMAP_INVALID && mapped != RT_RESERVOIR_NO_LIGHT) {
                            // M clamp: history cannot outvote the present.
                            float histM = min(hist.sampleCount, max(mCount, 1.0) * historyMaxMRatio);
                            // p̂ re-evaluated HERE. This is also the normal test
                            // the merge does not otherwise need: a light behind
                            // or out of range of the new surface evaluates to
                            // zero and contributes nothing but M.
                            const float pHere = clusterLightTarget(mapped, worldPos, normal);
                            const float w = pHere * hist.weight * histM;
                            const float mBefore = mCount;
                            mCount += 1.0;
                            if (w > 0.0) {
                                wSum += w;
                                if (rtNextFloat(rng) * wSum < w) {
                                    y = mapped;
                                    targetPdf = pHere;
                                }
                            }
                            mCount = mBefore + histM;
                        }
                    }
                }

                // W = w_sum / (M * p̂(y)).
                const float denom = mCount * targetPdf;
                outRes.lightIndex = y;
                outRes.weight = denom > 0.0 ? wSum / denom : 0.0;
                outRes.sampleCount = mCount;
                outRes.depth = viewDepth;

                // ---- 5. one shadow ray at the winner -------------------------
                if (y != RT_RESERVOIR_NO_LIGHT) {
                    const ClusterLightSample geom = clusterLightGeometry(y);
                    const float3 toLight = geom.position - worldPos;
                    const float dist = length(toLight);
                    const float3 L = toLight / max(dist, 1e-4);

                    // Sphere cap subtended by the emitter. sin(theta_max) =
                    // r / d, clamped at 1 for a surface inside the emitter —
                    // where cosThetaMax collapses to 0, the cone becomes the
                    // full hemisphere, and the answer is noise either way.
                    const float sinThetaMax = min(lightSourceRadius / max(dist, 1e-4), 1.0);
                    const float cosThetaMax = sqrt(saturate(1.0 - sinThetaMax * sinThetaMax));

                    const float ndotl = dot(normal, L);
                    if (ndotl <= 0.0) {
                        // Already unlit by N.L in the forward loop, and the
                        // target function agrees (it returns 0), so this can
                        // only be reached when every candidate was zero.
                        lit = true;
                    } else {
                        const float slope = saturate(1.0 - ndotl);
                        const float3 rayOrigin = worldPos +
                                                 normal * (normalBias * (1.0 + slope * 4.0)) +
                                                 L * depthBias;
                        const float2 xi = rtSample2D(uint2(coord), frameIndex);
                        const float3 rayDir = sampleCone(L, cosThetaMax, xi);

                        if (dot(normal, rayDir) <= 0.0) {
                            // The cap sample went below the horizon: that ray
                            // would start by burrowing into the surface it came
                            // from. Treat it as occluded, as the sun pass does.
                            lit = false;
                        } else {
                            // Stop short of the emitter itself. The light is not
                            // geometry, but whatever it is mounted on usually is.
                            const float rayLen = max(dist - depthBias - normalBias, 0.0);
                            lit = rayLen <= 0.0 ? true : !sceneOccluded(rayOrigin, rayDir, rayLen);
                        }
                    }
                }

                // ---- 6. the Full-DI tier's demodulated estimate -------------
                //
                // Deliberately AFTER the ray and outside its N.L early-out: the
                // BRDF is evaluated once, here, from the winner and the bit the
                // ray produced, so the two tiers cannot disagree about which
                // light was picked or whether it was visible.
                if (diMode >= 2u && y != RT_RESERVOIR_NO_LIGHT) {
                    const ClusterLightShading sh = clusterLightShading(y, worldPos);
                    const float ndotl = max(dot(normal, sh.L), 0.0);
                    if (sh.valid && ndotl > 0.0) {
                        const float3 Vdir = normalize(cameraPosition - worldPos);
                        const float NdotV = max(dot(normal, Vdir), 1e-4);
                        const float3 hsum = Vdir + sh.L;
                        const float3 H = hsum / max(length(hsum), 1e-5);
                        const float NdotH = max(dot(normal, H), 0.0);
                        const float LdotH = max(dot(sh.L, H), 0.0);

                        // The G-buffer's material half. It is the same
                        // roughness the forward pass shades with — the prepass
                        // writes the MR-modulated, specular-AA-widened value —
                        // which is what makes the split reconstructable there.
                        const float roughness = saturate(gbuf.z);
                        const float metallic = saturate(gbuf.w);

                        const float schlick = localDiSchlickFraction(LdotH);
                        const float Fd = localDiDielectricFresnel(schlick);
                        const float D = ndfGGX(NdotH, roughness);
                        const float Vis = visibilitySmithGGX(NdotV, ndotl, roughness);
                        // diffuseBurley with a white albedo IS the (ls*vs/PI)
                        // scalar the split calls for — called rather than
                        // re-typed, so pbr.hlsli stays the one owner of the
                        // curve the forward pass will multiply an albedo into.
                        const float burley =
                            diffuseBurley(float3(1.0, 1.0, 1.0), roughness, NdotV, ndotl, LdotH).x;

                        // E = the ReSTIR estimate of the WHOLE point+spot sum:
                        // the winner's incident radiance, its N.L, the traced
                        // visibility bit, and the reservoir's unbiasing weight.
                        const float3 E = sh.radiance * (ndotl * (lit ? 1.0 : 0.0) * outRes.weight);

                        diDiffuse.rgb = ((1.0 - Fd) * (1.0 - metallic) * burley) * E;
                        diSpecular.rgb = (D * Vis * Fd) * E;
                        diSpecular.a = schlick;
                    }
                }
            }
        }
    }

    if (inBounds) {
        reservoirOut[uint(coord.y) * outputSize.x + uint(coord.x)] = outRes;
        if (diMode >= 2u) {
            diDiffuseOut[coord] = diDiffuse;
            diSpecularOut[coord] = diSpecular;
        }
    }

    if (lit) {
        InterlockedOr(g_tileBits, 1u << gi);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi == 0) {
        localShadowBits[int2(gid.xy)] = g_tileBits;
    }
}
