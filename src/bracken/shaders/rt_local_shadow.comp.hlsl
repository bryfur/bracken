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
//  4b. SPATIAL MERGE (R7a) with k neighbours' temporal reservoirs, under
//     PAIRWISE MIS. See "TWO STAGES" below.
//  5. ONE any-hit ray at a point sampled on the winner's emitter sphere. Same
//     sampleCone the sun uses, with cosThetaMax from the emitter radius over the
//     distance instead of the solar disk — so the penumbra widens with distance
//     from the occluder for exactly the same reason it does for the sun.
//
// ---------------------------------------------------------------------------
// TWO STAGES, ONE PIPELINE (R7a, docs/rt_lighting_plan.md §5.3's deferred
// "optional spatial pass (3-5 neighbours, pairwise MIS)")
// ---------------------------------------------------------------------------
//
// This shader is dispatched TWICE per frame, with `stage` selecting which half
// runs:
//
//   stage 0 (RESAMPLE): steps 2-4. Writes a reservoir and nothing else.
//   stage 1 (REUSE + TRACE): step 4b, then step 5, then the DI block. Writes the
//                            final reservoir, the visibility bits and the three
//                            DI targets.
//
// WHY TWO DISPATCHES AND NOT ONE. A spatial pass reads its NEIGHBOURS' temporal
// output, and a compute dispatch has no cross-workgroup barrier — inside one
// dispatch a thread can only see what its own 8x4 group wrote. Reuse confined to
// a group would be neither the documented radius nor free of tile-boundary
// structure (the pixels at a tile edge would reuse a systematically different
// neighbourhood from the pixels at its centre, which reads as an 8x4 grid in the
// noise). So the split is forced by the memory model, not chosen.
//
// WHY THE RAY MOVED AFTER THE REUSE. §5.3 prescribes one ray for the FINAL
// winner. Tracing before the spatial merge would shadow a light the pixel then
// discards, and the estimate would pair one light's radiance with another
// light's visibility — a bias with no upper bound, worst exactly where the
// spatial pass helps most (a pixel whose own candidates were poor).
//
// WHY ONE PIPELINE AND NOT TWO. R5b's rt_local_shadow_gradient.comp.hlsl replays
// this pass's decisions, and every resolution it has to reproduce — the per-light
// emitter radius, the emissive epoch, the capsule set, the sample sequence — is
// written once here. A second module would be a second place for those to drift.
// The bindings are identical between the stages; what the CPU changes is which
// reservoir buffer sits in each of the two slots (stage 0 reads LAST FRAME'S and
// writes the temporal scratch; stage 1 reads the temporal scratch and writes the
// frame's final) and which stage the uniform names.
//
// WHAT THE GRADIENT REPLAY THEREFORE SEES, and the contract that keeps it a
// replay: `reservoirOut` after stage 1 holds THE LIGHT THE RAY WAS AIMED AT, and
// the ray's cap sample is `rtSample2D(coord, frameIndex)` at the pixel's OWN
// coordinate. Both statements were true before R7a and are true after it — the
// spatial merge changes WHICH light lands in that record, not the record's
// meaning nor the sample stream — so the replay reproduces the ray that ran
// without knowing spatial reuse happened at all. That is the whole reason the
// merge writes its result to the reservoir the gradient reads, rather than
// keeping a "pre-spatial" record around for the replay's benefit.
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
// ---------------------------------------------------------------------------
// EMISSIVE MESH LIGHTS (R6, docs/rt_lighting_plan.md §5.7) — AND WHY THEY ARE
// A FULL-TIER FEATURE
// ---------------------------------------------------------------------------
//
// A second candidate stream joins the froxel one: emissive triangles, drawn
// from a scene-wide power-weighted alias table (rt_emissive.hlsli). The
// reservoir, the ray and the estimator are otherwise unchanged; the RIS weights
// carry the balance, and that derivation is written out at the draw site below.
//
// THEY ONLY WORK IN THE FULL-DI TIER, and this is a real limitation rather than
// an unfinished corner. The Ratio tier's entire output is a scalar V in
// `Sigma(contribution x V) / Sigma(contribution)`, which the forward pass
// multiplies its analytic point+spot subtotal by. That subtotal does not
// contain the emissive triangle and cannot: the forward shader has no emissive
// light list, no way to build one, and no budget for one. A ratio estimator can
// only SCALE lighting the forward loop already evaluates — there is no value of
// V that ADDS a light. Any attempt to smuggle the emissive contribution into
// the numerator makes the mask a number that is no longer a visibility ratio,
// and the first thing it breaks is the R5c energy gate that keeps the two tiers
// honest against each other.
//
// The Full tier has no such problem: it writes demodulated RADIANCE, and
// radiance from any number of sources adds. So the CPU side sends
// `emissiveCount = 0` in Ratio, which makes every expression below collapse to
// the R5c one bit for bit.
//
// What Ratio still gets: emissive surfaces GLOW (rt_hit_shade.hlsli adds the
// emission term to every traced hit, so they are bright in reflections) and
// they FEED GI (a probe ray landing on one deposits its radiance into the probe
// field, and the field lights the frame). What they do not do in Ratio is cast
// their own direct shadows.
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
    /// R6 — emissive mesh lights. `emissiveCount` is 0 whenever the scene has
    /// no emitter above the power threshold OR the tier is Ratio, and 0 is what
    /// makes every weight below collapse to exactly the R5c expression.
    uint emissiveCount;
    /// Sum of every record's power — the alias table's normalizing constant,
    /// so a record's discrete pdf is `power / emissiveTotalPower`.
    float emissiveTotalPower;
    /// M_e, emissive candidates drawn per pixel.
    uint emissiveCandidateCount;
    /// 0 when the record set was renumbered since last frame (a TLAS topology
    /// rebuild), in which case an emissive index in the history names a
    /// different triangle and the history must be dropped.
    uint emissiveHistoryValid;
    /// Analytic capsule occluders in `shadowCapsules` (§5.8). 0 collapses the
    /// capsule loop to nothing and reproduces the pre-capsule trace exactly.
    uint shadowCapsuleCount;
    /// R7a. 0 = RESAMPLE (candidates + temporal merge), 1 = REUSE + TRACE.
    /// See "TWO STAGES, ONE PIPELINE" at the top of this file.
    uint stage;
    /// k — spatial neighbours attempted per pixel. 0 turns the reuse off and
    /// makes stage 1's merge the identity, which reproduces the pre-R7a image
    /// BIT for bit (render/lighting/reservoir.hpp's `PairwiseSpatialMerge`
    /// returns the canonical unchanged, and the ray's sample stream is
    /// untouched). That is the escape hatch and the A/B control at once.
    uint spatialNeighbors;
    /// Radius of the neighbour ring, in TRACED texels.
    float spatialRadius;
    uint _pad0;
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

// R6's emissive record type, so the binding block below can declare its buffer.
// The FUNCTIONS half comes after that declaration, and after
// cluster_light_load.hlsli above has defined `clusterTargetLuminance`.
#define RT_EMISSIVE_TYPES
#include "rt_emissive.hlsli"

/// One pixel's reservoir. Mirrors render::lighting::Reservoir (16 B) field for
/// field — that header carries the rationale, including why p̂ is NOT stored.
struct RtReservoir {
    uint lightIndex;
    float weight;
    float sampleCount;
    float depth;
};

#define RT_RESERVOIR_NO_LIGHT 0xFFFFFFFFu

/// SOURCE TAG (R6). A reservoir's selection now comes from one of two domains,
/// and the estimator has to know which — an analytic packed light index and an
/// emissive record index are both small integers and would otherwise be
/// indistinguishable. Bit 31 is free by construction: packed light indices are
/// bounded by MAX_LIGHTS (256) and record indices by k_max_emissive_records
/// (8192), so neither can ever collide with the flag or with each other.
///
/// RT_RESERVOIR_NO_LIGHT is all-ones and therefore also has the flag set, which
/// is why every test below checks it FIRST.
#define RT_RESERVOIR_EMISSIVE_FLAG 0x80000000u
#define RT_RESERVOIR_EMISSIVE_MASK 0x7FFFFFFFu

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
/// R7b's third DI target: the winning light's octahedral direction. RG16F
/// rather than a lane on one of the two above, because both alpha lanes already
/// carry data the forward pass cannot reconstruct. Like them, a 1x1 stand-in in
/// Ratio mode.
[[vk::image_format("rg16f")]]
[[vk::binding(3, 1)]] RWTexture2D<float2> diDirectionOut : register(u3, space1);
[[vk::binding(4, 1)]] RWStructuredBuffer<RtReservoir> reservoirOut : register(u4, space1);
/// THE RESERVOIRS THIS DISPATCH READS — READ ONLY, despite the read-write
/// declaration. The eight read-only storage-buffer slots SDL3 allows a compute
/// pipeline are exactly spent above (four acceleration buffers + four cluster
/// buffers), so this one lives in set 1 instead. Nothing writes it.
///
/// WHICH buffer it is depends on the stage, and the CPU decides: stage 0 reads
/// LAST FRAME's final reservoirs (the temporal history), stage 1 reads THIS
/// frame's stage-0 output (the spatial neighbourhood). One binding, two
/// meanings, because the two stages ask structurally the same question of it —
/// "what did some other reservoir select, and how confident is it".
[[vk::binding(5, 1)]] RWStructuredBuffer<RtReservoir> prevReservoirs : register(u5, space1);
/// R6's emissive light records + their alias rows — READ ONLY, for the same
/// reason `prevReservoirs` is: the eight read-only storage-buffer slots SDL3
/// allows a compute pipeline were already spent to the last one before this
/// phase existed. When the scene has no emitter this binds a 1-element stand-in
/// and `emissiveCount` is 0, so nothing indexes it.
[[vk::binding(6, 1)]] RWStructuredBuffer<EmissiveRecord> emissiveRecords : register(u6, space1);
/// The frame's analytic capsule occluders (docs/rt_lighting_plan.md §5.8) —
/// READ ONLY, riding a read-write slot for the third time in this shader and
/// for the third time for the same reason: the eight read-only storage-buffer
/// slots SDL3 allows a compute pipeline are spent to the last one above (four
/// acceleration + four cluster), and R2b set the precedent that a read-only use
/// of a read-write binding is legal and is what the budget leaves. The sun
/// pass, which has read-only headroom, binds the same data at t6 in set 0.
/// When the host emits no capsules this binds a 1-element stand-in and
/// `shadowCapsuleCount` is 0, so nothing indexes it.
[[vk::binding(7, 1)]] RWStructuredBuffer<ShadowCapsule> shadowCapsules : register(u7, space1);

#define RT_EMISSIVE_FUNCTIONS
#include "rt_emissive.hlsli"

// The buffers and `tlasNodeCount` above are what the walks bind to.
#define RT_TRACE_CAPSULES
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

// ---------------------------------------------------------------------------
// Surface reconstruction, factored because R7a needs it at the NEIGHBOURS too
// ---------------------------------------------------------------------------
//
// Pairwise MIS asks three questions per neighbour, and two of them
// ("what is my sample worth THERE", "what is its sample worth THERE") are
// evaluated at the neighbour's own shading point. So the spatial pass cannot
// merely gate on a stored depth the way the temporal one does — it has to
// reconstruct the neighbour's surface. That is the honest cost of pairwise
// weights over 1/Z, and it is why this used to be inline and is now a function.

struct RtSurface {
    float3 pos;
    float3 normal;
    float viewDepth;   ///< positive, through clusterParams.view — the binner's own
    float roughness;   ///< G-buffer B
    float metallic;    ///< G-buffer A
    bool valid;        ///< false for sky
};

RtSurface rtSurfaceAt(int2 coord) {
    RtSurface s;
    s.pos = float3(0.0, 0.0, 0.0);
    s.normal = float3(0.0, 1.0, 0.0);
    s.viewDepth = 0.0;
    s.roughness = 1.0;
    s.metallic = 0.0;
    s.valid = false;

    uint2 depthSize;
    sceneDepth.GetDimensions(depthSize.x, depthSize.y);
    const int2 depthCoord =
        int2((float2(coord) + 0.5) / float2(outputSize) * float2(depthSize));
    const float depth = sceneDepth.Load(int3(depthCoord, 0));
    // Reversed-Z: 0 is the far plane, so depth == 0 is sky.
    if (depth <= 0.0) {
        return s;
    }

    // The G-buffer's whole texel: RG = octahedral world normal, B = roughness,
    // A = metallic. Ratio only ever needed the normal; the DI block needs the
    // material half too, and it is the same fetch.
    const float4 gbuf = sceneNormal.Load(int3(depthCoord, 0));
    s.normal = (hasGBuffer != 0u) ? octDecode(gbuf.xy) : reconstructNormal(depthCoord, depthSize);
    s.roughness = saturate(gbuf.z);
    s.metallic = saturate(gbuf.w);

    const float2 uv = (float2(coord) + 0.5) / float2(outputSize);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    const float4 world = mul(invViewProj, float4(ndc, depth, 1.0));
    s.pos = world.xyz / world.w;
    // Positive distance forward of the camera, through the binner's own view
    // matrix (render/lighting/cluster_index.hpp's contract). A right-handed view
    // matrix looks down -z, hence the negation.
    s.viewDepth = -mul(clusterParams.view, float4(s.pos, 1.0)).z;
    s.valid = true;
    return s;
}

/// p̂ of a stored selection at an arbitrary surface point, in either domain.
/// The one function that lets everything downstream of the reservoir stay
/// source-agnostic — and the reason the pairwise weights below never have to
/// know whether they are pairing analytic lights or emissive triangles.
float rtReservoirTarget(uint y, float3 pos, float3 normal) {
    if (y == RT_RESERVOIR_NO_LIGHT) {
        return 0.0;
    }
    if ((y & RT_RESERVOIR_EMISSIVE_FLAG) != 0u) {
        const uint rec = y & RT_RESERVOIR_EMISSIVE_MASK;
        return rec < emissiveCount ? rtEmissiveTarget(emissiveRecords[rec], pos, normal) : 0.0;
    }
    return clusterLightTarget(y, pos, normal);
}

// ---------------------------------------------------------------------------
// The spatial neighbour ring (R7a)
// ---------------------------------------------------------------------------
//
// EIGHT fixed directions at 45-degree steps, at two alternating radii, walked
// with a STRIDE OF 3 from a rotating start. Every part of that is load-bearing:
//
//  * fixed and documented, rather than a Poisson draw, so the pattern is a pure
//    function of (pixel, frame) and two runs of the same frame reuse the same
//    neighbours — the replayability §5.0 requires and R5b's gradient depends on.
//  * stride 3 is coprime with 8, so any k <= 3 consecutive picks span 135
//    degrees of the ring rather than clustering in one quadrant. A stride of 1
//    would reuse three neighbours that see almost the same thing.
//  * the start rotates with `frameIndex` AND with the pixel, so a surface
//    accumulates a different neighbourhood every frame (which is what makes k=3
//    behave like a much wider gather after a few frames) without the whole
//    screen sampling in lockstep (which would print the ring into the noise).
//  * alternating radii, because a single ring is an annulus and shows as one.
static const float2 RT_SPATIAL_RING[8] = {
    float2(1.0, 0.0),        float2(0.70711, 0.70711),   float2(0.0, 1.0),
    float2(-0.70711, 0.70711), float2(-1.0, 0.0),        float2(-0.70711, -0.70711),
    float2(0.0, -1.0),       float2(0.70711, -0.70711),
};
static const float RT_SPATIAL_RING_SCALE[8] = {1.0, 0.55, 1.0, 0.55, 1.0, 0.55, 1.0, 0.55};

/// Relative view-depth difference beyond which a neighbour describes a
/// different surface. The temporal merge's tolerance, deliberately: this asks
/// the same question about the same kind of reprojection error.
#define RT_SPATIAL_DEPTH_TOLERANCE 0.05
/// Normal agreement below which a neighbour is not worth its three target
/// evaluations. NOTE THAT THIS IS A VARIANCE GATE, NOT A BIAS ONE: the pairwise
/// MIS weights are unbiased across a crease too, so a neighbour let through here
/// would not darken or brighten anything — it would just contribute a sample
/// with p̂_c near zero and cost three evaluations to do it. 0.9 is ~25 degrees.
#define RT_SPATIAL_NORMAL_TOLERANCE 0.9

// One tile per group: 8x4 = 32 threads, each owning one bit of the output word.
// Assembled with an atomic OR into groupshared rather than WaveActiveBallot,
// for the reason spelled out in rt_shadow.comp.hlsl: the wave width is not a
// compile-time constant on this stack, and an LDS OR is correct at every width.
groupshared uint g_tileBits;

[numthreads(8, 4, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    // `stage` is a uniform, so every thread in the dispatch takes the same
    // branch and the group barriers below stay uniformly reached.
    if (gi == 0) {
        g_tileBits = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    const int2 coord = int2(tid.xy);
    const bool inBounds = tid.x < outputSize.x && tid.y < outputSize.y;
    const uint index = uint(coord.y) * outputSize.x + uint(coord.x);

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
    // A direction is only meaningful where the diffuse alpha says so; +Y is the
    // "nothing here" encoding and is never read past that gate.
    float2 diDirection = localDiDirEncode(float3(0.0, 1.0, 0.0));

    RtSurface surf;
    surf.pos = float3(0.0, 0.0, 0.0);
    surf.normal = float3(0.0, 1.0, 0.0);
    surf.viewDepth = 0.0;
    surf.roughness = 1.0;
    surf.metallic = 0.0;
    surf.valid = false;
    if (inBounds) {
        surf = rtSurfaceAt(coord);
    }
    // R6 widened the second half of this test. `clusterTotalLights() > 0` still
    // guards every froxel buffer read below — with zero binned lights they hold
    // LAST frame's contents — but it must no longer gate the whole pass,
    // because an emissive mesh is a light the cluster grid knows nothing about.
    // A scene lit only by glowing windows has zero binned lights and is not dark.
    const bool shaded =
        inBounds && surf.valid && (clusterTotalLights() > 0u || emissiveCount > 0u);

    if (stage == 0u) {
        // =================================================================
        // STAGE 0 — RESAMPLE: candidates from the froxel + the temporal merge.
        // Writes a reservoir and nothing else. No ray, no BVH, no DI.
        // =================================================================
        if (shaded) {
            const float2 uv = (float2(coord) + 0.5) / float2(outputSize);
            const float3 worldPos = surf.pos;
            const float3 normal = surf.normal;
            const float viewDepth = surf.viewDepth;

            uint froxel = 0u;
            uint listCount = 0u;
            if (clusterTotalLights() > 0u) {
                froxel = clusterFroxelIndex(uv, viewDepth);
                listCount = clusterFroxelLightCount(froxel);
            }

            // ---- the two candidate budgets --------------------------------
            //
            // M_a from the froxel, M_e from the emissive alias table.
            // `emissiveCount` is 0 in the Ratio tier by construction (the CPU
            // side never sends records there — see the tier note in
            // docs/rt_lighting_plan.md §5.7), so M_e is 0, `mCur` is `mA`, both
            // scale factors below are exactly 1.0, and every expression in this
            // block reduces to the R5c one BIT FOR BIT rather than merely
            // closely.
            const uint mA = min(candidateCount, listCount);
            const uint mE = min(emissiveCandidateCount, emissiveCount);
            const float mCur = float(mA + mE);

            // ---- MULTI-SOURCE RIS: where the balance actually lives --------
            //
            // Two candidate streams over DISJOINT domains — analytic lights and
            // emissive records — whose integrals must be SUMMED, not averaged.
            // That distinction is the whole subtlety here, and getting it wrong
            // is an energy error rather than a noise one:
            //
            //   * The temporal merge below combines two estimates of the SAME
            //     integral, so a confidence-weighted convex combination
            //     (M_prev/(M_cur+M_prev) and friends) is exactly right, and it
            //     is what R2b shipped.
            //   * The two CURRENT streams estimate DIFFERENT integrals. Feeding
            //     them into one reservoir with a shared 1/M normalization would
            //     deliver (M_a*I_a + M_e*I_e)/(M_a+M_e) — a convex combination
            //     of the two halves, i.e. the whole frame dimmed toward
            //     whichever source was drawn more often.
            //
            // The fix is the balance heuristic evaluated per candidate:
            // m_i(x) = p_i(x) / sum_j p_j(x) over all M draws. With disjoint
            // domains every cross term is zero, so m collapses to 1/M_a for an
            // analytic candidate and 1/M_e for an emissive one. Folding those
            // into the stream weights and then keeping the SHIPPED finalize
            // (W = wSum / (M_total * p̂)) means each weight carries a factor
            // M_cur/M_source — which is these two scalars, and nothing else in
            // the estimator changes.
            //
            //   analytic i:  w = (M_cur/M_a) * p̂(x_i) * listCount
            //   emissive j:  w = (M_cur/M_e) * p̂(y_j) * totalPower / power_j
            //   history:     w = M_prev * p̂_here(y_prev) * W_prev   (unchanged)
            //   W = wSum / ((M_a + M_e + M_prev) * p̂(y))            (unchanged)
            //
            // Expectation check, which is the thing worth re-deriving if anyone
            // touches this: E[sum of analytic weights]/M_total = I_a, likewise
            // I_e, and the history term contributes the convex share the
            // confidence weighting intends. Sum = I_a + I_e. The M_a and M_e
            // cancel against the counts, which is why the code below is two
            // multiplications rather than a restructured estimator.
            const float sourceScaleA = mA > 0u ? mCur / float(mA) : 0.0;
            const float sourceScaleE = mE > 0u ? mCur / float(mE) : 0.0;

            if (mA + mE > 0u) {
                // ---- 3. RIS over M uniform draws from the froxel list -------
                // Source pdf is 1/listCount, so the RIS weight of a candidate
                // is p̂ * listCount. Drawing WITH replacement is deliberate: it
                // keeps the estimator's source pdf exactly uniform whatever the
                // relationship between M and the list length, where sampling
                // without replacement would need a different normalization for
                // short lists.
                uint rng = rtSeed(uint2(coord), frameIndex);
                const uint m = mA;
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
                    const float w = p * listCountF * sourceScaleA;
                    if (w > 0.0) {
                        wSum += w;
                        if (rtNextFloat(rng) * wSum < w) {
                            y = packed;
                            targetPdf = p;
                        }
                    }
                }

                // ---- 3b. EMISSIVE candidates from the alias table (R6) ------
                //
                // The distribution is GLOBAL — power over the whole scene, not
                // over the pixel's froxel — because an emissive triangle is not
                // binned into the cluster grid and cannot be: the grid holds
                // point/spot lights with a range, and a triangle has neither.
                // Power weighting is what keeps a global draw useful anyway:
                // the bright emitters near the camera dominate the table, and a
                // record on the far side of the village is drawn with
                // proportionally tiny probability AND scores p̂ ~ 0 when it is.
                // A per-froxel emissive list is the obvious next step and is
                // recorded as such in the plan; it is not needed at this scale.
                //
                // Note the RNG discipline: these draws come AFTER the analytic
                // loop and are skipped entirely when mE is 0, so the sample
                // stream a no-emissive scene sees is untouched.
                for (uint e = 0; e < mE; ++e) {
                    const float u1 = rtNextFloat(rng);
                    const float u2 = rtNextFloat(rng);
                    const uint rec = rtEmissiveSampleIndex(emissiveCount, u1, u2);
                    const EmissiveRecord er = emissiveRecords[rec];
                    const float p = rtEmissiveTarget(er, worldPos, normal);
                    mCount += 1.0;
                    // Discrete source pdf: power / totalPower. Its reciprocal is
                    // what turns p̂ into an unbiased RIS weight.
                    const float pdf = er.power / max(emissiveTotalPower, 1e-6);
                    const float w = pdf > 0.0 ? p * sourceScaleE / pdf : 0.0;
                    if (w > 0.0) {
                        wSum += w;
                        if (rtNextFloat(rng) * wSum < w) {
                            y = rec | RT_RESERVOIR_EMISSIVE_FLAG;
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

                        // The stored index is LAST frame's, in whichever of the
                        // two domains its source tag names. Resolve it, or drop
                        // the history — the same rule for both domains, with
                        // two different translations:
                        //
                        //   analytic: through the prev->current light remap, so
                        //     a vanished light does not silently become
                        //     whatever now occupies its packed slot.
                        //   emissive: through the record EPOCH. Record indices
                        //     are stable while the TLAS topology is, and a
                        //     topology rebuild renumbers them wholesale, so
                        //     `emissiveHistoryValid` is a single frame-wide
                        //     answer rather than a per-record table. Dropping
                        //     the history on a rebuild is CORRECT and not a
                        //     limitation: the alternative is a reservoir
                        //     pointing at an unrelated triangle, which is
                        //     exactly the failure the light remap exists to
                        //     prevent on the other side.
                        uint mapped = RT_RESERVOIR_NO_LIGHT;
                        bool histEmissive = false;
                        if (depthOk && hist.lightIndex != RT_RESERVOIR_NO_LIGHT &&
                            hist.sampleCount > 0.0) {
                            if ((hist.lightIndex & RT_RESERVOIR_EMISSIVE_FLAG) != 0u) {
                                const uint rec = hist.lightIndex & RT_RESERVOIR_EMISSIVE_MASK;
                                if (emissiveHistoryValid != 0u && rec < emissiveCount) {
                                    mapped = rec;
                                    histEmissive = true;
                                }
                            } else {
                                mapped = clusterRemapPrevLight(hist.lightIndex);
                            }
                        }

                        if (mapped != LIGHT_REMAP_INVALID && mapped != RT_RESERVOIR_NO_LIGHT) {
                            // M clamp: history cannot outvote the present.
                            float histM = min(hist.sampleCount, max(mCount, 1.0) * historyMaxMRatio);
                            // p̂ re-evaluated HERE. This is also the normal test
                            // the merge does not otherwise need: a light behind
                            // or out of range of the new surface evaluates to
                            // zero and contributes nothing but M.
                            const float pHere =
                                histEmissive
                                    ? rtEmissiveTarget(emissiveRecords[mapped], worldPos, normal)
                                    : clusterLightTarget(mapped, worldPos, normal);
                            const float w = pHere * hist.weight * histM;
                            const float mBefore = mCount;
                            mCount += 1.0;
                            if (w > 0.0) {
                                wSum += w;
                                if (rtNextFloat(rng) * wSum < w) {
                                    y = histEmissive ? (mapped | RT_RESERVOIR_EMISSIVE_FLAG)
                                                     : mapped;
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
            }
        }

        if (inBounds) {
            reservoirOut[index] = outRes;
        }
        return;
    }

    // =====================================================================
    // STAGE 1 — REUSE + TRACE.
    // =====================================================================
    if (shaded) {
        const float3 worldPos = surf.pos;
        const float3 normal = surf.normal;

        // The DI validity gate, set for every SHADED pixel and not only for
        // pixels a light reached: a froxel that has just emptied must report
        // "geometry here, no local light" (rgb 0, alpha > 0) so the denoiser
        // accumulates toward black, where "no answer" (alpha 0) would freeze
        // the previous frame's lighting in place.
        diDiffuse.a = surf.viewDepth;

        // ---- 4b. SPATIAL MERGE, pairwise MIS ---------------------------
        //
        // Transliterated from render::lighting::PairwiseSpatialMerge, which
        // derives the weights, proves they partition unity at every point
        // (including the two degenerate ones), and is tested against brute
        // force. The shape here is that struct's begin/add_neighbor/finalize,
        // inlined because a reservoir does not survive a function boundary in
        // HLSL without costing registers to no purpose.
        const RtReservoir canonical = prevReservoirs[index];
        const float pCanonical = rtReservoirTarget(canonical.lightIndex, worldPos, normal);

        outRes = canonical;

        if (spatialNeighbors > 0u) {
            // A stream decorrelated from stage 0's: reusing rtSeed(coord,
            // frameIndex) would draw the SAME uniforms the candidate loop drew,
            // which correlates "which candidate won" with "which neighbour
            // won" for no benefit. The golden-ratio constant is the same one
            // rtSeed itself mixes with.
            uint sRng = rtSeed(uint2(coord), frameIndex ^ 0x9E3779B9u);
            const uint start =
                (frameIndex + uint(coord.x) * 3u + uint(coord.y) * 5u) & 7u;

            uint y = RT_RESERVOIR_NO_LIGHT;
            float targetPdf = 0.0;
            float wSum = 0.0;
            float mSum = 0.0;
            float canonicalMis = 0.0;
            uint k = 0u;

            for (uint i = 0u; i < spatialNeighbors; ++i) {
                const uint ring = (start + i * 3u) & 7u;
                const int2 nCoord =
                    coord + int2(round(RT_SPATIAL_RING[ring] * spatialRadius *
                                       RT_SPATIAL_RING_SCALE[ring]));
                // Rejected rather than clamped: clamping would make every edge
                // pixel reuse ITSELF, which is a merge with a perfectly
                // correlated sample and buys nothing but a wrong confidence.
                if (nCoord.x < 0 || nCoord.y < 0 || nCoord.x >= int(outputSize.x) ||
                    nCoord.y >= int(outputSize.y)) {
                    continue;
                }
                const RtReservoir other =
                    prevReservoirs[uint(nCoord.y) * outputSize.x + uint(nCoord.x)];
                if (other.lightIndex == RT_RESERVOIR_NO_LIGHT || other.sampleCount <= 0.0) {
                    continue; // holds nothing: not an input, and not counted in k
                }
                const RtSurface ns = rtSurfaceAt(nCoord);
                if (!ns.valid) {
                    continue;
                }
                if (abs(ns.viewDepth - surf.viewDepth) >
                    RT_SPATIAL_DEPTH_TOLERANCE * max(surf.viewDepth, 1.0)) {
                    continue;
                }
                if (dot(ns.normal, normal) < RT_SPATIAL_NORMAL_TOLERANCE) {
                    continue;
                }

                ++k;
                // The three target evaluations pairwise MIS needs.
                const float pHere = rtReservoirTarget(other.lightIndex, worldPos, normal);
                const float pThere = rtReservoirTarget(other.lightIndex, ns.pos, ns.normal);
                const float pCanonThere =
                    rtReservoirTarget(canonical.lightIndex, ns.pos, ns.normal);

                // a_i(X_i): the neighbour's share of its own sample.
                const float aNum = other.sampleCount * pThere;
                const float aDen = aNum + canonical.sampleCount * pHere;
                const float ai = aDen > 0.0 ? aNum / aDen : 0.0;

                // b_i(X_c): the canonical's share of ITS sample, this pairing.
                const float bNum = canonical.sampleCount * pCanonical;
                const float bDen = bNum + other.sampleCount * pCanonThere;
                canonicalMis += bDen > 0.0 ? bNum / bDen : 0.0;

                mSum += other.sampleCount;

                const float w = ai * pHere * other.weight;
                if (w > 0.0) {
                    wSum += w;
                    if (rtNextFloat(sRng) * wSum < w) {
                        y = other.lightIndex;
                        targetPdf = pHere;
                    }
                }
            }

            if (k > 0u) {
                // The canonical streams LAST: its MIS weight is a sum over
                // every accepted neighbour and cannot be known before them.
                const float wc = canonicalMis * pCanonical * canonical.weight;
                mSum += canonical.sampleCount;
                if (wc > 0.0) {
                    wSum += wc;
                    if (rtNextFloat(sRng) * wSum < wc) {
                        y = canonical.lightIndex;
                        targetPdf = pCanonical;
                    }
                }
                // W = sum(w) / (k * p̂_c(y)). NO division by M: the MIS weights
                // already normalize, which is the structural difference from
                // the temporal finalize above. The 1/k is factored out of every
                // weight so it cannot touch the selection.
                const float denom = float(k) * targetPdf;
                outRes.lightIndex = y;
                outRes.weight = denom > 0.0 ? wSum / denom : 0.0;
                outRes.sampleCount = mSum;
                // outRes.depth stays the canonical's — the record is this
                // pixel's from here on, and a neighbour's depth would answer
                // next frame's disocclusion test about the wrong point.
            }
        }

        const uint y = outRes.lightIndex;

        // ---- 5. one shadow ray at the FINAL winner ----------------------
        //
        // An EMISSIVE winner takes the branch below instead: the ray
        // goes at a point sampled uniformly on the triangle, which is
        // the same trick the sphere cap plays for an analytic light —
        // scatter the ray across the emitter and the penumbra comes out
        // of the sampling rather than out of a filter.
        if (y != RT_RESERVOIR_NO_LIGHT && (y & RT_RESERVOIR_EMISSIVE_FLAG) != 0u) {
            const EmissiveRecord er = emissiveRecords[y & RT_RESERVOIR_EMISSIVE_MASK];
            const float2 xi = rtSample2D(uint2(coord), frameIndex);
            const float3 pt = rtEmissiveSamplePoint(er, xi);
            const float3 toPt = pt - worldPos;
            const float dist = length(toPt);
            const float3 L = toPt / max(dist, 1e-4);
            const float ndotl = dot(normal, L);
            if (ndotl <= 0.0 || dist <= 1e-4) {
                lit = true; // the target function already scored this 0
            } else {
                const float slope = saturate(1.0 - ndotl);
                const float3 rayOrigin = worldPos +
                                         normal * (normalBias * (1.0 + slope * 4.0)) +
                                         L * depthBias;
                // THE EMITTER MUST NOT OCCLUDE ITS OWN LIGHT. Unlike an
                // analytic light, this one IS geometry in the TLAS, and
                // the ray is aimed exactly at a point on it — so an
                // any-hit walk to the full distance hits the emitter
                // every single time and every emissive light is fully
                // shadowed. Stopping short by a RELATIVE margin (rather
                // than the absolute bias the analytic path uses) is what
                // keeps that true across four orders of magnitude of
                // scene scale: 0.2% of the distance is under a
                // millimetre at arm's length and still clears the
                // triangle at 500 units. Anything genuinely between the
                // surface and the emitter is longer than that margin by
                // construction, so nothing real is missed.
                const float rayLen = min(dist * 0.998, dist - depthBias) - normalBias;
                lit = rayLen <= 0.0 ? true : !sceneOccluded(rayOrigin, L, rayLen);
            }
        } else if (y != RT_RESERVOIR_NO_LIGHT) {
            const ClusterLightSample geom = clusterLightGeometry(y);
            const float3 toLight = geom.position - worldPos;
            const float dist = length(toLight);
            const float3 L = toLight / max(dist, 1e-4);

            // Sphere cap subtended by the emitter. sin(theta_max) =
            // r / d, clamped at 1 for a surface inside the emitter —
            // where cosThetaMax collapses to 0, the cone becomes the
            // full hemisphere, and the answer is noise either way.
            //
            // R6: the radius is PER LIGHT now
            // (contracts/light.hpp, `source_radius`), with 0 meaning
            // "the host did not author one" and falling back to the
            // pass's global default — which is what every light got
            // before this phase, so an unauthored scene is unchanged.
            const float emitterRadius =
                geom.source_radius > 0.0 ? geom.source_radius : lightSourceRadius;
            const float sinThetaMax = min(emitterRadius / max(dist, 1e-4), 1.0);
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

        // ---- 6. the Full-DI tier's demodulated estimate -----------------
        //
        // Deliberately AFTER the ray and outside its N.L early-out: the
        // BRDF is evaluated once, here, from the winner and the bit the
        // ray produced, so the two tiers cannot disagree about which
        // light was picked or whether it was visible.
        if (diMode >= 2u && y != RT_RESERVOIR_NO_LIGHT) {
            // Both domains answer through the same three fields, in the
            // same units — which is the point of rt_emissive.hlsli
            // mirroring ClusterLightShading's shape rather than
            // inventing its own: everything below this line is
            // source-agnostic, so the emissive tier cannot drift from
            // the analytic one's demodulation.
            ClusterLightShading sh;
            if ((y & RT_RESERVOIR_EMISSIVE_FLAG) != 0u) {
                const EmissiveSample es =
                    rtEmissiveShading(emissiveRecords[y & RT_RESERVOIR_EMISSIVE_MASK], worldPos);
                sh.L = es.L;
                sh.radiance = es.radiance;
                sh.dist = es.dist;
                sh.valid = es.valid;
            } else {
                sh = clusterLightShading(y, worldPos);
            }
            const float ndotl = max(dot(normal, sh.L), 0.0);
            if (sh.valid && ndotl > 0.0) {
                const float3 Vdir = normalize(cameraPosition - worldPos);
                const float NdotV = max(dot(normal, Vdir), 1e-4);
                const float3 hsum = Vdir + sh.L;
                const float3 H = hsum / max(length(hsum), 1e-5);
                const float NdotH = max(dot(normal, H), 0.0);
                const float LdotH = max(dot(sh.L, H), 0.0);

                // The G-buffer's material half. It is the same roughness the
                // forward pass shades with — the prepass writes the
                // MR-modulated, specular-AA-widened value — which is what makes
                // the split reconstructable there.
                const float roughness = surf.roughness;
                const float metallic = surf.metallic;

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

                // E = the ReSTIR estimate of the WHOLE point+spot sum: the
                // winner's incident radiance, its N.L, the traced visibility
                // bit, and the reservoir's unbiasing weight. UNCHANGED by R7b —
                // the cosine stays here, and the forward pass corrects it with a
                // bounded ratio built from the direction target below. Moving it
                // out instead inflates this value by 1/cos at a grazing light,
                // which is a firefly source in a buffer about to be blurred;
                // render/lighting/local_di.hpp records that A/B.
                const float3 E = sh.radiance * (ndotl * (lit ? 1.0 : 0.0) * outRes.weight);

                diDiffuse.rgb = ((1.0 - Fd) * (1.0 - metallic) * burley) * E;
                diSpecular.rgb = (D * Vis * Fd) * E;
                diSpecular.a = schlick;
                diDirection = localDiDirEncode(sh.L);
            }
        }
    }

    if (inBounds) {
        reservoirOut[index] = outRes;
        if (diMode >= 2u) {
            diDiffuseOut[coord] = diDiffuse;
            diSpecularOut[coord] = diSpecular;
            diDirectionOut[coord] = diDirection;
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
