// rt_shadow.comp.hlsl - ray-traced sun shadows over a software two-level BVH.
//
// There is no hardware ray tracing in SDL3 GPU, so this traverses a BVH the
// engine builds on the CPU (render/raytracing/bvh.hpp). Every design choice
// here exists to keep that affordable:
//
//  * NO PRIMARY RAYS. World position is reconstructed from the depth prepass,
//    so the only ray cast is the shadow ray itself. That is 1.44M rays saved
//    per frame at 1600x900 before anything else.
//  * ANY-HIT, NOT CLOSEST-HIT. A shadow ray only asks "is anything in the
//    way", so traversal returns on the first triangle it crosses instead of
//    sorting for the nearest. On occluded pixels this typically ends after a
//    handful of nodes.
//  * RAYS THAT CANNOT MATTER ARE NEVER CAST. A surface facing away from the
//    sun is already unlit, and sky pixels have no surface at all. Both are
//    rejected before traversal, which removes roughly half the dispatch.
//  * TWO LEVELS. The TLAS is walked in world space; only when an instance is
//    entered is the ray transformed into model space and its BLAS walked. The
//    scene's 6.2M drawn triangles are ~600k unique ones behind instancing.
//  * SHORT STACKS. Traversal stacks are sized to the deepest tree the builder
//    can produce, not to a comfortable round number: stack space is per-lane
//    registers, and register pressure caps occupancy, which caps latency
//    hiding, which is the whole game in a memory-bound traversal.
//
// SOFT, NOT HARD. The ray is jittered across the sun's disk (about half a
// degree), one sample per pixel per frame, so the output is a stochastic
// estimate of sun visibility rather than a binary answer. That is what
// produces real penumbrae — contact-sharp under a foot, wide under a distant
// roofline — and it costs exactly the same one ray as the hard version did.
// A single sample is of course pure noise on its own; rt_shadow_denoise_*.comp
// turns it back into an image, and NOTHING here is usable without them.
//
// Output is one BIT per pixel, packed 8x4 into a uint tile (rt_sampling.hlsli).
// The tile classifier downstream reads 32 pixels of visibility per fetch,
// which is what lets it skip filtering entirely over the large uniform regions
// that dominate a sunlit frame.
//
// THIS PASS DELIBERATELY OWNS NOTHING BUT VISIBILITY. It used to also read the
// GTAO texture and copy it into R of an RG8 mask, so the lighting pass could
// get both terms from one sampler slot. That coupling was a bug factory: the
// AO fetch used a texel coordinate derived from the DEPTH texture's dimensions
// while indexing the AO texture, so the two only agreed by luck. With AO off
// the engine binds a 1x1 white dummy there, every thread but (0,0) read out of
// bounds, an out-of-bounds Load returns 0.0, and the whole screen came back
// "fully occluded" — the entire ambient/IBL term zeroed scene-wide. It also
// point-downsampled full-res GTAO to the mask's resolution on every frame ray
// tracing was on. The mask now has its own R8 texture and its own fragment
// sampler slot (the "11-sampler DXC ceiling" that motivated the packing does
// not reproduce: model.frag compiles clean at 12).

#include "rt_sampling.hlsli"

// The BVH structs and the traversal walks are shared with every other traced
// pass — see rt_trace.hlsli for why it is included in two phases.
#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtShadowUniforms : register(b0, space2) {
    float4x4 invViewProj;   // clip -> world, for depth reconstruction
    float3 sunDirection;    // direction light TRAVELS (engine convention)
    float rayMaxDistance;   // ray length; shorter = cheaper and misses less
    float3 cameraPosition;
    float normalBias;       // push along the normal to stop self-shadowing
    uint2 outputSize;       // traced resolution (may be half the screen)
    uint tlasNodeCount;     // 0 = no acceleration structure this frame
    float depthBias;
    /// cos of the sun's angular RADIUS. 1.0 collapses the cone to a point and
    /// reproduces the old hard shadow exactly, which is what the deterministic
    /// smoke run pins so its output stays comparable frame to frame.
    float sunCosThetaMax;
    /// Advances the sample sequence. Must change every frame or the temporal
    /// filter accumulates the same sample forever and never converges.
    uint frameIndex;
    /// 0 when no prepass G-buffer exists this frame, in which case the normal
    /// is reconstructed from depth taps as it used to be — worse, but better
    /// than not running.
    uint hasGBuffer;
    /// Analytic capsule occluders in `shadowCapsules` (docs/rt_lighting_plan.md
    /// §5.8). 0 means the host emitted none this frame, which collapses the
    /// capsule loop to nothing and reproduces the pre-capsule trace exactly.
    uint shadowCapsuleCount;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float> sceneDepth : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState sceneDepthSampler : register(s0, space0);
// Prepass G-buffer: RG = octahedral world normal, B = roughness, A = metallic.
// Reading the REAL normal here rather than reconstructing one from four depth
// taps is the single biggest quality fix in this pass — see the allocation
// comment in scene_renderer.cpp for what the reconstruction did to foliage.
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> sceneNormal : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState sceneNormalSampler : register(s1, space0);

[[vk::binding(2, 0)]] StructuredBuffer<BvhNodeGpu> tlasNodes : register(t2, space0);
[[vk::binding(3, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t3, space0);
[[vk::binding(4, 0)]] StructuredBuffer<BvhNodeGpu> blasNodes : register(t4, space0);
[[vk::binding(5, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t5, space0);
// The frame's animated capsule occluders. A FIFTH read-only storage buffer,
// which this pass has room for (SDL3 GPU's ceiling is eight and the four
// acceleration buffers are all it spent) — unlike rt_local_shadow, which was
// already at the wall and has to ride a read-write slot for the same data.
[[vk::binding(6, 0)]] StructuredBuffer<ShadowCapsule> shadowCapsules : register(t6, space0);

// One bit of sun visibility per pixel, 8x4 pixels per uint. 1 = lit.
[[vk::binding(0, 1)]] RWTexture2D<uint> shadowBits : register(u0, space1);

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

// Geometric normal from neighbouring depth taps. Picks the closer neighbour on
// each axis so a depth discontinuity does not smear the normal across it,
// which would put shadow halos on every silhouette.
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
//
// The word is assembled in groupshared memory with an atomic OR rather than
// with WaveActiveBallot. Ballot would be one instruction, but it is only
// equivalent to this when the group maps onto the wave the way you assumed,
// and the wave width here is not a constant — this machine reports 32 while
// its own driver advertises a subgroup size of 64. An atomic OR into LDS is
// correct at every wave width, and against a BVH traversal its cost does not
// register.
groupshared uint g_tileBits;

[numthreads(8, 4, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    if (gi == 0) {
        g_tileBits = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    const int2 coord = int2(tid.xy);
    const bool inBounds = tid.x < outputSize.x && tid.y < outputSize.y;

    // "Lit" is the default for everything that is not a shaded surface facing
    // the sun: out-of-bounds padding lanes, and sky. Sky must read lit rather
    // than shadowed or the fog and skybox pick up a dark wash.
    bool lit = true;

    if (inBounds) {
        uint2 depthSize;
        sceneDepth.GetDimensions(depthSize.x, depthSize.y);
        const int2 depthCoord =
            int2((float2(coord) + 0.5) / float2(outputSize) * float2(depthSize));
        const float depth = sceneDepth.Load(int3(depthCoord, 0));

        // Reversed-Z: 0 is the far plane, so depth == 0 is sky. Casting a ray
        // from the far plane is the most expensive possible way to learn that.
        if (depth > 0.0) {
            float3 normal;
            if (hasGBuffer != 0u) {
                normal = octDecode(sceneNormal.Load(int3(depthCoord, 0)).xy);
            } else {
                normal = reconstructNormal(depthCoord, depthSize);
            }

            const float3 toLight = -sunDirection; // sunDirection is where light TRAVELS
            const float ndotl = dot(normal, toLight);

            // Facing away from the sun: already unlit by N.L in the lighting
            // pass, so a traversal would only confirm the dot product.
            if (ndotl <= 0.0) {
                lit = false;
            } else {
                float2 ndc = (float2(coord) + 0.5) / float2(outputSize) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                const float4 world = mul(invViewProj, float4(ndc, depth, 1.0));
                const float3 worldPos = world.xyz / world.w;

                // Bias along the normal, scaled by grazing angle. A constant
                // bias either acne-stripes flat surfaces or detaches contact
                // shadows on steep ones.
                const float slope = saturate(1.0 - ndotl);
                const float3 rayOrigin = worldPos +
                                         normal * (normalBias * (1.0 + slope * 4.0)) +
                                         toLight * depthBias;

                // The jittered sun-disk ray. Sampling the cone rather than its
                // axis is the entire difference between a hard shadow and a
                // penumbra that widens with occluder distance.
                const float2 xi = rtSample2D(uint2(coord), frameIndex);
                const float3 rayDir = sampleCone(toLight, sunCosThetaMax, xi);

                // A cone sample can land below the horizon on a grazing
                // surface; that ray would start by burrowing into the geometry
                // it came from, so treat it as occluded instead of tracing it.
                if (dot(normal, rayDir) <= 0.0) {
                    lit = false;
                } else {
                    lit = !sceneOccluded(rayOrigin, rayDir, rayMaxDistance);
                }
            }
        }
    }

    if (lit) {
        InterlockedOr(g_tileBits, 1u << gi);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi == 0) {
        shadowBits[int2(gid.xy)] = g_tileBits;
    }
}
