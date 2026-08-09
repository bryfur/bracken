// Full-resolution resolve for the ray-traced reflection signal.
//
// WHAT THIS PASS IS depends on where `GraphicsSettings::rt_reflection_resolution`
// is set, and it is FULL by default. At Full the bilinear reconstruction below
// collapses to a single tap of weight 1 (the source texel IS the pixel), and
// what remains — which is not incidental — is the full-resolution material gate
// and the receiving surface's BRDF. At Half/Quarter the four-tap
// depth/normal/material-guided upsample is additionally what keeps a reflecting
// stone texel off the grass and foliage beside it: a plain bilinear sample in
// the final composite blurred it straight across silhouettes.
//
// WHERE IT SITS is true at any resolution and is the other half of the reason
// it exists: running directly after opaque forward lighting and before
// water/TAA/bloom/exposure lets the traced signal accumulate with the rest of
// the frame, instead of shimmering as a late overlay drawn after TAA.

#include "reflection_gate.hlsli"
#include "depth_linear.hlsli"

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer RtReflectionResolveUniforms {
    float4x4 invViewProj;
    float3 cameraPosition;
    float maxRoughness;
    float2 outputSize;
    float2 reflectionSize;
    float depthA;
    float depthB;
    float fadeStart;
    float intensity;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float4> reflectionTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState reflectionSampler;

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D<float> depthTexture;
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState depthSampler;

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D<float4> normalRoughMetalTexture;
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState normalRoughMetalSampler;

float3 octDecode(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, float2(-t, -t), float2(t, t));
    return normalize(n);
}

float linearizeDepth(float d) {
    // The depthB guard is this pass's own: with the terms unset it falls back
    // to raw device depth rather than to 0, because the taps below are
    // compared against each other and 0-vs-0 would accept every one of them.
    return abs(depthB) > 1e-8 ? brackenLinearViewDepth(d, depthA, depthB) : d;
}

float2 envBRDFApprox(float roughness, float NoV) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

float4 PSMain(PSInput input) : SV_Target {
    int2 coord = clamp(int2(input.position.xy), int2(0, 0), int2(outputSize) - 1);
    float centerDepth = depthTexture.Load(int3(coord, 0));
    if (centerDepth <= 0.0) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float4 centerG = normalRoughMetalTexture.Load(int3(coord, 0));
    float3 centerNormal = octDecode(centerG.xy);
    float centerRoughness = centerG.z;
    float centerMetallic = centerG.w;
    float centerGate = rtReflectionWeight(centerRoughness, fadeStart, maxRoughness);
    if (centerGate <= 0.0) {
        // This full-resolution test is the material boundary. Grass, MASK
        // foliage and terrain write roughness 1, so a neighbouring low-res
        // stone reflection cannot leak onto them.
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float2 sourcePos = (float2(coord) + 0.5) * reflectionSize / outputSize - 0.5;
    int2 sourceBase = int2(floor(sourcePos));
    float2 f = frac(sourcePos);
    float spatial[4] = {
        (1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y,
        f.x * f.y
    };
    const int2 offsets[4] = {int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1)};

    float centerZ = linearizeDepth(centerDepth);
    float3 radiance = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    [unroll]
    for (int i = 0; i < 4; ++i) {
        int2 tap = clamp(sourceBase + offsets[i], int2(0, 0), int2(reflectionSize) - 1);
        float4 traced = reflectionTexture.Load(int3(tap, 0));
        if (traced.a <= 0.0) {
            continue;
        }

        // This is exactly the source full-resolution texel the trace pass used
        // for this sparse texel (integer truncation after center mapping).
        int2 sourceCoord = clamp(
            int2((float2(tap) + 0.5) / reflectionSize * outputSize),
            int2(0, 0), int2(outputSize) - 1);
        float tapDepth = depthTexture.Load(int3(sourceCoord, 0));
        if (tapDepth <= 0.0) {
            continue;
        }
        float4 tapG = normalRoughMetalTexture.Load(int3(sourceCoord, 0));
        float3 tapNormal = octDecode(tapG.xy);

        float tapZ = linearizeDepth(tapDepth);
        float relativeDepth = abs(tapZ - centerZ) / max(centerZ, 1e-3);
        float depthWeight = exp(-relativeDepth * 80.0);
        float normalWeight = pow(saturate(dot(centerNormal, tapNormal)), 32.0);
        float materialWeight = exp(-abs(tapG.z - centerRoughness) * 24.0
                                   -abs(tapG.w - centerMetallic) * 12.0);
        float w = spatial[i] * depthWeight * normalWeight * materialWeight;
        radiance += traced.rgb * w;
        weightSum += w;
    }

    if (weightSum <= 1e-5) {
        // A reflection from another material or across a silhouette is worse
        // than none, so the preference for zero is right — but be clear about
        // what zero costs, because the second half of this comment used to
        // claim "TAA fills stable sub-pixel gaps over subsequent frames" and
        // that is false. The rejection is a deterministic function of the
        // G-buffer, not of the jitter: a thin smooth railing whose sparse taps
        // all land on the background is rejected EVERY frame, and there is no
        // traced data anywhere near it for TAA to converge onto. Worse, the
        // forward pass has already surrendered its share of this pixel
        // (`iblSpecularScale = 1 - rtReflectionWeight(...)`, computed from the
        // pixel's own roughness in a pass that runs before this one), so the
        // pixel ends up with `gate` of nothing plus `1 - gate` of IBL — the
        // sum-to-one invariant in reflection_gate.hlsli broken on the dark
        // side. Fixing it properly means this pass supplying the IBL half
        // itself, which needs the prefiltered cube bound here.
        //
        // UNREACHABLE AT THE DEFAULT Full resolution, which is why it has
        // stayed invisible: there `sourcePos == coord` exactly, tap 0 is the
        // pixel itself, its depth/normal/material weights are all 1, and the
        // trace wrote alpha from the same G-buffer texel through the same gate,
        // so `weightSum` is 1.
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    radiance /= weightSum;

    float2 ndc = (float2(coord) + 0.5) / outputSize * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 world = mul(invViewProj, float4(ndc, centerDepth, 1.0));
    float3 worldPos = world.xyz / world.w;
    float3 V = normalize(cameraPosition - worldPos);
    float NoV = saturate(dot(centerNormal, V));
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), float3(1.0, 1.0, 1.0), centerMetallic);
    float2 ab = envBRDFApprox(centerRoughness, NoV);
    float3 specularWeight = (F0 * ab.x + ab.y) * (intensity * centerGate);

    // RGB is already the exact additive contribution. The pipeline uses an
    // RGB-add / alpha-preserve blend state, so alpha is intentionally zero.
    return float4(radiance * specularWeight, 0.0);
}
