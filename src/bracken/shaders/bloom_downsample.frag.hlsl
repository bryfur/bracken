/**
 * Bloom Downsample Fragment Shader
 *
 * Extracts bright pixels above a post-exposure luminance threshold and
 * downsamples using a 13-tap filter for high quality (Jimenez 2014 / Call of
 * Duty). Each mip level reads from the previous level.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer BloomDownsampleUniforms : register(b0, space3) {
    float2 srcTexelSize;   // 1.0 / source resolution
    float threshold;       // post-exposure luminance threshold (first pass only)
    float isFirstPass;     // 1.0 for first pass (apply threshold), 0.0 otherwise
    float exposure;        // composite-pass MANUAL exposure bias
    float autoExposureEnabled; // 1 when exposureState below holds a live EV
    // The CPU struct (gpu::BloomDownsampleUniforms) pads to 32 bytes and the
    // engine pushes all 32, so the cbuffer declares the pad too — same
    // convention as ShadowFragUniforms / UIFragmentUniforms. Named scalars,
    // not float[2]: a cbuffer array would stride at 16 B per element.
    float _padding0;
    float _padding1;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D srcTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState srcSampler : register(s0, space2);

// Auto-exposure state written by exposure_adapt.comp, the SAME buffer and the
// same layout composite.frag.hlsl reads: x = adapted EV, y = target EV,
// z = avg log2 luminance, w = valid. Storage buffers follow the samplers in
// space2, so with one sampler above this is slot 1 (composite has four
// samplers and lands on t4). Only read when autoExposureEnabled — the slot
// binds a zeroed dummy when auto exposure is off or unavailable.
[[vk::binding(1, 2)]]
StructuredBuffer<float4> exposureState : register(t1, space2);

float bloomLuminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Karis-average tap weight: stabilizes fireflies in HDR bloom prefilter.
float karisWeight(float3 c) {
    return 1.0 / (1.0 + max(max(c.r, c.g), c.b));
}

// Soft threshold: smooth knee around the threshold value, then keep only the
// part above the threshold (max-component cut for HDR robustness).
// The threshold decision is made in POST-exposure luminance so "what blooms"
// tracks what will actually be bright after the composite pass applies
// exposure (an exposure-dropped night scene stops blooming mid-grey surfaces).
// POST-exposure means BOTH of composite's factors — the manual bias AND the
// adapted EV — because composite multiplies by `exposure * exp2(EV)`. Seeing
// only the bias is what made this cut scene-referred: at a night-adapted
// EV of +4 the effective threshold was 16x its nominal value, four stops past
// display white, so nothing but the emitters themselves bloomed.
// The returned color stays in pre-exposure scene units — composite multiplies
// bloom by exposure later, so exposure must not be baked in twice.
float3 applyThreshold(float3 color) {
    float exposureMul = exposure;
    if (autoExposureEnabled > 0.5) {
        exposureMul *= exp2(exposureState[0].x);
    }
    float lum = bloomLuminance(color) * exposureMul;
    float knee = threshold * 0.5;
    float soft = lum - threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.00001);
    float contribution = max(soft, lum - threshold) / max(lum, 0.00001);
    return color * max(contribution, 0.0);
}

float4 PSMain(PSInput input) : SV_Target {
    float2 uv = input.texCoord;
    float2 ts = srcTexelSize;

    // 13-tap downsample filter (Jimenez 2014)
    // Center sample
    float3 a = srcTexture.SampleLevel(srcSampler, uv, 0).rgb;

    // Inner ring (4 samples at half-texel offsets)
    float3 b = srcTexture.SampleLevel(srcSampler, uv + float2(-0.5, -0.5) * ts, 0).rgb;
    float3 c = srcTexture.SampleLevel(srcSampler, uv + float2( 0.5, -0.5) * ts, 0).rgb;
    float3 d = srcTexture.SampleLevel(srcSampler, uv + float2(-0.5,  0.5) * ts, 0).rgb;
    float3 e = srcTexture.SampleLevel(srcSampler, uv + float2( 0.5,  0.5) * ts, 0).rgb;

    // Outer ring (8 samples at 1-texel offsets)
    float3 f = srcTexture.SampleLevel(srcSampler, uv + float2(-1.0, -1.0) * ts, 0).rgb;
    float3 g = srcTexture.SampleLevel(srcSampler, uv + float2( 0.0, -1.0) * ts, 0).rgb;
    float3 h = srcTexture.SampleLevel(srcSampler, uv + float2( 1.0, -1.0) * ts, 0).rgb;
    float3 i = srcTexture.SampleLevel(srcSampler, uv + float2(-1.0,  0.0) * ts, 0).rgb;
    float3 j = srcTexture.SampleLevel(srcSampler, uv + float2( 1.0,  0.0) * ts, 0).rgb;
    float3 k = srcTexture.SampleLevel(srcSampler, uv + float2(-1.0,  1.0) * ts, 0).rgb;
    float3 l = srcTexture.SampleLevel(srcSampler, uv + float2( 0.0,  1.0) * ts, 0).rgb;
    float3 m = srcTexture.SampleLevel(srcSampler, uv + float2( 1.0,  1.0) * ts, 0).rgb;

    float3 color;
    if (isFirstPass > 0.5) {
        // Karis-average the five 2x2 groups (firefly suppression for HDR).
        float3 group_center = (b + c + d + e) * 0.25;
        float3 group_tl = (a + g + f + i) * 0.25;
        float3 group_tr = (a + h + g + j) * 0.25;
        float3 group_bl = (a + i + k + l) * 0.25;
        float3 group_br = (a + j + l + m) * 0.25;

        float w_center = karisWeight(group_center) * 0.5;
        float w_tl     = karisWeight(group_tl)     * 0.125;
        float w_tr     = karisWeight(group_tr)     * 0.125;
        float w_bl     = karisWeight(group_bl)     * 0.125;
        float w_br     = karisWeight(group_br)     * 0.125;

        float w_sum = w_center + w_tl + w_tr + w_bl + w_br;
        color = (group_center * w_center + group_tl * w_tl + group_tr * w_tr +
                 group_bl * w_bl + group_br * w_br) / max(w_sum, 1e-5);

        color = applyThreshold(color);
    } else {
        // Standard Jimenez weights for subsequent mips.
        float3 group_center = (b + c + d + e) * 0.25;
        float3 group_tl = (a + g + f + i) * 0.25;
        float3 group_tr = (a + h + g + j) * 0.25;
        float3 group_bl = (a + i + k + l) * 0.25;
        float3 group_br = (a + j + l + m) * 0.25;
        color = group_center * 0.5 + (group_tl + group_tr + group_bl + group_br) * 0.125;
    }

    return float4(color, 1.0);
}
