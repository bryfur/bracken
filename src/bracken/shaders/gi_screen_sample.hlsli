// gi_screen_sample.hlsli - read the screen-probe SH atlas: pixel + normal ->
// diffuse irradiance, and how much of it to believe.
//
// THE CONSUMPTION PATH OF PHASE R4b (docs/rt_lighting_plan.md §5.5), the layer
// above `gi_sample.hlsli`. Both return the SAME UNIT — cosine-weighted mean
// incoming radiance, irradiance/pi, the IBL irradiance cube's convention — so
// the including shader blends one into the other with no scale factor:
//
//     float3 gi = giSampleDiffuse(P, N, V, ambientSky, ambientGround);
//     float  w;
//     float3 sp = giScreenSampleDiffuse(pixel, P, N, bentN, cameraPos, w);
//     gi = lerp(gi, sp, w);
//
// The weight is the whole contract. Screen probes are SHARPER than the world
// field where they have an answer and have NO answer at all in three cases the
// field handles fine — a tile that was all sky, a pixel whose four probes sit
// on other surfaces (thin railings, silhouettes), and geometry that was
// disoccluded this frame. `w` going to zero in those cases is not a
// degradation, it is the tier below taking over.
//
// ---------------------------------------------------------------------------
// WHAT THE INCLUDING SHADER MUST DECLARE FIRST
// ---------------------------------------------------------------------------
// Unlike gi_sample.hlsli, this file OWNS its binding — a fifth fragment storage
// buffer immediately after cluster_lighting.hlsli's four. Define
// `CLUSTER_SSBO_BINDING_BASE` and include cluster_lighting.hlsli BEFORE this
// file, and bind the buffer at fragment storage-buffer slot 4.
//
// A STORAGE BUFFER, NOT A SAMPLER, and that is a platform limit rather than a
// preference. This tier's design called for one new fragment sampler in each
// scene shader (model/instanced 16 -> 17). SDL3 GPU refuses:
// `SDL_CreateGPUShader` asserts "Shader sampler count cannot be higher than
// 16!" and never creates the pipeline. Seventeen is DXC's legalization ceiling;
// SDL's is sixteen, and SDL's is the one that binds. A buffer is also the
// better fit once forced — this is probe-INDEXED data, read with a plain index
// and never filtered.
//
// In a cbuffer, by these names:
//
//   float giScreenEnabled;   // 0 = never call giScreenSampleDiffuse
//   float giScreenProbesX;   // the probe grid, as floats because the scene
//   float giScreenProbesY;   // cbuffers have float pads to spend and not ints

#ifndef BRACKEN_GI_SCREEN_SAMPLE_HLSLI
#define BRACKEN_GI_SCREEN_SAMPLE_HLSLI

#include "gi_screen_probe.hlsli"

#ifndef CLUSTER_SSBO_BINDING_BASE
#error "Define CLUSTER_SSBO_BINDING_BASE and include cluster_lighting.hlsli before gi_screen_sample.hlsli"
#endif

/// The published per-probe L1 SH, one 64-byte record per probe, row-major.
/// Bound at fragment storage-buffer slot 4, right after the cluster quartet.
[[vk::binding(CLUSTER_SSBO_BINDING_BASE + 4, 2)]]
StructuredBuffer<GiScreenProbeSh> giScreenShBuf;

/**
 * Diffuse irradiance/pi at a pixel from the screen-probe SH atlas.
 *
 * `pixel` is SV_Position.xy (pixel centres at +0.5). `weight` reports how much
 * of the result to use; the caller lerps from the world field toward it.
 *
 * MANUAL BILINEAR over the 2x2 nearest probes, because there is nothing a
 * hardware bilinear could do here even if this were a texture: the four taps
 * must be REJECTED individually on depth and normal, and a filtered fetch has
 * already averaged them. Four 64-byte record loads per pixel is the cost,
 * against the world field's twenty-four texture taps in the same shader.
 *
 * CPU mirror of the weight and the SH evaluation:
 * render/gi/screen_probe_math.hpp (`screen_probe_bilateral_weight`,
 * `sh_l1_irradiance_over_pi`).
 */
float3 giScreenSampleDiffuse(float2 pixel, float3 worldPos, float3 N, float3 sampleDir,
                             float3 cameraPosition, out float weight) {
    weight = 0.0;
    if (giScreenEnabled <= 0.0) {
        return float3(0, 0, 0);
    }
    int2 grid = int2(int(giScreenProbesX), int(giScreenProbesY));
    if (grid.x <= 0 || grid.y <= 0) {
        return float3(0, 0, 0);
    }

    // A probe represents its tile's CENTRE, so the bilinear cage starts half a
    // tile back. Getting this offset wrong is the classic way to produce a
    // visible tile grid: every pixel would weight its own tile's probe by 1.
    float2 t = pixel / float(GI_SP_TILE) - 0.5;
    float2 baseF = floor(t);
    float2 f = t - baseF;
    int2 base = int2(baseF);

    float pixelDistance = distance(worldPos, cameraPosition);

    float3 acc = float3(0, 0, 0);
    float wsum = 0.0;
    [unroll]
    for (int i = 0; i < 4; ++i) {
        int2 offset = int2(i & 1, (i >> 1) & 1);
        int2 pc = base + offset;
        if (any(pc < 0) || any(pc >= grid)) {
            continue;
        }
        GiScreenProbeSh probe = giScreenShBuf[giSpIndex(pc, grid)];
        if (probe.meta.w <= 0.0) {
            continue; // sky tile: no probe was placed
        }
        float2 bl = lerp(1.0 - f, f, float2(offset));
        float w = bl.x * bl.y * giSpBilateral(pixelDistance, probe.meta.w, N, probe.meta.xyz,
                                              GI_SP_FORWARD_NORMAL_POWER);
        if (w <= 1e-5) {
            continue;
        }
        acc += w * giSpShEval(probe.shR, probe.shG, probe.shB, sampleDir);
        wsum += w;
    }

    if (wsum <= 1e-5) {
        return float3(0, 0, 0);
    }
    // RENORMALISE over the probes that answered, then RAMP by how much of the
    // bilinear cage that was. Same two-step as gi_sample.hlsli's cascade
    // confidence, and the same reason: dividing by the full bilinear total
    // would darken every silhouette, while trusting one survivor completely
    // would stretch its radiance across the pixels the others covered.
    weight = saturate(wsum / GI_SP_WEIGHT_FLOOR);
    return acc / wsum;
}

#endif // BRACKEN_GI_SCREEN_SAMPLE_HLSLI
