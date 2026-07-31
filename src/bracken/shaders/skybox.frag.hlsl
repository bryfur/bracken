/**
 * Skybox Fragment Shader - SDL3 GPU API
 *
 * Procedural sky on a fullscreen triangle. The whole radiance function
 * (gradient, clouds, sun disc, haze, stars) lives in sky_common.hlsli and is
 * SHARED with the IBL sky-capture pass — both must evaluate the identical
 * sky. This pass only reconstructs the per-pixel ray and adds the
 * screen-space IGN dither.
 */

#include "lighting.hlsli"
#include "sky_common.hlsli"

// Fragment input (from vertex shader)
struct PSInput {
    float4 position : SV_Position;
    float2 clipPos : TEXCOORD0;
};

// Uniform buffer - SDL3 GPU SPIR-V requires fragment uniforms in set 3.
// Mirrors gpu::SkyboxFragmentUniforms (gpu_uniforms.hpp).
[[vk::binding(0, 3)]]
cbuffer SkyUniforms {
    float4x4 invViewProjection;
    float3 sunDirection;  // Direction TO sun (normalized)
    float time;
    float3 sunColor;
    float _pad0;
    float3 zenithColor;
    float _pad1;
    float3 horizonColor;
    float _pad2;
    float3 groundColor;
    float _pad3;
    float3 fogHorizonColor;
    float _pad4;
    float3 fogSunColor;
    float _pad5;
    float cloudCoverage;  // EnvironmentState::cloud_coverage
    float cloudDensity;   // EnvironmentState::cloud_density
    float cloudScale;     // EnvironmentState::cloud_scale
    float hazeStrength;   // EnvironmentState::haze_strength
};

float4 PSMain(PSInput input) : SV_Target {
    // Compute world-space ray direction per-pixel
    float4 worldPos = mul(invViewProjection, float4(input.clipPos, 1.0, 1.0));
    float3 dir = normalize(worldPos.xyz / worldPos.w);

    float3 sky = evaluateSkyRadiance(dir, sunDirection, time, sunColor, zenithColor,
                                     horizonColor, groundColor, fogHorizonColor, fogSunColor,
                                     cloudCoverage, cloudDensity, cloudScale, hazeStrength);

    // IGN dither (+/- 0.5/255 in linear) hides gradient banding after tonemap.
    // Offset decorrelates from the composite pass's post-encode IGN dither so
    // the two patterns average out on sky pixels instead of stacking.
    sky += (interleavedGradientNoise(input.position.xy + 71.0) - 0.5) * (1.0 / 255.0);

    return float4(max(sky, 0.0), 1.0);
}
