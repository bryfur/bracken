/**
 * IBL Sky Capture Fragment Shader
 *
 * Renders one cubemap face of the procedural sky (sky_common.hlsli — the
 * exact function the skybox pass evaluates, so the prefiltered environment
 * can never drift from the visible sky). Drawn as a fullscreen triangle into
 * a single face/mip of the RGBA16F capture cube; render::IBL supplies the
 * face basis so the direction math matches the Vulkan/D3D cube lookup
 * convention (CPU reference: ibl_face_direction in render/ibl.hpp).
 */

#include "sky_common.hlsli"

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

// Mirrors gpu::SkyCaptureUniforms (gpu_uniforms.hpp).
[[vk::binding(0, 3)]]
cbuffer SkyCaptureUniforms {
    float3 faceForward;
    float _pad0;
    float3 faceRight;
    float _pad1;
    float3 faceUp;
    float _pad2;
    float3 sunDirection; // direction TO the sun
    float time;          // sky_time snapshot at chain start (freezes cloud scroll)
    float3 sunColor;
    float _pad3;
    float3 zenithColor;
    float _pad4;
    float3 horizonColor;
    float _pad5;
    float3 groundColor;
    float _pad6;
    float3 fogHorizonColor;
    float _pad7;
    float3 fogSunColor;
    float _pad8;
    float cloudCoverage;  // IblSkyParams::cloud_coverage
    float cloudDensity;   // IblSkyParams::cloud_density
    float cloudScale;     // IblSkyParams::cloud_scale
    float hazeStrength;   // IblSkyParams::haze_strength
};

float4 PSMain(PSInput input) : SV_Target {
    // texCoord.y grows downward, matching the cube face t axis.
    float2 st = input.texCoord * 2.0 - 1.0;
    float3 dir = normalize(faceForward + st.x * faceRight + st.y * faceUp);

    float3 sky = evaluateSkyRadiance(dir, sunDirection, time, sunColor, zenithColor,
                                     horizonColor, groundColor, fogHorizonColor, fogSunColor,
                                     cloudCoverage, cloudDensity, cloudScale, hazeStrength);
    return float4(max(sky, 0.0), 1.0);
}
