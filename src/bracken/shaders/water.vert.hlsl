// water.vert.hlsl - water surface grid.
//
// The mesh is a flat world-space XZ grid covering the water region; the
// vertex shader lifts it to the water level and displaces it vertically by
// the simulation height field (inside the sim window; the displacement fades
// to zero at the window edge so the surface is seamless where the sim ends).

[[vk::binding(0, 1)]]
cbuffer WaterVertexUniforms : register(b0, space1) {
    float4x4 viewProj;
    float4 simWindow;   // xy = window min (world XZ), z = 1/window size, w = unused
    float4 params;      // x = water level (world Y), y = displacement scale,
                        // z = sim edge fade width (uv), w = unused
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D<float2> simState : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState simSampler : register(s0, space0);

struct VSInput {
    float2 xz : TEXCOORD0; // world-space XZ of the grid vertex
};

struct VSOutput {
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 simUVFade : TEXCOORD1; // xy = sim uv, z = sim influence (0 outside window)
};

VSOutput VSMain(VSInput input) {
    VSOutput output;

    float2 simUV = (input.xz - simWindow.xy) * simWindow.z;
    // 1 inside the window, easing to 0 at its edge (and clamped outside).
    float fadeW = max(params.z, 1e-4);
    float2 edge = min(simUV, 1.0 - simUV) / fadeW;
    float fade = saturate(min(edge.x, edge.y));

    float h = simState.SampleLevel(simSampler, saturate(simUV), 0).x;
    float y = params.x + h * params.y * fade;

    float3 world = float3(input.xz.x, y, input.xz.y);
    output.position = mul(viewProj, float4(world, 1.0));
    output.worldPos = world;
    output.simUVFade = float3(simUV, fade);
    return output;
}
