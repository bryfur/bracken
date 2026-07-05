/**
 * Skybox Vertex Shader - SDL3 GPU API
 *
 * Fullscreen triangle approach: vertices are clip-space positions.
 * Ray direction is computed per-pixel in the fragment shader.
 */

// Vertex input - position is a clip-space triangle covering the screen
struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
};

// Vertex output / Fragment input
struct VSOutput {
    float4 position : SV_Position;
    float2 clipPos : TEXCOORD0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    // Reversed-Z: emit z = 0.0 (the far plane) so the GREATER_OR_EQUAL depth
    // test passes only where the buffer still holds the 0.0 clear value —
    // i.e. wherever no geometry was drawn.
    output.position = float4(input.position.xy, 0.0, 1.0);
    output.clipPos = input.position.xy;
    return output;
}
