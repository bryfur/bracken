/**
 * Depth-Prepass Velocity Fragment Shader - SDL3 GPU API
 *
 * Shared by every depth-prepass pipeline (model / skinned / instanced /
 * terrain / grass). Writes screen-space motion vectors to the RG16F velocity
 * target while the raster pipeline lays down scene depth.
 *
 * Velocity convention (mirrored by scene/velocity_math.hpp — keep in sync):
 *   velocity = uv_current - uv_previous          (UV space, v grows downward)
 * so a TAA resolve fetches history at prev_uv = uv - velocity. Both clip
 * positions come from UNJITTERED view-projection matrices (the vertex stage
 * guarantees this); jitter never leaks into velocity.
 */

struct PSInput {
    float4 position : SV_Position;
    float4 currClip : TEXCOORD0; // unjittered current clip position
    float4 prevClip : TEXCOORD1; // unjittered previous clip position
};

float2 PSMain(PSInput input) : SV_Target0 {
    // Guard against degenerate reprojection (point behind a clip plane after
    // extreme motion); zero velocity keeps consumers stable.
    if (input.currClip.w <= 0.0 || input.prevClip.w <= 0.0) {
        return float2(0.0, 0.0);
    }
    float2 currNdc = input.currClip.xy / input.currClip.w;
    float2 prevNdc = input.prevClip.xy / input.prevClip.w;
    // NDC delta -> UV delta: u = ndc.x*0.5+0.5, v = 0.5-ndc.y*0.5.
    return (currNdc - prevNdc) * float2(0.5, -0.5);
}
