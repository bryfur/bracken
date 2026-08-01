/**
 * Depth-Prepass G-Buffer Fragment Shader, rough-surface variant.
 *
 * For geometry that is never a mirror -- grass blades. Writes the same
 * velocity as prepass_velocity.frag.hlsl and a roughness of 1, which puts
 * these pixels above any sane RtReflectionSettings::max_roughness so the
 * reflection pass skips them without tracing a ray. The normal is a
 * placeholder that is never read as a result.
 *
 * This exists because a render pass and every pipeline inside it must agree on
 * attachment count: with the G-buffer attached, grass needs SOME second-target
 * write even though it has nothing useful to say.
 */

struct PSInput {
    float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
};

struct PSOutput {
    float2 velocity : SV_Target0;
    float4 normalRough : SV_Target1;
};

PSOutput PSMain(PSInput input) {
    PSOutput output;
    if (input.currClip.w <= 0.0 || input.prevClip.w <= 0.0) {
        output.velocity = float2(0.0, 0.0);
    } else {
        float2 currNdc = input.currClip.xy / input.currClip.w;
        float2 prevNdc = input.prevClip.xy / input.prevClip.w;
        output.velocity = (currNdc - prevNdc) * float2(0.5, -0.5);
    }
    output.normalRough = float4(0.5, 1.0, 0.5, 1.0); // +Y encoded, fully rough
    return output;
}
