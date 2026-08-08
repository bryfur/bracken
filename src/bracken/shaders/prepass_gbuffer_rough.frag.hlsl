/**
 * Depth-Prepass G-Buffer Fragment Shader, rough-surface variant.
 *
 * For geometry that is never a mirror -- grass blades. Writes the same
 * velocity as prepass_velocity.frag.hlsl and a roughness of 1, which puts
 * these pixels above any sane RtReflectionSettings::max_roughness so the
 * reflection pass skips them without tracing a ray.
 *
 * This exists because a render pass and every pipeline inside it must agree on
 * attachment count: with the G-buffer attached, grass needs SOME second-target
 * write even though it has nothing useful to say.
 *
 * THE NORMAL IS A CONSTANT +Y AND IS READ. The header used to claim it is
 * "never read as a result", which was true when the G-buffer served only the
 * reflection pass and false afterwards -- rt_shadow, rt_local_shadow's
 * `surfaceAt`, both gradient replays, the DI spatial filter's bilateral weight
 * and screen-probe placement all decode it. prepass_gbuffer_masked.frag.hlsl
 * had the identical comment and the identical constant, and there it was doing
 * real damage, so it now carries a real normal.
 *
 * Kept constant HERE on purpose: grass blades genuinely are near-vertical
 * quads whose interpolated normal is close to up, the error is bounded and
 * unbiased across a tuft rather than per-card as it was for foliage, and the
 * grass vertex format would have to grow to fix it. If the DI filter's normal
 * weight ever needs to separate grass from the ground it stands on, this is the
 * line to change -- but it is a trade-off now, not an oversight.
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
    // oct(+Y) = (0.75, 0.75); fully rough, non-metal — the reflection pass
    // rejects these on roughness before the normal is ever used.
    output.normalRough = float4(0.75, 0.75, 1.0, 0.0);
    return output;
}
