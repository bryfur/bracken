/**
 * Depth-Prepass G-Buffer Fragment Shader, terrain variant.
 *
 * Same velocity + octahedral-normal write as prepass_gbuffer.frag.hlsl, but
 * with the roughness/metallic constants terrain.frag.hlsl actually shades
 * with baked in instead of read from a uniform:
 *
 *     terrain.frag.hlsl:  roughness = 0.9 + (noise - 0.5) * 0.1   -> 0.85..0.95
 *                         TERRAIN_METALLIC = 0
 *
 * 0.9 is the centre of that band, and the traced/probe handover is a fade, so
 * the noise the constant drops is well inside the blend's tolerance.
 *
 * IT HAS NO FRAGMENT UNIFORM BUFFER, AND THAT IS THE POINT. The terrain
 * G-buffer pipeline used to share prepass_gbuffer.frag with the model paths,
 * which declares one -- and TerrainRenderer::render_prepass only ever called
 * SDL_PushGPUVertexUniformData. SDL3's Metal backend reacts to a bound
 * pipeline declaring a fragment UBO by acquiring a buffer from its recycled
 * uniform pool and binding it with nothing written into it; the pool is never
 * zeroed. So `roughness` read as whatever that slot last held -- measured at
 * roughly 0, far below RtReflectionSettings::max_roughness -- and the ENTIRE
 * visible terrain was classified as a perfect mirror and given a full traced
 * reflection. Non-deterministically, since it depends on pool recycling.
 *
 * That is the "reflections on the grass" half of the report: the reflection
 * pass point-samples this full-res target at a quarter-res stride, so one
 * terrain texel's roughness governs a 4x4 block, and the composite's linear
 * magnification then bleeds the mirror onto the fence rails, grass tufts and
 * flower stems standing in front of the ground.
 *
 * Giving terrain a shader with `.fs = k_none` makes the failure mode
 * unrepresentable rather than merely fixed: there is no uniform to forget.
 */

struct PSInput {
    float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
    float3 worldNormal : TEXCOORD2;
};

struct PSOutput {
    float2 velocity : SV_Target0;
    float4 normalRough : SV_Target1;
};

// Octahedral normal encoding (Cigolle et al. 2014) — must match the decode in
// rt_reflection.comp.hlsl and the encode in prepass_gbuffer.frag.hlsl.
float2 octEncode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, float2(1.0, 1.0), float2(-1.0, -1.0));
    }
    return e * 0.5 + 0.5;
}

PSOutput PSMain(PSInput input) {
    PSOutput output;

    // Identical to prepass_velocity.frag.hlsl.
    if (input.currClip.w <= 0.0 || input.prevClip.w <= 0.0) {
        output.velocity = float2(0.0, 0.0);
    } else {
        float2 currNdc = input.currClip.xy / input.currClip.w;
        float2 prevNdc = input.prevClip.xy / input.prevClip.w;
        output.velocity = (currNdc - prevNdc) * float2(0.5, -0.5);
    }

    float3 n = normalize(input.worldNormal);
    output.normalRough = float4(octEncode(n), 0.9, 0.0);
    return output;
}
