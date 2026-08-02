/**
 * Depth-Prepass G-Buffer Fragment Shader (velocity + normal/roughness/metallic).
 *
 * The G-buffer variant of prepass_velocity.frag.hlsl, used only when
 * ray-traced reflections are on (render/rt_renderer.hpp). SV_Target0 is
 * byte-for-byte the same velocity write as the base shader -- keep the two in
 * sync -- and SV_Target1 adds what the reflection pass needs to decide whether
 * a pixel is worth tracing, and how much of the environment it reflects:
 *
 *   RG = world normal, octahedral-encoded to [0,1]
 *   B  = linear roughness
 *   A  = metallic
 *
 * Octahedral rather than the obvious xyz*0.5+0.5 because two 8-bit channels
 * hold a unit vector more accurately than three do, and it frees the fourth
 * channel for metallic -- without which the reflection pass cannot tell a
 * mirror from grass and weights every surface as if it were chrome.
 *
 * ROUGHNESS IS SAMPLED, NOT TAKEN FROM THE FACTOR. This shader used to write
 * `roughnessFactor` straight through, on the reasoning that a threshold test
 * only needs to separate "mirror" from "probe" and the factor decides that.
 * In real glTF content the factor decides nothing: of the 79 materials in the
 * staged medieval/nature packs, 77 have roughnessFactor exactly 1.0 and all 79
 * have metallicFactor 0.0 -- every bit of actual roughness and metalness lives
 * in the 14 metallicRoughness/ORM textures. So the G-buffer classified 100% of
 * the village and nature art as fully rough, the gate rejected every stone,
 * brick, plaster, tile, wood and rock pixel, and the ONLY geometry in the
 * frame that could ever reflect was the playground's own procedural chrome
 * sphere. Sampling the map here costs one fetch per opaque prepass pixel -- the
 * same order as the base-colour fetch the masked variants already pay -- and
 * it is what makes the prepass classify the surface the forward pass actually
 * shades.
 */

#include "pbr.hlsli" // specularAARoughness

struct PSInput {
    float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
    float3 worldNormal : TEXCOORD2;
    float2 uv : TEXCOORD3;
};

struct PSOutput {
    float2 velocity : SV_Target0;
    float4 normalRough : SV_Target1;
};

[[vk::binding(0, 3)]]
cbuffer PrepassGBufferUniforms {
    float roughness; // material factor; modulated by the MR map below
    float metallic;  // material factor; ditto
    float hasMRMap;  // 0 = the 1x1 white dummy is bound, factors stand alone
    float _pad2;
};

// glTF metallicRoughness: G = roughness, B = metallic. Same texture and same
// convention as model.frag.hlsl's sampler 4; the white dummy is bound when the
// material has no map, so this slot is always valid.
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D metallicRoughnessTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState metallicRoughnessSampler;

// Octahedral normal encoding (Cigolle et al. 2014). Unit vector -> [0,1]^2.
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

    // Interpolation shortens the normal; renormalize before encoding.
    float3 n = normalize(input.worldNormal);

    // Operation-for-operation what model.frag.hlsl / model_instanced.frag.hlsl
    // do, so the two sides classify identically. The one residual difference
    // is that specular AA runs on the geometric normal here and on the
    // normal-mapped one there; the traced/probe handover is a fade, not a
    // cutoff, precisely so that small disagreement is a small blend-ratio
    // error rather than a binary flip.
    float rough = roughness;
    float metal = metallic;
    if (hasMRMap > 0.5) {
        float3 mr = metallicRoughnessTexture.Sample(metallicRoughnessSampler, input.uv).rgb;
        rough *= mr.g;
        metal *= mr.b;
    }
    rough = specularAARoughness(n, rough);

    output.normalRough = float4(octEncode(n), saturate(rough), saturate(metal));
    return output;
}
