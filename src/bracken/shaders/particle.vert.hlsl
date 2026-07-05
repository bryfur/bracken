/**
 * Particle Vertex Shader - SDL3 GPU API (instanced)
 *
 * Draws every particle of one (emitter, mesh) batch in a single instanced
 * draw. Per-particle transform + color come from a frame-global storage
 * buffer of ParticleInstance records (uploaded once per frame in the
 * FrameUploads copy pass); instanceOffset selects the batch's first record.
 * Particles are unlit and unfogged, so only position/texCoord/color are
 * consumed, and the instance tint + opacity are folded into the vertex color
 * here (additive blending fades by scaling rgb).
 */

// Vertex input - locations match get_vertex3d_attributes() in gpu_types.hpp.
// The normal (1) and tangent (4) attributes exist in the vertex layout but
// are not consumed (particles are unlit).
struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
    [[vk::location(3)]] float4 color : COLOR0;
};

// Vertex output / Fragment input (particle.frag PSInput must byte-match)
struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float4 color : TEXCOORD1; // vertex color * instance tint, opacity folded into rgb
};

// Camera uniforms, pushed once per pass - layout matches
// gpu::InstancedCameraUniforms (shared with model_instanced.vert.hlsl).
[[vk::binding(0, 1)]]
cbuffer CameraUniforms {
    float4x4 view;
    float4x4 projection;
    float3 cameraPos; // unused (no fog on particles); kept for layout sharing
    float _padding0;
};

// Per-batch uniforms, pushed once per (emitter, mesh) batch.
[[vk::binding(1, 1)]]
cbuffer BatchUniforms {
    uint instanceOffset; // first ParticleInstance record of this batch
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Per-particle data, tightly packed for the whole frame
// (vertex storage buffers live in set 0; matches model_instanced.vert.hlsl).
struct ParticleInstance {
    float4x4 model;
    float4 color; // rgb = tint, a = opacity
};

[[vk::binding(0, 0)]]
StructuredBuffer<ParticleInstance> instances;

VSOutput VSMain(VSInput input, uint instanceID : SV_InstanceID) {
    VSOutput output;

    ParticleInstance inst = instances[instanceOffset + instanceID];

    float4 worldPos = mul(inst.model, float4(input.position, 1.0));
    output.position = mul(projection, mul(view, worldPos));
    output.texCoord = input.texCoord;

    // Additive (ONE, ONE) blending ignores alpha, so opacity fades by
    // scaling rgb; alpha is fixed at 1.0 (see particle.frag).
    output.color = float4(input.color.rgb * inst.color.rgb * inst.color.a, 1.0);

    return output;
}
