// The reference consumer's own shader — the thing bracken_compile_shaders
// exists for. It is compiled by the ENGINE's pinned shadercross/DXC and staged
// into `shaders_playground/` next to the exe, deliberately NOT into `shaders/`
// (bracken_copy_shaders wipes that directory every build) and deliberately not
// into the engine's own SPIR-V output directory (gpu/uniform_layout.test.cpp
// enumerates that one exhaustively and would fail on a host cbuffer).
//
// NOTHING REGISTERS IT AT RUNTIME. It exists so the toolchain path is exercised
// by every build: if bracken_compile_shaders breaks, the playground stops
// building, which is a louder and earlier signal than a game discovering it.
// Registering it would move the `--smoke` screenshot baseline, and that
// baseline is the standing proof this whole feature is additive.
//
// Pair it with `fullscreen.vert.spv` from the engine's own set to draw a
// full-screen triangle; see docs/gap_pipelines.md for the registration call.

struct PSInput {
    float4 position : SV_Position;
    float2 texcoord : TEXCOORD0;
};

// space3/b0 is the FRAGMENT-stage uniform slot in SDL3 GPU's binding model
// (vertex uniforms are space1). See src/bracken/shaders/README.md — set-0
// binding ORDER is the trap there, and it applies to host shaders identically.
cbuffer TintUniforms : register(b0, space3) {
    float4 tintColor; // linear, premultiplied by the caller
};

float4 PSMain(PSInput input) : SV_Target {
    // A vignette in the host's tint colour: enough to be visibly a HOST pass
    // rather than an engine one, and cheap enough to be honest about.
    const float2 centered = input.texcoord * 2.0f - 1.0f;
    const float falloff = saturate(1.0f - dot(centered, centered));
    return float4(tintColor.rgb * falloff, tintColor.a * falloff);
}
