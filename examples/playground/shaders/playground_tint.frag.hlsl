// The reference consumer's own shader — the thing bracken_compile_shaders
// exists for. It is compiled by the ENGINE's pinned shadercross/DXC and staged
// into `shaders_playground/` next to the exe, deliberately NOT into `shaders/`
// (bracken_copy_shaders wipes that directory every build) and deliberately not
// into the engine's own SPIR-V output directory (gpu/uniform_layout.test.cpp
// enumerates that one exhaustively and would fail on a host cbuffer).
//
// It is compiled by EVERY build, and registered at runtime only when
// PLAYGROUND_CUSTOM_PASS=1 — see examples/playground/custom_pass.cpp, which is
// the worked example of the whole loop. Off by default because `--smoke` is the
// engine's screenshot regression gate and this paints on the frame.
//
// Paired with the engine's own `fullscreen.vert.spv` (search roots resolve
// engine-first, so a host pipeline can name an engine shader), drawn with
// BlendMode::AdditivePreserveAlpha into the RGBA16F linear HDR scene target.
// That blend adds RGB and leaves DESTINATION ALPHA alone, because scene alpha
// is metadata later passes read — so this shader writes PREMULTIPLIED RGB and
// its own alpha output is discarded by the blend state. Do not "fix" the alpha
// here; it is not read.

struct PSInput {
    float4 position : SV_Position;
    float2 texcoord : TEXCOORD0;
};

// space3/b0 is the FRAGMENT-stage uniform slot in SDL3 GPU's binding model
// (vertex uniforms are space1). See src/bracken/shaders/README.md — set-0
// binding ORDER is the trap there, and it applies to host shaders identically.
cbuffer TintUniforms : register(b0, space3) {
    float4 tintColor; // linear HDR; .a is the overall strength
};

float4 PSMain(PSInput input) : SV_Target {
    // Edge-weighted vignette: zero in the middle of the frame, strongest in the
    // corners. Deliberately not centre-weighted — a wash over the whole image
    // would be hard to tell apart from an exposure change, and the point of the
    // example is to be visibly a HOST pass.
    const float2 centered = input.texcoord * 2.0f - 1.0f;
    const float falloff = saturate(dot(centered, centered) - 0.25f);
    return float4(tintColor.rgb * (tintColor.a * falloff), 0.0f);
}
