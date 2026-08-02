// hzb_copy.comp.hlsl — mip 0 of the hierarchical-Z pyramid.
//
// A straight copy of the depth prepass into the R32_FLOAT pyramid, and it has
// to be a copy rather than an aliased view for one reason: the prepass depth is
// D32_FLOAT, and a depth format cannot be a compute STORAGE texture. So this
// pass reads the depth through a SAMPLER (Load(), no filtering) and writes the
// storage level. Everything above mip 0 is hzb_reduce.comp.hlsl.
//
// No conversion happens here. The value stored is the raw reversed-Z device
// depth, near = 1, far = 0 — see the reduction-direction derivation in
// render/hzb.hpp. Linearizing at build time would bake this frame's projection
// into the pyramid; the march that consumes it knows its own projection.

[[vk::binding(0, 2)]]
cbuffer HzbCopyUniforms : register(b0, space2) {
    uint2 mipSize;    // dimensions of mip 0 == the depth texture's
    uint2 _pad0;
};

[[vk::binding(0, 0)]] Texture2D<float> sceneDepth : register(t0, space0);
[[vk::binding(0, 0)]] SamplerState sceneDepthSampler : register(s0, space0);

// Write-only, so the format qualifier is not strictly required — declared
// anyway so both HZB shaders name the same format at the same binding and a
// future read here does not become a silent Unknown-image read.
[[vk::image_format("r32f")]]
[[vk::binding(0, 1)]] RWTexture2D<float> hzbMip0 : register(u0, space1);

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= mipSize.x || tid.y >= mipSize.y) {
        return;
    }
    hzbMip0[int2(tid.xy)] = sceneDepth.Load(int3(int2(tid.xy), 0));
}
