// voxel_mip.comp.hlsl - build one level of the radiance volume's mip chain.
// Phase RG2 of the raster GI path.
//
// WHY THIS EXISTS RATHER THAN SDL_GenerateMipmapsForGPUTexture. That function
// FAULTS on a 3D texture. The volume can be created mipmapped and bound
// read-write at mip 0 perfectly well — both were verified in isolation — but the
// generation call itself is a blit chain that does not handle the third
// dimension, and it takes the process down inside the driver with no diagnostic.
// Isolating it cost a bisect through four other suspects, so: the chain is built
// here, one dispatch per level, and SDL's generator is never called on a volume.
//
// THE CHAIN IS THE TECHNIQUE, not an optimization. A cone takes ONE filtered tap
// at the level whose voxels match its width, standing in for the whole
// cross-section it represents — that substitution is what makes a cone cost tens
// of samples instead of the thousands of rays it stands for. Without the chain
// the cone tracer has nothing to widen into.
//
// FILTERED BY THE SAMPLER, not by eight manual loads. Sampling the source level
// at the destination texel's centre gives exactly the 2x2x2 box average, in the
// texture unit, in one instruction — the source is already SAMPLER-usable
// because the cones read it that way. Opacity averages with the colour, which is
// what a coarser voxel means: partial coverage.

#include "voxel_common.hlsli"

[[vk::binding(0, 2)]]
cbuffer VoxelMipUniforms : register(b0, space2) {
    /// Resolution of the level being WRITTEN.
    uint3 voxelMipDstSize;
    /// Level being READ. The destination is this + 1.
    float voxelMipSrcLevel;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture3D<float4> voxelMipSrc : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState voxelMipSrcSampler : register(s0, space0);

[[vk::image_format("rgba16f")]]
[[vk::binding(0, 1)]] RWTexture3D<float4> voxelMipDst : register(u0, space1);

[numthreads(4, 4, 4)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= voxelMipDstSize)) {
        return;
    }
    // The destination texel's centre in normalized coordinates. Sampling the
    // source level there lands exactly between its eight contributing texels, so
    // trilinear filtering returns their average — the box filter, for free.
    const float3 uvw = (float3(tid) + 0.5) / float3(voxelMipDstSize);
    voxelMipDst[tid] = voxelMipSrc.SampleLevel(voxelMipSrcSampler, uvw, voxelMipSrcLevel);
}
