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
// FILTERED BY THE SAMPLER WHEN THE SOURCE IS EVEN, which is all but one level.
// Sampling the source at the destination texel's centre lands exactly between
// its eight contributors, so trilinear returns their average — the 2x2x2 box
// filter, in the texture unit, in one instruction, and the source is already
// SAMPLER-usable because the cones read it that way. Opacity averages with the
// colour, which is what a coarser voxel means: partial coverage.
//
// 96 IS NOT A POWER OF TWO, so the chain is 96, 48, 24, 12, 6, 3, 1 and the last
// step reduces 3^3 rather than 2^3. There the centre tap lands exactly ON the
// middle texel and would present 1 of 27 contributors as their mean, so an odd
// source is reduced by an explicit loop instead. The general lo/hi partition
// below degenerates to [2*tid, 2*tid+2) on the even path, so the two branches
// agree by construction.
//
// The RGB stored here is PREMULTIPLIED by coverage — that is not a choice made
// here but a consequence of this filter: injection writes black for an empty
// voxel, so averaging children attenuates the colour by the occupied fraction
// exactly as it sets the alpha to it. voxel_common.hlsli's compositor is
// written against that convention.

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
    const int srcLevel = int(voxelMipSrcLevel);
    const uint srcExtent = uint(max(int(VOXEL_RESOLUTION) >> srcLevel, 1));
    const uint3 srcSize = uint3(srcExtent, srcExtent, srcExtent);

    if ((srcExtent & 1u) == 0u) {
        // EVEN SOURCE. The destination texel's centre in normalized coordinates
        // lands exactly between its eight contributors, so trilinear filtering
        // returns their average — the box filter, for free.
        const float3 uvw = (float3(tid) + 0.5) / float3(voxelMipDstSize);
        voxelMipDst[tid] = voxelMipSrc.SampleLevel(voxelMipSrcSampler, uvw, voxelMipSrcLevel);
        return;
    }

    // ODD SOURCE — only 3 -> 1 at VOXEL_RESOLUTION = 96, a single thread doing
    // 27 loads. A centre tap here would coincide with the middle texel's own
    // centre and report it as the mean of the block.
    const uint3 lo = tid * srcSize / voxelMipDstSize;
    const uint3 hi = (tid + 1) * srcSize / voxelMipDstSize;
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    float count = 0.0;
    for (uint z = lo.z; z < hi.z; ++z) {
        for (uint y = lo.y; y < hi.y; ++y) {
            for (uint x = lo.x; x < hi.x; ++x) {
                sum += voxelMipSrc.Load(int4(int(x), int(y), int(z), srcLevel));
                count += 1.0;
            }
        }
    }
    voxelMipDst[tid] = sum / max(count, 1.0);
}
