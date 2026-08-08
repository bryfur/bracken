// voxel_clear.comp.hlsl - zero one cascade's albedo+opacity volume before it is
// revoxelized. Phase RG1 of the raster GI path.
//
// WHY A CLEAR IS REQUIRED AT ALL. The voxelizer composites with InterlockedMax,
// which is what makes it order-independent and therefore reproducible — but max
// is monotone, so without a clear a voxel could only ever get MORE opaque. A
// building that was demolished, or simply scrolled out of the window and back
// in over different ground, would leave its occlusion behind forever.
//
// Cheap enough not to bother being clever about: one thread per voxel over
// 96^3 is ~885k lanes of a single store, and it only runs on the frames a
// cascade actually revoxelizes.

#include "voxel_common.hlsli"

[[vk::image_format("r32ui")]]
[[vk::binding(0, 1)]] RWTexture3D<uint> voxelAlbedo : register(u0, space1);

[numthreads(4, 4, 4)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= uint(VOXEL_RESOLUTION))) {
        return;
    }
    // No window test and no wrap: this clears STORAGE, not a world region, and
    // every texel of the volume belongs to the cascade being cleared.
    voxelAlbedo[tid] = 0u;
}
