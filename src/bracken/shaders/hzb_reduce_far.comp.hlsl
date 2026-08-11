// hzb_reduce_far.comp.hlsl — one mip of the FARTHEST-surface Hi-Z pyramid.
//
// Byte-for-byte hzb_reduce.comp.hlsl with ONE operator flipped: max -> min.
// Two files rather than one behind a #define because the pipeline catalogue is
// a table of (name, shader file, resources) rows and a second entry point in
// one file would have to be a second row anyway — and because the two are the
// two halves of a contract that must never accidentally become the same
// shader. The C++ mirrors are hzb_reduce() and hzb_reduce_far() in
// render/hzb.hpp, and render/hzb.test.cpp pins `far <= near` texel for texel.
//
// ---------------------------------------------------------------------------
// THE REDUCTION IS A MIN, AND THAT IS THE OPPOSITE OF THE OTHER ONE. Scene
// depth is reversed-Z zero-to-one: near maps to 1, far to 0, so a SMALLER value
// is a FARTHER surface. Draw occlusion culling may declare a box hidden only
// when the box's nearest point is behind the FARTHEST occluder anywhere in its
// footprint — anything weaker culls a box that peeks between two occluders that
// share a texel. So a coarse texel must retain that farthest surface, which
// here is the MIN. Use the closest-surface chain (hzb_reduce.comp.hlsl) for
// this and the culler either culls nothing or, if someone "fixes" the compare,
// culls visible geometry. Full derivation: render/hzb.hpp.
//
// THE FOOTPRINT IS NOT ALWAYS 2x2. Mip N is max(1, size >> N), so an odd source
// dimension loses its last row/column to integer division — a 5-wide source
// reduces to 2 and a plain 2x2 gather never reads column 4. Clamping the gather
// to bounds does not fix that; it re-reads a column already covered. The last
// destination texel therefore EXTENDS to the end of the source. A source
// dimension is either 2*dst or 2*dst+1, so the extension is at most one texel
// and the gather is 2x2, 3x2, 2x3 or 3x3. The C++ mirror of this indexing —
// hzb_source_span() in render/hzb.hpp — is what the tests brute-force against.

[[vk::binding(0, 2)]]
cbuffer HzbReduceUniforms : register(b0, space2) {
    uint2 dstSize;    // dimensions of the mip being written
    uint2 srcSize;    // dimensions of the mip being read
};

// u0 = source mip (read only, despite the binding), u1 = destination mip.
//
// The explicit [[vk::image_format]] is what makes the READ legal: a storage
// image declared Unknown may only be WRITTEN unless the device advertises the
// optional shaderStorageImageReadWithoutFormat feature. Same reason the GI
// blend passes carry one. It must name the texture's real format (R32_FLOAT).
[[vk::image_format("r32f")]]
[[vk::binding(0, 1)]] RWTexture2D<float> hzbSrc : register(u0, space1);
[[vk::image_format("r32f")]]
[[vk::binding(1, 1)]] RWTexture2D<float> hzbDst : register(u1, space1);

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= dstSize.x || tid.y >= dstSize.y) {
        return;
    }
    const int2 dst = int2(tid.xy);
    const int2 srcExtent = int2(srcSize);
    const int2 dstExtent = int2(dstSize);

    // Mirrors hzb_source_span(): [begin, end) per axis, with the last
    // destination texel absorbing an odd source's trailing texel. dstExtent >=
    // srcExtent is the degenerate 1-to-1 step an axis that already hit 1 takes.
    int2 begin;
    int2 end;
    [unroll]
    for (int a = 0; a < 2; ++a) {
        if (dstExtent[a] >= srcExtent[a]) {
            begin[a] = dst[a];
            end[a] = dst[a] + 1;
        } else {
            begin[a] = dst[a] * 2;
            end[a] = (dst[a] == dstExtent[a] - 1) ? srcExtent[a] : min(begin[a] + 2, srcExtent[a]);
        }
    }

    // The footprint is at most 3x3, and the loop bounds are data-dependent only
    // at the two trailing edges.
    float farthest = hzbSrc[int2(begin.x, begin.y)];
    for (int y = begin.y; y < end.y; ++y) {
        for (int x = begin.x; x < end.x; ++x) {
            farthest = min(farthest, hzbSrc[int2(x, y)]);
        }
    }
    hzbDst[dst] = farthest;
}
