// rt_shadow_denoise_classify.comp.hlsl — tile classification, pass 1 of 3.
//
// One thread per 8x4 tile. Decides whether the tile needs to be filtered at
// all, and writes that verdict where the temporal and filter passes can read
// it in a single fetch.
//
// WHY THIS PASS EXISTS. In a sunlit exterior most of the frame is either
// entirely lit or entirely in shadow; only the penumbra bands are actually
// noisy. A denoiser that filters uniformly spends nearly all of its time
// averaging 25 identical values together. Classifying first turns the filter
// into a pass that runs at full cost on the few percent of tiles that carry
// signal and at near-zero cost everywhere else.
//
// A tile is only skippable if ITS NEIGHBOURS agree with it too — the filter
// reads across tile boundaries, so a uniform tile bordering a noisy one still
// has to run or the penumbra gets a hard edge exactly on the tile grid. Hence
// the 3x3 gather rather than a single load.

#include "rt_sampling.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtShadowClassifyUniforms : register(b0, space2) {
    uint2 tileCount;   // dimensions of the packed-bit texture
    uint2 _pad0;
};

[[vk::binding(0, 0)]] Texture2D<uint> shadowBits : register(t0, space0);

// 0 = uniformly lit, 1 = uniformly shadowed, 2 = mixed (must be filtered).
[[vk::binding(0, 1)]] RWTexture2D<uint> tileClass : register(u0, space1);

static const uint TILE_LIT = 0u;
static const uint TILE_SHADOW = 1u;
static const uint TILE_MIXED = 2u;

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= tileCount.x || tid.y >= tileCount.y) {
        return;
    }
    const int2 tile = int2(tid.xy);

    // Accumulate the AND and the OR of the whole 3x3 neighbourhood. all-ones
    // means every pixel in every neighbour is lit; all-zeros means every one
    // is shadowed. Anything else is mixed.
    uint andBits = 0xFFFFFFFFu;
    uint orBits = 0u;
    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            // Clamp rather than skip: an out-of-bounds neighbour must not be
            // able to make an edge tile look uniform when it is not.
            const int2 n = clamp(tile + int2(dx, dy), int2(0, 0), int2(tileCount) - 1);
            const uint bits = shadowBits.Load(int3(n, 0));
            andBits &= bits;
            orBits |= bits;
        }
    }

    uint verdict = TILE_MIXED;
    if (andBits == 0xFFFFFFFFu) {
        verdict = TILE_LIT;
    } else if (orBits == 0u) {
        verdict = TILE_SHADOW;
    }
    tileClass[tile] = verdict;
}
