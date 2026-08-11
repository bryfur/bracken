// TextureCache: the byte estimator and the release path's refusals.
//
// WHAT IS AND IS NOT COVERED, stated up front because the gap matters.
// `acquire()` needs a live `gpu::GPUDevice` — it creates a GPU texture — and
// there is no device in the test suite (no window, no swapchain). So the
// refcount's INCREMENT side is not exercised here; what is exercised is
// everything reachable without one: the size estimate that feeds the residency
// budget, and the release path's behaviour on null, foreign and unknown
// pointers, which is where the eviction walk actually spends its time.
//
// The refcount's decrement-to-zero behaviour is guarded by the fact that
// destruction is OFF by default (`set_destroy_on_release`), and the header says
// why. Enabling it is a measured step for a project that has verified its own
// pairing with `stats()`, not something a unit test can bless.

#include <gtest/gtest.h>

#include "bracken/texture_cache.hpp"

#include <cstdint>

using namespace bracken;

TEST(TextureBytesEstimate, UnmippedIsExactlyWidthTimesHeightTimesTexel) {
    EXPECT_EQ(texture_bytes_estimate(4, 4, gpu::TextureFormat::RGBA8, false), 4u * 4u * 4u);
    EXPECT_EQ(texture_bytes_estimate(4, 4, gpu::TextureFormat::R8, false), 4u * 4u * 1u);
    EXPECT_EQ(texture_bytes_estimate(4, 4, gpu::TextureFormat::R16, false), 4u * 4u * 2u);
    EXPECT_EQ(texture_bytes_estimate(256, 128, gpu::TextureFormat::RGBA8_SRGB, false), 256u * 128u * 4u);
}

TEST(TextureBytesEstimate, MipChainIsTheEXACTSumNotTheFourThirdsLimit) {
    // 4/3 is the limit of a square chain, and it is wrong for anything else.
    // A 1024x1 strip's chain is 1024+512+256+...+1 = 2047 texels, nowhere near
    // 4/3 of 1024 — and a residency budget built on the limit would under-count
    // exactly the textures (UI atlases, gradient ramps) that are shaped that way.
    std::size_t expect = 0;
    for (int w = 1024; w >= 1; w /= 2) {
        expect += static_cast<std::size_t>(w);
    }
    EXPECT_EQ(texture_bytes_estimate(1024, 1, gpu::TextureFormat::R8, true), expect);

    // A square chain: 64x64 + 32x32 + ... + 1x1 == 5461 texels.
    std::size_t square = 0;
    for (int s = 64; s >= 1; s /= 2) {
        square += static_cast<std::size_t>(s) * static_cast<std::size_t>(s);
    }
    EXPECT_EQ(texture_bytes_estimate(64, 64, gpu::TextureFormat::R8, true), square);
    EXPECT_EQ(square, 5461u);
}

TEST(TextureBytesEstimate, IsMonotoneInEveryDimension) {
    // The property the budget actually needs: twice the pixels is more bytes.
    // Exactness would be a lie (SDL exposes no allocation size); monotonicity
    // is what makes "evict the big one" mean anything.
    for (int w = 1; w <= 512; w *= 2) {
        EXPECT_LT(texture_bytes_estimate(w, 8, gpu::TextureFormat::RGBA8, true),
                  texture_bytes_estimate(w * 2, 8, gpu::TextureFormat::RGBA8, true));
        EXPECT_LT(texture_bytes_estimate(8, w, gpu::TextureFormat::RGBA8, true),
                  texture_bytes_estimate(8, w * 2, gpu::TextureFormat::RGBA8, true));
    }
    EXPECT_GT(texture_bytes_estimate(64, 64, gpu::TextureFormat::RGBA8, true),
              texture_bytes_estimate(64, 64, gpu::TextureFormat::RGBA8, false));
}

TEST(TextureBytesEstimate, DegenerateSizesAreZero) {
    EXPECT_EQ(texture_bytes_estimate(0, 16, gpu::TextureFormat::RGBA8, true), 0u);
    EXPECT_EQ(texture_bytes_estimate(16, 0, gpu::TextureFormat::RGBA8, true), 0u);
    EXPECT_EQ(texture_bytes_estimate(-4, -4, gpu::TextureFormat::RGBA8, true), 0u);
}

TEST(TextureCache, AnEmptyCacheReportsNothing) {
    TextureCache cache;
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(cache.hits(), 0u);
    EXPECT_EQ(cache.bytes_resident(), 0u);
    const TextureCache::Stats s = cache.stats();
    EXPECT_EQ(s.images, 0u);
    EXPECT_EQ(s.references, 0u);
    EXPECT_EQ(s.bytes_resident, 0u);
}

TEST(TextureCache, DestructionOnReleaseIsOffByDefault) {
    // The whole risk posture of the refcount, pinned as a test rather than left
    // as a comment: Mesh/PBRMaterial hold four non-owning GPUTexture* each, and
    // a missed pairing with destruction ON is a live descriptor pointing at a
    // freed texture — a GPU fault, not an assert.
    TextureCache cache;
    EXPECT_FALSE(cache.destroys_on_release());
    cache.set_destroy_on_release(true);
    EXPECT_TRUE(cache.destroys_on_release());
    cache.set_destroy_on_release(false);
    EXPECT_FALSE(cache.destroys_on_release());
}

TEST(TextureCache, ReleasingNullIsASilentNoOp) {
    // The eviction walk releases four slots per mesh and most of them are null.
    // Counting those as "unknown" would bury the diagnostic that matters.
    TextureCache cache;
    cache.release(nullptr);
    cache.release(nullptr);
    EXPECT_EQ(cache.stats().unknown_releases, 0u);
    EXPECT_EQ(cache.refcount(nullptr), 0u);
    EXPECT_EQ(cache.texture_bytes(nullptr), 0u);
}

TEST(TextureCache, ReleasingAForeignPointerIsCountedAndSurvived) {
    // A pointer this cache does not own — another subsystem's texture, or one
    // an earlier eviction already dropped. Counted, warned once, never fatal.
    TextureCache cache;
    // Never dereferenced: release() only looks the pointer up in a map.
    auto* foreign = reinterpret_cast<gpu::GPUTexture*>(static_cast<std::uintptr_t>(0xF00D));
    cache.release(foreign);
    cache.release(foreign);
    EXPECT_EQ(cache.stats().unknown_releases, 2u);
    EXPECT_EQ(cache.refcount(foreign), 0u);
    EXPECT_EQ(cache.texture_bytes(foreign), 0u);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TextureCache, ClearResetsEveryCounter) {
    TextureCache cache;
    auto* foreign = reinterpret_cast<gpu::GPUTexture*>(static_cast<std::uintptr_t>(0xBEEF));
    cache.release(foreign);
    ASSERT_EQ(cache.stats().unknown_releases, 1u);
    cache.clear();
    const TextureCache::Stats s = cache.stats();
    EXPECT_EQ(s.images, 0u);
    EXPECT_EQ(s.bytes_resident, 0u);
    EXPECT_EQ(s.released_to_zero, 0u);
    EXPECT_EQ(s.destroyed, 0u);
    EXPECT_EQ(s.unknown_releases, 0u);
    EXPECT_EQ(cache.hits(), 0u);
}
