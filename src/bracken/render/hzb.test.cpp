// CPU mirror tests for the hierarchical-Z pyramid's index and reduction math
// (render/hzb.hpp, mirrored by shaders/hzb_reduce.comp.hlsl).
//
// The pyramid is a conservative structure: a texel that under-reports the
// closest surface in its footprint is not "slightly blurry", it is a hole a
// screen-space march walks through. The two ways to open one are getting the
// reduction direction backwards under reversed-Z and dropping the last
// row/column of an odd source, so both are brute-forced here rather than
// spot-checked.

#include "bracken/render/hzb.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using bracken::render::HzbSpan;
using bracken::render::hzb_mip_count;
using bracken::render::hzb_mip_size;
using bracken::render::hzb_reduce_far;
using bracken::render::hzb_reduce;
using bracken::render::hzb_source_span;

/// One level of a CPU pyramid: dimensions + row-major texels.
struct Level {
    int width = 0;
    int height = 0;
    std::vector<float> texels;

    float at(int x, int y) const { return texels[static_cast<size_t>(y) * static_cast<size_t>(width) + x]; }
};

/// Build the whole chain exactly the way HzbPyramid::build dispatches it.
std::vector<Level> build_pyramid(const Level& mip0) {
    std::vector<Level> mips;
    mips.push_back(mip0);
    const int levels = hzb_mip_count(mip0.width, mip0.height);
    for (int level = 1; level < levels; ++level) {
        const Level& src = mips.back();
        Level dst;
        dst.width = hzb_mip_size(mip0.width, level);
        dst.height = hzb_mip_size(mip0.height, level);
        dst.texels.resize(static_cast<size_t>(dst.width) * static_cast<size_t>(dst.height));
        for (int y = 0; y < dst.height; ++y) {
            for (int x = 0; x < dst.width; ++x) {
                const HzbSpan sx = hzb_source_span(x, dst.width, src.width);
                const HzbSpan sy = hzb_source_span(y, dst.height, src.height);
                float v = -std::numeric_limits<float>::infinity();
                for (int sy_i = sy.begin; sy_i < sy.end; ++sy_i) {
                    for (int sx_i = sx.begin; sx_i < sx.end; ++sx_i) {
                        v = hzb_reduce(v, src.at(sx_i, sy_i));
                    }
                }
                dst.texels[static_cast<size_t>(y) * static_cast<size_t>(dst.width) + x] = v;
            }
        }
        mips.push_back(std::move(dst));
    }
    return mips;
}

Level random_depth(int width, int height, uint32_t seed) {
    Level l;
    l.width = width;
    l.height = height;
    l.texels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& t : l.texels) {
        t = dist(rng);
    }
    return l;
}

/// The mip-0 region destination texel (x, y) of `level` transitively covers,
/// derived by walking the spans back down the chain rather than by assuming
/// a power-of-two block.
HzbSpan mip0_span(int coord, int level, int size0) {
    HzbSpan span{coord, coord + 1};
    for (int l = level; l > 0; --l) {
        const int dst_extent = hzb_mip_size(size0, l);
        const int src_extent = hzb_mip_size(size0, l - 1);
        const HzbSpan lo = hzb_source_span(span.begin, dst_extent, src_extent);
        const HzbSpan hi = hzb_source_span(span.end - 1, dst_extent, src_extent);
        span = {lo.begin, hi.end};
    }
    return span;
}

const int kSizes[][2] = {
    {1, 1},  {2, 1},   {1, 2},   {3, 3},   {5, 5},    {7, 3},    {4, 4},
    {16, 9}, {17, 11}, {64, 64}, {63, 65}, {100, 37}, {1600, 900},
};

} // namespace

// -----------------------------------------------------------------------------
// Mip-count / mip-size formulas
// -----------------------------------------------------------------------------

TEST(HzbMath, MipCountReachesOneByOne) {
    for (const auto& s : kSizes) {
        SCOPED_TRACE(testing::Message() << s[0] << "x" << s[1]);
        const int levels = hzb_mip_count(s[0], s[1]);
        EXPECT_GE(levels, 1);
        // The last level is 1x1 and no earlier level is.
        EXPECT_EQ(hzb_mip_size(s[0], levels - 1), 1);
        EXPECT_EQ(hzb_mip_size(s[1], levels - 1), 1);
        if (levels > 1) {
            EXPECT_TRUE(hzb_mip_size(s[0], levels - 2) > 1 || hzb_mip_size(s[1], levels - 2) > 1);
        }
    }
}

TEST(HzbMath, MipCountMatchesFloorLog2OfTheLongerAxis) {
    // floor(log2(max(w, h))) + 1, spelled without floating point.
    for (const auto& s : kSizes) {
        SCOPED_TRACE(testing::Message() << s[0] << "x" << s[1]);
        int longest = s[0] > s[1] ? s[0] : s[1];
        int expected = 1;
        while (longest > 1) {
            longest /= 2;
            ++expected;
        }
        EXPECT_EQ(hzb_mip_count(s[0], s[1]), expected);
    }
}

TEST(HzbMath, MipCountForThePlaygroundBackbuffer) {
    // The number quoted in docs/rt_lighting_plan.md §5.5 and printed on the
    // [smoke] gi line: 1600x900 -> 11 levels.
    EXPECT_EQ(hzb_mip_count(1600, 900), 11);
    EXPECT_EQ(hzb_mip_size(1600, 10), 1);
    EXPECT_EQ(hzb_mip_size(900, 10), 1);
    EXPECT_EQ(hzb_mip_count(3, 3), 2);
    EXPECT_EQ(hzb_mip_count(1, 1), 1);
}

TEST(HzbMath, MipSizeNeverReachesZero) {
    for (const auto& s : kSizes) {
        for (int level = 0; level < 20; ++level) {
            EXPECT_GE(hzb_mip_size(s[0], level), 1);
            EXPECT_GE(hzb_mip_size(s[1], level), 1);
        }
    }
}

// -----------------------------------------------------------------------------
// Footprints
// -----------------------------------------------------------------------------

TEST(HzbMath, SourceSpansPartitionTheSourceExactly) {
    // Every source texel belongs to exactly one destination texel. This is the
    // property the whole conservative claim rests on: gaps drop geometry,
    // overlaps would merely be wasteful.
    for (int src_extent = 1; src_extent <= 200; ++src_extent) {
        const int dst_extent = src_extent > 1 ? src_extent / 2 : 1;
        std::vector<int> owners(static_cast<size_t>(src_extent), 0);
        for (int d = 0; d < dst_extent; ++d) {
            const HzbSpan span = hzb_source_span(d, dst_extent, src_extent);
            ASSERT_GT(span.count(), 0) << "empty span at dst " << d << " of " << src_extent;
            ASSERT_GE(span.begin, 0);
            ASSERT_LE(span.end, src_extent);
            for (int s = span.begin; s < span.end; ++s) {
                owners[static_cast<size_t>(s)] += 1;
            }
        }
        for (int s = 0; s < src_extent; ++s) {
            EXPECT_EQ(owners[static_cast<size_t>(s)], 1)
                << "source texel " << s << " of " << src_extent << " covered " << owners[static_cast<size_t>(s)]
                << " times";
        }
    }
}

TEST(HzbMath, TheLastRowOrColumnOfAnOddSourceIsNeverDropped) {
    // The specific off-by-one this function exists to prevent: a plain
    // dst*2 + {0,1} gather over an odd source never reads its final texel.
    for (int src_extent = 3; src_extent <= 201; src_extent += 2) {
        const int dst_extent = src_extent / 2;
        const HzbSpan last = hzb_source_span(dst_extent - 1, dst_extent, src_extent);
        EXPECT_EQ(last.end, src_extent) << "src " << src_extent;
        EXPECT_EQ(last.count(), 3) << "src " << src_extent; // the one extended footprint
    }
}

TEST(HzbMath, EvenSourcesGatherExactlyTwo) {
    for (int src_extent = 2; src_extent <= 200; src_extent += 2) {
        const int dst_extent = src_extent / 2;
        for (int d = 0; d < dst_extent; ++d) {
            const HzbSpan span = hzb_source_span(d, dst_extent, src_extent);
            EXPECT_EQ(span, (HzbSpan{d * 2, d * 2 + 2})) << "src " << src_extent << " dst " << d;
        }
    }
}

TEST(HzbMath, DegenerateAndOutOfRangeSpans) {
    // An axis that already reached 1 does not halve again: the step is 1-to-1.
    EXPECT_EQ(hzb_source_span(0, 1, 1), (HzbSpan{0, 1}));
    // Out-of-range destinations yield empty spans, not wrapped ones.
    EXPECT_EQ(hzb_source_span(-1, 4, 8).count(), 0);
    EXPECT_EQ(hzb_source_span(4, 4, 8).count(), 0);
    EXPECT_EQ(hzb_source_span(0, 0, 8).count(), 0);
    EXPECT_EQ(hzb_source_span(0, 4, 0).count(), 0);
}

// -----------------------------------------------------------------------------
// The reduction direction
// -----------------------------------------------------------------------------

TEST(HzbMath, ReductionKeepsTheClosestSurfaceUnderReversedZ) {
    // Reversed-Z: larger stored depth == closer to the camera. The pyramid must
    // keep the CLOSEST surface in the footprint, so the reduction is a MAX.
    constexpr float far_surface = 0.05f;  // nearly at the far plane
    constexpr float near_surface = 0.95f; // nearly at the near plane
    EXPECT_FLOAT_EQ(hzb_reduce(far_surface, near_surface), near_surface);
    EXPECT_FLOAT_EQ(hzb_reduce(near_surface, far_surface), near_surface);
    // Sky (cleared to 0.0, the far plane) must never win over real geometry.
    EXPECT_FLOAT_EQ(hzb_reduce(0.0f, far_surface), far_surface);
}

TEST(HzbMath, ACoarseTexelNeverHidesANearSurface) {
    // One near texel dropped into an otherwise-far 1600x900 depth buffer must
    // survive to the 1x1 top of the pyramid.
    Level mip0 = random_depth(1600, 900, 7u);
    for (float& t : mip0.texels) {
        t *= 0.1f; // everything far
    }
    mip0.texels[static_cast<size_t>(899) * 1600 + 1599] = 0.99f; // the last texel, deliberately
    const std::vector<Level> mips = build_pyramid(mip0);
    EXPECT_FLOAT_EQ(mips.back().at(0, 0), 0.99f);
    // And at every intermediate level, some texel carries it.
    for (const Level& l : mips) {
        float best = 0.0f;
        for (float t : l.texels) {
            best = hzb_reduce(best, t);
        }
        EXPECT_FLOAT_EQ(best, 0.99f) << "lost at " << l.width << "x" << l.height;
    }
}

// -----------------------------------------------------------------------------
// Full-pyramid correctness against a brute-force reference
// -----------------------------------------------------------------------------

TEST(HzbPyramidMath, EveryMipTexelIsTheMaxOfItsImmediateSourceFootprint) {
    for (const auto& s : kSizes) {
        SCOPED_TRACE(testing::Message() << s[0] << "x" << s[1]);
        const std::vector<Level> mips = build_pyramid(random_depth(s[0], s[1], 1234u));
        for (size_t level = 1; level < mips.size(); ++level) {
            const Level& dst = mips[level];
            const Level& src = mips[level - 1];
            for (int y = 0; y < dst.height; ++y) {
                for (int x = 0; x < dst.width; ++x) {
                    const HzbSpan sx = hzb_source_span(x, dst.width, src.width);
                    const HzbSpan sy = hzb_source_span(y, dst.height, src.height);
                    float reference = -std::numeric_limits<float>::infinity();
                    for (int j = sy.begin; j < sy.end; ++j) {
                        for (int i = sx.begin; i < sx.end; ++i) {
                            reference = std::max(reference, src.at(i, j));
                        }
                    }
                    ASSERT_FLOAT_EQ(dst.at(x, y), reference)
                        << "level " << level << " texel " << x << "," << y;
                }
            }
        }
    }
}

TEST(HzbPyramidMath, EveryMipTexelIsTheMaxOverItsWholeMip0Region) {
    // The transitive claim, which is what a march actually relies on: a texel
    // at any level bounds the closest surface over EVERY mip-0 texel beneath
    // it, not merely over its immediate parents. Brute-forced against the
    // mip-0 grid directly.
    const int cases[][2] = {{3, 3}, {5, 7}, {17, 11}, {63, 65}, {100, 37}};
    for (const auto& s : cases) {
        SCOPED_TRACE(testing::Message() << s[0] << "x" << s[1]);
        const std::vector<Level> mips = build_pyramid(random_depth(s[0], s[1], 99u));
        for (size_t level = 1; level < mips.size(); ++level) {
            const Level& dst = mips[level];
            for (int y = 0; y < dst.height; ++y) {
                for (int x = 0; x < dst.width; ++x) {
                    const HzbSpan sx = mip0_span(x, static_cast<int>(level), s[0]);
                    const HzbSpan sy = mip0_span(y, static_cast<int>(level), s[1]);
                    float reference = -std::numeric_limits<float>::infinity();
                    for (int j = sy.begin; j < sy.end; ++j) {
                        for (int i = sx.begin; i < sx.end; ++i) {
                            reference = std::max(reference, mips[0].at(i, j));
                        }
                    }
                    ASSERT_FLOAT_EQ(dst.at(x, y), reference)
                        << "level " << level << " texel " << x << "," << y;
                }
            }
        }
    }
}

TEST(HzbPyramidMath, TheMip0RegionsOfALevelTileTheWholeDepthBuffer) {
    // Restates "no row or column is dropped" at the level that matters: the
    // union of every texel's mip-0 coverage is the entire buffer, at EVERY mip.
    const int cases[][2] = {{3, 3}, {5, 5}, {1600, 900}};
    for (const auto& s : cases) {
        SCOPED_TRACE(testing::Message() << s[0] << "x" << s[1]);
        const int levels = hzb_mip_count(s[0], s[1]);
        for (int level = 1; level < levels; ++level) {
            for (int axis = 0; axis < 2; ++axis) {
                const int size0 = s[axis];
                const int extent = hzb_mip_size(size0, level);
                int covered = 0;
                int expected_next = 0;
                for (int c = 0; c < extent; ++c) {
                    const HzbSpan span = mip0_span(c, level, size0);
                    ASSERT_EQ(span.begin, expected_next) << "gap/overlap at level " << level << " coord " << c;
                    expected_next = span.end;
                    covered += span.count();
                }
                EXPECT_EQ(covered, size0);
                EXPECT_EQ(expected_next, size0);
            }
        }
    }
}

TEST(HzbPyramidMath, PathologicalThreeByThree) {
    // 3x3 reduces to 1x1 in one step, so the single destination texel gathers
    // the whole 3x3 grid. A 2x2 gather here would see 4 of 9 texels.
    Level mip0;
    mip0.width = 3;
    mip0.height = 3;
    mip0.texels = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    const std::vector<Level> mips = build_pyramid(mip0);
    ASSERT_EQ(mips.size(), 2u);
    EXPECT_EQ(mips[1].width, 1);
    EXPECT_EQ(mips[1].height, 1);
    EXPECT_FLOAT_EQ(mips[1].at(0, 0), 0.9f); // the bottom-right texel a 2x2 gather misses
}

TEST(HzbPyramidMath, PyramidIsDeterministic) {
    const Level mip0 = random_depth(1600, 900, 4242u);
    const std::vector<Level> a = build_pyramid(mip0);
    const std::vector<Level> b = build_pyramid(mip0);
    ASSERT_EQ(a.size(), b.size());
    for (size_t level = 0; level < a.size(); ++level) {
        ASSERT_EQ(a[level].width, b[level].width);
        ASSERT_EQ(a[level].height, b[level].height);
        // Bitwise, not approximately: max() over the same values in the same
        // order has no room to differ, and the smoke gate depends on it.
        EXPECT_EQ(a[level].texels, b[level].texels) << "level " << level;
    }
}

// -----------------------------------------------------------------------------
// The SECOND reduction: farthest-surface-wins, for draw occlusion culling
// (docs/gap_streaming.md piece c).
//
// These mirror the closest-surface tests above one for one, plus the ordering
// test — which is the cheap guard against someone wiring the culler to the
// wrong pyramid. That mistake is silent in one direction (nothing is ever
// culled, reads as "occlusion culling does not help here") and catastrophic in
// the other (visible geometry disappears, and mean luma does not move).
// -----------------------------------------------------------------------------

namespace {

/// The far pyramid, built with the same walk and the same footprints — only the
/// reduction differs, which is exactly the claim under test.
std::vector<Level> build_far_pyramid(const Level& mip0) {
    std::vector<Level> mips;
    mips.push_back(mip0);
    const int levels = hzb_mip_count(mip0.width, mip0.height);
    for (int level = 1; level < levels; ++level) {
        const Level& src = mips.back();
        Level dst;
        dst.width = hzb_mip_size(mip0.width, level);
        dst.height = hzb_mip_size(mip0.height, level);
        dst.texels.resize(static_cast<size_t>(dst.width) * static_cast<size_t>(dst.height));
        for (int y = 0; y < dst.height; ++y) {
            for (int x = 0; x < dst.width; ++x) {
                const HzbSpan sx = hzb_source_span(x, dst.width, src.width);
                const HzbSpan sy = hzb_source_span(y, dst.height, src.height);
                float v = std::numeric_limits<float>::infinity();
                for (int sy_i = sy.begin; sy_i < sy.end; ++sy_i) {
                    for (int sx_i = sx.begin; sx_i < sx.end; ++sx_i) {
                        v = hzb_reduce_far(v, src.at(sx_i, sy_i));
                    }
                }
                dst.texels[static_cast<size_t>(y) * static_cast<size_t>(dst.width) + x] = v;
            }
        }
        mips.push_back(std::move(dst));
    }
    return mips;
}

} // namespace

TEST(HzbMath, FarReductionKeepsTheFarthestSurfaceUnderReversedZ) {
    constexpr float far_surface = 0.05f;  // nearly at the far plane
    constexpr float near_surface = 0.95f; // nearly at the near plane
    EXPECT_FLOAT_EQ(hzb_reduce_far(far_surface, near_surface), far_surface);
    EXPECT_FLOAT_EQ(hzb_reduce_far(near_surface, far_surface), far_surface);
    // Sky (cleared to 0.0 = the far plane) DOES win here, and must: a texel
    // with sky in it occludes nothing, which is what makes a prop against the
    // sky un-cullable for free.
    EXPECT_FLOAT_EQ(hzb_reduce_far(0.0f, near_surface), 0.0f);
}

TEST(HzbMath, TheTwoReductionsAreOpposites) {
    // The guard against a flipped operator anywhere in the chain: for any pair,
    // far <= near, with equality only when the pair is equal.
    Level mip0 = random_depth(64, 37, 31337u);
    for (size_t i = 0; i + 1 < mip0.texels.size(); ++i) {
        const float a = mip0.texels[i];
        const float b = mip0.texels[i + 1];
        EXPECT_LE(hzb_reduce_far(a, b), hzb_reduce(a, b));
    }
}

TEST(HzbPyramidMath, EveryFarMipTexelIsTheMinOfItsImmediateSourceFootprint) {
    for (const auto& s : kSizes) {
        SCOPED_TRACE(testing::Message() << s[0] << "x" << s[1]);
        const std::vector<Level> mips = build_far_pyramid(random_depth(s[0], s[1], 1234u));
        for (size_t level = 1; level < mips.size(); ++level) {
            const Level& dst = mips[level];
            const Level& src = mips[level - 1];
            for (int y = 0; y < dst.height; ++y) {
                for (int x = 0; x < dst.width; ++x) {
                    const HzbSpan sx = hzb_source_span(x, dst.width, src.width);
                    const HzbSpan sy = hzb_source_span(y, dst.height, src.height);
                    float reference = std::numeric_limits<float>::infinity();
                    for (int j = sy.begin; j < sy.end; ++j) {
                        for (int i = sx.begin; i < sx.end; ++i) {
                            reference = std::min(reference, src.at(i, j));
                        }
                    }
                    ASSERT_FLOAT_EQ(dst.at(x, y), reference) << "level " << level << " texel " << x << "," << y;
                }
            }
        }
    }
}

TEST(HzbPyramidMath, EveryFarMipTexelIsTheMinOverItsWholeMip0Region) {
    // The transitive claim the CULLER relies on: a texel at any level bounds
    // the FARTHEST surface over every mip-0 texel beneath it. If this ever
    // fails, a box can be declared hidden by a surface that is not actually in
    // front of it everywhere in the footprint.
    const int cases[][2] = {{3, 3}, {5, 7}, {17, 11}, {63, 65}, {100, 37}};
    for (const auto& s : cases) {
        SCOPED_TRACE(testing::Message() << s[0] << "x" << s[1]);
        const std::vector<Level> mips = build_far_pyramid(random_depth(s[0], s[1], 99u));
        for (size_t level = 1; level < mips.size(); ++level) {
            const Level& dst = mips[level];
            for (int y = 0; y < dst.height; ++y) {
                for (int x = 0; x < dst.width; ++x) {
                    const HzbSpan sx = mip0_span(x, static_cast<int>(level), s[0]);
                    const HzbSpan sy = mip0_span(y, static_cast<int>(level), s[1]);
                    float reference = std::numeric_limits<float>::infinity();
                    for (int j = sy.begin; j < sy.end; ++j) {
                        for (int i = sx.begin; i < sx.end; ++i) {
                            reference = std::min(reference, mips[0].at(i, j));
                        }
                    }
                    ASSERT_FLOAT_EQ(dst.at(x, y), reference) << "level " << level << " texel " << x << "," << y;
                }
            }
        }
    }
}

TEST(HzbPyramidMath, TheFarPyramidIsTexelWiseNoGreaterThanTheNearOne) {
    // THE WRONG-PYRAMID GUARD. Wire the culler to the closest-surface chain and
    // it culls nothing (harmless, looks like "no occluders here"); flip the
    // comparison to make it cull and it removes visible geometry. This is the
    // one assertion that would catch either.
    for (const auto& s : kSizes) {
        SCOPED_TRACE(testing::Message() << s[0] << "x" << s[1]);
        const Level mip0 = random_depth(s[0], s[1], 555u);
        const std::vector<Level> near_mips = build_pyramid(mip0);
        const std::vector<Level> far_mips = build_far_pyramid(mip0);
        ASSERT_EQ(near_mips.size(), far_mips.size());
        for (size_t level = 0; level < near_mips.size(); ++level) {
            ASSERT_EQ(near_mips[level].texels.size(), far_mips[level].texels.size());
            for (size_t i = 0; i < near_mips[level].texels.size(); ++i) {
                ASSERT_LE(far_mips[level].texels[i], near_mips[level].texels[i])
                    << "level " << level << " texel " << i;
            }
        }
        // Mip 0 is a copy either way, so the two chains must agree there and
        // (for a random buffer) diverge above it — otherwise the "opposite
        // reductions" claim is vacuous on this input.
        EXPECT_EQ(near_mips[0].texels, far_mips[0].texels);
        if (near_mips.size() > 1) {
            EXPECT_NE(near_mips[1].texels, far_mips[1].texels);
        }
    }
}
