// The instance-batch packing order (SceneRenderer::for_each_instance_batch).
//
// Four sites must agree on it — the packer in scene_renderer_upload.cpp and
// the forward / depth-prepass / shadow draw walks — and after the
// translation-unit split they live in four different files. The walk is one
// function so there is one definition; this pins what that definition is.
//
// The failure this guards against is silent and ugly: if a draw walk SKIPS a
// bin whose handle no longer resolves instead of skipping the draw and
// advancing past its packed range, every later bin renders with another
// model's transforms. Nothing upstream of binning can detect it, and no test
// that only inspects bins can either, so the offsets are asserted here by
// value.

#include <gtest/gtest.h>

#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/scene/scene_renderer.hpp"

#include <cstdint>
#include <utility>
#include <vector>

using namespace bracken;
using namespace bracken::scene;

namespace {

using Bins = std::vector<std::vector<gpu::InstanceData>>;

// A dense bin array where bins[h] holds `counts[h]` instances (0 = empty).
Bins make_bins(std::initializer_list<std::size_t> counts) {
    Bins bins;
    for (std::size_t n : counts) {
        bins.emplace_back(n);
    }
    return bins;
}

struct Visit {
    ModelHandle handle;
    std::uint32_t base_instance;
    std::size_t count;
    bool operator==(const Visit&) const = default;
};

std::vector<Visit> walk(const Bins& bins, std::uint32_t base = 0, std::uint32_t* end_out = nullptr) {
    std::vector<Visit> seen;
    const std::uint32_t end = SceneRenderer::for_each_instance_batch(
        bins, base, [&](ModelHandle h, const std::vector<gpu::InstanceData>& instances, std::uint32_t b) {
            seen.push_back({h, b, instances.size()});
        });
    if (end_out) {
        *end_out = end;
    }
    return seen;
}

} // namespace

TEST(InstanceBatchWalk, VisitsNonEmptyBinsInAscendingHandleOrder) {
    const Bins bins = make_bins({2, 0, 3, 0, 0, 1});
    const std::vector<Visit> expected = {{0, 0, 2}, {2, 2, 3}, {5, 5, 1}};
    EXPECT_EQ(walk(bins), expected);
}

TEST(InstanceBatchWalk, EmptyBinsConsumeNoBaseInstanceRange) {
    // Leading, interior and trailing holes must all be invisible to the
    // offsets — the packer writes nothing for them.
    const Bins bins = make_bins({0, 0, 4, 0, 2, 0, 0});
    const std::vector<Visit> expected = {{2, 0, 4}, {4, 4, 2}};
    EXPECT_EQ(walk(bins), expected);
}

TEST(InstanceBatchWalk, ReturnsTheOffsetOnePastTheLastBin) {
    std::uint32_t end = 0;
    const Bins bins = make_bins({3, 0, 5});
    (void)walk(bins, 0, &end);
    EXPECT_EQ(end, 8u) << "the shadow packer carries this into the next cascade's base";

    // Trailing empties do not extend the packed range.
    const Bins trailing = make_bins({3, 0, 5, 0, 0});
    (void)walk(trailing, 0, &end);
    EXPECT_EQ(end, 8u);
}

TEST(InstanceBatchWalk, SeedBaseOffsetsEveryVisit) {
    // Shadow cascades are packed contiguously into one buffer: cascade c
    // starts at shadow_cascade_base_instance_[c].
    const Bins bins = make_bins({2, 1});
    const std::vector<Visit> expected = {{0, 100, 2}, {1, 102, 1}};
    EXPECT_EQ(walk(bins, 100), expected);
}

TEST(InstanceBatchWalk, EmptyBinArrayVisitsNothingAndKeepsTheBase) {
    std::uint32_t end = 0;
    const Bins bins;
    EXPECT_TRUE(walk(bins, 42, &end).empty());
    EXPECT_EQ(end, 42u);
}

TEST(InstanceBatchWalk, AllEmptyBinsVisitNothingAndKeepTheBase) {
    std::uint32_t end = 0;
    const Bins bins = make_bins({0, 0, 0});
    EXPECT_TRUE(walk(bins, 7, &end).empty());
    EXPECT_EQ(end, 7u);
}

// The load-bearing case, stated by value: a bin in the MIDDLE whose model no
// longer resolves. The packer emits its instances, so the draw walk must skip
// the draw and still advance — which is what the drawable form does by
// building on this walk rather than re-implementing the loop.
TEST(InstanceBatchWalk, AnUndrawableBinStillConsumesItsPackedRange) {
    const Bins bins = make_bins({2, 3, 4});
    // Pretend handle 1 does not resolve: the caller simply returns without
    // drawing. The base_instance handed to handle 2 must be unchanged.
    std::vector<Visit> drawn;
    const std::uint32_t end = SceneRenderer::for_each_instance_batch(
        bins, 0, [&](ModelHandle h, const std::vector<gpu::InstanceData>& instances, std::uint32_t b) {
            if (h == 1) {
                return; // "model_manager_->get_model(h) == nullptr"
            }
            drawn.push_back({h, b, instances.size()});
        });
    const std::vector<Visit> expected = {{0, 0, 2}, {2, 5, 4}};
    EXPECT_EQ(drawn, expected) << "handle 2 must start at 5 (2 + 3), not at 2";
    EXPECT_EQ(end, 9u);
}

// The packer and the draw walks are the same function, so a shadow-instance
// bin array must produce identical offsets to a main-instance one.
TEST(InstanceBatchWalk, OffsetsDoNotDependOnTheInstanceType) {
    std::vector<std::vector<gpu::ShadowInstanceData>> shadow_bins;
    for (std::size_t n : {2u, 0u, 3u, 1u}) {
        shadow_bins.emplace_back(n);
    }
    std::vector<Visit> seen;
    const std::uint32_t end = SceneRenderer::for_each_instance_batch(
        shadow_bins, 0, [&](ModelHandle h, const std::vector<gpu::ShadowInstanceData>& instances, std::uint32_t b) {
            seen.push_back({h, b, instances.size()});
        });
    const std::vector<Visit> expected = {{0, 0, 2}, {2, 2, 3}, {3, 5, 1}};
    EXPECT_EQ(seen, expected);
    EXPECT_EQ(end, 6u);
}
