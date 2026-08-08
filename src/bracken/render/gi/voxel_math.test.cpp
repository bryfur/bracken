#include "bracken/render/gi/voxel_math.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <set>

// The voxel radiance clipmap's addressing and cone math (RG0). Every function
// here is transliterated into shaders/voxel_common.hlsli, where it cannot be
// tested; this is the half that can be. The properties pinned below are the
// ones a transliteration silently breaks — the torus wrap at the origin, the
// containment test that is the only thing standing between an out-of-window
// sample and a confident wrong answer, and the cone's mip/step relationship,
// which is what makes the technique affordable at all.

namespace bracken::render::gi {
namespace {

TEST(VoxelMath, CascadesNestAndDouble) {
    for (int c = 1; c < VOXEL_CASCADE_COUNT; ++c) {
        EXPECT_FLOAT_EQ(voxel_size(c), 2.0f * voxel_size(c - 1)) << "cascade " << c;
        EXPECT_FLOAT_EQ(voxel_cascade_extent(c), 2.0f * voxel_cascade_extent(c - 1));
    }
    // Same texel count in every cascade — that is what "clipmap" means here, and
    // what lets one shader index all three.
    EXPECT_FLOAT_EQ(voxel_cascade_extent(0), VOXEL_BASE_SIZE * float(VOXEL_RESOLUTION));
}

TEST(VoxelMath, WrapIsEuclideanAcrossTheOrigin) {
    // C++ % truncates toward zero, so the naive form folds -1 onto 1 instead of
    // onto RESOLUTION-1 and the volume tears along the world axes. Walk a range
    // that straddles zero and require the sequence to be contiguous.
    for (int v = -2 * VOXEL_RESOLUTION; v < 2 * VOXEL_RESOLUTION; ++v) {
        const int w = voxel_wrap_axis(v, VOXEL_RESOLUTION);
        EXPECT_GE(w, 0) << "at " << v;
        EXPECT_LT(w, VOXEL_RESOLUTION) << "at " << v;
        // Adjacent inputs must land on adjacent texels (mod resolution).
        const int next = voxel_wrap_axis(v + 1, VOXEL_RESOLUTION);
        EXPECT_EQ(next, (w + 1) % VOXEL_RESOLUTION) << "discontinuity at " << v;
    }
}

TEST(VoxelMath, WrapIsABijectionOverOneWindow) {
    // Every texel used exactly once across a full window — if two lattice
    // coordinates aliased, two points in the world would share a voxel.
    std::set<int> seen;
    for (int v = -37; v < -37 + VOXEL_RESOLUTION; ++v) {
        seen.insert(voxel_wrap_axis(v, VOXEL_RESOLUTION));
    }
    EXPECT_EQ(seen.size(), static_cast<std::size_t>(VOXEL_RESOLUTION));
}

TEST(VoxelMath, BaseAdvancesOneVoxelAtATimeThroughTheOrigin) {
    // `floor` rather than truncation, and this is what it buys: the base must
    // be monotone as the camera crosses zero, or the whole volume shifts by a
    // voxel and every stored radiance is suddenly in the wrong place.
    const float vs = voxel_size(0);
    int previous = voxel_cascade_base(0, glm::vec3(-4.0f * vs, 0.0f, 0.0f)).x;
    for (float x = -4.0f * vs; x <= 4.0f * vs; x += vs * 0.25f) {
        const int base = voxel_cascade_base(0, glm::vec3(x, 0.0f, 0.0f)).x;
        EXPECT_GE(base, previous) << "base went backwards at x=" << x;
        EXPECT_LE(base - previous, 1) << "base skipped a voxel at x=" << x;
        previous = base;
    }
}

TEST(VoxelMath, SubVoxelCameraMotionDoesNotScroll) {
    // The scroll snaps to whole voxels precisely so that a camera drifting
    // within one voxel leaves every stored voxel exactly where it was. A
    // fractional base would move every voxel's world position every frame and
    // invalidate the entire volume continuously.
    const float vs = voxel_size(0);
    const glm::ivec3 a = voxel_cascade_base(0, glm::vec3(1000.0f, 50.0f, -20.0f));
    const glm::ivec3 b =
        voxel_cascade_base(0, glm::vec3(1000.0f + vs * 0.49f, 50.0f, -20.0f + vs * 0.49f));
    EXPECT_EQ(a, b);
}

TEST(VoxelMath, CenterRoundTripsThroughCoordAt) {
    // A voxel's centre must land back in that same voxel. This is the pairing
    // the sampler depends on: a trilinear tap at `voxel_center` should return
    // that voxel and not a blend with its neighbour.
    for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
        for (const glm::ivec3 coord :
             {glm::ivec3{0, 0, 0}, glm::ivec3{5, -3, 17}, glm::ivec3{-129, 64, -1}}) {
            EXPECT_EQ(voxel_coord_at(c, voxel_center(c, coord)), coord)
                << "cascade " << c << " coord " << coord.x << "," << coord.y << "," << coord.z;
        }
    }
}

TEST(VoxelMath, InWindowRejectsWhatTheTorusWouldHappilyAnswer) {
    // THE POINT OF THE TEST. Torus addressing wraps ANY coordinate to a valid
    // texel, so without this containment test a sample one voxel outside the
    // window reads the opposite side of the world with full confidence.
    const glm::ivec3 base{100, -50, 7};
    EXPECT_TRUE(voxel_in_window(base, base));
    EXPECT_TRUE(voxel_in_window(base + glm::ivec3(VOXEL_RESOLUTION - 1), base));
    EXPECT_FALSE(voxel_in_window(base - glm::ivec3(1, 0, 0), base));
    EXPECT_FALSE(voxel_in_window(base + glm::ivec3(VOXEL_RESOLUTION, 0, 0), base));
    // And the aliasing it prevents: one window away wraps to the SAME texel.
    const glm::ivec3 outside = base + glm::ivec3(VOXEL_RESOLUTION, 0, 0);
    EXPECT_EQ(voxel_wrap(outside), voxel_wrap(base));
    EXPECT_FALSE(voxel_in_window(outside, base));
}

TEST(VoxelMath, CascadeSelectionPicksTheFinestThatFits) {
    const glm::vec3 camera(0.0f);
    EXPECT_EQ(voxel_select_cascade(camera, camera), 0);

    // Just inside cascade 0's margin.
    const float inner = voxel_cascade_half_extent(0) - VOXEL_CASCADE_MARGIN * voxel_size(0) - 1.0f;
    EXPECT_EQ(voxel_select_cascade(camera + glm::vec3(inner, 0, 0), camera), 0);

    // Past cascade 0 but inside cascade 1.
    const float beyond0 = voxel_cascade_half_extent(0) + 1.0f;
    EXPECT_EQ(voxel_select_cascade(camera + glm::vec3(beyond0, 0, 0), camera), 1);

    // Past everything.
    const float beyondAll = voxel_cascade_half_extent(VOXEL_CASCADE_COUNT - 1) + 1.0f;
    EXPECT_EQ(voxel_select_cascade(camera + glm::vec3(beyondAll, 0, 0), camera), -1);
}

TEST(VoxelMath, SelectedCascadeAlwaysHasMarginToFilterIn) {
    // The margin is not politeness: a trilinear tap reaches half a voxel past
    // the sample and a mip tap reaches further, so a sample accepted at the
    // very edge would filter against the opposite side of the torus. Whatever
    // cascade is chosen, the point must be inside its window by at least the
    // margin — checked against the WINDOW, not against the selection rule, so
    // the two cannot drift into agreeing with each other and not with reality.
    const glm::vec3 camera(2500.0f, 120.0f, -700.0f);
    for (float d = 0.0f; d < voxel_cascade_half_extent(VOXEL_CASCADE_COUNT - 1); d += 37.0f) {
        const glm::vec3 p = camera + glm::normalize(glm::vec3(0.6f, 0.3f, -0.74f)) * d;
        const int c = voxel_select_cascade(p, camera);
        if (c < 0) {
            continue;
        }
        const glm::ivec3 base = voxel_cascade_base(c, camera);
        const glm::ivec3 coord = voxel_coord_at(c, p);
        EXPECT_TRUE(voxel_in_window(coord, base)) << "cascade " << c << " at distance " << d;
        const glm::ivec3 local = coord - base;
        const int m = static_cast<int>(VOXEL_CASCADE_MARGIN);
        EXPECT_GE(std::min({local.x, local.y, local.z}), m - 1) << "distance " << d;
        EXPECT_LE(std::max({local.x, local.y, local.z}), VOXEL_RESOLUTION - m) << "distance " << d;
    }
}

TEST(VoxelMath, AWideConeIsPushedToACoarserCascadeRatherThanWrapping) {
    // THE SEAM BUG, pinned. Storage is a torus, so a tap whose filter footprint
    // crosses the window edge reads the opposite side of the world — which drew
    // a hard black line across the map at the cascade boundary. A fixed margin
    // covers a mip-0 tap and nothing wider; the footprint term is what makes it
    // scale with the level a cone actually samples.
    const glm::vec3 camera(0.0f);
    // A point comfortably inside cascade 0 for a pinpoint tap...
    const float inside = voxel_cascade_half_extent(0) - 4.0f * voxel_size(0);
    const glm::vec3 p = camera + glm::vec3(inside, 0.0f, 0.0f);
    EXPECT_EQ(voxel_select_cascade_for_footprint(p, camera, 0.0f), 0);
    // ...is NOT safe for a cone whose footprint reaches past the window edge,
    // and must be answered by a coarser cascade instead of wrapping.
    const int coarse = voxel_select_cascade_for_footprint(p, camera, 8.0f * voxel_size(0));
    EXPECT_GT(coarse, 0) << "a wide cone must not be answered by the cascade it would wrap in";
}

TEST(VoxelMath, EveryConeSampleStaysInsideItsCascadeWithItsFootprint) {
    // The property the fix has to guarantee, checked against the WINDOW rather
    // than against the selection rule, so the two cannot agree with each other
    // and not with reality. Walks a real cone: widening footprint, increasing
    // distance, exactly as voxel_cone_trace does.
    const glm::vec3 camera(1234.0f, 60.0f, -900.0f);
    const glm::vec3 dir = glm::normalize(glm::vec3(0.5f, 0.2f, -0.84f));
    const float tan_half = std::tan(glm::radians(30.0f));
    for (float t = VOXEL_BASE_SIZE; t < voxel_cascade_half_extent(VOXEL_CASCADE_COUNT - 1);
         t += voxel_cone_step(t, tan_half)) {
        const glm::vec3 world = camera + dir * t;
        const float diameter = voxel_cone_diameter(t, tan_half);
        const int c = voxel_select_cascade_for_footprint(world, camera, diameter);
        if (c < 0) {
            break; // left the clipmap; the cone terminates
        }
        const glm::ivec3 base = voxel_cascade_base(c, camera);
        // The sample AND its whole footprint must be inside the window.
        const float half = diameter * 0.5f;
        for (const glm::vec3 corner :
             {glm::vec3(-half, -half, -half), glm::vec3(half, half, half)}) {
            const glm::ivec3 coord = voxel_coord_at(c, world + corner);
            EXPECT_TRUE(voxel_in_window(coord, base))
                << "cone footprint escaped cascade " << c << " at t=" << t;
        }
    }
}

TEST(VoxelMath, ConeWidensAndSelectsCoarserMips) {
    const float tan_half = std::tan(glm::radians(30.0f));
    float previous_mip = -1.0f;
    for (float t = 0.0f; t < 4000.0f; t += 50.0f) {
        const float d = voxel_cone_diameter(t, tan_half);
        const float mip = voxel_cone_mip(d, 0);
        EXPECT_GE(mip, 0.0f) << "mip must never go below the base level, at t=" << t;
        EXPECT_GE(mip, previous_mip - 1e-6f) << "cone mip must be monotone, at t=" << t;
        previous_mip = mip;
    }
    // A cone one voxel wide sits exactly on mip 0 — the anchor that ties the
    // mip scale to the voxel scale.
    EXPECT_FLOAT_EQ(voxel_cone_mip(voxel_size(0), 0), 0.0f);
    EXPECT_FLOAT_EQ(voxel_cone_mip(voxel_size(0) * 4.0f, 0), 2.0f);
}

TEST(VoxelMath, ConeStepIsBoundedBelowSoTracingTerminates) {
    // A step proportional to the cone width is what keeps the sample count
    // logarithmic in distance, but at t=0 that is zero and the trace would spin
    // forever. The floor is the thing that makes the loop finite.
    const float tan_half = std::tan(glm::radians(30.0f));
    EXPECT_GT(voxel_cone_step(0.0f, tan_half), 0.0f);
    // And a degenerate (zero-angle) cone still advances.
    EXPECT_GT(voxel_cone_step(0.0f, 0.0f), 0.0f);
    EXPECT_GT(voxel_cone_step(1000.0f, 0.0f), 0.0f);

    // Bounded step count over the coarsest cascade, which is the real claim.
    float t = 0.0f;
    int steps = 0;
    const float limit = voxel_cascade_extent(VOXEL_CASCADE_COUNT - 1);
    while (t < limit && steps < 10000) {
        t += voxel_cone_step(t, tan_half);
        ++steps;
    }
    EXPECT_LT(steps, 200) << "a diffuse cone should cross the whole clipmap in tens of steps";
}

TEST(VoxelMath, AccumulateIsFrontToBackAndSaturates) {
    const glm::vec4 clear(0.0f);
    // An empty sample changes nothing.
    EXPECT_EQ(voxel_cone_accumulate(clear, glm::vec4(1.0f, 1.0f, 1.0f, 0.0f)), clear);

    // A fully opaque sample takes the whole result...
    const glm::vec4 opaque =
        voxel_cone_accumulate(clear, glm::vec4(0.25f, 0.5f, 0.75f, 1.0f));
    EXPECT_FLOAT_EQ(opaque.a, 1.0f);
    EXPECT_FLOAT_EQ(opaque.r, 0.25f);

    // ...and nothing behind it contributes, which is what stops a cone that hit
    // a wall from reporting the room beyond.
    const glm::vec4 behind =
        voxel_cone_accumulate(opaque, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    EXPECT_FLOAT_EQ(behind.r, opaque.r);
    EXPECT_FLOAT_EQ(behind.a, 1.0f);

    // Alpha is monotone and never exceeds 1 over a long march.
    glm::vec4 accum(0.0f);
    for (int i = 0; i < 500; ++i) {
        const glm::vec4 next = voxel_cone_accumulate(accum, glm::vec4(1.0f, 1.0f, 1.0f, 0.05f));
        EXPECT_GE(next.a, accum.a);
        EXPECT_LE(next.a, 1.0f + 1e-5f);
        accum = next;
    }
}

TEST(VoxelMath, HitAlphaIsBelowTerminationSoADistanceIsAlwaysRecorded) {
    // The probe field's Chebyshev moments need a distance from every cone. If
    // the hit threshold were at or above the termination threshold, a cone that
    // terminated early would have no distance to report and the visibility
    // atlas would fill with the miss value.
    EXPECT_LT(VOXEL_CONE_HIT_ALPHA, VOXEL_CONE_TERMINATE_ALPHA);
}

TEST(VoxelMath, ClipmapMemoryIsWithinBudget) {
    // EXACTLY what VoxelVolume allocates: an RGBA8 albedo+opacity per cascade
    // with no mip chain (only injection reads it) and an RGBA16F radiance per
    // cascade WITH one (that is what the cones sample). No normal volume — see
    // VOXEL_RESOLUTION.
    const std::uint64_t albedo = voxel_cascade_bytes(4, false) * VOXEL_CASCADE_COUNT;
    const std::uint64_t radiance = voxel_cascade_bytes(8, true) * VOXEL_CASCADE_COUNT;
    const double mb = static_cast<double>(albedo + radiance) / (1024.0 * 1024.0);
    // The budget this feature is held to: it exists to be the cheaper
    // alternative to ray tracing, and the probe field it feeds already costs
    // ~30 MB. A regression here is a design change, not a tuning change.
    EXPECT_LT(mb, 48.0) << "clipmap is " << mb << " MB";
    EXPECT_GT(mb, 1.0) << "sanity: the volumes are not empty";

    // The mip chain converges to 8/7 of the base level, not to 2x — the reason
    // mipmapping the radiance volume is affordable at all.
    EXPECT_LT(voxel_cascade_bytes(8, true), voxel_cascade_bytes(8, false) * 8 / 7 + 4096);
    EXPECT_GT(voxel_cascade_bytes(8, true), voxel_cascade_bytes(8, false));
}

// ---------------------------------------------------------------------------
// The scroll's invalidation accounting (VoxelVolume::scroll), which is pure
// arithmetic over the base corners and is what decides how much revoxelization
// a frame owes. Reproduced here rather than reached through the class, because
// the class owns GPU resources and this property does not need them.
// ---------------------------------------------------------------------------

namespace {

/// The shell a scroll uncovers: window volume minus the overlap with where the
/// window used to be. Statement for statement what `VoxelVolume::scroll` does.
[[nodiscard]] std::uint64_t invalidated_by(const glm::ivec3& from, const glm::ivec3& to) {
    const auto window =
        static_cast<std::uint64_t>(VOXEL_RESOLUTION) * VOXEL_RESOLUTION * VOXEL_RESOLUTION;
    const glm::ivec3 d = glm::abs(to - from);
    const glm::ivec3 overlap = glm::max(glm::ivec3(VOXEL_RESOLUTION) - d, glm::ivec3(0));
    const auto kept = static_cast<std::uint64_t>(overlap.x) *
                      static_cast<std::uint64_t>(overlap.y) *
                      static_cast<std::uint64_t>(overlap.z);
    return window - kept;
}

} // namespace

TEST(VoxelScroll, StationaryCameraInvalidatesNothing) {
    // The entire point of snapping the base to whole voxels. If this ever
    // reports work for a still camera, the volume is being rebuilt every frame
    // and the clipmap has become a very expensive way to store nothing.
    const glm::ivec3 base{12, -3, 400};
    EXPECT_EQ(invalidated_by(base, base), 0u);
}

TEST(VoxelScroll, OneVoxelStepInvalidatesExactlyOneSlab) {
    const glm::ivec3 base{0, 0, 0};
    const auto slab =
        static_cast<std::uint64_t>(VOXEL_RESOLUTION) * VOXEL_RESOLUTION; // one face
    EXPECT_EQ(invalidated_by(base, base + glm::ivec3(1, 0, 0)), slab);
    EXPECT_EQ(invalidated_by(base, base + glm::ivec3(0, 0, -1)), slab);
}

TEST(VoxelScroll, AJumpBeyondTheWindowInvalidatesAllOfIt) {
    // A teleport keeps nothing, and must not report a negative or wrapped
    // overlap — the clamp at zero is what guarantees that.
    const glm::ivec3 base{0, 0, 0};
    const auto window =
        static_cast<std::uint64_t>(VOXEL_RESOLUTION) * VOXEL_RESOLUTION * VOXEL_RESOLUTION;
    EXPECT_EQ(invalidated_by(base, base + glm::ivec3(VOXEL_RESOLUTION, 0, 0)), window);
    EXPECT_EQ(invalidated_by(base, base + glm::ivec3(9999, -9999, 9999)), window);
}

TEST(VoxelScroll, InvalidationGrowsMonotonicallyWithDistance) {
    const glm::ivec3 base{0, 0, 0};
    std::uint64_t previous = 0;
    for (int step = 0; step <= VOXEL_RESOLUTION; ++step) {
        const std::uint64_t n = invalidated_by(base, base + glm::ivec3(step, 0, 0));
        EXPECT_GE(n, previous) << "at step " << step;
        previous = n;
    }
}

TEST(VoxelScroll, CoarseCascadesScrollFarLessOftenThanFineOnes) {
    // The reason a clipmap is affordable: cascade c's voxels are 2^c times
    // wider, so the same camera motion moves its base 2^c times less. Walk a
    // realistic camera path and count.
    std::array<int, VOXEL_CASCADE_COUNT> moves{};
    std::array<glm::ivec3, VOXEL_CASCADE_COUNT> bases{};
    for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
        bases[static_cast<std::size_t>(c)] = voxel_cascade_base(c, glm::vec3(0.0f));
    }
    for (int frame = 1; frame <= 400; ++frame) {
        const glm::vec3 camera(static_cast<float>(frame) * 5.0f, 0.0f, 0.0f); // 5 units/frame
        for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
            const glm::ivec3 next = voxel_cascade_base(c, camera);
            if (next != bases[static_cast<std::size_t>(c)]) {
                ++moves[static_cast<std::size_t>(c)];
                bases[static_cast<std::size_t>(c)] = next;
            }
        }
    }
    for (int c = 1; c < VOXEL_CASCADE_COUNT; ++c) {
        EXPECT_LT(moves[static_cast<std::size_t>(c)], moves[static_cast<std::size_t>(c - 1)])
            << "cascade " << c << " scrolled as often as " << (c - 1);
    }
    EXPECT_GT(moves[0], 0) << "the finest cascade must actually track the camera";
}

} // namespace
} // namespace bracken::render::gi
