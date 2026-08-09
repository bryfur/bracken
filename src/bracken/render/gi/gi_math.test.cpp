// The probe field's addressing, tested with no GPU, no window and no SDL.
//
// This is the half of phase R1 that is easy to get subtly wrong and impossible
// to SEE wrong once radiance sits on top of it: a torus wrap that is off by one
// puts a probe's irradiance in the neighbouring probe's texels, and the result
// is a slightly wrong ambient term that looks like a shading bug for a week.
// So every mapping R1b's shaders will mirror is pinned here first, while there
// is nothing else in the frame that could be blamed.

#include "bracken/render/gi/gi_math.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <vector>

using namespace bracken::render::gi;

namespace {

glm::vec3 camera_at_probe(const glm::vec3& probe_steps, int cascade) {
    return probe_steps * cascade_spacing(cascade);
}

} // namespace

// --- Dimensions and the atlas tiling ---------------------------------------

TEST(GiMath, GridAndAtlasDimensions) {
    EXPECT_EQ(CASCADE_COUNT, 3);
    EXPECT_EQ(PROBES_PER_CASCADE, 24 * 12 * 24);
    EXPECT_EQ(PROBES_PER_CASCADE, 6912);
    EXPECT_EQ(PROBES_TOTAL, 20736);

    // The border ring is why the payload and the tile differ; both numbers are
    // load-bearing for the shader's texel math.
    EXPECT_EQ(IRRADIANCE_PROBE_TEXELS, 8);
    EXPECT_EQ(VISIBILITY_PROBE_TEXELS, 16);

    EXPECT_EQ(PROBES_PER_ATLAS_ROW * ATLAS_ROWS, PROBES_PER_CASCADE);
    EXPECT_EQ(atlas_width(IRRADIANCE_PROBE_TEXELS), 2304);
    EXPECT_EQ(atlas_band_height(IRRADIANCE_PROBE_TEXELS), 192);
    EXPECT_EQ(atlas_width(VISIBILITY_PROBE_TEXELS), 4608);
    EXPECT_EQ(atlas_band_height(VISIBILITY_PROBE_TEXELS), 384);

    // R1c: one texture per quantity, every cascade banded into it.
    EXPECT_EQ(ATLAS_ROWS_ALL_CASCADES, ATLAS_ROWS * CASCADE_COUNT);
    EXPECT_EQ(atlas_height(IRRADIANCE_PROBE_TEXELS), 576);
    EXPECT_EQ(atlas_height(VISIBILITY_PROBE_TEXELS), 1152);
}

TEST(GiMath, CascadeBandsTileTheAtlasWithoutOverlap) {
    // The banding is what buys the forward shaders three sampler slots instead
    // of nine, so "cascade c's tiles are disjoint from cascade c''s" is a
    // correctness property of the whole consumption path, not a detail.
    std::set<std::pair<int, int>> tiles;
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        for (int slot = 0; slot < PROBES_PER_CASCADE; ++slot) {
            const glm::ivec2 origin = probe_atlas_origin(slot, IRRADIANCE_PROBE_TEXELS, c);
            EXPECT_GE(origin.y, c * atlas_band_height(IRRADIANCE_PROBE_TEXELS));
            EXPECT_LT(origin.y, (c + 1) * atlas_band_height(IRRADIANCE_PROBE_TEXELS));
            EXPECT_LT(origin.y, atlas_height(IRRADIANCE_PROBE_TEXELS));
            EXPECT_TRUE(tiles.emplace(origin.x, origin.y).second)
                << "cascade " << c << " slot " << slot << " collides with another band";
        }
    }
    EXPECT_EQ(tiles.size(), static_cast<std::size_t>(PROBES_PER_CASCADE * CASCADE_COUNT));

    // The one-texel planes use the same banding at a tile size of 1, which is
    // what lets the CPU upload all three states as one contiguous plane.
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        for (int slot : {0, 1, PROBES_PER_ATLAS_ROW, PROBES_PER_CASCADE - 1}) {
            const glm::ivec2 t = probe_state_texel(slot, c);
            EXPECT_EQ(t.x, slot % PROBES_PER_ATLAS_ROW);
            EXPECT_EQ(t.y, slot / PROBES_PER_ATLAS_ROW + c * ATLAS_ROWS);
            // Row-major linear index into the CPU mirror ProbeVolume uploads.
            EXPECT_EQ(t.y * PROBES_PER_ATLAS_ROW + t.x, c * PROBES_PER_CASCADE + slot);
        }
    }
}

TEST(GiMath, CascadeSpacingDoubles) {
    EXPECT_FLOAT_EQ(cascade_spacing(0), 150.0f);
    EXPECT_FLOAT_EQ(cascade_spacing(1), 300.0f);
    EXPECT_FLOAT_EQ(cascade_spacing(2), 600.0f);
    // Every coarser lattice is a subset of the finer one — a cascade boundary
    // therefore never falls between two lattices.
    for (int c = 1; c < CASCADE_COUNT; ++c) {
        const glm::ivec3 coarse{3, -2, 5};
        const glm::vec3 world = probe_world_position(coarse, c);
        const glm::vec3 fine = world_to_probe_space(world, c - 1);
        EXPECT_FLOAT_EQ(fine.x, std::round(fine.x));
        EXPECT_FLOAT_EQ(fine.y, std::round(fine.y));
        EXPECT_FLOAT_EQ(fine.z, std::round(fine.z));
    }
}

TEST(GiMath, ProbeIndexIsABijectionOntoTheAtlas) {
    std::set<std::pair<int, int>> tiles;
    for (int i = 0; i < PROBES_PER_CASCADE; ++i) {
        const glm::ivec3 c = probe_coord(i);
        EXPECT_EQ(probe_index(c), i);
        EXPECT_GE(c.x, 0);
        EXPECT_LT(c.x, GRID_X);
        EXPECT_GE(c.y, 0);
        EXPECT_LT(c.y, GRID_Y);
        EXPECT_GE(c.z, 0);
        EXPECT_LT(c.z, GRID_Z);

        const glm::ivec2 origin = probe_atlas_origin(i, IRRADIANCE_PROBE_TEXELS, 0);
        EXPECT_GE(origin.x, 0);
        EXPECT_LT(origin.x, atlas_width(IRRADIANCE_PROBE_TEXELS));
        EXPECT_GE(origin.y, 0);
        EXPECT_LT(origin.y, atlas_band_height(IRRADIANCE_PROBE_TEXELS));
        EXPECT_TRUE(tiles.emplace(origin.x, origin.y).second) << "two probes share atlas tile at " << i;
    }
    EXPECT_EQ(tiles.size(), static_cast<std::size_t>(PROBES_PER_CASCADE));
}

TEST(GiMath, PayloadTexelsStayInsideTheirTile) {
    // The border ring exists so a bilinear tap at the payload edge does not
    // reach the next probe. That only holds if the payload really is inset.
    for (int slot : {0, 1, PROBES_PER_ATLAS_ROW, PROBES_PER_CASCADE - 1}) {
        const glm::ivec2 origin = probe_atlas_origin(slot, VISIBILITY_PROBE_TEXELS, 1);
        for (int y = 0; y < VISIBILITY_PAYLOAD_TEXELS; ++y) {
            for (int x = 0; x < VISIBILITY_PAYLOAD_TEXELS; ++x) {
                const glm::ivec2 t = probe_payload_texel(slot, VISIBILITY_PROBE_TEXELS, 1, {x, y});
                EXPECT_GT(t.x, origin.x);
                EXPECT_GT(t.y, origin.y);
                EXPECT_LT(t.x, origin.x + VISIBILITY_PROBE_TEXELS - 1);
                EXPECT_LT(t.y, origin.y + VISIBILITY_PROBE_TEXELS - 1);
            }
        }
    }
}

// --- Torus addressing -------------------------------------------------------

TEST(GiMath, WrapIsEuclideanAndPeriodic) {
    // C++ `%` truncates toward zero, so -1 % 24 is -1. Every scroll below zero
    // depends on this being 23 instead.
    EXPECT_EQ(wrap_axis(-1, GRID_X), GRID_X - 1);
    EXPECT_EQ(wrap_axis(-GRID_X, GRID_X), 0);
    EXPECT_EQ(wrap_axis(-GRID_X - 1, GRID_X), GRID_X - 1);
    EXPECT_EQ(wrap_axis(GRID_X, GRID_X), 0);

    for (int k = -3; k <= 3; ++k) {
        const glm::ivec3 a{7, 3, 19};
        const glm::ivec3 shifted = a + glm::ivec3(k * GRID_X, k * GRID_Y, k * GRID_Z);
        EXPECT_EQ(wrapped_probe_index(shifted), wrapped_probe_index(a));
    }
}

// Scroll a long way out and back: every probe must land on the slot it had at
// the start. This is the property the whole "scrolling is free" claim rests on.
TEST(GiMath, TorusRoundTripAfterScrollingFarAndBack) {
    ProbeCascade cascade;
    cascade.reset(0);
    const glm::vec3 origin{1234.0f, 210.0f, -880.0f};
    cascade.update(origin);
    const glm::ivec3 base_before = cascade.base();

    std::vector<glm::vec3> slot_world_before(static_cast<std::size_t>(PROBES_PER_CASCADE));
    for (int s = 0; s < PROBES_PER_CASCADE; ++s) {
        slot_world_before[static_cast<std::size_t>(s)] = cascade.slot_world_position(s);
    }

    // Out past several grid widths in every axis, in many steps, then back.
    for (int step = 1; step <= 40; ++step) {
        cascade.update(origin + glm::vec3(static_cast<float>(step) * 260.0f,
                                          static_cast<float>(step) * 170.0f,
                                          static_cast<float>(step) * -310.0f));
    }
    for (int step = 39; step >= 0; --step) {
        cascade.update(origin + glm::vec3(static_cast<float>(step) * 260.0f,
                                          static_cast<float>(step) * 170.0f,
                                          static_cast<float>(step) * -310.0f));
    }

    EXPECT_EQ(cascade.base(), base_before);
    for (int s = 0; s < PROBES_PER_CASCADE; ++s) {
        EXPECT_EQ(cascade.slot_world_position(s), slot_world_before[static_cast<std::size_t>(s)])
            << "slot " << s << " does not hold the same probe after a round trip";
    }
}

// Every slot holds exactly one probe of the current window, and every probe of
// the window is in exactly one slot — at any base.
TEST(GiMath, SlotsPartitionTheWindowAtEveryBase) {
    ProbeCascade cascade;
    cascade.reset(1);
    for (const glm::vec3& camera : {glm::vec3(0.0f), glm::vec3(4321.0f, -900.0f, 77.0f),
                                    glm::vec3(-100000.0f, 5000.0f, 100000.0f)}) {
        cascade.update(camera);
        std::set<int> slots;
        for (int z = 0; z < GRID_Z; ++z) {
            for (int y = 0; y < GRID_Y; ++y) {
                for (int x = 0; x < GRID_X; ++x) {
                    slots.insert(wrapped_probe_index(cascade.base() + glm::ivec3(x, y, z)));
                }
            }
        }
        EXPECT_EQ(slots.size(), static_cast<std::size_t>(PROBES_PER_CASCADE));
    }
}

// --- Scrolling and invalidation ---------------------------------------------

TEST(GiMath, FirstUpdatePlacesTheCascadeAndInvalidatesEverything) {
    ProbeCascade cascade;
    cascade.reset(0);
    EXPECT_FALSE(cascade.placed());
    EXPECT_EQ(cascade.update(glm::vec3(0.0f)), static_cast<std::uint32_t>(PROBES_PER_CASCADE));
    EXPECT_TRUE(cascade.placed());
    EXPECT_TRUE(cascade.state_dirty());
    // Standing still costs nothing at all.
    EXPECT_EQ(cascade.update(glm::vec3(0.0f)), 0u);
}

TEST(GiMath, SubProbeMotionDoesNotScroll) {
    ProbeCascade cascade;
    cascade.reset(0);
    cascade.update(glm::vec3(0.0f));
    const glm::ivec3 base = cascade.base();
    // Anything short of half a spacing keeps the same snap.
    EXPECT_EQ(cascade.update(glm::vec3(40.0f, -20.0f, 60.0f)), 0u);
    EXPECT_EQ(cascade.base(), base);
}

// One spacing along +x enters exactly one plane: GRID_Y * GRID_Z probes, all of
// them on the leading x face of the new window, and nothing else.
TEST(GiMath, OneStepAlongXInvalidatesExactlyOnePlane) {
    ProbeCascade cascade;
    cascade.reset(0);
    cascade.update(glm::vec3(0.0f));
    const glm::ivec3 base_before = cascade.base();

    std::vector<glm::ivec3> entered;
    const glm::ivec3 base_after = base_before + glm::ivec3(1, 0, 0);
    const std::uint32_t n =
        for_each_entering_probe(base_before, base_after, [&](const glm::ivec3& a) { entered.push_back(a); });

    EXPECT_EQ(n, static_cast<std::uint32_t>(GRID_Y * GRID_Z));
    EXPECT_EQ(entered.size(), static_cast<std::size_t>(GRID_Y * GRID_Z));
    std::set<int> distinct_x;
    for (const glm::ivec3& a : entered) {
        distinct_x.insert(a.x);
        EXPECT_FALSE(window_contains(base_before, a));
        EXPECT_TRUE(window_contains(base_after, a));
    }
    // A single plane: one x value, the far face of the new window.
    ASSERT_EQ(distinct_x.size(), 1u);
    EXPECT_EQ(*distinct_x.begin(), base_before.x + GRID_X);

    // ...and the cascade itself agrees when driven by a camera move.
    EXPECT_EQ(cascade.update(camera_at_probe(glm::vec3(1.0f, 0.0f, 0.0f), 0)),
              static_cast<std::uint32_t>(GRID_Y * GRID_Z));
    EXPECT_EQ(cascade.base(), base_after);
}

TEST(GiMath, NegativeStepEntersTheOppositeFace) {
    const glm::ivec3 base{5, -3, 11};
    std::set<int> distinct_z;
    const std::uint32_t n = for_each_entering_probe(base, base - glm::ivec3(0, 0, 1),
                                                    [&](const glm::ivec3& a) { distinct_z.insert(a.z); });
    EXPECT_EQ(n, static_cast<std::uint32_t>(GRID_X * GRID_Y));
    ASSERT_EQ(distinct_z.size(), 1u);
    EXPECT_EQ(*distinct_z.begin(), base.z - 1);
}

// A diagonal step enters an L-shaped slab whose arms share a corner. Counting
// per axis and summing would double-count that corner; the set-difference
// formulation cannot.
TEST(GiMath, DiagonalStepDoesNotDoubleCountTheSharedCorner) {
    const glm::ivec3 base{0, 0, 0};
    const glm::ivec3 moved = base + glm::ivec3(1, 0, 1);
    std::set<std::tuple<int, int, int>> entered;
    const std::uint32_t n = for_each_entering_probe(
        base, moved, [&](const glm::ivec3& a) { entered.emplace(a.x, a.y, a.z); });

    // |plane_x| + |plane_z| - |shared corner column|
    const std::uint32_t expected =
        static_cast<std::uint32_t>(GRID_Y * GRID_Z) + static_cast<std::uint32_t>(GRID_X * GRID_Y) -
        static_cast<std::uint32_t>(GRID_Y);
    EXPECT_EQ(n, expected);
    EXPECT_EQ(entered.size(), static_cast<std::size_t>(n)) << "for_each_entering_probe visited a probe twice";
}

TEST(GiMath, ThreeAxisStepMatchesInclusionExclusion) {
    const glm::ivec3 base{-7, 2, 40};
    const glm::ivec3 moved = base + glm::ivec3(1, 1, 1);
    std::uint32_t n = for_each_entering_probe(base, moved, [](const glm::ivec3&) {});
    // |A u B u C| for three unit-thick slabs of the new window.
    const int a = GRID_Y * GRID_Z; // x plane
    const int b = GRID_X * GRID_Z; // y plane
    const int c = GRID_X * GRID_Y; // z plane
    const int ab = GRID_Z, ac = GRID_Y, bc = GRID_X, abc = 1;
    EXPECT_EQ(n, static_cast<std::uint32_t>(a + b + c - ab - ac - bc + abc));
}

TEST(GiMath, JumpLargerThanTheGridInvalidatesEverything) {
    ProbeCascade cascade;
    cascade.reset(0);
    cascade.update(glm::vec3(0.0f));
    // A teleport well past one grid width on a single axis: no probe of the new
    // window was in the old one.
    const float far_x = cascade_spacing(0) * static_cast<float>(GRID_X + 5);
    EXPECT_EQ(cascade.update(glm::vec3(far_x, 0.0f, 0.0f)),
              static_cast<std::uint32_t>(PROBES_PER_CASCADE));
    // Exactly one grid width is the boundary case: the windows touch but do not
    // overlap, so it is still a full invalidation.
    const glm::vec3 one_grid_over = camera_at_probe(glm::vec3(static_cast<float>(GRID_X), 0.0f, 0.0f), 0) +
                                    glm::vec3(far_x, 0.0f, 0.0f);
    EXPECT_EQ(cascade.update(one_grid_over), static_cast<std::uint32_t>(PROBES_PER_CASCADE));
}

TEST(GiMath, EveryInvalidatedProbeIsMarkedUninitialized) {
    ProbeCascade cascade;
    cascade.reset(0);
    cascade.update(glm::vec3(0.0f));
    // Pretend R1b filled the field.
    std::vector<int> filled_slots;
    for (int z = 0; z < GRID_Z; ++z) {
        for (int y = 0; y < GRID_Y; ++y) {
            for (int x = 0; x < GRID_X; ++x) {
                filled_slots.push_back(wrapped_probe_index(cascade.base() + glm::ivec3(x, y, z)));
            }
        }
    }
    // (The state buffer is const from outside; drive the check off the coords.)
    const glm::ivec3 base_before = cascade.base();
    const std::uint32_t n = cascade.update(camera_at_probe(glm::vec3(2.0f, 0.0f, 0.0f), 0));
    EXPECT_EQ(n, static_cast<std::uint32_t>(2 * GRID_Y * GRID_Z));

    std::set<int> should_be_uninitialized;
    for_each_entering_probe(base_before, cascade.base(), [&](const glm::ivec3& a) {
        should_be_uninitialized.insert(wrapped_probe_index(a));
    });
    EXPECT_EQ(should_be_uninitialized.size(), static_cast<std::size_t>(n));
    for (int slot : should_be_uninitialized) {
        EXPECT_EQ(cascade.state()[static_cast<std::size_t>(slot)], PROBE_UNINITIALIZED);
    }
}

// The smoke gate depends on determinism, and so will R1b's ray scheduling: the
// same camera path must produce the same invalidation sequence, every run.
TEST(GiMath, InvalidationSequenceIsDeterministic) {
    auto run = [] {
        ProbeCascade cascade;
        cascade.reset(0);
        std::vector<std::uint32_t> counts;
        std::vector<int> first_slots;
        for (int f = 0; f < 120; ++f) {
            const float t = static_cast<float>(f) * 0.1f;
            const glm::vec3 camera{900.0f * std::cos(t), 60.0f * std::sin(0.3f * t), 900.0f * std::sin(t)};
            const glm::ivec3 before = cascade.base();
            const bool was_placed = cascade.placed();
            counts.push_back(cascade.update(camera));
            int first = -1;
            if (was_placed) {
                for_each_entering_probe(before, cascade.base(), [&](const glm::ivec3& a) {
                    if (first < 0) {
                        first = wrapped_probe_index(a);
                    }
                });
            }
            first_slots.push_back(first);
        }
        return std::pair{counts, first_slots};
    };
    const auto a = run();
    const auto b = run();
    EXPECT_EQ(a.first, b.first);
    EXPECT_EQ(a.second, b.second);
    // ...and the path actually scrolled, or the check above is vacuous.
    std::uint32_t total = 0;
    for (std::uint32_t c : a.first) {
        total += c;
    }
    EXPECT_GT(total, static_cast<std::uint32_t>(PROBES_PER_CASCADE));
}

TEST(GiMath, CoarserCascadesScrollLessOften) {
    std::array<ProbeCascade, CASCADE_COUNT> cascades;
    std::array<std::uint32_t, CASCADE_COUNT> totals{};
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        cascades[static_cast<std::size_t>(c)].reset(c);
        cascades[static_cast<std::size_t>(c)].update(glm::vec3(0.0f));
    }
    for (int f = 1; f <= 200; ++f) {
        const glm::vec3 camera{static_cast<float>(f) * 30.0f, 0.0f, 0.0f};
        for (int c = 0; c < CASCADE_COUNT; ++c) {
            totals[static_cast<std::size_t>(c)] += cascades[static_cast<std::size_t>(c)].update(camera);
        }
    }
    // Twice the spacing, half the scroll events, half the invalidations.
    EXPECT_GT(totals[0], totals[1]);
    EXPECT_GT(totals[1], totals[2]);
}

// --- World <-> probe mapping ------------------------------------------------

TEST(GiMath, WorldToProbeRoundTripsAcrossCascades) {
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        for (const glm::ivec3& a : {glm::ivec3(0), glm::ivec3(1, -1, 2), glm::ivec3(-40, 9, 133)}) {
            const glm::vec3 world = probe_world_position(a, c);
            const glm::vec3 back = world_to_probe_space(world, c);
            EXPECT_FLOAT_EQ(back.x, static_cast<float>(a.x));
            EXPECT_FLOAT_EQ(back.y, static_cast<float>(a.y));
            EXPECT_FLOAT_EQ(back.z, static_cast<float>(a.z));
        }
    }
}

// The camera must sit inside its own cascade — near the centre, and never
// outside — at every scale and on both sides of the origin.
TEST(GiMath, CameraStaysInsideEveryCascade) {
    const glm::vec3 cameras[] = {{0.0f, 0.0f, 0.0f},        {75.0f, 74.9f, -75.0f},
                                 {-5000.0f, 300.0f, 2500.0f}, {123456.0f, -700.0f, -98765.0f}};
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        for (const glm::vec3& camera : cameras) {
            const glm::ivec3 base = cascade_base(camera, c);
            const glm::vec3 lo = probe_world_position(base, c);
            const glm::vec3 hi = probe_world_position(base + glm::ivec3(GRID_X - 1, GRID_Y - 1, GRID_Z - 1), c);
            EXPECT_GE(camera.x, lo.x);
            EXPECT_LE(camera.x, hi.x);
            EXPECT_GE(camera.y, lo.y);
            EXPECT_LE(camera.y, hi.y);
            EXPECT_GE(camera.z, lo.z);
            EXPECT_LE(camera.z, hi.z);
        }
    }
}

TEST(GiMath, SlotWorldPositionAgreesWithTheWindow) {
    ProbeCascade cascade;
    cascade.reset(2);
    cascade.update(glm::vec3(9000.0f, -400.0f, -3000.0f));
    for (int z = 0; z < GRID_Z; ++z) {
        for (int y = 0; y < GRID_Y; ++y) {
            for (int x = 0; x < GRID_X; ++x) {
                const glm::ivec3 a = cascade.base() + glm::ivec3(x, y, z);
                const int slot = wrapped_probe_index(a);
                EXPECT_EQ(cascade.slot_world_position(slot), probe_world_position(a, 2));
            }
        }
    }
}

// --- Octahedral mapping -----------------------------------------------------

// Must match `octEncode` (shaders/prepass_gbuffer.frag.hlsl) / `octDecode`
// (shaders/rt_trace.hlsli). A CPU round trip cannot prove the HLSL agrees, but
// it does pin the convention this side commits to, which is what R1b will
// transliterate.
TEST(GiMath, OctahedralRoundTripOverADirectionSweep) {
    double worst_rad = 0.0;
    int samples = 0;
    for (int i = 0; i < 96; ++i) {
        for (int j = 0; j < 96; ++j) {
            // Uniform-ish sphere sweep (equal-area in cos(theta)).
            const float u = (static_cast<float>(i) + 0.5f) / 96.0f;
            const float v = (static_cast<float>(j) + 0.5f) / 96.0f;
            const float z = 2.0f * u - 1.0f;
            const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
            const float phi = 6.2831853f * v;
            const glm::vec3 dir{r * std::cos(phi), r * std::sin(phi), z};

            const glm::vec2 e = oct_encode(dir);
            EXPECT_GE(e.x, 0.0f);
            EXPECT_LE(e.x, 1.0f);
            EXPECT_GE(e.y, 0.0f);
            EXPECT_LE(e.y, 1.0f);

            const glm::vec3 back = oct_decode(e);
            const float cos_a = std::clamp(glm::dot(dir, back), -1.0f, 1.0f);
            worst_rad = std::max(worst_rad, static_cast<double>(std::acos(cos_a)));
            ++samples;
        }
    }
    EXPECT_EQ(samples, 96 * 96);
    // Float round trip through a [0,1]^2 parameterization; anything above a
    // milliradian means the two halves disagree, not that floats are floats.
    EXPECT_LT(worst_rad, 1e-3) << "worst angular error " << worst_rad << " rad";
}

TEST(GiMath, OctahedralAxisDirectionsAreExact) {
    // +Z is the centre of the [0,1]^2 square, -Z is its corners — the defining
    // property of the octahedral layout, and the thing a sign flip breaks.
    EXPECT_NEAR(oct_encode({0.0f, 0.0f, 1.0f}).x, 0.5f, 1e-6f);
    EXPECT_NEAR(oct_encode({0.0f, 0.0f, 1.0f}).y, 0.5f, 1e-6f);
    const glm::vec3 centre = oct_decode({0.5f, 0.5f});
    EXPECT_NEAR(centre.z, 1.0f, 1e-6f);
    for (const glm::vec2& corner : {glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 0.0f), glm::vec2(0.0f, 1.0f),
                                    glm::vec2(1.0f, 1.0f)}) {
        EXPECT_NEAR(oct_decode(corner).z, -1.0f, 1e-6f);
    }
}

TEST(GiMath, PayloadTexelDirectionsCoverTheSphere) {
    // Every payload texel must be a unit direction, and the set must not
    // collapse onto a hemisphere (which is what a missing fold would do).
    int upper = 0;
    int lower = 0;
    for (int y = 0; y < IRRADIANCE_PAYLOAD_TEXELS; ++y) {
        for (int x = 0; x < IRRADIANCE_PAYLOAD_TEXELS; ++x) {
            const glm::vec3 d = probe_texel_direction({x, y}, IRRADIANCE_PAYLOAD_TEXELS);
            EXPECT_NEAR(glm::length(d), 1.0f, 1e-5f);
            (d.z >= 0.0f ? upper : lower)++;
        }
    }
    EXPECT_EQ(upper + lower, IRRADIANCE_PAYLOAD_TEXELS * IRRADIANCE_PAYLOAD_TEXELS);
    EXPECT_GT(upper, 0);
    EXPECT_GT(lower, 0);
}

// ===========================================================================
// R1b — what the update passes mirror
// ===========================================================================

namespace {

/// Deterministic unit directions, so a coverage claim is reproducible rather
/// than "it passed on my machine".
std::vector<glm::vec3> sample_sphere(int count) {
    std::vector<glm::vec3> out;
    out.reserve(static_cast<std::size_t>(count));
    // A second, unrelated low-discrepancy set: using spherical Fibonacci to
    // test spherical Fibonacci would only prove it agrees with itself.
    for (int i = 0; i < count; ++i) {
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(count);
        const float v = std::fmod(static_cast<float>(i) * 0.7548776662466927f, 1.0f);
        const float z = 1.0f - 2.0f * u;
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = 6.283185307179586f * v;
        out.push_back({std::cos(phi) * r, std::sin(phi) * r, z});
    }
    return out;
}

} // namespace

TEST(GiMath, SphericalFibonacciIsDeterministicAndUnit) {
    for (const int count : {32, 64, 96}) {
        for (int i = 0; i < count; ++i) {
            const glm::vec3 a = spherical_fibonacci(i, count);
            const glm::vec3 b = spherical_fibonacci(i, count);
            EXPECT_EQ(a, b) << "same inputs must give bit-identical directions";
            EXPECT_NEAR(glm::length(a), 1.0f, 1e-5f);
        }
    }
}

TEST(GiMath, SphericalFibonacciCoversTheSphereEvenly) {
    // Three claims, one per failure mode a hand-written direction set has:
    //  * COVERAGE — no direction of the sphere is far from every ray, or a
    //    probe has a blind spot that never gets sampled.
    //  * SEPARATION — no two rays are nearly the same, or the budget is spent
    //    twice on one direction.
    //  * BALANCE — the set's centroid is at the origin, so the estimate has no
    //    built-in directional bias.
    // The thresholds are the measured values with headroom, not aspirations.
    const std::vector<glm::vec3> queries = sample_sphere(4000);
    struct Case {
        int count;
        float min_best_dot;
    };
    for (const Case& c : {Case{32, 0.85f}, Case{64, 0.90f}, Case{96, 0.93f}}) {
        std::vector<glm::vec3> dirs;
        dirs.reserve(static_cast<std::size_t>(c.count));
        for (int i = 0; i < c.count; ++i) {
            dirs.push_back(spherical_fibonacci(i, c.count));
        }

        float worst = 1.0f;
        for (const glm::vec3& q : queries) {
            float best = -1.0f;
            for (const glm::vec3& d : dirs) {
                best = std::max(best, glm::dot(d, q));
            }
            worst = std::min(worst, best);
        }
        EXPECT_GT(worst, c.min_best_dot) << "coverage gap at count " << c.count;

        float max_pair = -1.0f;
        glm::vec3 centroid{0.0f};
        for (std::size_t i = 0; i < dirs.size(); ++i) {
            centroid += dirs[i];
            for (std::size_t j = i + 1; j < dirs.size(); ++j) {
                max_pair = std::max(max_pair, glm::dot(dirs[i], dirs[j]));
            }
        }
        EXPECT_LT(max_pair, 0.97f) << "two rays nearly coincide at count " << c.count;
        EXPECT_LT(glm::length(centroid) / static_cast<float>(c.count), 0.02f)
            << "direction set is biased at count " << c.count;
    }
}

TEST(GiMath, RayRotationIsDeterministicAndOrthonormal) {
    for (const std::uint32_t frame : {0u, 1u, 7u, 1000u, 65535u}) {
        const glm::mat3 a = probe_ray_rotation(frame);
        const glm::mat3 b = probe_ray_rotation(frame);
        EXPECT_EQ(a, b) << "the rotation must be a pure function of the frame counter";

        // A rotation, not a general matrix: lengths preserved, handedness kept.
        // A non-orthonormal "rotation" would stretch the ray set and quietly
        // break the cosine weighting the blend passes apply to it.
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(glm::length(a[c]), 1.0f, 1e-4f);
        }
        EXPECT_NEAR(glm::dot(a[0], a[1]), 0.0f, 1e-4f);
        EXPECT_NEAR(glm::dot(a[0], a[2]), 0.0f, 1e-4f);
        EXPECT_NEAR(glm::dot(a[1], a[2]), 0.0f, 1e-4f);
        EXPECT_NEAR(glm::determinant(a), 1.0f, 1e-4f);
    }
}

TEST(GiMath, RayRotationActuallyMovesTheSetBetweenFrames) {
    // The whole point of rotating is that consecutive updates of one probe
    // sample DIFFERENT directions; a rotation that barely moved would leave the
    // estimate converging to a fixed 64-direction quadrature of the scene.
    float worst = 0.0f;
    for (std::uint32_t frame = 0; frame < 32; ++frame) {
        const glm::vec3 a = probe_ray_direction(0, 64, frame);
        const glm::vec3 b = probe_ray_direction(0, 64, frame + 1);
        worst = std::max(worst, glm::dot(a, b));
        EXPECT_NEAR(glm::length(a), 1.0f, 1e-4f);
    }
    EXPECT_LT(worst, 0.999f);
}

TEST(GiMath, OctahedralCosineWeightsIntegrateToPi) {
    // The blend divides by the weight total, so what has to be true is that the
    // texel set is a fair quadrature of the sphere: sum of max(0, cos) over the
    // payload, times the per-texel solid angle 4pi/N, must come out at
    // integral(cos) over the hemisphere = pi. The octahedral map is only
    // APPROXIMATELY equal-area, which is where the tolerance comes from (the
    // measured spread over 300 normals is 2.86 .. 3.25).
    const float solid_angle = 4.0f * 3.14159265358979f /
                              static_cast<float>(IRRADIANCE_PAYLOAD_TEXELS * IRRADIANCE_PAYLOAD_TEXELS);
    for (const glm::vec3& n : sample_sphere(300)) {
        float sum = 0.0f;
        for (int y = 0; y < IRRADIANCE_PAYLOAD_TEXELS; ++y) {
            for (int x = 0; x < IRRADIANCE_PAYLOAD_TEXELS; ++x) {
                sum += std::max(0.0f, glm::dot(probe_texel_direction({x, y}, IRRADIANCE_PAYLOAD_TEXELS), n));
            }
        }
        EXPECT_NEAR(sum * solid_angle, 3.14159265358979f, 0.35f);
    }
}

TEST(GiMath, BorderReplicatesTheOctahedralWrap) {
    constexpr int P = IRRADIANCE_PAYLOAD_TEXELS;
    constexpr int T = P + 2 * PROBE_BORDER_TEXELS;

    // Corners take the OPPOSITE corner, which is NOT clamp-to-edge. All four
    // square corners are the same POINT on the sphere (the -Z pole), but the
    // payload texels around them sit in four different azimuth quadrants, so the
    // diagonal continuation past tile corner (0,0) comes back out at (P-1, P-1).
    // Returning (0,0) here is the clamp-to-edge bug this test exists to catch.
    EXPECT_EQ(probe_border_source({0, 0}, P), glm::ivec2(P - 1, P - 1));
    EXPECT_EQ(probe_border_source({T - 1, 0}, P), glm::ivec2(0, P - 1));
    EXPECT_EQ(probe_border_source({0, T - 1}, P), glm::ivec2(P - 1, 0));
    EXPECT_EQ(probe_border_source({T - 1, T - 1}, P), glm::ivec2(0, 0));

    // Edges MIRROR along the edge — not clamp-to-edge, which is the bug this
    // test exists to catch: walking off the top of the octahedron's square
    // re-enters the top from the opposite side.
    for (int x = 1; x < T - 1; ++x) {
        const glm::ivec2 top = probe_border_source({x, 0}, P);
        EXPECT_EQ(top.y, 0);
        EXPECT_EQ(top.x, P - 1 - (x - 1));
        const glm::ivec2 bottom = probe_border_source({x, T - 1}, P);
        EXPECT_EQ(bottom.y, P - 1);
        EXPECT_EQ(bottom.x, P - 1 - (x - 1));
    }
    for (int y = 1; y < T - 1; ++y) {
        const glm::ivec2 left = probe_border_source({0, y}, P);
        EXPECT_EQ(left.x, 0);
        EXPECT_EQ(left.y, P - 1 - (y - 1));
        const glm::ivec2 right = probe_border_source({T - 1, y}, P);
        EXPECT_EQ(right.x, P - 1);
        EXPECT_EQ(right.y, P - 1 - (y - 1));
    }

    // Every border source must be a real payload texel — an out-of-range value
    // would sample garbage on the GPU with no diagnostic at all.
    for (int y = 0; y < T; ++y) {
        for (int x = 0; x < T; ++x) {
            const glm::ivec2 s = probe_border_source({x, y}, P);
            EXPECT_GE(s.x, 0);
            EXPECT_GE(s.y, 0);
            EXPECT_LT(s.x, P);
            EXPECT_LT(s.y, P);
        }
    }
}

TEST(GiMath, BorderMatchesTheOctahedronsOwnGluing) {
    // An INDEPENDENT reference for the whole ring, derived from the sphere
    // rather than from the branch structure of probe_border_source: the
    // octahedral square's edges are glued to themselves, so leaving through
    // x = 1 at (1+s, y) is the same direction as re-entering at (1-s, -y). Fold
    // the border texel's e-space centre back into the domain, decode it, and the
    // payload texel it must replicate is simply the one pointing closest to it.
    //
    // The fold is NOT the same thing as evaluating oct_decode outside [-1,1]^2 —
    // that gives a different vector and would certify the clamp-to-edge bug.
    const auto fold = [](glm::vec2 e) {
        if (e.x > 1.0f) {
            e = {2.0f - e.x, -e.y};
        } else if (e.x < -1.0f) {
            e = {-2.0f - e.x, -e.y};
        }
        if (e.y > 1.0f) {
            e = {-e.x, 2.0f - e.y};
        } else if (e.y < -1.0f) {
            e = {-e.x, -2.0f - e.y};
        }
        return e;
    };

    for (const int P : {IRRADIANCE_PAYLOAD_TEXELS, VISIBILITY_PAYLOAD_TEXELS}) {
        const int T = P + 2 * PROBE_BORDER_TEXELS;
        for (int y = 0; y < T; ++y) {
            for (int x = 0; x < T; ++x) {
                const bool border = x < PROBE_BORDER_TEXELS || x >= T - PROBE_BORDER_TEXELS ||
                                    y < PROBE_BORDER_TEXELS || y >= T - PROBE_BORDER_TEXELS;
                if (!border) {
                    continue;
                }
                const glm::vec2 e{
                    2.0f * (static_cast<float>(x - PROBE_BORDER_TEXELS) + 0.5f) / static_cast<float>(P) -
                        1.0f,
                    2.0f * (static_cast<float>(y - PROBE_BORDER_TEXELS) + 0.5f) / static_cast<float>(P) -
                        1.0f};
                const glm::vec3 want = oct_decode(fold(e) * 0.5f + 0.5f);

                glm::ivec2 best{0, 0};
                float best_dot = -2.0f;
                for (int py = 0; py < P; ++py) {
                    for (int px = 0; px < P; ++px) {
                        const float d = glm::dot(probe_texel_direction({px, py}, P), want);
                        if (d > best_dot) {
                            best_dot = d;
                            best = {px, py};
                        }
                    }
                }
                EXPECT_EQ(probe_border_source({x, y}, P), best)
                    << "payload " << P << " border texel (" << x << ", " << y << ")";
            }
        }
    }
}

TEST(GiMath, HysteresisCurveSpansTheDocumentedRange) {
    EXPECT_FLOAT_EQ(probe_blend_hysteresis(0.0f), PROBE_HYSTERESIS_MAX);
    EXPECT_FLOAT_EQ(probe_blend_hysteresis(PROBE_CHANGE_SATURATION), PROBE_HYSTERESIS_MIN);
    EXPECT_FLOAT_EQ(probe_blend_hysteresis(10.0f), PROBE_HYSTERESIS_MIN);
    // Monotonic: more disagreement can only ever mean less history.
    float previous = PROBE_HYSTERESIS_MAX + 1.0f;
    for (int i = 0; i <= 20; ++i) {
        const float h = probe_blend_hysteresis(static_cast<float>(i) * 0.1f);
        EXPECT_LE(h, previous);
        previous = h;
    }
}

TEST(GiMath, BlendConvergesAtBothHysteresisExtremes) {
    // The two ends of the curve, as pure EMAs, so the documented time constants
    // are pinned rather than asserted in a comment: 0.97 reaches 90% of a step
    // in 76 updates, 0.70 in 7.
    const auto steps_to_90 = [](float hysteresis) {
        float v = 0.0f;
        for (int i = 1; i <= 500; ++i) {
            v = 1.0f + (v - 1.0f) * hysteresis;
            if (v >= 0.9f) {
                return i;
            }
        }
        return -1;
    };
    EXPECT_EQ(steps_to_90(PROBE_HYSTERESIS_MAX), 76);
    EXPECT_EQ(steps_to_90(PROBE_HYSTERESIS_MIN), 7);

    // And the adaptive blend itself, on a full step from dark to lit: it starts
    // at the fast end because the disagreement is total, and eases toward the
    // slow end as the estimate closes — 16 updates, which at cascade 0's
    // every-frame rota is a quarter of a second.
    glm::vec3 v{0.0f};
    int steps = -1;
    for (int i = 1; i <= 500; ++i) {
        v = probe_blend(v, glm::vec3(1.0f, 1.0f, 1.0f));
        if (v.x >= 0.9f) {
            steps = i;
            break;
        }
    }
    EXPECT_EQ(steps, 16);

    // A SMALL change is treated as noise and integrated slowly — that is the
    // half of the curve that makes a 32-ray estimate usable at all.
    glm::vec3 w{1.0f};
    int slow = -1;
    for (int i = 1; i <= 500; ++i) {
        w = probe_blend(w, glm::vec3(1.05f));
        if (w.x >= 1.0f + 0.9f * 0.05f) {
            slow = i;
            break;
        }
    }
    EXPECT_GT(slow, 40) << "a 5% change must not be adopted at the fast rate";
}

TEST(GiMath, ZeroSigmaLeavesTheBlendCurveExactlyWhereItWas) {
    // The gate is opt-in by construction: every caller that cannot estimate its
    // own standard error passes 0 and gets the pre-gate curve to the bit.
    const glm::vec3 old_value(0.4f, 0.9f, 1.7f);
    const glm::vec3 estimate(0.9f, 0.3f, 2.4f);
    for (const float reset : {0.0f, 0.35f, 1.0f}) {
        const glm::vec3 a = probe_blend(old_value, estimate, reset);
        const glm::vec3 b = probe_blend(old_value, estimate, reset, 0.0f);
        EXPECT_FLOAT_EQ(a.x, b.x);
        EXPECT_FLOAT_EQ(a.y, b.y);
        EXPECT_FLOAT_EQ(a.z, b.z);
    }
}

TEST(GiMath, TheBlendsNoiseGateHoldsTheWindowOpenOnAStaticProbe) {
    // THE DEFECT THIS PINS. `new_value` is one update's estimate over a ray set
    // that is re-rotated every frame, so |new - old| has a strictly positive
    // floor even for a bit-static scene: the estimator's own dispersion. Fed raw
    // into the change-adaptive curve, that floor reads as "the lighting changed"
    // and collapses the accumulation window permanently — the field never runs
    // at PROBE_HYSTERESIS_MAX at all. Both sides are asserted, because a test
    // that only checks the gated number pins nothing about the gate.
    constexpr float SIGMA_REL = 0.15f; // a realistic per-update relative error
    const glm::vec3 truth(1.0f, 1.0f, 1.0f);

    const auto mean_hysteresis = [&](bool gated) {
        std::uint32_t rng = 12345u;
        glm::vec3 value = truth;
        double sum = 0.0;
        int samples = 0;
        for (int i = 0; i < 20000; ++i) {
            rng = rng * 1664525u + 1013904223u;
            // Uniform on [-sqrt(3), sqrt(3)]: zero mean, unit variance.
            const float u =
                (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f) * 1.7320508f;
            const glm::vec3 estimate = truth * (1.0f + SIGMA_REL * u);
            const float sigma = gated ? SIGMA_REL * glm::length(estimate) : 0.0f;
            const glm::vec3 blended = probe_blend(value, estimate, 0.0f, sigma);
            // h back out of the EMA: blended = new + (old - new) * h.
            if (std::abs(value.x - estimate.x) > 1e-5f) {
                sum += static_cast<double>((blended.x - estimate.x) / (value.x - estimate.x));
                ++samples;
            }
            value = blended;
        }
        return static_cast<float>(sum / static_cast<double>(samples));
    };

    const float ungated = mean_hysteresis(false);
    const float gated = mean_hysteresis(true);
    EXPECT_LT(ungated, 0.95f) << "the ungated curve must demonstrably collapse on pure noise, "
                                 "or this test is not exercising the defect";
    EXPECT_GT(gated, 0.95f) << "the gate must recover most of the documented window";
    EXPECT_GT(gated, ungated + 0.015f);
    EXPECT_LE(gated, PROBE_HYSTERESIS_MAX);
}

// --- Relocation and classification -----------------------------------------

TEST(GiMath, TheFullRayDirectionSetHasNoResultantWorthNudgingAlong) {
    // The premise of the coherence guard, stated as a number: a probe that flags
    // EVERY ray hands the classifier the resultant of the whole rotated
    // Fibonacci set, whose length is numerical residue and whose direction
    // re-randomizes every frame. This is the regression that would catch a
    // future change to the direction set re-opening the hole.
    for (const int rays : {32, 64, 96}) {
        for (std::uint32_t frame = 0; frame < 8; ++frame) {
            glm::vec3 sum(0.0f);
            for (int i = 0; i < rays; ++i) {
                sum += probe_ray_direction(i, rays, frame);
            }
            EXPECT_LT(glm::length(sum), PROBE_RELOCATE_MIN_COHERENCE * static_cast<float>(rays))
                << rays << " rays, frame " << frame;
        }
    }
}

TEST(GiMath, AnEnclosedProbeIsWrittenOffRatherThanRandomWalked) {
    constexpr int RAYS = 64;
    const float spacing = cascade_spacing(0);
    const float max_offset = PROBE_MAX_OFFSET_FRACTION * spacing;

    // Two consecutive frames, because the failure mode being closed is a WALK:
    // the resultant's direction changes with the ray rotation, so the old code
    // moved the probe somewhere different on every update while the sampler read
    // that moving offset as the probe's position.
    for (const std::uint32_t frame : {7u, 8u}) {
        glm::vec3 sum(0.0f);
        for (int i = 0; i < RAYS; ++i) {
            sum += probe_ray_direction(i, RAYS, frame);
        }
        const ProbeRelocation r = probe_relocation_step(glm::vec3(0.0f), sum, RAYS, RAYS, spacing,
                                                        max_offset, /*was_inside=*/false);
        EXPECT_FALSE(r.relocated);
        EXPECT_FLOAT_EQ(glm::length(r.offset), 0.0f);
        // And it must still be written off — gating the verdict on the offset
        // reaching the clamp would leave an enclosed probe PROBE_CLASS_OK
        // forever, at full weight in every cage, contributing black.
        EXPECT_FLOAT_EQ(r.classification, PROBE_CLASS_INSIDE_GEOMETRY);
    }
}

TEST(GiMath, AOneSidedBackfaceSetStillNudgesAwayFromIt) {
    constexpr int RAYS = 64;
    const float spacing = cascade_spacing(0);
    const float max_offset = PROBE_MAX_OFFSET_FRACTION * spacing;

    // A probe flush against a wall on its +X side: the rays into +X come back as
    // backfaces, the rest do not. This is the case relocation exists for, and
    // the guard must not touch it.
    glm::vec3 sum(0.0f);
    std::uint32_t backfaces = 0;
    for (int i = 0; i < RAYS; ++i) {
        const glm::vec3 d = probe_ray_direction(i, RAYS, 3u);
        if (d.x > 0.0f) {
            sum += d;
            ++backfaces;
        }
    }
    ASSERT_GT(static_cast<float>(backfaces) / static_cast<float>(RAYS),
              PROBE_RELOCATE_BACKFACE_FRACTION);
    ASSERT_LT(static_cast<float>(backfaces) / static_cast<float>(RAYS),
              PROBE_INSIDE_BACKFACE_FRACTION);

    const ProbeRelocation r = probe_relocation_step(glm::vec3(0.0f), sum, backfaces, RAYS, spacing,
                                                    max_offset, /*was_inside=*/false);
    EXPECT_TRUE(r.relocated);
    EXPECT_LT(r.offset.x, 0.0f) << "the nudge is away from the occluding shell";
    EXPECT_NEAR(glm::length(r.offset), 0.25f * spacing, 1e-2f);
    EXPECT_FLOAT_EQ(r.classification, PROBE_CLASS_OK);
}

TEST(GiMath, RelocationStopsAtTheOffsetClampAndThenWritesTheProbeOff) {
    constexpr int RAYS = 64;
    const float spacing = cascade_spacing(0);
    const float max_offset = PROBE_MAX_OFFSET_FRACTION * spacing;

    // Coherent, still mostly backfaces, and already at the clamp: the pre-guard
    // write-off path, which the guard must leave exactly as it was.
    glm::vec3 sum(0.0f);
    std::uint32_t backfaces = 0;
    for (int i = 0; i < RAYS; ++i) {
        const glm::vec3 d = probe_ray_direction(i, RAYS, 3u);
        if (d.x > -0.6f) {
            sum += d;
            ++backfaces;
        }
    }
    ASSERT_GT(static_cast<float>(backfaces) / static_cast<float>(RAYS),
              PROBE_INSIDE_BACKFACE_FRACTION);
    const glm::vec3 at_clamp(-max_offset, 0.0f, 0.0f);
    const ProbeRelocation r = probe_relocation_step(at_clamp, sum, backfaces, RAYS, spacing,
                                                    max_offset, /*was_inside=*/false);
    EXPECT_TRUE(r.relocated);
    EXPECT_NEAR(glm::length(r.offset), max_offset, 1e-2f);
    EXPECT_FLOAT_EQ(r.classification, PROBE_CLASS_INSIDE_GEOMETRY);
}

// --- Per-cascade ray reach -------------------------------------------------

TEST(GiMath, RayReachIsPerCascadeAndOnlyEverACeiling) {
    // The finest cascade gives up reach it cannot use; the coarsest keeps the
    // caller's range exactly. Never longer than asked for, on any cascade.
    constexpr float REQUESTED = 4000.0f;
    EXPECT_FLOAT_EQ(cascade_ray_max_distance(0, REQUESTED), 12.0f * 150.0f);
    EXPECT_FLOAT_EQ(cascade_ray_max_distance(1, REQUESTED), 12.0f * 300.0f);
    EXPECT_FLOAT_EQ(cascade_ray_max_distance(2, REQUESTED), REQUESTED);
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        EXPECT_LE(cascade_ray_max_distance(c, REQUESTED), REQUESTED);
        // And it must clear the range the visibility moments can even represent,
        // or a miss would record as an occluder at the clamp distance.
        EXPECT_GT(cascade_ray_max_distance(c, REQUESTED),
                  PROBE_VISIBILITY_DISTANCE_SCALE * cascade_spacing(c));
    }
    // A caller asking for less than a cascade's own reach still gets less.
    EXPECT_FLOAT_EQ(cascade_ray_max_distance(0, 500.0f), 500.0f);
}

// --- The update rota -------------------------------------------------------

TEST(GiMath, UninitializedProbesJumpTheUpdateQueue) {
    ProbeCascade cascade;
    cascade.reset(0);
    cascade.update(glm::vec3(0.0f));
    ASSERT_TRUE(cascade.placed());

    // A freshly placed cascade is entirely uninitialized, so the first turns
    // are pure phase A: slots in order, budget at a time, and the round-robin
    // cursor has not moved because phase B never ran.
    std::vector<ProbeUpdateSlot> picked;
    EXPECT_EQ(cascade.select_updates(100, picked), 100u);
    ASSERT_EQ(picked.size(), 100u);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(picked[static_cast<std::size_t>(i)].slot, i);
        EXPECT_TRUE(picked[static_cast<std::size_t>(i)].was_uninitialized);
    }
    EXPECT_EQ(cascade.update_cursor(), 0);

    picked.clear();
    EXPECT_EQ(cascade.select_updates(100, picked), 100u);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(picked[static_cast<std::size_t>(i)].slot, 100 + i);
    }
}

TEST(GiMath, SteadyStateUpdatesRoundRobinFromAPersistentCursor) {
    ProbeCascade cascade;
    cascade.reset(0);
    cascade.update(glm::vec3(0.0f));

    // One enormous turn initializes the whole cascade, so what follows is pure
    // phase B.
    std::vector<ProbeUpdateSlot> picked;
    EXPECT_EQ(cascade.select_updates(PROBES_PER_CASCADE, picked), PROBES_PER_CASCADE);

    for (int turn = 0; turn < 3; ++turn) {
        picked.clear();
        EXPECT_EQ(cascade.select_updates(100, picked), 100u);
        for (int i = 0; i < 100; ++i) {
            EXPECT_EQ(picked[static_cast<std::size_t>(i)].slot, turn * 100 + i);
            EXPECT_FALSE(picked[static_cast<std::size_t>(i)].was_uninitialized);
        }
        EXPECT_EQ(cascade.update_cursor(), (turn + 1) * 100);
    }

    // The cursor wraps rather than stalling at the end of the cascade.
    picked.clear();
    const int remaining = PROBES_PER_CASCADE - 300;
    EXPECT_EQ(cascade.select_updates(static_cast<std::uint32_t>(remaining) + 50u, picked),
              static_cast<std::uint32_t>(remaining) + 50u);
    EXPECT_EQ(picked.back().slot, 49);
    EXPECT_EQ(cascade.update_cursor(), 50);
}

TEST(GiMath, ScrolledInProbesPreemptTheRoundRobin) {
    ProbeCascade cascade;
    cascade.reset(0);
    cascade.update(glm::vec3(0.0f));
    std::vector<ProbeUpdateSlot> picked;
    cascade.select_updates(PROBES_PER_CASCADE, picked); // everything live

    // Move one whole probe step along X: exactly one plane enters the window.
    const std::uint32_t entered = cascade.update(glm::vec3(cascade_spacing(0), 0.0f, 0.0f));
    ASSERT_EQ(entered, static_cast<std::uint32_t>(GRID_Y * GRID_Z));

    picked.clear();
    cascade.select_updates(64, picked);
    ASSERT_EQ(picked.size(), 64u);
    for (const ProbeUpdateSlot& slot : picked) {
        EXPECT_TRUE(slot.was_uninitialized)
            << "an entering probe must outrank a probe that already holds valid radiance";
    }
}

TEST(GiMath, UpdateSelectionIsDeterministicAndDuplicateFree) {
    const auto run = [] {
        ProbeCascade cascade;
        cascade.reset(1);
        cascade.update(glm::vec3(0.0f));
        std::vector<int> order;
        for (int turn = 0; turn < 12; ++turn) {
            std::vector<ProbeUpdateSlot> picked;
            cascade.select_updates(700, picked);
            // A slot chosen twice in ONE turn would be two threadgroups writing
            // one probe's texels in the same dispatch — a data race, not just
            // duplicated work.
            std::set<int> unique;
            for (const ProbeUpdateSlot& s : picked) {
                EXPECT_TRUE(unique.insert(s.slot).second) << "duplicate slot in one turn";
                order.push_back(s.slot);
            }
            cascade.update(glm::vec3(static_cast<float>(turn) * 40.0f, 0.0f, 0.0f));
        }
        return order;
    };
    EXPECT_EQ(run(), run());
}

// ---------------------------------------------------------------------------
// R1c — the SAMPLER. The CPU mirror of shaders/gi_sample.hlsli.
// ---------------------------------------------------------------------------
//
// These are the tests the shader cannot have. The consumption path's failure
// modes are all continuity failures — a seam at a cascade boundary, a halo
// where a probe was written off, a step where a cage crosses a cell — and every
// one of them is invisible in a still frame and obvious in motion. Asserting
// them numerically here is the only way to catch them before a player does.

namespace {

/// A field where every probe is usable, has no relocation offset, reports the
/// same irradiance, and sees nothing (moments at the distance clamp, so
/// Chebyshev never attenuates). The arithmetic baseline.
struct UniformField {
    glm::vec3 value{1.0f, 2.0f, 3.0f};
    ProbeReading operator()(int cascade, const glm::ivec3&, int, const glm::vec3&) const {
        ProbeReading r;
        const float far = PROBE_VISIBILITY_DISTANCE_SCALE * cascade_spacing(cascade);
        r.moments = {far, far * far};
        r.irradiance = value;
        return r;
    }
};

/// Same, but a caller-chosen set of absolute coordinates is written off as
/// sitting inside geometry, and each cascade answers with its own colour so the
/// blend can be read off the result.
struct BandedField {
    /// Keyed by (cascade, x, y, z): the coarse lattices are SUBSETS of the fine
    /// one, so a cage killed by absolute coordinate alone would take the
    /// next-coarser cascade's cage with it and the handover under test would
    /// never happen.
    std::set<std::tuple<int, int, int, int>> dead;
    std::array<glm::vec3, CASCADE_COUNT> per_cascade{glm::vec3(1.0f, 0.0f, 0.0f),
                                                     glm::vec3(0.0f, 1.0f, 0.0f),
                                                     glm::vec3(0.0f, 0.0f, 1.0f)};
    ProbeReading operator()(int cascade, const glm::ivec3& a, int, const glm::vec3&) const {
        ProbeReading r;
        const float far = PROBE_VISIBILITY_DISTANCE_SCALE * cascade_spacing(cascade);
        r.moments = {far, far * far};
        r.irradiance = per_cascade[static_cast<std::size_t>(cascade)];
        if (dead.count({cascade, a.x, a.y, a.z}) != 0) {
            r.classification = PROBE_CLASS_INSIDE_GEOMETRY;
        }
        return r;
    }
};

/// Cascade bases for a camera at the world origin — what ProbeVolume would
/// report after one update() there.
std::array<glm::ivec3, CASCADE_COUNT> bases_at(const glm::vec3& camera) {
    std::array<glm::ivec3, CASCADE_COUNT> b{};
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        b[static_cast<std::size_t>(c)] = cascade_base(camera, c);
    }
    return b;
}

constexpr glm::vec3 UP{0.0f, 1.0f, 0.0f};
constexpr glm::vec3 SKY{0.2f, 0.4f, 0.9f};
constexpr glm::vec3 GROUND{0.1f, 0.08f, 0.06f};

} // namespace

TEST(GiSample, AUniformFieldReproducesItsOwnValue) {
    // Trilinear over eight identical probes, renormalised, must be exactly the
    // probe value at ANY position inside the window — including positions that
    // are not on the lattice. If it is not, the weights are not a partition.
    const auto bases = bases_at(glm::vec3(0.0f));
    const UniformField field;
    for (const glm::vec3& p : {glm::vec3(0.0f), glm::vec3(37.0f, 12.0f, -91.0f),
                               glm::vec3(-450.0f, 300.0f, 620.0f)}) {
        const glm::vec3 got = sample_field(bases, p, UP, UP, SKY, GROUND, field);
        EXPECT_NEAR(got.x, field.value.x, 1e-4f) << "at " << p.x << "," << p.y << "," << p.z;
        EXPECT_NEAR(got.y, field.value.y, 1e-4f);
        EXPECT_NEAR(got.z, field.value.z, 1e-4f);
    }
}

TEST(GiSample, CascadeSelectionTakesTheFinestCascadeThatContainsThePoint) {
    const auto bases = bases_at(glm::vec3(0.0f));
    BandedField field;
    // Near the camera every cascade contains the point, so cascade 0 answers
    // alone and the coarser ones never get any weight.
    const glm::vec3 near_result = sample_field(bases, glm::vec3(0.0f), UP, UP, SKY, GROUND, field);
    EXPECT_NEAR(near_result.x, 1.0f, 1e-4f);
    EXPECT_NEAR(near_result.y, 0.0f, 1e-4f);

    // Cascade 0 reaches 24 * 150 = 3600 units across, centred, so ~+1600 on X
    // is outside it but well inside cascade 1 (7200) — which must answer alone.
    const glm::vec3 mid_result =
        sample_field(bases, glm::vec3(1600.0f, 0.0f, 0.0f), UP, UP, SKY, GROUND, field);
    EXPECT_NEAR(mid_result.x, 0.0f, 1e-4f);
    EXPECT_NEAR(mid_result.y, 1.0f, 1e-4f);

    // Past cascade 2's reach nothing answers and the hemisphere fallback — the
    // engine's GI-off tier — takes the whole weight, so a pixel the field
    // cannot cover looks exactly as it did before this phase.
    const glm::vec3 far_result =
        sample_field(bases, glm::vec3(50000.0f, 0.0f, 0.0f), UP, UP, SKY, GROUND, field);
    const glm::vec3 hemisphere = glm::mix(GROUND, SKY, UP.y * 0.5f + 0.5f);
    EXPECT_NEAR(far_result.x, hemisphere.x, 1e-5f);
    EXPECT_NEAR(far_result.y, hemisphere.y, 1e-5f);
    EXPECT_NEAR(far_result.z, hemisphere.z, 1e-5f);
}

TEST(GiSample, ACascadeStopsAnsweringExactlyWhereItsCageWouldLeaveTheWindow) {
    // The margin is what keeps a trilinear tap from wrapping around the torus
    // and reading the opposite side of the world. Walk out along +X until
    // cascade 0 gives up, and check the last position it answered for still has
    // its whole 8-tap cage inside the window.
    const auto bases = bases_at(glm::vec3(0.0f));
    const UniformField field;
    const float sp = cascade_spacing(0);
    float last_answering = 0.0f;
    for (float x = 0.0f; x < 4000.0f; x += 1.0f) {
        const CascadeSample s = sample_cascade(0, bases[0], glm::vec3(x, 0.0f, 0.0f), UP, UP, field);
        if (s.confidence > 0.0f) {
            last_answering = x;
        } else {
            break;
        }
    }
    const glm::vec3 biased = sample_bias_position(glm::vec3(last_answering, 0.0f, 0.0f), UP, UP, 0);
    const glm::vec3 local = biased / sp - glm::vec3(bases[0]);
    EXPECT_GE(local.x, SAMPLE_CASCADE_MARGIN_PROBES);
    EXPECT_LE(std::floor(local.x) + 1.0f, static_cast<float>(GRID_X - 1));
}

TEST(GiSample, DeadProbesAreExcludedAndTheSurvivorsRenormalise) {
    // Half the cage written off must NOT halve the answer: the live weights are
    // renormalised, so four survivors reporting the same value still report
    // that value. Dividing by the full trilinear total instead would produce a
    // dark halo around every wall the relocation pass gave up on.
    const auto bases = bases_at(glm::vec3(0.0f));
    const glm::vec3 P(37.0f, 12.0f, -91.0f);
    BandedField field;
    const CascadeSample all_live = sample_cascade(0, bases[0], P, UP, UP, field);
    ASSERT_GT(all_live.confidence, 0.0f);

    const glm::vec3 local = sample_bias_position(P, UP, UP, 0) / cascade_spacing(0) - glm::vec3(bases[0]);
    const glm::ivec3 corner = bases[0] + glm::ivec3(static_cast<int>(std::floor(local.x)),
                                                    static_cast<int>(std::floor(local.y)),
                                                    static_cast<int>(std::floor(local.z)));
    for (int i = 0; i < 8; i += 2) { // kill four of the eight
        const glm::ivec3 a = corner + glm::ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        field.dead.insert({0, a.x, a.y, a.z});
    }
    const CascadeSample half_dead = sample_cascade(0, bases[0], P, UP, UP, field);
    EXPECT_NEAR(half_dead.irradiance.x, all_live.irradiance.x, 1e-5f);
    // What it DOES cost is confidence: less of the cage answered, so more of
    // the result comes from the next-coarser cascade.
    EXPECT_LT(half_dead.confidence, all_live.confidence + 1e-6f);

    // All eight dead: the cascade abstains entirely rather than returning zero
    // radiance, which is the difference between "ask someone else" and "black".
    for (int i = 0; i < 8; ++i) {
        const glm::ivec3 a = corner + glm::ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        field.dead.insert({0, a.x, a.y, a.z});
    }
    EXPECT_EQ(sample_cascade(0, bases[0], P, UP, UP, field).confidence, 0.0f);
    const glm::vec3 blended = sample_field(bases, P, UP, UP, SKY, GROUND, field);
    EXPECT_NEAR(blended.y, 1.0f, 1e-4f) << "cascade 1 should have taken over";
}

TEST(GiSample, ChebyshevOccludesPastTheMeanAndPassesInFrontOfIt) {
    const float sp = cascade_spacing(0);
    // A probe that consistently sees a wall 100 units away. The stored moments
    // are NORMALIZED BY SPACING — see ProbeReading::moments — so the wall's mean
    // is 100/sp and its variance 25/sp^2. `dist` stays in world units.
    const float wall = 100.0f;
    const float mean = wall / sp;
    const float mean2 = mean * mean + 25.0f / (sp * sp);
    EXPECT_FLOAT_EQ(chebyshev_visibility(50.0f, mean, mean2, sp), 1.0f);
    EXPECT_FLOAT_EQ(chebyshev_visibility(wall, mean, mean2, sp), 1.0f);
    EXPECT_LT(chebyshev_visibility(400.0f, mean, mean2, sp), 0.05f);
    // Monotonically non-increasing in distance, which is what makes it usable
    // as a weight at all — a non-monotone occlusion term produces rings.
    float previous = 1.0f;
    for (float d = 0.0f; d < 1000.0f; d += 5.0f) {
        const float v = chebyshev_visibility(d, mean, mean2, sp);
        EXPECT_LE(v, previous + 1e-6f) << "at " << d;
        previous = v;
    }
    // The variance floor is what keeps a texel that has only ever seen one
    // distance from collapsing into a hard binary test that flickers as the ray
    // set rotates.
    EXPECT_GT(chebyshev_visibility(wall + 0.5f * sp, mean, mean * mean, sp), 0.0f);
}

TEST(GiSample, VisibilityMomentsStayInsideHalfFloatOnEveryCascade) {
    // WHY THIS TEST EXISTS. The visibility atlas is R16G16_FLOAT (probe_volume.cpp),
    // whose finite maximum is 65504, and the moments used to be stored as raw
    // world distances. The clamp is PROBE_VISIBILITY_DISTANCE_SCALE * spacing
    // over spacings of 150 / 300 / 600, so the SECOND moment reached 50625 /
    // 202500 / 810000 and the two coarse cascades saturated to infinity — after
    // which `mean2 - mean^2` is NaN or negative, pins at the variance floor, and
    // the Chebyshev bound degenerates into the hard binary test that flickers as
    // the ray set rotates. Normalizing by spacing bounds the pair by
    // (1.5, 2.25) on every cascade, with three orders of magnitude to spare.
    constexpr float HALF_MAX = 65504.0f;
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        const float sp = cascade_spacing(c);
        const float clamp_world = PROBE_VISIBILITY_DISTANCE_SCALE * sp;
        const float mean = clamp_world / sp;
        const float mean2 = mean * mean;
        EXPECT_FLOAT_EQ(mean, PROBE_VISIBILITY_DISTANCE_SCALE) << "cascade " << c;
        EXPECT_LT(mean2, HALF_MAX / 1000.0f)
            << "cascade " << c << " second moment must have orders of magnitude of headroom";
        // And the test still answers softly at the clamp rather than collapsing:
        // a point at the clamp distance is exactly what the probe measured.
        EXPECT_FLOAT_EQ(chebyshev_visibility(clamp_world, mean, mean2, sp), 1.0f);
    }
}

TEST(GiSample, ChebyshevIsScaleInvariantAcrossCascades) {
    // The payoff of normalizing: the same geometry expressed in each cascade's
    // own units must give the same visibility. Before normalization the bias and
    // variance floor were multiplied by spacing and spacing^2 at the use site
    // against world-unit moments, which was also scale-invariant in exact
    // arithmetic — the difference is that it is now scale-invariant in fp16 too.
    const float mean = 0.5f;             // half a spacing away, whichever cascade
    const float mean2 = mean * mean + 0.01f;
    float reference = -1.0f;
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        const float sp = cascade_spacing(c);
        const float v = chebyshev_visibility(1.2f * sp, mean, mean2, sp);
        if (reference < 0.0f) {
            reference = v;
        } else {
            EXPECT_NEAR(v, reference, 1e-6f) << "cascade " << c;
        }
    }
    EXPECT_GT(reference, 0.0f);
    EXPECT_LT(reference, 1.0f) << "1.2 spacings is past a wall at 0.5, so it must attenuate";
}

TEST(GiSample, TheCascadeBlendIsContinuousAcrossABoundary) {
    // NO VISIBLE SEAM AT A CASCADE BOUNDARY MEANS THE WEIGHTS ARE C0, and that
    // is a numeric claim, so it is made numerically. Walk a fine line along +X
    // straight through cascade 0's edge into cascade 1's interior, with each
    // cascade answering a DIFFERENT colour so any discontinuity in the handover
    // shows up at full contrast, and assert the frame-to-frame delta stays
    // bounded by the step size.
    const auto bases = bases_at(glm::vec3(0.0f));
    BandedField field;
    constexpr float STEP = 2.0f;
    glm::vec3 previous = sample_field(bases, glm::vec3(0.0f), UP, UP, SKY, GROUND, field);
    float worst = 0.0f;
    bool saw_cascade0 = false;
    bool saw_cascade1 = false;
    bool saw_mixture = false;
    for (float x = STEP; x <= 3400.0f; x += STEP) {
        const glm::vec3 got = sample_field(bases, glm::vec3(x, 0.0f, 0.0f), UP, UP, SKY, GROUND, field);
        worst = std::max(worst, glm::length(got - previous));
        saw_cascade0 = saw_cascade0 || got.x > 0.99f;
        saw_cascade1 = saw_cascade1 || got.y > 0.99f;
        saw_mixture = saw_mixture || (got.x > 0.05f && got.y > 0.05f);
        previous = got;
    }
    EXPECT_TRUE(saw_cascade0) << "the walk never started inside cascade 0";
    EXPECT_TRUE(saw_cascade1) << "the walk never reached cascade 1 alone";
    EXPECT_TRUE(saw_mixture) << "the handover was a step, not a blend";
    // The fade band is one probe of cascade 0 = 150 units, over which the
    // result travels a chromatic distance of sqrt(2); at 2 units per step that
    // bounds a step at ~0.019. Ten times that is a generous ceiling that still
    // fails loudly on any hard switch, which would jump the full sqrt(2).
    EXPECT_LT(worst, 0.19f) << "cascade handover is discontinuous";
}

TEST(GiSample, TheSampleIsContinuousAcrossACellBoundary) {
    // The other seam: the 8-tap cage changes membership every probe spacing.
    // It stays continuous only because a probe leaving the cage does so with a
    // trilinear factor of exactly zero — so make the probes disagree wildly and
    // check the sample still cannot jump.
    struct NoisyField {
        ProbeReading operator()(int cascade, const glm::ivec3& a, int, const glm::vec3&) const {
            ProbeReading r;
            const float far = PROBE_VISIBILITY_DISTANCE_SCALE * cascade_spacing(cascade);
            r.moments = {far, far * far};
            const int h = (a.x * 73856093) ^ (a.y * 19349663) ^ (a.z * 83492791);
            r.irradiance = glm::vec3(static_cast<float>((h >> 3) & 0xFF) / 255.0f);
            return r;
        }
    } field;
    const auto bases = bases_at(glm::vec3(0.0f));
    constexpr float STEP = 1.0f;
    glm::vec3 previous = sample_field(bases, glm::vec3(-600.0f, 0.0f, 0.0f), UP, UP, SKY, GROUND, field);
    float worst = 0.0f;
    for (float x = -600.0f + STEP; x <= 600.0f; x += STEP) {
        const glm::vec3 got = sample_field(bases, glm::vec3(x, 0.0f, 0.0f), UP, UP, SKY, GROUND, field);
        worst = std::max(worst, glm::length(got - previous));
        previous = got;
    }
    // One step is 1/150th of a spacing, and the trilinear factor of the probe
    // being swapped is zero at the crossing, so the bound is the Lipschitz
    // constant of the interpolation alone: at most (max - min) / 150 per step
    // in each channel, i.e. well under 0.05 for a sqrt(3)-long unit cube.
    EXPECT_LT(worst, 0.05f) << "cell crossing is discontinuous";
}

TEST(GiSample, TheBiasPushesAlongTheNormalAndTowardTheCamera) {
    // Both halves, and both scaled by the cascade's spacing rather than fixed —
    // a coarse cascade's acceptable bias is proportionally larger.
    const glm::vec3 P(10.0f, 20.0f, 30.0f);
    const glm::vec3 N(0.0f, 1.0f, 0.0f);
    const glm::vec3 V(1.0f, 0.0f, 0.0f);
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        const glm::vec3 biased = sample_bias_position(P, N, V, c);
        const float sp = cascade_spacing(c);
        EXPECT_FLOAT_EQ(biased.y - P.y, SAMPLE_NORMAL_BIAS_FRACTION * sp);
        EXPECT_FLOAT_EQ(biased.x - P.x, SAMPLE_VIEW_BIAS_FRACTION * sp);
        EXPECT_FLOAT_EQ(biased.z, P.z);
    }
}

TEST(GiSample, RelocatedProbesAreSampledWhereTheyActuallySat) {
    // The R1c contract from probe_volume.hpp: a probe nudged out of a wall is
    // not at its lattice position, and a sampler that ignores the offset reads
    // it as if it were. Prove the offset reaches the weights by giving ONE
    // probe all the radiance and then pushing it past the distance its own
    // visibility moments call occluded — its weight must collapse, which cannot
    // happen unless the offset is being applied to its world position.
    // The probe the shading normal actually points at: N is +Y, so the cage
    // corner ABOVE the sample point is the one carrying the weight.
    static constexpr glm::ivec3 TARGET{0, 1, 0};
    struct OffsetField {
        glm::vec3 offset{0.0f};
        ProbeReading operator()(int cascade, const glm::ivec3& a, int, const glm::vec3&) const {
            ProbeReading r;
            // A wall 120 units out in every direction. The biased sample point
            // sits ~70 units under probe (0,1,0), inside that; pushing that
            // probe up by the maximum relocation puts it ~138 units away, i.e.
            // outside, which is the whole point of the test.
            //
            // Moments are stored NORMALIZED BY SPACING (ProbeReading::moments),
            // so the wall goes in as 120/spacing rather than 120.
            const float wall = 120.0f / cascade_spacing(cascade);
            r.moments = {wall, wall * wall};
            const bool target = a == TARGET;
            r.irradiance = target ? glm::vec3(1.0f) : glm::vec3(0.0f);
            if (target) {
                r.offset = offset;
            }
            return r;
        }
    } field;
    const auto bases = bases_at(glm::vec3(0.0f));
    const glm::vec3 P(5.0f, 5.0f, 5.0f);
    const CascadeSample unmoved = sample_cascade(0, bases[0], P, UP, UP, field);
    ASSERT_GT(unmoved.irradiance.x, 0.3f) << "the probe above should dominate the cage";

    field.offset = glm::vec3(0.0f, PROBE_MAX_OFFSET_FRACTION * cascade_spacing(0), 0.0f);
    const CascadeSample moved = sample_cascade(0, bases[0], P, UP, UP, field);
    EXPECT_LT(moved.irradiance.x, unmoved.irradiance.x * 0.5f)
        << "the relocation offset never reached the weights";
}
