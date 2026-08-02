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
    EXPECT_EQ(atlas_height(IRRADIANCE_PROBE_TEXELS), 192);
    EXPECT_EQ(atlas_width(VISIBILITY_PROBE_TEXELS), 4608);
    EXPECT_EQ(atlas_height(VISIBILITY_PROBE_TEXELS), 384);
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

        const glm::ivec2 origin = probe_atlas_origin(i, IRRADIANCE_PROBE_TEXELS);
        EXPECT_GE(origin.x, 0);
        EXPECT_LT(origin.x, atlas_width(IRRADIANCE_PROBE_TEXELS));
        EXPECT_GE(origin.y, 0);
        EXPECT_LT(origin.y, atlas_height(IRRADIANCE_PROBE_TEXELS));
        EXPECT_TRUE(tiles.emplace(origin.x, origin.y).second) << "two probes share atlas tile at " << i;
    }
    EXPECT_EQ(tiles.size(), static_cast<std::size_t>(PROBES_PER_CASCADE));
}

TEST(GiMath, PayloadTexelsStayInsideTheirTile) {
    // The border ring exists so a bilinear tap at the payload edge does not
    // reach the next probe. That only holds if the payload really is inset.
    for (int slot : {0, 1, PROBES_PER_ATLAS_ROW, PROBES_PER_CASCADE - 1}) {
        const glm::ivec2 origin = probe_atlas_origin(slot, VISIBILITY_PROBE_TEXELS);
        for (int y = 0; y < VISIBILITY_PAYLOAD_TEXELS; ++y) {
            for (int x = 0; x < VISIBILITY_PAYLOAD_TEXELS; ++x) {
                const glm::ivec2 t = probe_payload_texel(slot, VISIBILITY_PROBE_TEXELS, {x, y});
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
