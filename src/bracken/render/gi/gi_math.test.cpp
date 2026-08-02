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

    // Corners take the OPPOSITE corner (the octahedron's -Z point is all four
    // square corners, so they are the same direction).
    EXPECT_EQ(probe_border_source({0, 0}, P), glm::ivec2(0, 0));
    EXPECT_EQ(probe_border_source({T - 1, 0}, P), glm::ivec2(P - 1, 0));
    EXPECT_EQ(probe_border_source({0, T - 1}, P), glm::ivec2(0, P - 1));
    EXPECT_EQ(probe_border_source({T - 1, T - 1}, P), glm::ivec2(P - 1, P - 1));

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
