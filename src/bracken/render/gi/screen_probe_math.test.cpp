// Screen-probe gather math (R4b, docs/rt_lighting_plan.md §5.5), headless.
//
// Everything checked here is mirrored texel for texel in
// shaders/gi_screen_probe.hlsli. The reason this file exists rather than
// "looking at the screenshot" is that every failure mode of this code is
// INVISIBLE in a screenshot and lethal to the signal: a march step that lands
// on the wrong cell boundary either hangs or tunnels; a jitter that repeats
// makes the temporal EMA converge to 64 directions instead of to the scene; an
// SH basis with a sign flipped produces indirect light that is subtly wrong
// everywhere and obviously wrong nowhere.
//
// No GPU, no window, no SDL.

#include <gtest/gtest.h>

#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/render/gi/screen_probe_math.hpp"
#include "bracken/render/hzb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace {

using namespace bracken::render;
using namespace bracken::render::gi;

// ===========================================================================
// The grid and the tiling
// ===========================================================================

TEST(ScreenProbeGrid, CoversEveryPixelIncludingAPartialEdgeTile) {
    // 1600x900 is the smoke resolution: 900 is not a multiple of 16, so the
    // last row of tiles is partial and MUST still get a probe — dropping it
    // would leave a 4-pixel band with no screen GI at the bottom of the frame.
    const glm::ivec2 grid = screen_probe_grid(1600, 900);
    EXPECT_EQ(grid.x, 100);
    EXPECT_EQ(grid.y, 57); // ceil(900 / 16) == 57, not 56
    EXPECT_EQ(screen_probe_count(grid), 5700);
    EXPECT_GE(grid.y * SCREEN_PROBE_TILE, 900);

    EXPECT_EQ(screen_probe_grid(0, 0), glm::ivec2(0, 0));
    EXPECT_EQ(screen_probe_grid(-4, 100), glm::ivec2(0, 0));
    EXPECT_EQ(screen_probe_grid(16, 16), glm::ivec2(1, 1));
    EXPECT_EQ(screen_probe_grid(17, 1), glm::ivec2(2, 1));
}

TEST(ScreenProbeGrid, IndexAndCoordRoundTrip) {
    const glm::ivec2 grid = screen_probe_grid(1600, 900);
    for (int y = 0; y < grid.y; ++y) {
        for (int x = 0; x < grid.x; ++x) {
            const int index = screen_probe_index({x, y}, grid);
            ASSERT_GE(index, 0);
            ASSERT_LT(index, screen_probe_count(grid));
            EXPECT_EQ(screen_probe_coord(index, grid), glm::ivec2(x, y));
        }
    }
}

TEST(ScreenProbeAtlas, RadianceTexelsPartitionTheAtlas) {
    const glm::ivec2 grid{7, 5};
    const glm::ivec2 size = screen_probe_radiance_atlas_size(grid);
    std::set<std::pair<int, int>> seen;
    for (int p = 0; p < screen_probe_count(grid); ++p) {
        const glm::ivec2 probe = screen_probe_coord(p, grid);
        for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
            const glm::ivec2 texel = screen_probe_radiance_texel(probe, t);
            ASSERT_GE(texel.x, 0);
            ASSERT_GE(texel.y, 0);
            ASSERT_LT(texel.x, size.x);
            ASSERT_LT(texel.y, size.y);
            EXPECT_TRUE(seen.insert({texel.x, texel.y}).second)
                << "two (probe, texel) pairs share an atlas texel";
        }
    }
    EXPECT_EQ(seen.size(), static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y));
}

// ===========================================================================
// Anchor placement
// ===========================================================================

TEST(ScreenProbeAnchor, SearchOrderIsInsideTheTileStartsAtTheCentreAndNeverRepeats) {
    std::set<std::pair<int, int>> seen;
    for (int attempt = 0; attempt < SCREEN_PROBE_ANCHOR_ATTEMPTS; ++attempt) {
        const glm::ivec2 o = screen_probe_anchor_offset(attempt);
        EXPECT_GE(o.x, 0);
        EXPECT_GE(o.y, 0);
        EXPECT_LT(o.x, SCREEN_PROBE_TILE);
        EXPECT_LT(o.y, SCREEN_PROBE_TILE);
        EXPECT_TRUE(seen.insert({o.x, o.y}).second) << "attempt " << attempt << " repeats a pixel";
    }
    // The centre is first: it is the pixel whose interpolation weight is
    // highest for the whole tile, so it is the one to prefer when it is valid.
    EXPECT_EQ(screen_probe_anchor_offset(0), glm::ivec2(SCREEN_PROBE_TILE / 2, SCREEN_PROBE_TILE / 2));
    // Candidates walk OUTWARD: the four quarter-points before the four edge
    // midpoints, so a tile whose centre is sky but whose corner is a rooftop
    // takes the rooftop nearest the middle rather than an arbitrary edge.
    for (int attempt = 1; attempt <= 4; ++attempt) {
        const glm::ivec2 o = screen_probe_anchor_offset(attempt);
        EXPECT_GT(o.x, 0);
        EXPECT_GT(o.y, 0);
        EXPECT_LT(o.x, SCREEN_PROBE_TILE - 1);
        EXPECT_LT(o.y, SCREEN_PROBE_TILE - 1);
    }
    // Out of range clamps to the centre rather than reading past the table.
    EXPECT_EQ(screen_probe_anchor_offset(-1), screen_probe_anchor_offset(0));
    EXPECT_EQ(screen_probe_anchor_offset(999), screen_probe_anchor_offset(0));
}

TEST(ScreenProbeAnchor, FlagsRoundTripEveryOffsetTheSearchCanProduce) {
    for (int attempt = 0; attempt < SCREEN_PROBE_ANCHOR_ATTEMPTS; ++attempt) {
        const glm::ivec2 o = screen_probe_anchor_offset(attempt);
        const std::uint32_t flags = screen_probe_pack_flags(o, true);
        EXPECT_TRUE((flags & SCREEN_PROBE_FLAG_VALID) != 0u);
        EXPECT_EQ(screen_probe_unpack_anchor(flags), o);
        const std::uint32_t invalid = screen_probe_pack_flags(o, false);
        EXPECT_TRUE((invalid & SCREEN_PROBE_FLAG_VALID) == 0u);
        EXPECT_EQ(screen_probe_unpack_anchor(invalid), o);
    }
    // Four bits per axis is EXACTLY a 16-pixel tile. If the tile size ever
    // changes, this packing has to change with it — which is what this asserts.
    static_assert(SCREEN_PROBE_TILE == 16, "the anchor packing is four bits per axis");
    EXPECT_EQ(screen_probe_unpack_anchor(screen_probe_pack_flags({15, 15}, true)), glm::ivec2(15, 15));
}

// ===========================================================================
// Ray directions
// ===========================================================================

TEST(ScreenProbeDirections, JitterIsDeterministicInTheFrameCounterAndDoesNotRepeatQuickly) {
    for (std::uint32_t f = 0; f < 64; ++f) {
        const glm::vec2 a = screen_probe_jitter(f);
        const glm::vec2 b = screen_probe_jitter(f);
        EXPECT_EQ(a, b) << "the jitter must be a pure function of the frame index";
        EXPECT_GE(a.x, 0.0f);
        EXPECT_LT(a.x, 1.0f);
        EXPECT_GE(a.y, 0.0f);
        EXPECT_LT(a.y, 1.0f);
    }
    // A repeat inside the temporal filter's own window (~22 frames at
    // alpha 0.1) would make the EMA converge to a fixed quadrature of the scene
    // rather than to the scene.
    float closest = 1e9f;
    for (std::uint32_t a = 0; a < 64; ++a) {
        for (std::uint32_t b = a + 1; b < 64; ++b) {
            closest = std::min(closest, glm::length(screen_probe_jitter(a) - screen_probe_jitter(b)));
        }
    }
    EXPECT_GT(closest, 0.005f) << "two frames within the EMA window sample the same sub-texel point";
}

TEST(ScreenProbeDirections, TexelsCoverTheWholeSphereAndStayInTheirOwnTexel) {
    // Every direction is unit length, every texel maps back to itself under the
    // octahedral encode, and the set spans the sphere (a probe has no normal —
    // one probe serves a whole tile, whose pixels do not share one).
    float min_y = 1.0f;
    float max_y = -1.0f;
    for (std::uint32_t frame : {0u, 1u, 7u, 1000u}) {
        for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
            const glm::vec3 d = screen_probe_texel_direction(t, frame);
            ASSERT_NEAR(glm::length(d), 1.0f, 1e-5f);
            min_y = std::min(min_y, d.y);
            max_y = std::max(max_y, d.y);
            // The jitter must stay INSIDE the texel: a direction that leaks
            // into a neighbour's cell is sampled twice and its own never.
            const glm::vec2 oct = oct_encode(d) * static_cast<float>(SCREEN_PROBE_OCT);
            const glm::ivec2 cell = glm::ivec2(glm::floor(oct));
            EXPECT_EQ(cell.x, t % SCREEN_PROBE_OCT) << "texel " << t << " frame " << frame;
            EXPECT_EQ(cell.y, t / SCREEN_PROBE_OCT) << "texel " << t << " frame " << frame;
        }
    }
    EXPECT_LT(min_y, -0.8f);
    EXPECT_GT(max_y, 0.8f);
}

TEST(ScreenProbeDirections, HalfTheSetIsTracedForAnyAnchorNormal) {
    // The hemisphere restriction is a cost claim as well as a correctness one:
    // it is what makes a probe 32 rays and not 64.
    const glm::vec3 normals[] = {
        {0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {0.577f, 0.577f, 0.577f}, {-0.6f, 0.8f, 0.0f}};
    for (const glm::vec3& n : normals) {
        int traced = 0;
        for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
            if (screen_probe_traces_texel(screen_probe_texel_direction(t, 3), glm::normalize(n))) {
                ++traced;
            }
        }
        EXPECT_GE(traced, SCREEN_PROBE_RAYS / 4);
        EXPECT_LE(traced, 3 * SCREEN_PROBE_RAYS / 4);
    }
}

// ===========================================================================
// The march's step function, against a brute-force scan
// ===========================================================================

TEST(ScreenTraceStep, LandsOnTheNEARESTCellBoundaryAhead) {
    // The one piece of the march that is pure arithmetic, and the one where an
    // off-by-one either hangs the loop (revisiting its own cell) or tunnels
    // (skipping one). Checked against a fine scan of the ray rather than
    // against a second closed form.
    std::mt19937 rng(20260802);
    std::uniform_real_distribution<float> unit(0.02f, 0.98f);
    std::uniform_real_distribution<float> dir(-1.5f, 1.5f);
    const glm::vec2 grids[] = {{1600, 900}, {200, 113}, {50, 29}, {13, 8}, {3, 2}};

    for (const glm::vec2& cells : grids) {
        for (int trial = 0; trial < 400; ++trial) {
            const glm::vec2 p{unit(rng), unit(rng)};
            glm::vec2 d{dir(rng), dir(rng)};
            if (std::abs(d.x) < 1e-3f && std::abs(d.y) < 1e-3f) {
                d = {1.0f, 0.0f};
            }
            // A p sitting within a hundredth of a cell of a boundary is skipped,
            // and the reason is float32 cancellation rather than tolerance for
            // a bug: `(boundary - p)` there subtracts two numbers that agree to
            // six digits, so the QUOTIENT carries a percent-level relative
            // error however exact the formula is. The step is a hundredth of a
            // texel in that case and the shader's epsilon swamps it. Testing it
            // would pin a rounding mode, not an algorithm.
            const glm::dvec2 exact = glm::dvec2(p) * glm::dvec2(cells);
            if (std::abs(exact.x - std::round(exact.x)) < 0.01 ||
                std::abs(exact.y - std::round(exact.y)) < 0.01) {
                continue;
            }
            const float t = screen_trace_cell_step(p, d, cells);
            ASSERT_TRUE(std::isfinite(t)) << "a ray with a nonzero direction always crosses";
            ASSERT_GE(t, 0.0f);

            const glm::ivec2 start =
                glm::ivec2(glm::floor(glm::dvec2(p) * glm::dvec2(cells)));

            // (a) NOTHING EARLIER: every sample strictly before t is still in
            //     the starting cell. This is what stops the march tunnelling.
            //
            //     Scanned in DOUBLE. The quantity under test is the step value,
            //     not float32's evaluation of the ray: at a 1600-cell grid a
            //     boundary is not exactly representable, so a float scan
            //     straddles it by a fraction of a texel and reports a crossing
            //     the step function did not make. (The shader's own epsilon is
            //     what covers that on the GPU, and (b) below is what pins it.)
            const glm::dvec2 pd(p);
            const glm::dvec2 dd(d);
            const glm::dvec2 cd(cells);
            //     The last 1% of the interval is excluded because `t` itself is
            //     a float32 quotient of a near-cancelling subtraction. A
            //     wrong-PLANE bug — the failure this catches — overshoots by a
            //     whole cell, i.e. roughly doubles t, not by a hundredth of it.
            const int samples = 512;
            for (int i = 1; i < samples; ++i) {
                const double ti = static_cast<double>(t) * 0.99 *
                                  (static_cast<double>(i) / static_cast<double>(samples));
                const glm::ivec2 cell = glm::ivec2(glm::floor((pd + dd * ti) * cd));
                ASSERT_EQ(cell, start) << "crossed a boundary before the reported step";
            }

            // (a2) IT IS A BOUNDARY: p + d*t lands on a cell edge on at least
            //      one axis. Together with (a) this pins WHICH plane the step
            //      chose, which is the thing a scan alone cannot distinguish
            //      from "some later boundary".
            const glm::dvec2 landed = (pd + dd * static_cast<double>(t)) * cd;
            const double off_x = std::abs(landed.x - std::round(landed.x));
            const double off_y = std::abs(landed.y - std::round(landed.y));
            EXPECT_LT(std::min(off_x, off_y), 1e-3)
                << "the step did not land on a cell boundary at all";

            // (b) SOMETHING AT t: a nudge past it leaves the cell. The nudge is
            //     the same shape the shader uses (a fraction of a texel along
            //     the ray), so this pins the epsilon's job as well as the step's.
            const float eps = 0.5f / std::max(glm::length(d * cells), 1e-6f);
            const glm::ivec2 after = glm::ivec2(
                glm::floor((glm::dvec2(p) + glm::dvec2(d) * static_cast<double>(t + eps)) *
                           glm::dvec2(cells)));
            EXPECT_NE(after, start) << "the step plus the march's epsilon did not leave the cell";
        }
    }
}

TEST(ScreenTraceStep, AnAxisWithNoMotionNeverDecidesTheStep) {
    const glm::vec2 cells{100.0f, 50.0f};
    const glm::vec2 p{0.315f, 0.717f};
    // Pure-X motion: the answer must come from the X boundary alone.
    const float tx = screen_trace_cell_step(p, {0.25f, 0.0f}, cells);
    ASSERT_TRUE(std::isfinite(tx));
    EXPECT_NEAR(std::floor((p.x + 0.25f * tx) * cells.x), (p.x + 0.25f * tx) * cells.x, 1e-3f);
    // Both axes still: the ray cannot advance at all, and the caller must see
    // that rather than a plausible small number.
    EXPECT_FALSE(std::isfinite(screen_trace_cell_step(p, {0.0f, 0.0f}, cells)));
}

TEST(ScreenTraceStep, ReversedZLinearisationAgreesWithTheProjectionItCameFrom) {
    // near/far swapped, glm RH_ZO — the engine's convention (CLAUDE.md).
    const float near_plane = 10.0f;
    const float far_plane = 40000.0f;
    const glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, far_plane, near_plane);
    const float a = proj[2][2];
    const float b = proj[3][2];
    for (float view_z : {20.0f, 150.0f, 1000.0f, 12000.0f}) {
        // Project a point at -view_z (RH: the camera looks down -Z).
        const glm::vec4 clip = proj * glm::vec4(0.0f, 0.0f, -view_z, 1.0f);
        const float device = clip.z / clip.w;
        EXPECT_NEAR(screen_trace_linear_depth(device, a, b), view_z, view_z * 1e-3f);
    }
    // The cleared far plane (sky) must not produce a NaN that propagates.
    EXPECT_TRUE(std::isfinite(screen_trace_linear_depth(0.0f, a, b)));
}

TEST(ScreenTraceThickness, IsAFloorPlusAFractionAndNeverZero) {
    EXPECT_FLOAT_EQ(screen_trace_thickness(0.0f), SCREEN_TRACE_THICKNESS_MIN);
    EXPECT_FLOAT_EQ(screen_trace_thickness(100.0f), SCREEN_TRACE_THICKNESS_MIN);
    // Past the crossover the relative term takes over: at 40 m (4000 units) the
    // slab is 200 units, not 50.
    EXPECT_FLOAT_EQ(screen_trace_thickness(4000.0f), 4000.0f * SCREEN_TRACE_THICKNESS_REL);
    EXPECT_GT(screen_trace_thickness(-4000.0f), 0.0f);
}

// ===========================================================================
// The hierarchical march itself, mirrored and checked against a brute marcher
// ===========================================================================
//
// The shader's loop transliterated over a synthetic pyramid, against a
// texel-by-texel mip-0 scan of the same data. The claim under test is the only
// one that matters for a CONSERVATIVE pyramid: the hierarchical walk must never
// step OVER a surface the brute walk finds. It is allowed to stop earlier (the
// mip-0 confirmation is the same in both), and it is allowed to run out of
// steps — a handoff is not a wrong answer, it is the BVH's turn.

struct Pyramid {
    int width = 0;
    int height = 0;
    int levels = 0;
    std::vector<std::vector<float>> mips; // reversed-Z: larger == closer

    float at(int level, int x, int y) const {
        const int w = hzb_mip_size(width, level);
        const int h = hzb_mip_size(height, level);
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        return mips[static_cast<std::size_t>(level)][static_cast<std::size_t>(y * w + x)];
    }
};

/// Build the MAX pyramid exactly as hzb_reduce.comp.hlsl does, through
/// hzb.hpp's own span/reduce functions — so this test cannot drift from the
/// pyramid the GPU actually builds.
Pyramid build_pyramid(int width, int height, const std::vector<float>& mip0) {
    Pyramid p;
    p.width = width;
    p.height = height;
    p.levels = hzb_mip_count(width, height);
    p.mips.resize(static_cast<std::size_t>(p.levels));
    p.mips[0] = mip0;
    for (int level = 1; level < p.levels; ++level) {
        const int sw = hzb_mip_size(width, level - 1);
        const int sh = hzb_mip_size(height, level - 1);
        const int dw = hzb_mip_size(width, level);
        const int dh = hzb_mip_size(height, level);
        std::vector<float> dst(static_cast<std::size_t>(dw * dh), 0.0f);
        for (int y = 0; y < dh; ++y) {
            for (int x = 0; x < dw; ++x) {
                const HzbSpan sx = hzb_source_span(x, dw, sw);
                const HzbSpan sy = hzb_source_span(y, dh, sh);
                float acc = -1e30f;
                for (int v = sy.begin; v < sy.end; ++v) {
                    for (int u = sx.begin; u < sx.end; ++u) {
                        acc = hzb_reduce(acc, p.mips[static_cast<std::size_t>(level - 1)]
                                                  [static_cast<std::size_t>(v * sw + u)]);
                    }
                }
                dst[static_cast<std::size_t>(y * dw + x)] = acc;
            }
        }
        p.mips[static_cast<std::size_t>(level)] = std::move(dst);
    }
    return p;
}

/// The shader's ladder, transliterated. Returns the screen parameter it stopped
/// at, whether it converged to mip 0 inside the screen, and the last position it
/// PROVED clear (what a handoff resumes from).
struct MarchResult {
    bool converged = false;
    float s = 0.0f;
    float s_clear = 0.0f;
    int steps = 0;
};

MarchResult hierarchical_march(const Pyramid& p, glm::vec3 P0, glm::vec3 D, int start_mip) {
    MarchResult r;
    const glm::vec2 screen(static_cast<float>(p.width), static_cast<float>(p.height));
    const float texel_length = glm::length(glm::vec2(D) * screen);
    if (texel_length <= 1.0f) {
        return r;
    }
    const float eps = 0.25f / texel_length;
    float s = std::min(2.0f / texel_length, 1.0f);
    float s_clear = s;
    int level = start_mip;

    for (int iter = 0; iter < SCREEN_TRACE_MAX_STEPS && level >= 0; ++iter) {
        r.steps = iter + 1;
        const glm::vec2 uv = glm::vec2(P0) + glm::vec2(D) * s;
        if (s >= 1.0f || uv.x < 0.0f || uv.y < 0.0f || uv.x > 1.0f || uv.y > 1.0f) {
            r.s_clear = s_clear;
            return r;
        }
        const glm::ivec2 mip_size(hzb_mip_size(p.width, level), hzb_mip_size(p.height, level));
        const glm::vec2 cells = screen_trace_cell_grid({p.width, p.height}, level);
        const glm::vec2 cell_before = glm::floor(uv * cells);
        const glm::ivec2 lookup = screen_trace_cell(uv, cells, mip_size);
        const float hz = p.at(level, lookup.x, lookup.y);

        // Reversed-Z over a MAX pyramid: the ray is in front of everything in
        // this footprint, so nothing here can be hit and the position is proven
        // clear.
        const bool ahead = hz <= P0.z + D.z * s;
        if (ahead) {
            s_clear = s;
        }
        float next = s;
        if (ahead && std::abs(D.z) > 1e-9f) {
            next = std::max(s, (hz - P0.z) / D.z);
        }
        const glm::vec2 cell_after = glm::floor((glm::vec2(P0) + glm::vec2(D) * next) * cells);
        // The depth plane can only advance a RECEDING ray; a ray travelling
        // toward the camera (D.z > 0 in reversed-Z) has the max() above clamp it
        // to s, so the cell boundary is its only forward progress. Stepping it
        // is exactly conservative — rayZ only rises, so a ray in front of this
        // cell's closest surface stays in front across the whole cell. Extending
        // it to !ahead would tunnel: that is the descend-to-refine path.
        if (cell_after != cell_before || (ahead && D.z > -1e-9f)) {
            const float dt = screen_trace_cell_step(uv, glm::vec2(D), cells);
            if (!std::isfinite(dt)) {
                r.s_clear = s_clear;
                return r;
            }
            next = s + dt + eps;
            level = std::min(level + 2, p.levels - 1);
        }
        s = next;
        --level;
    }
    r.s_clear = s_clear;
    if (level < 0 && s < 1.0f) {
        const glm::vec2 uv = glm::vec2(P0) + glm::vec2(D) * s;
        if (uv.x >= 0.0f && uv.y >= 0.0f && uv.x <= 1.0f && uv.y <= 1.0f) {
            r.converged = true;
            r.s = s;
        }
    }
    return r;
}

/// How far behind a surface the brute walk needs the ray before it calls the
/// sample a crossing, in device depth.
///
/// NOT tolerance for a bug. The ladder's test is per CELL — it advances only
/// where the cell's MAX depth is at or behind the ray — while the reference
/// point-samples mip 0, and the two can only disagree where the ray runs
/// TANGENT to a smooth surface and dips a ten-thousandth of a unit under it.
/// There the "first crossing" is a coin flip between two discretizations, and
/// the shader's thickness heuristic is what decides anyway. A real occluder in
/// this fixture stands 0.2..0.7 proud of its surroundings, so the margin costs
/// the reference nothing it was pinning.
constexpr float BRUTE_CROSSING_MARGIN = 0.005f;

/// A texel-at-a-time mip-0 walk: the reference the ladder must not undercut.
float brute_first_crossing(const Pyramid& p, glm::vec3 P0, glm::vec3 D) {
    const glm::vec2 screen(static_cast<float>(p.width), static_cast<float>(p.height));
    const float texel_length = glm::length(glm::vec2(D) * screen);
    if (texel_length <= 1.0f) {
        return -1.0f;
    }
    const int steps = static_cast<int>(std::ceil(texel_length * 2.0f));
    const float start = std::min(2.0f / texel_length, 1.0f);
    for (int i = 0; i <= steps; ++i) {
        const float s = start + (1.0f - start) * (static_cast<float>(i) / static_cast<float>(steps));
        if (s >= 1.0f) {
            break;
        }
        const glm::vec2 uv = glm::vec2(P0) + glm::vec2(D) * s;
        if (uv.x < 0.0f || uv.y < 0.0f || uv.x > 1.0f || uv.y > 1.0f) {
            return -1.0f; // left the screen: a handoff, not a hit
        }
        const int x = std::clamp(static_cast<int>(uv.x * screen.x), 0, p.width - 1);
        const int y = std::clamp(static_cast<int>(uv.y * screen.y), 0, p.height - 1);
        const float ray = P0.z + D.z * s;
        if (ray < p.at(0, x, y) - BRUTE_CROSSING_MARGIN) { // reversed-Z: ray behind the surface
            return s;
        }
    }
    return -1.0f;
}

/// The test fixture: a ground plane whose depth falls off with y, plus a couple
/// of surfaces that jump much closer. Reversed-Z, so larger is closer.
Pyramid build_scene_pyramid(int W, int H) {
    std::vector<float> mip0(static_cast<std::size_t>(W * H));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float d = 0.2f + 0.5f * (static_cast<float>(y) / static_cast<float>(H));
            if (x > 40 && x < 52) {
                d = 0.92f; // a near wall
            }
            if (y > 50 && y < 56 && x > 80) {
                d = 0.85f; // a near ledge
            }
            mip0[static_cast<std::size_t>(y * W + x)] = d;
        }
    }
    return build_pyramid(W, H, mip0);
}

TEST(ScreenTraceMarch, NeverStepsOverASurfaceTheBruteWalkFinds) {
    const int W = 128;
    const int H = 72;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> unit(0.05f, 0.95f);
    std::uniform_real_distribution<float> delta(-1.2f, 1.2f);
    // DEPTH SYMMETRICALLY, and this is not a detail: a fixed D.z of -0.9 covers
    // only rays travelling AWAY from the camera, and the toward-camera half —
    // which for a surface facing the eye is its whole upper hemisphere — went
    // unchecked for a release. Every failure the ladder can have is available in
    // both halves; only one of them was being asked for.
    std::uniform_real_distribution<float> dz(-1.2f, 1.2f);
    std::uniform_real_distribution<float> depth0(0.3f, 0.9f);

    const Pyramid p = build_scene_pyramid(W, H);

    int compared = 0;
    int toward = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const glm::vec3 P0{unit(rng), unit(rng), depth0(rng)};
        glm::vec3 D{delta(rng), delta(rng), dz(rng)};
        if (std::abs(D.x) < 1e-3f && std::abs(D.y) < 1e-3f) {
            continue;
        }
        const float brute = brute_first_crossing(p, P0, D);
        const MarchResult hier = hierarchical_march(p, P0, D, SCREEN_TRACE_START_MIP);
        if (brute < 0.0f) {
            continue; // the reference itself found nothing to compare against
        }
        ++compared;
        if (D.z > 0.0f) {
            ++toward;
        }
        // The ladder may stop short of the brute crossing (it converges on the
        // first mip-0 cell it cannot rule out, and the thickness test in the
        // shader is what decides whether that is a hit). It must NOT sail past
        // it: that is the tunnelling failure a conservative pyramid exists to
        // prevent. Two texels of slack for the epsilon nudges.
        const float slack = 2.0f / glm::length(glm::vec2(D) * glm::vec2(W, H));
        if (hier.converged) {
            EXPECT_LE(hier.s, brute + slack)
                << "the hierarchical march converged past the first surface (trial " << trial << ")";
        }
        // AND THE HANDOFF MUST NOT EITHER. `s_clear` is what the shader gives
        // the BVH as its resume distance on a reject, and the whole point of
        // carrying it separately from `s` is that resuming past a surface the
        // march itself found makes that surface invisible to both estimators.
        EXPECT_LE(hier.s_clear, brute + slack)
            << "the handoff resumed past the first surface (trial " << trial << ")";
        EXPECT_LE(hier.s_clear, hier.s) << "the resume overtook the march itself";
        EXPECT_LE(hier.steps, SCREEN_TRACE_MAX_STEPS);
    }
    EXPECT_GT(compared, 500) << "the fixture produced too few comparable rays to mean anything";
    EXPECT_GT(toward, 200) << "the toward-camera half of the ray set is not being exercised";
}

TEST(ScreenTraceMarch, AdvancesRaysTravellingTowardTheCamera) {
    // THE PIN FOR THE STALL. In reversed-Z a ray travelling toward the camera
    // has D.z > 0, so `hz <= rayZ` implies `(hz - P0.z)/D.z <= s` and the
    // depth-plane advance's max() clamps it to s. Without the cell-boundary
    // fallback such a ray never moves: it burns its three levels refining the
    // texel it started in, reports "converged" two texels from the anchor, and
    // is handed to the BVH from there. Measured on this fixture, the shipped
    // ladder landed on 0 of ~800 crossings that stand more than eight texels
    // ahead; with the fallback it lands on ~98% of them.
    const int W = 128;
    const int H = 72;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> unit(0.05f, 0.95f);
    std::uniform_real_distribution<float> delta(-1.2f, 1.2f);
    std::uniform_real_distribution<float> dz(0.05f, 1.2f); // toward the camera only
    std::uniform_real_distribution<float> depth0(0.3f, 0.9f);

    const Pyramid p = build_scene_pyramid(W, H);

    int far = 0;
    int landed = 0;
    for (int trial = 0; trial < 6000; ++trial) {
        const glm::vec3 P0{unit(rng), unit(rng), depth0(rng)};
        const glm::vec3 D{delta(rng), delta(rng), dz(rng)};
        const float texels = glm::length(glm::vec2(D) * glm::vec2(W, H));
        if (texels <= 1.0f) {
            continue;
        }
        const float brute = brute_first_crossing(p, P0, D);
        if (brute < 0.0f) {
            continue;
        }
        // Only crossings the ladder has to TRAVEL to; one within a couple of
        // texels of the start is found by standing still.
        const float start = std::min(2.0f / texels, 1.0f);
        if ((brute - start) * texels < 8.0f) {
            continue;
        }
        ++far;
        const MarchResult hier = hierarchical_march(p, P0, D, SCREEN_TRACE_START_MIP);
        if (hier.converged && std::abs(hier.s - brute) * texels <= 4.0f) {
            ++landed;
        }
    }
    ASSERT_GT(far, 200) << "the fixture produced too few distant crossings to mean anything";
    EXPECT_GT(landed, (far * 9) / 10)
        << "toward-camera rays are not reaching the surfaces ahead of them: " << landed << " of "
        << far;
}

TEST(ScreenTraceMarch, TheHandoffResumesBeforeTheSurfaceTheMarchStoppedAt) {
    // The ladder's last advance lands the ray ON a surface's depth plane. If the
    // thickness test then rejects it, or the reprojection cannot be read, the
    // ray is handed to the BVH — and handing it off AT that plane makes the very
    // geometry the march declined to adjudicate invisible to both estimators
    // (the BVH's own ray epsilon rejects a t at the origin). `s_clear` is the
    // last sample at which the ray was proven in front of everything in its
    // footprint, which is the position a reject may resume from.
    //
    // Measured on this fixture over rays that travelled and then landed on a
    // real crossing: the resume sits at least half a texel short of it for 91%
    // of them, against 29% if the stop position is handed off instead.
    const int W = 128;
    const int H = 72;
    std::mt19937 rng(20260809);
    std::uniform_real_distribution<float> unit(0.05f, 0.95f);
    std::uniform_real_distribution<float> delta(-1.2f, 1.2f);
    std::uniform_real_distribution<float> depth0(0.3f, 0.9f);

    const Pyramid p = build_scene_pyramid(W, H);

    int landed = 0;
    int backed_off = 0;
    for (int trial = 0; trial < 8000; ++trial) {
        const glm::vec3 P0{unit(rng), unit(rng), depth0(rng)};
        const glm::vec3 D{delta(rng), delta(rng), delta(rng)};
        const float texels = glm::length(glm::vec2(D) * glm::vec2(W, H));
        if (texels <= 1.0f) {
            continue;
        }
        const float brute = brute_first_crossing(p, P0, D);
        const MarchResult hier = hierarchical_march(p, P0, D, SCREEN_TRACE_START_MIP);
        if (brute < 0.0f || !hier.converged) {
            continue;
        }
        const float start = std::min(2.0f / texels, 1.0f);
        // Only rays that actually walked to a surface: one that converged where
        // it started never advanced onto a plane and has nothing to back off.
        if ((hier.s - start) * texels < 4.0f || std::abs(hier.s - brute) * texels > 4.0f) {
            continue;
        }
        ++landed;
        if ((brute - hier.s_clear) * texels > 0.5f) {
            ++backed_off;
        }
    }
    ASSERT_GT(landed, 200) << "the fixture produced too few walked-to crossings to mean anything";
    EXPECT_GT(backed_off, (landed * 7) / 10)
        << "the handoff is being taken at the surface rather than before it: " << backed_off
        << " of " << landed;
}

TEST(ScreenTraceMarch, TerminatesOnEveryRayItIsGiven) {
    // A hang here is a GPU hang. The step budget is the guarantee, and this
    // pins that no path through the ladder can spin without consuming one.
    const int W = 64;
    const int H = 64;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> depth(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> delta(-2.0f, 2.0f);

    std::vector<float> mip0(static_cast<std::size_t>(W * H));
    for (float& d : mip0) {
        d = depth(rng);
    }
    const Pyramid p = build_pyramid(W, H, mip0);

    for (int trial = 0; trial < 5000; ++trial) {
        const glm::vec3 P0{unit(rng), unit(rng), unit(rng)};
        const glm::vec3 D{delta(rng), delta(rng), delta(rng) * 0.5f};
        const MarchResult r = hierarchical_march(p, P0, D, SCREEN_TRACE_START_MIP);
        ASSERT_LE(r.steps, SCREEN_TRACE_MAX_STEPS);
    }
}

// ===========================================================================
// L1 spherical harmonics
// ===========================================================================

TEST(ShL1, UniformSphereReproducesItsOwnRadiance) {
    // A probe that sees the same radiance in every direction must report that
    // radiance as its irradiance/pi, whatever normal it is asked about. This is
    // the unit check: get the 1 and the 2/3 wrong and everything is uniformly
    // too bright or too dim, which reads as a tuning problem for a week.
    glm::vec3 coeffs[4] = {};
    const glm::vec3 radiance{0.3f, 0.6f, 0.9f};
    for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
        sh_l1_accumulate(coeffs, screen_probe_texel_direction(t, 0), radiance,
                         screen_probe_texel_solid_angle(t, 0));
    }
    for (const glm::vec3& n : {glm::vec3{0, 1, 0}, glm::vec3{1, 0, 0}, glm::vec3{0, 0, -1},
                               glm::normalize(glm::vec3{1, 1, 1})}) {
        const glm::vec3 e = sh_l1_irradiance_over_pi(coeffs, n);
        EXPECT_NEAR(e.r, radiance.r, 0.02f);
        EXPECT_NEAR(e.g, radiance.g, 0.02f);
        EXPECT_NEAR(e.b, radiance.b, 0.02f);
    }
}

TEST(ShL1, UniformHemisphereIsExactForTheAnchorNormal) {
    // THE JUSTIFICATION FOR TRACING ONLY HALF THE TEXELS. Zero radiance below
    // the anchor's horizon, projected onto L1 and convolved with the cosine
    // lobe, reproduces the hemisphere integral EXACTLY for that normal — so the
    // 32 untraced rays are not an approximation, they are a saving.
    const glm::vec3 normal{0.0f, 1.0f, 0.0f};
    const glm::vec3 radiance{1.0f, 1.0f, 1.0f};
    glm::vec3 coeffs[4] = {};
    for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
        const glm::vec3 d = screen_probe_texel_direction(t, 0);
        const glm::vec3 value = screen_probe_traces_texel(d, normal) ? radiance : glm::vec3(0.0f);
        sh_l1_accumulate(coeffs, d, value, screen_probe_texel_solid_angle(t, 0));
    }
    const glm::vec3 e = sh_l1_irradiance_over_pi(coeffs, normal);
    // E = integral of L cos over the hemisphere = pi * L, so E/pi = L = 1.
    EXPECT_NEAR(e.r, 1.0f, 0.03f);
    EXPECT_NEAR(e.g, 1.0f, 0.03f);
    EXPECT_NEAR(e.b, 1.0f, 0.03f);
}

TEST(ShL1, TheTexelSolidAngleSumsToTheWholeSphereButIsNotUniform) {
    constexpr float FOUR_PI = 4.0f * 3.14159265358979323846f;
    constexpr float UNIFORM = FOUR_PI / static_cast<float>(SCREEN_PROBE_RAYS);

    // The SUM is 4*pi, which the old constant weight also had — and that is
    // precisely why a uniform sphere and a uniform hemisphere both reproduced
    // and hid the fact that the DISTRIBUTION was wrong.
    float centres = 0.0f;
    float lo = 1e9f;
    float hi = -1e9f;
    for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
        const float sa = screen_probe_texel_solid_angle(t, 0);
        ASSERT_GT(sa, 0.0f);
        centres += sa;
        lo = std::min(lo, sa / UNIFORM);
        hi = std::max(hi, sa / UNIFORM);
    }
    EXPECT_NEAR(centres, FOUR_PI, 1e-3f);
    // A jittered frame samples the Jacobian off-centre, so its sum wanders by
    // about a percent; the expectation over the jitter sequence is the exact
    // solid angle, and the temporal EMA is what takes that expectation.
    for (const std::uint32_t f : {1u, 7u, 33u, 1000u}) {
        float sum = 0.0f;
        for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
            sum += screen_probe_texel_solid_angle(t, f);
        }
        EXPECT_NEAR(sum, FOUR_PI, 0.02f * FOUR_PI) << "frame " << f;
    }
    // dOmega = dA/|v|^3 with |v| from 1/sqrt(3) to 1, so the true weight is
    // roughly half the uniform one at the world axes and half again as much on
    // the face diagonals. This is the assertion the "equal-area by
    // construction" comment claimed was unnecessary.
    EXPECT_LT(lo, 0.6f);
    EXPECT_GT(hi, 1.4f);
}

TEST(ShL1, AConcentratedSourceReconstructsTheSameOnAWorldAxisAsOnADiagonal) {
    // THE DEFECT THE UNIFORM WEIGHT HAD, stated as a property. The octahedral
    // map is applied to WORLD directions with no rotation, so a per-texel weight
    // error is fixed to the world axes: with the constant 4*pi/64 the same
    // source reconstructs at 2.15x the correct L1 answer about +Y and 0.62x
    // about (1,1,1)/sqrt(3). With the true solid angle the only error left is
    // the L1 truncation, which is rotation-invariant — so the two ratios agree.
    constexpr float HALF_ANGLE = 25.0f * 3.14159265358979323846f / 180.0f;
    const float cos_half = std::cos(HALF_ANGLE);
    // E(n) for a cap of half-angle theta about n is pi*sin^2(theta), so E/pi is
    // sin^2(theta) — an exact reference, no quadrature.
    const float reference = std::sin(HALF_ANGLE) * std::sin(HALF_ANGLE);
    constexpr int FRAMES = 64; // what the temporal EMA integrates over

    const auto ratio_about = [&](const glm::vec3& axis) {
        glm::vec3 coeffs[4] = {};
        for (std::uint32_t f = 0; f < FRAMES; ++f) {
            for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
                const glm::vec3 d = screen_probe_texel_direction(t, f);
                const float value = glm::dot(d, axis) >= cos_half ? 1.0f : 0.0f;
                sh_l1_accumulate(coeffs, d, glm::vec3(value),
                                 screen_probe_texel_solid_angle(t, f) /
                                     static_cast<float>(FRAMES));
            }
        }
        return sh_l1_irradiance_over_pi(coeffs, axis).r / reference;
    };

    const float on_axis = ratio_about({0.0f, 1.0f, 0.0f});
    const float on_diagonal = ratio_about(glm::normalize(glm::vec3{1.0f, 1.0f, 1.0f}));
    // ~0.76 both ways: an L1 fit cannot hold a 25-degree cap, and that shortfall
    // is inherent, isotropic and not what this test is about.
    EXPECT_GT(on_axis, 0.6f);
    EXPECT_LT(on_axis, 0.9f);
    EXPECT_NEAR(on_axis, on_diagonal, 0.08f)
        << "the reconstruction depends on where in the octahedral map the source "
           "landed: " << on_axis << " vs " << on_diagonal;
}

TEST(ShL1, ADirectionalLobeProjectsBackWhereItCameFrom) {
    // Project a cosine lobe about +X; the reconstruction must peak facing +X,
    // be near zero facing -X, and never go negative (an L1 fit to a one-sided
    // distribution does, and a negative irradiance is a hole).
    const glm::vec3 axis = glm::normalize(glm::vec3{1.0f, 0.2f, -0.3f});
    glm::vec3 coeffs[4] = {};
    for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
        const glm::vec3 d = screen_probe_texel_direction(t, 5);
        const float lobe = std::max(0.0f, glm::dot(d, axis));
        sh_l1_accumulate(coeffs, d, glm::vec3(lobe), screen_probe_texel_solid_angle(t, 5));
    }
    const float along = sh_l1_irradiance_over_pi(coeffs, axis).r;
    const float against = sh_l1_irradiance_over_pi(coeffs, -axis).r;
    const float across = sh_l1_irradiance_over_pi(coeffs, glm::normalize(glm::cross(axis, {0, 1, 0}))).r;
    EXPECT_GT(along, across);
    EXPECT_GT(across, against);
    EXPECT_GE(against, 0.0f) << "the clamp is what stops a negative irradiance";

    // Sweep the sphere: nothing negative, and the maximum is within a few
    // degrees of the lobe axis.
    float best = -1.0f;
    glm::vec3 best_dir{0.0f};
    for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
        const glm::vec3 d = screen_probe_texel_direction(t, 9);
        const float v = sh_l1_irradiance_over_pi(coeffs, d).r;
        EXPECT_GE(v, 0.0f);
        if (v > best) {
            best = v;
            best_dir = d;
        }
    }
    EXPECT_GT(glm::dot(best_dir, axis), 0.8f);
}

TEST(ShL1, BasisMatchesTheStoredCoefficientOrder) {
    // (Y00, Y1-1, Y10, Y11) — the order the SH record stores and the shader's
    // giSpShBasis reproduces. A permutation here rotates every probe's lobe.
    const glm::vec4 y = sh_l1_basis({0.0f, 1.0f, 0.0f});
    EXPECT_NEAR(y[0], 0.28209479f, 1e-6f);
    EXPECT_NEAR(y[1], 0.48860251f, 1e-6f); // Y1-1 carries y
    EXPECT_NEAR(y[2], 0.0f, 1e-6f);
    EXPECT_NEAR(y[3], 0.0f, 1e-6f);
    const glm::vec4 z = sh_l1_basis({0.0f, 0.0f, 1.0f});
    EXPECT_NEAR(z[2], 0.48860251f, 1e-6f); // Y10 carries z
    const glm::vec4 x = sh_l1_basis({1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(x[3], 0.48860251f, 1e-6f); // Y11 carries x
}

// ===========================================================================
// The fused spatial filter (gi_screen_probe_project.comp.hlsl)
// ===========================================================================

TEST(ScreenProbeSpatialFilter, DoesNotAverageATexelAgainstANeighboursBelowHorizonZero) {
    // The 3x3 gather is PER DIRECTION: thread i averages texel i across nine
    // probes. But the march traces only the hemisphere above each anchor and
    // writes a RESOLVED zero elsewhere, so two anchors whose normals differ by
    // theta disagree about the horizon over a band sin(theta) of the hemisphere
    // wide — 42% of it at 25 degrees, where cos^8 still passes the weight test
    // comfortably. Folding a neighbour's structural zero in at full weight is a
    // fabricated sample, and it darkens the tier on exactly the geometry
    // (detailed, normal-varying) the screen tier exists to serve.
    const glm::vec3 anchor_normal{0.0f, 1.0f, 0.0f};
    constexpr float TILT = 25.0f * 3.14159265358979323846f / 180.0f;
    constexpr int FRAMES = 64;

    std::vector<glm::vec3> neighbours;
    for (int k = 0; k < 8; ++k) {
        const float phi = static_cast<float>(k) * 3.14159265358979323846f / 4.0f;
        neighbours.push_back(glm::normalize(glm::vec3{std::sin(TILT) * std::cos(phi),
                                                      std::cos(TILT),
                                                      std::sin(TILT) * std::sin(phi)}));
    }
    // Sky above, dim ground below: the field a probe on open ground sees, and
    // the one that makes a fabricated zero cost the most.
    const auto field = [](const glm::vec3& d) { return d.y > 0.0f ? 1.0f : 0.1f; };

    const auto reconstruct = [&](bool guard) {
        glm::vec3 coeffs[4] = {};
        for (std::uint32_t f = 0; f < FRAMES; ++f) {
            for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
                const glm::vec3 d = screen_probe_texel_direction(t, f);
                float radiance = 0.0f;
                if (!guard || screen_probe_traces_texel(d, anchor_normal)) {
                    // The centre tap: its own sample at weight 1, which is zero
                    // below its own horizon whether or not the guard is on.
                    float acc = screen_probe_traces_texel(d, anchor_normal) ? field(d) : 0.0f;
                    float wsum = 1.0f;
                    for (const glm::vec3& n : neighbours) {
                        const bool traced = screen_probe_traces_texel(d, n);
                        if (guard && !traced) {
                            continue; // no sample here, not a black one
                        }
                        const float w = screen_probe_bilateral_weight(
                            1000.0f, 1000.0f, anchor_normal, n, SCREEN_PROBE_NORMAL_POWER);
                        if (w <= 1e-4f) {
                            continue;
                        }
                        acc += (traced ? field(d) : 0.0f) * w;
                        wsum += w;
                    }
                    radiance = wsum > 1e-6f ? acc / wsum : 0.0f;
                }
                sh_l1_accumulate(coeffs, d, glm::vec3(radiance),
                                 screen_probe_texel_solid_angle(t, f) / static_cast<float>(FRAMES));
            }
        }
        return sh_l1_irradiance_over_pi(coeffs, anchor_normal).r;
    };

    // Everything above this anchor's horizon carries radiance 1, so the answer
    // is 1 — which the guarded gather reproduces because the hemisphere
    // restriction stays exact (see ShL1.UniformHemisphereIsExactForTheAnchorNormal).
    EXPECT_NEAR(reconstruct(true), 1.0f, 0.02f);
    // The counter-case, and the reason this test exists: averaging the
    // neighbours' structural zeros in loses several percent of the tier's
    // energy, all of it on normal-varying geometry.
    EXPECT_LT(reconstruct(false), 0.96f);
}

// ===========================================================================
// The firefly clamp's sample count (gi_screen_probe_filter.comp.hlsl)
// ===========================================================================

TEST(ScreenProbeFirefly, TheClampMeanIsTakenOverTheTexelsThatCarryASample) {
    // The clamp divides by the number of LIVE samples, not by SCREEN_PROBE_RAYS.
    // Half the payload is below the anchor's horizon and is written as a
    // resolved zero — a true zero for the SH projection, which integrates the
    // whole sphere, but not a sample of anything. Dividing by 64 halves the
    // statistic and with it the ceiling SCREEN_PROBE_FIREFLY_MAX is documented
    // to be a multiple of.
    int lowest = SCREEN_PROBE_RAYS;
    int highest = 0;
    long long total = 0;
    int samples = 0;
    std::mt19937 rng(20260809);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    for (int trial = 0; trial < 400; ++trial) {
        glm::vec3 n{unit(rng), unit(rng), unit(rng)};
        if (glm::length(n) < 1e-3f) {
            continue;
        }
        n = glm::normalize(n);
        for (const std::uint32_t f : {0u, 1u, 7u, 33u}) {
            const int live = screen_probe_live_sample_count(n, f);
            lowest = std::min(lowest, live);
            highest = std::max(highest, live);
            total += live;
            ++samples;
        }
    }
    const double mean = static_cast<double>(total) / static_cast<double>(samples);
    // ~32 of 64 on average — but never exactly 32, which is why the shader
    // reduces the count alongside the luminance rather than assuming one.
    EXPECT_NEAR(mean, 0.5 * SCREEN_PROBE_RAYS, 0.5);
    EXPECT_GE(lowest, 20);
    EXPECT_LE(highest, 44);
    EXPECT_LT(lowest, SCREEN_PROBE_RAYS / 2);
    EXPECT_GT(highest, SCREEN_PROBE_RAYS / 2);

    // A probe whose lower hemisphere is all structural zeros must report the
    // mean of what it MEASURED, not half of it.
    const glm::vec3 up{0.0f, 1.0f, 0.0f};
    const int live = screen_probe_live_sample_count(up, 0);
    float sum = 0.0f;
    for (int t = 0; t < SCREEN_PROBE_RAYS; ++t) {
        sum += screen_probe_traces_texel(screen_probe_texel_direction(t, 0), up) ? 4.0f : 0.0f;
    }
    EXPECT_NEAR(sum / static_cast<float>(live), 4.0f, 1e-4f);
    EXPECT_LT(sum / static_cast<float>(SCREEN_PROBE_RAYS), 2.2f) << "the divisor that was shipped";
}

// ===========================================================================
// Miss-buffer sizing and the bilateral weight
// ===========================================================================

TEST(ScreenProbeMissBuffer, IsSizedForEveryTracedRayMissing) {
    const glm::ivec2 grid = screen_probe_grid(1600, 900);
    const int capacity = screen_probe_miss_capacity(grid);
    // Worst case: every probe valid, every traced ray handed off. Only the
    // hemisphere is traced, so that is probes x RAYS/2 and the `dropped`
    // counter should therefore read zero on every frame ever.
    EXPECT_EQ(capacity, screen_probe_count(grid) * (SCREEN_PROBE_RAYS / 2));
    EXPECT_EQ(capacity, 182400);
    // The buffer's byte size, including the 16-byte count header the shader
    // reads its record count from.
    const std::size_t bytes = bracken::gpu::GI_SCREEN_PROBE_MISS_HEADER_BYTES +
                              static_cast<std::size_t>(capacity) *
                                  sizeof(bracken::gpu::GiScreenProbeMiss);
    EXPECT_EQ(bytes, 16u + 182400u * 16u);
    EXPECT_LT(bytes, 4u * 1024u * 1024u) << "the worst case is not worth a heuristic";
}

TEST(ScreenProbeRecords, StridesAreWhatTheShadersIndexBy) {
    // Both sides of every buffer the six passes share. The header is indexed by
    // probe id in five shaders and the miss record is addressed by hand with
    // Store4/Load4 at a 16-byte pitch, so a size change here is not a wrong
    // value, it is a different probe.
    static_assert(sizeof(bracken::gpu::GiScreenProbeHeader) == 32);
    static_assert(sizeof(bracken::gpu::GiScreenProbeMiss) == 16);
    static_assert(sizeof(bracken::gpu::GiScreenProbeSh) == 64);
    static_assert(sizeof(bracken::gpu::GiScreenProbeCounters) == 16);
    static_assert(bracken::gpu::GI_SCREEN_PROBE_MISS_HEADER_BYTES ==
                  sizeof(bracken::gpu::GiScreenProbeMiss));
    SUCCEED();
}

TEST(ScreenProbeBilateral, RejectsOtherSurfacesAndAcceptsTheSameOne) {
    const glm::vec3 up{0.0f, 1.0f, 0.0f};
    const glm::vec3 side{1.0f, 0.0f, 0.0f};
    // Same plane, same distance: full weight.
    EXPECT_NEAR(screen_probe_bilateral_weight(1000.0f, 1000.0f, up, up, SCREEN_PROBE_NORMAL_POWER),
                1.0f, 1e-5f);
    // Perpendicular normals: nothing, whatever the depths say.
    EXPECT_NEAR(screen_probe_bilateral_weight(1000.0f, 1000.0f, up, side, SCREEN_PROBE_NORMAL_POWER),
                0.0f, 1e-5f);
    // RELATIVE depth tolerance: the same absolute gap is a different surface up
    // close and the same one far away.
    const float near_gap =
        screen_probe_bilateral_weight(200.0f, 260.0f, up, up, SCREEN_PROBE_NORMAL_POWER);
    const float far_gap =
        screen_probe_bilateral_weight(6000.0f, 6060.0f, up, up, SCREEN_PROBE_NORMAL_POWER);
    EXPECT_LT(near_gap, far_gap);
    EXPECT_GT(far_gap, 0.7f);
    EXPECT_LT(near_gap, 0.5f);
    // The forward gather is deliberately more permissive about normals than the
    // probe-space filter: a pixel's normal-mapped normal legitimately differs
    // from its tile anchor's geometric one.
    const glm::vec3 tilted = glm::normalize(glm::vec3{0.4f, 1.0f, 0.0f});
    EXPECT_GT(screen_probe_bilateral_weight(1000.0f, 1000.0f, up, tilted,
                                            SCREEN_PROBE_FORWARD_NORMAL_POWER),
              screen_probe_bilateral_weight(1000.0f, 1000.0f, up, tilted, SCREEN_PROBE_NORMAL_POWER));
}

} // namespace
