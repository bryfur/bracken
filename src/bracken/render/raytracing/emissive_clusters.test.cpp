#include "bracken/render/raytracing/emissive_clusters.hpp"

#include "bracken/contracts/camera_state.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/render/lighting/alias_table.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
#include "bracken/render/lighting/reservoir.hpp"

#include <gtest/gtest.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include <cmath>
#include <vector>

using bracken::render::lighting::ClusterBounds;
using bracken::render::lighting::ClusterParams;
using bracken::render::lighting::ris_source_scale;
using bracken::render::rt::emissive_cluster_table_size;
using bracken::render::rt::emissive_draw_mixture;
using bracken::render::rt::emissive_froxel_holds;
using bracken::render::rt::emissive_froxel_share;
using bracken::render::rt::emissive_mixture_pdf;
using bracken::render::rt::emissive_sample_index;
using bracken::render::rt::emissive_target;
using bracken::render::rt::EmissiveClusterGrid;
using bracken::render::rt::EmissiveTriangle;
using bracken::render::rt::k_emissive_froxel_capacity;
using bracken::render::rt::k_emissive_froxel_stride;

namespace {

EmissiveTriangle make_tri(glm::vec3 v0, glm::vec3 e1, glm::vec3 e2, glm::vec3 radiance) {
    EmissiveTriangle t;
    t.v0 = v0;
    t.e1 = e1;
    t.e2 = e2;
    t.radiance = radiance;
    t.area = 0.5f * glm::length(glm::cross(e1, e2));
    t.power = bracken::render::lighting::target_luminance(radiance) * t.area;
    return t;
}

/// Give a record set its Vose rows, exactly as RtScene::rebuild_emissive_alias
/// does — the mixture's global half reads them straight out of the records.
float build_alias_into(std::vector<EmissiveTriangle>& records) {
    std::vector<float> weights(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        weights[i] = records[i].power;
    }
    std::vector<bracken::render::lighting::AliasEntry> table;
    const float total = bracken::render::lighting::build_alias_table(weights, table);
    for (std::size_t i = 0; i < records.size(); ++i) {
        records[i].alias_prob = table[i].prob;
        records[i].alias_index = table[i].alias;
    }
    return total;
}

/// The same deterministic LCG emissive_lights.test.cpp uses, so a failure here
/// is a real failure rather than an unlucky seed.
struct Lcg {
    uint32_t s = 0x9E3779B9u;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }
};

bracken::scene::CameraState make_camera() {
    bracken::scene::CameraState c;
    c.view = glm::mat4(1.0f); // at the origin, looking down -Z
    c.projection = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 4000.0f);
    c.view_projection = c.projection * c.view;
    c.position = glm::vec3(0.0f);
    return c;
}

/// A scene of emitters spread down the view axis, dense enough that some froxel
/// somewhere exceeds the per-froxel cap.
std::vector<EmissiveTriangle> make_scene(int per_row = 40) {
    std::vector<EmissiveTriangle> records;
    for (int i = 0; i < per_row; ++i) {
        const float x = static_cast<float>(i % 8) * 3.0f - 12.0f;
        const float z = -60.0f - static_cast<float>(i / 8) * 4.0f;
        records.push_back(make_tri({x, 0.0f, z}, {2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f},
                                   {4.0f + 0.1f * static_cast<float>(i), 3.0f, 2.0f}));
    }
    return records;
}

} // namespace

// ---------------------------------------------------------------------------
// The table layout — the CPU/HLSL mirror that a screenshot cannot check
// ---------------------------------------------------------------------------

TEST(EmissiveClusterTable, LayoutMatchesTheShader) {
    // shaders/rt_emissive.hlsli spells these as RT_EMISSIVE_FROXEL_CAPACITY and
    // RT_EMISSIVE_FROXEL_STRIDE. A silent divergence here reads a froxel's block
    // at the wrong offset, which looks like "the lists are empty" rather than
    // like a bug.
    EXPECT_EQ(k_emissive_froxel_capacity, 16u);
    EXPECT_EQ(k_emissive_froxel_stride, 17u);
    EXPECT_EQ(emissive_cluster_table_size(3456), 3456u * 17u);
}

// ---------------------------------------------------------------------------
// The mixture source pdf
// ---------------------------------------------------------------------------

TEST(EmissiveMixture, PdfIsAProperDistributionOverEveryRecord) {
    // THE claim the estimator rests on: the mixture is a pmf over the WHOLE
    // record set, so every record has positive probability and the total is 1.
    // Asserted by summing it, which is the only way to catch a missing term.
    std::vector<EmissiveTriangle> records{
        make_tri({0, 0, -10}, {1, 0, 0}, {0, 1, 0}, {5, 5, 5}),
        make_tri({4, 0, -12}, {2, 0, 0}, {0, 2, 0}, {1, 1, 1}),
        make_tri({-6, 1, -20}, {3, 0, 0}, {0, 1, 0}, {9, 4, 1}),
        make_tri({30, 0, -300}, {6, 0, 0}, {0, 6, 0}, {2, 2, 2}), // the "lighthouse"
    };
    const float total = build_alias_into(records);
    const std::vector<uint32_t> list{0u, 2u}; // ascending, as the binner packs

    for (const float share : {0.0f, 0.25f, 0.75f, 1.0f}) {
        double sum = 0.0;
        for (uint32_t i = 0; i < records.size(); ++i) {
            const bool in_list = emissive_froxel_holds(list, i);
            const float p = emissive_mixture_pdf(share, static_cast<uint32_t>(list.size()), in_list,
                                                 records[i].power, total);
            EXPECT_GE(p, 0.0f);
            // Positive for EVERY record whenever any global mass is left — that
            // is the coverage guarantee the lighthouse case depends on.
            if (share < 1.0f) {
                EXPECT_GT(p, 0.0f) << "record " << i << " unreachable at share " << share;
            }
            sum += p;
        }
        EXPECT_NEAR(sum, 1.0, 1e-5) << "share " << share;
    }
}

TEST(EmissiveMixture, EmptyListTakesNoneOfTheDrawAndIsNotADivisionByZero) {
    EXPECT_EQ(emissive_froxel_share(0.75f, 0), 0.0f);
    EXPECT_EQ(emissive_froxel_share(1.0f, 0), 0.0f);
    EXPECT_EQ(emissive_froxel_share(0.75f, 4), 0.75f);

    std::vector<EmissiveTriangle> records{
        make_tri({0, 0, -10}, {1, 0, 0}, {0, 1, 0}, {5, 5, 5}),
        make_tri({4, 0, -12}, {2, 0, 0}, {0, 2, 0}, {1, 1, 1}),
    };
    const float total = build_alias_into(records);
    // An unbinned froxel behaves EXACTLY as R6 did: pure global.
    double sum = 0.0;
    for (uint32_t i = 0; i < records.size(); ++i) {
        const float p = emissive_mixture_pdf(1.0f, 0, false, records[i].power, total);
        EXPECT_FLOAT_EQ(p, records[i].power / total);
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0, 1e-5);
}

TEST(EmissiveMixture, ShareZeroIsTheR6DrawBitForBit) {
    // The A/B control has to REPRODUCE the pre-R8 figures, not approach them —
    // which means the drawn index and the pdf must both be identical to the
    // plain alias draw for every pair of uniforms, not merely close.
    std::vector<EmissiveTriangle> records{
        make_tri({0, 0, -10}, {1, 0, 0}, {0, 1, 0}, {5, 5, 5}),
        make_tri({4, 0, -12}, {2, 0, 0}, {0, 2, 0}, {1, 1, 1}),
        make_tri({-6, 1, -20}, {3, 0, 0}, {0, 1, 0}, {9, 4, 1}),
    };
    const float total = build_alias_into(records);
    const std::vector<uint32_t> list{0u, 1u, 2u};

    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            const float u1 = (static_cast<float>(i) + 0.5f) / 64.0f;
            const float u2 = (static_cast<float>(j) + 0.5f) / 64.0f;
            const auto d = emissive_draw_mixture(list, records, total, 0.0f, u1, u2);
            const uint32_t expect = emissive_sample_index(records, u1, u2);
            EXPECT_EQ(d.index, expect);
            EXPECT_EQ(d.pdf, records[expect].power / total); // exact, not NEAR
        }
    }
}

TEST(EmissiveMixture, DrawFrequenciesMatchTheStatedPdf) {
    // Integrating the sampler over its own uniforms rather than drawing randoms:
    // the same technique alias_table.test.cpp uses, and the only one that pins
    // "the pdf the weight divides by is the pdf the draw actually has".
    std::vector<EmissiveTriangle> records{
        make_tri({0, 0, -10}, {1, 0, 0}, {0, 1, 0}, {5, 5, 5}),
        make_tri({4, 0, -12}, {2, 0, 0}, {0, 2, 0}, {1, 1, 1}),
        make_tri({-6, 1, -20}, {3, 0, 0}, {0, 1, 0}, {9, 4, 1}),
        make_tri({30, 0, -300}, {6, 0, 0}, {0, 6, 0}, {2, 2, 2}),
    };
    const float total = build_alias_into(records);
    const std::vector<uint32_t> list{1u, 3u};

    constexpr int N = 400;
    for (const float share : {0.0f, 0.5f, 0.75f}) {
        std::vector<double> hits(records.size(), 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const float u1 = (static_cast<float>(i) + 0.5f) / N;
                const float u2 = (static_cast<float>(j) + 0.5f) / N;
                hits[emissive_draw_mixture(list, records, total, share, u1, u2).index] += 1.0;
            }
        }
        for (uint32_t r = 0; r < records.size(); ++r) {
            const float expected = emissive_mixture_pdf(share, 2, emissive_froxel_holds(list, r),
                                                        records[r].power, total);
            EXPECT_NEAR(hits[r] / (N * N), expected, 0.01)
                << "record " << r << " at share " << share;
        }
    }
}

// ---------------------------------------------------------------------------
// The estimator. Unbiasedness is a mathematical claim and is pinned like one.
// ---------------------------------------------------------------------------

TEST(EmissiveMixture, EstimatorConvergesToTheWHOLERecordSumWheneverGlobalMassRemains) {
    // The load-bearing claim of the mixture: localizing the DRAW does not shrink
    // the DOMAIN as long as q_g > 0. Ground truth is closed form here, because
    // the target function IS the integrand.
    const glm::vec3 P{0.0f, 0.0f, 0.0f};
    const glm::vec3 N{0.0f, 1.0f, 0.0f};

    std::vector<EmissiveTriangle> records{
        make_tri({-30, 80, -30}, {60, 0, 0}, {0, 0, 60}, {2.0f, 1.4f, 0.5f}),
        make_tri({10, 20, 40}, {8, 0, 0}, {0, 6, 0}, {9, 9, 9}),
        make_tri({-5, 200, 5}, {30, 0, 0}, {0, 0, 30}, {0.3f, 0.3f, 0.6f}),
        make_tri({120, 300, -80}, {40, 0, 0}, {0, 0, 40}, {1.0f, 1.0f, 1.0f}),
    };
    const float total = build_alias_into(records);
    // A froxel list holding only two of the four — the interesting case, because
    // the other two are reachable ONLY through the global half.
    const std::vector<uint32_t> list{0u, 1u};

    double truth = 0.0;
    for (const EmissiveTriangle& r : records) {
        truth += emissive_target(r, P, N);
    }
    ASSERT_GT(truth, 0.0);

    const uint32_t m_e = 2;
    const float scale_e = ris_source_scale(m_e, m_e); // one stream: exactly 1
    ASSERT_EQ(scale_e, 1.0f);

    Lcg rng;
    constexpr int TRIALS = 400000;
    double acc = 0.0;
    for (int t = 0; t < TRIALS; ++t) {
        bracken::render::lighting::ReservoirBuilder res;
        for (uint32_t i = 0; i < m_e; ++i) {
            const float u1 = rng.next();
            const float u2 = rng.next();
            const auto d = emissive_draw_mixture(list, records, total, 0.75f, u1, u2);
            const float p = emissive_target(records[d.index], P, N);
            res.update(d.index, d.pdf > 0.0f ? p * scale_e / d.pdf : 0.0f, p, rng.next());
        }
        res.finalize();
        acc += static_cast<double>(res.target_pdf) * static_cast<double>(res.r.weight);
    }
    EXPECT_NEAR((acc / TRIALS) / truth, 1.0, 0.02);
}

TEST(EmissiveMixture, ShareOneConvergesToTheLISTSumAndThatIsTheDocumentedBias) {
    // The shipped default drops the global half where a froxel list exists, so
    // the estimator is unbiased for the LIST's sum and not the scene's. That is
    // R8's honest cost (docs/rt_lighting_plan.md §5.7) and it is asserted at the
    // specific value rather than merely "different", so that anyone who later
    // restores full coverage has to come here and say so.
    const glm::vec3 P{0.0f, 0.0f, 0.0f};
    const glm::vec3 N{0.0f, 1.0f, 0.0f};

    std::vector<EmissiveTriangle> records{
        make_tri({-30, 80, -30}, {60, 0, 0}, {0, 0, 60}, {2.0f, 1.4f, 0.5f}),
        make_tri({10, 20, 40}, {8, 0, 0}, {0, 6, 0}, {9, 9, 9}),
        make_tri({-5, 200, 5}, {30, 0, 0}, {0, 0, 30}, {0.3f, 0.3f, 0.6f}),
    };
    const float total = build_alias_into(records);
    const std::vector<uint32_t> list{0u, 1u};

    double list_truth = 0.0;
    for (const uint32_t i : list) {
        list_truth += emissive_target(records[i], P, N);
    }
    double all_truth = 0.0;
    for (const EmissiveTriangle& r : records) {
        all_truth += emissive_target(r, P, N);
    }
    ASSERT_GT(all_truth, list_truth); // record 2 really is outside the list

    Lcg rng;
    constexpr int TRIALS = 200000;
    double acc = 0.0;
    for (int t = 0; t < TRIALS; ++t) {
        bracken::render::lighting::ReservoirBuilder res;
        for (uint32_t i = 0; i < 2; ++i) {
            const float u1 = rng.next();
            const float u2 = rng.next();
            const auto d = emissive_draw_mixture(list, records, total, 1.0f, u1, u2);
            const float p = emissive_target(records[d.index], P, N);
            res.update(d.index, d.pdf > 0.0f ? p / d.pdf : 0.0f, p, rng.next());
        }
        res.finalize();
        acc += static_cast<double>(res.target_pdf) * static_cast<double>(res.r.weight);
    }
    EXPECT_NEAR((acc / TRIALS) / list_truth, 1.0, 0.02);
}

// ---------------------------------------------------------------------------
// The binner
// ---------------------------------------------------------------------------

namespace {

/// Drive ClusterGrid's CPU half far enough to own a froxel-bounds table. No
/// init(), so no GPU: begin_frame() sizes its scratch lazily for exactly this.
struct CpuFroxelGrid {
    bracken::render::lighting::ClusterGrid grid;
    void run() { grid.begin_frame(make_camera(), 1600, 900, 0.1f, 4000.0f); }
    const ClusterParams& params() const { return grid.params(); }
    std::span<const ClusterBounds> bounds() const { return grid.cluster_bounds(); }
};

} // namespace

TEST(EmissiveClusterBinner, BinsRecordsIntoTheFroxelsTheyReach) {
    CpuFroxelGrid froxels;
    froxels.run();
    ASSERT_EQ(froxels.bounds().size(), bracken::render::lighting::CLUSTER_COUNT);

    const auto records = make_scene();
    EmissiveClusterGrid bins;
    bins.build(records, froxels.params(), froxels.bounds());

    EXPECT_TRUE(bins.live());
    EXPECT_GT(bins.stats().froxels_filled, 0u);
    EXPECT_GT(bins.stats().entries, 0u);
    EXPECT_EQ(bins.table().size(), emissive_cluster_table_size(
                                       static_cast<uint32_t>(froxels.bounds().size())));

    // Every packed index is in range, every list is within the cap, and every
    // list is ASCENDING — which the shader's membership scan early-outs on.
    uint32_t counted = 0;
    for (uint32_t f = 0; f < froxels.bounds().size(); ++f) {
        const auto list = bins.froxel_list(f);
        EXPECT_LE(list.size(), k_emissive_froxel_capacity);
        counted += static_cast<uint32_t>(list.size());
        for (std::size_t i = 0; i < list.size(); ++i) {
            EXPECT_LT(list[i], records.size());
            if (i > 0) {
                EXPECT_LT(list[i - 1], list[i]) << "froxel " << f << " entry " << i;
            }
        }
    }
    EXPECT_EQ(counted, bins.stats().entries);
}

TEST(EmissiveClusterBinner, IsBitIdenticalAtAnyWorkerCount) {
    // The house contract for every fanned-out loop (docs/multithreading_plan.md
    // §6.2). The per-froxel top-k replaces only on a STRICTLY greater score, so
    // ties keep the lower record index and the survivor set is a pure function
    // of the record order.
    CpuFroxelGrid froxels;
    froxels.run();
    const auto records = make_scene(200);

    bracken::core::jobs::JobSystem one;
    one.init(1);
    bracken::core::jobs::JobSystem many;
    many.init(4);
    ASSERT_TRUE(one.is_initialized());
    ASSERT_TRUE(many.is_initialized());

    EmissiveClusterGrid a;
    EmissiveClusterGrid b;
    a.build(one, records, froxels.params(), froxels.bounds());
    b.build(many, records, froxels.params(), froxels.bounds());

    ASSERT_EQ(a.table().size(), b.table().size());
    for (std::size_t i = 0; i < a.table().size(); ++i) {
        ASSERT_EQ(a.table()[i], b.table()[i]) << "table entry " << i;
    }
    EXPECT_EQ(a.stats().entries, b.stats().entries);
    EXPECT_EQ(a.stats().froxels_filled, b.stats().froxels_filled);
    EXPECT_EQ(a.stats().entries_dropped, b.stats().entries_dropped);
    EXPECT_EQ(a.stats().records_culled, b.stats().records_culled);

    one.shutdown();
    many.shutdown();
}

TEST(EmissiveClusterBinner, OverCapInsertionsAreCountedRatherThanSwallowed) {
    // The no-silent-drops rule. Enough coincident emitters that a froxel cannot
    // hold them all, and the count has to say so.
    CpuFroxelGrid froxels;
    froxels.run();
    std::vector<EmissiveTriangle> records;
    for (int i = 0; i < static_cast<int>(k_emissive_froxel_capacity) * 4; ++i) {
        records.push_back(make_tri({0.05f * static_cast<float>(i), 0.0f, -80.0f}, {1, 0, 0}, {0, 1, 0},
                                   {6.0f, 6.0f, 6.0f}));
    }
    EmissiveClusterGrid bins;
    bins.build(records, froxels.params(), froxels.bounds());
    EXPECT_GT(bins.stats().entries_dropped, 0u);
    for (uint32_t f = 0; f < froxels.bounds().size(); ++f) {
        EXPECT_LE(bins.froxel_list(f).size(), k_emissive_froxel_capacity);
    }
}

TEST(EmissiveClusterBinner, RecordsBehindTheCameraAreCulledNotBinned) {
    CpuFroxelGrid froxels;
    froxels.run();
    std::vector<EmissiveTriangle> records{
        make_tri({0, 0, -80}, {2, 0, 0}, {0, 2, 0}, {6, 6, 6}),   // in front
        make_tri({0, 0, 400}, {2, 0, 0}, {0, 2, 0}, {6, 6, 6}),   // behind
        make_tri({9000, 0, -80}, {2, 0, 0}, {0, 2, 0}, {6, 6, 6}) // far off to the side
    };
    EmissiveClusterGrid bins;
    bins.build(records, froxels.params(), froxels.bounds());
    EXPECT_GE(bins.stats().records_culled, 2u);
    for (uint32_t f = 0; f < froxels.bounds().size(); ++f) {
        for (const uint32_t r : bins.froxel_list(f)) {
            EXPECT_EQ(r, 0u) << "only the in-frustum record may be binned";
        }
    }
}

TEST(EmissiveClusterBinner, NoRecordsLeavesTheTableDEADRatherThanStale) {
    // `live()` false is the whole degradation path: the trace then sends
    // emissiveFroxelShare = 0 and draws exactly as R6 did. A table that claimed
    // to be live while describing a different record set would aim reservoirs at
    // renumbered triangles.
    CpuFroxelGrid froxels;
    froxels.run();
    EmissiveClusterGrid bins;
    bins.build({}, froxels.params(), froxels.bounds());
    EXPECT_FALSE(bins.live());
    EXPECT_EQ(bins.stats().entries, 0u);

    bins.build(make_scene(), froxels.params(), froxels.bounds());
    EXPECT_TRUE(bins.live());
    bins.invalidate();
    EXPECT_FALSE(bins.live());
    EXPECT_EQ(bins.stats().entries, 0u);
}

TEST(EmissiveClusterBinner, AMismatchedGridIsRefusedRatherThanIndexedPastTheEnd) {
    CpuFroxelGrid froxels;
    froxels.run();
    std::vector<ClusterBounds> too_few(4);
    EmissiveClusterGrid bins;
    bins.build(make_scene(), froxels.params(), too_few);
    EXPECT_FALSE(bins.live());
}
