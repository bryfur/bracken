// Determinism parity tests for the SceneRenderer CPU-prepare fan-out
// (docs/multithreading_plan.md §4.3 / §6.2): the prep cores must produce
// BIT-IDENTICAL outputs at any worker count. A synthetic sealed FramePacket
// (deterministic LCG transforms/bounds, a mix of culled/visible/instanced/
// non-instanced/skinned commands) is prepared with caller-owned JobSystem
// pools — uninitialized (pure serial reference), js(1) and js(4) — and every
// output (bin contents AND order, touched lists, non-instanced list, stats,
// resolved skinned entries, palette copy bytes) is compared byte-for-byte.
//
// The cores are device-free statics on SceneRenderer precisely so this test
// can reach them without a GPU device (models are registered into a
// ModelManager with no device set — prep only reads bounds/skeleton).

#include <gtest/gtest.h>

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render_stats.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/frustum.hpp"
#include "bracken/scene/motion_history.hpp"
#include "bracken/scene/scene_renderer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace bracken;
using namespace bracken::scene;
namespace jobs = bracken::core::jobs;

namespace {

// Minimal deterministic LCG (numerical-recipes constants) — reproducible
// inputs without <random> (same pattern as scene.bench.cpp).
struct Lcg {
    uint32_t state = 0x12345678u;
    uint32_t next_u32() {
        state = state * 1664525u + 1013904223u;
        return state;
    }
    float next_float01() { return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f); }
    float next_float(float lo, float hi) { return lo + next_float01() * (hi - lo); }
    // LCG low bits are short-period — draw decision bits from the high half.
    uint32_t next_bits(uint32_t mask) { return (next_u32() >> 20) & mask; }
};

// Bitwise equality helpers — the contract is bit-identity, so compare bytes,
// not operator== (which would treat -0.0f == 0.0f as equal).
template<typename T> bool bits_equal(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

bool stats_equal(const FrameStatsPartial& a, const FrameStatsPartial& b) {
    return a.draw_calls == b.draw_calls && a.triangle_count == b.triangle_count &&
           a.entities_rendered == b.entities_rendered && a.entities_distance_culled == b.entities_distance_culled &&
           a.entities_frustum_culled == b.entities_frustum_culled;
}

// ---------------------------------------------------------------------------
// Fixture world: a ModelManager (no device) + a sealed synthetic packet.
// ---------------------------------------------------------------------------

struct PrepWorld {
    ModelManager models;
    std::vector<ModelHandle> static_handles;
    std::vector<ModelHandle> skinned_handles;
    FramePacket packet{1 << 20};
    MotionHistory history;

    CameraState camera{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(0.0f)};
    Frustum main_frustum{};
    std::array<Frustum, SceneRenderer::MAX_SHADOW_CASCADES> cascade_frustums{};
    int active_cascades = 3;

    SceneRenderer::PrepParams params(bool collect_stats = true) const {
        SceneRenderer::PrepParams p;
        p.models = &models;
        p.motion_history = &history;
        p.camera_pos = camera.position;
        p.main_frustum = &main_frustum;
        p.cascade_frustums = cascade_frustums.data();
        p.active_cascades = active_cascades;
        p.draw_dist_sq = 700.0f * 700.0f;
        p.do_frustum_cull = true;
        p.collect_stats = collect_stats;
        p.prev_palette_valid = true;
        return p;
    }
};

std::unique_ptr<Model> make_static_model(Lcg& rng) {
    auto m = std::make_unique<Model>();
    m->min_x = -rng.next_float(0.5f, 8.0f);
    m->min_y = 0.0f;
    m->min_z = -rng.next_float(0.5f, 8.0f);
    m->max_x = rng.next_float(0.5f, 8.0f);
    m->max_y = rng.next_float(1.0f, 12.0f);
    m->max_z = rng.next_float(0.5f, 8.0f);
    m->compute_bounding_sphere();
    return m;
}

std::unique_ptr<Model> make_skinned_model(Lcg& rng, uint32_t joints) {
    auto m = make_static_model(rng);
    m->has_skeleton = true;
    m->skeleton.joints.resize(joints);
    return m;
}

glm::mat4 make_transform(Lcg& rng) {
    // Centers straddle the frustum AND the draw distance so culling sees a
    // realistic inside/outside mix; a few get non-uniform scale + rotation.
    glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(rng.next_float(-900.0f, 900.0f),
                                                            rng.next_float(-50.0f, 80.0f),
                                                            rng.next_float(-1200.0f, 500.0f)));
    if (rng.next_bits(3u) == 0) {
        t = glm::rotate(t, rng.next_float(0.0f, 6.28f), glm::vec3(0, 1, 0));
        t = glm::scale(t, glm::vec3(rng.next_float(0.5f, 3.0f), rng.next_float(0.5f, 3.0f),
                                    rng.next_float(0.5f, 3.0f)));
    }
    return t;
}

// Build the world: models, motion history, camera/cascade frustums, and a
// sealed packet with `model_count` static + `skinned_count` skinned commands.
void build_world(PrepWorld& w, size_t model_count, size_t skinned_count) {
    Lcg rng{0xC0FFEEu};

    for (int i = 0; i < 6; ++i) {
        w.static_handles.push_back(w.models.register_model("static_" + std::to_string(i), make_static_model(rng)));
    }
    for (int i = 0; i < 3; ++i) {
        w.skinned_handles.push_back(
            w.models.register_model("skinned_" + std::to_string(i), make_skinned_model(rng, 8u + 8u * i)));
    }

    // Camera + frustums (main perspective; cascades = shifted ortho volumes).
    glm::mat4 view = glm::lookAt(glm::vec3(0, 50, 100), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(glm::radians(70.0f), 16.0f / 9.0f, 0.1f, 1000.0f);
    w.camera = CameraState{view, proj, proj * view, glm::vec3(0, 50, 100)};
    w.main_frustum.extract_from_matrix(w.camera.view_projection);
    for (int c = 0; c < w.active_cascades; ++c) {
        const float ext = 60.0f * static_cast<float>(1 << c);
        glm::mat4 lview = glm::lookAt(glm::vec3(100, 300, 100), glm::vec3(0, 0, -100.0f * c), glm::vec3(0, 1, 0));
        glm::mat4 lproj = glm::ortho(-ext, ext, -ext, ext, 1.0f, 800.0f);
        w.cascade_frustums[c].extract_from_matrix(lproj * lview);
    }

    // Previous-frame motion history for a subset of motion ids (statics get a
    // transform only, skinned also a palette offset), then promote.
    for (uint32_t id = 1; id <= 200; id += 2) {
        Lcg r{id * 0x9E3779B9u};
        w.history.record(id, make_transform(r), (id % 4 == 1) ? id * 3u : MotionHistory::NO_PALETTE);
    }
    w.history.end_frame();

    FrameBuilder fb(w.packet);
    for (size_t i = 0; i < model_count; ++i) {
        // ~1/16 unregistered handle (no-bounds path), ~1/8 non-instanced,
        // ~1/4 with motion id, ~1/8 no_fog.
        const bool unregistered = rng.next_bits(15u) == 0;
        const ModelHandle h =
            unregistered ? static_cast<ModelHandle>(4321) : w.static_handles[rng.next_u32() % w.static_handles.size()];
        const bool non_instanced = rng.next_bits(7u) == 0;
        const bool no_fog = rng.next_bits(7u) == 0;
        const uint32_t motion_id = (rng.next_bits(3u) == 0) ? 1u + (rng.next_u32() % 200u) : 0u;
        const glm::vec4 tint(rng.next_float01(), rng.next_float01(), rng.next_float01(), 1.0f);
        fb.draw_model(h, make_transform(rng), tint, non_instanced, no_fog, motion_id);
    }
    FrameBuilder::BonePaletteAllocation shared{};
    for (size_t i = 0; i < skinned_count; ++i) {
        const ModelHandle h = w.skinned_handles[rng.next_u32() % w.skinned_handles.size()];
        // Every 5th draw shares the previous palette id (build-time sharing).
        FrameBuilder::BonePaletteAllocation alloc = shared;
        if (i % 5 != 4 || !shared.valid()) {
            const uint32_t joints = 4u + (rng.next_u32() % 28u);
            alloc = fb.lane(0).alloc_bone_palette(joints);
            ASSERT_TRUE(alloc.valid());
            for (glm::mat4& m : alloc.matrices) {
                m = make_transform(rng);
            }
            shared = alloc;
        }
        const uint32_t motion_id = (rng.next_bits(1u) == 0) ? 1u + (rng.next_u32() % 200u) : 0u;
        fb.draw_skinned(h, make_transform(rng), alloc.id, {1, 1, 1, 1}, false, motion_id);
    }
    fb.seal();
}

// ---------------------------------------------------------------------------
// One full prep run: instance batches + skinned resolve + palette copy.
// ---------------------------------------------------------------------------

struct PrepRun {
    std::vector<std::vector<gpu::InstanceData>> bins;
    std::vector<ModelHandle> touched;
    std::array<std::vector<std::vector<gpu::ShadowInstanceData>>, SceneRenderer::MAX_SHADOW_CASCADES> shadow_bins;
    std::array<std::vector<ModelHandle>, SceneRenderer::MAX_SHADOW_CASCADES> shadow_touched;
    std::vector<const ModelDrawCommand*> non_instanced;
    FrameStatsPartial stats;
    std::vector<SceneRenderer::ResolvedSkinnedEntry> skinned;
    FrameStatsPartial skinned_stats;
    std::vector<glm::mat4> palette_copy;
};

void run_prep(jobs::JobSystem& js, const PrepWorld& w, SceneRenderer::PrepScratch& scratch, PrepRun& run) {
    const SceneRenderer::PrepParams p = w.params();
    const SceneRenderer::InstanceBinOutputs out{run.bins,          run.touched,       run.shadow_bins,
                                                run.shadow_touched, run.non_instanced, run.stats};
    run.stats = {};
    SceneRenderer::prep_instance_batches(js, w.packet.model_commands(), p, scratch, out);

    run.skinned_stats = {};
    SceneRenderer::prep_resolve_skinned(js, w.packet.skinned_commands(), w.packet.palettes(), p, scratch.skinned,
                                        run.skinned, run.skinned_stats);

    // Sentinel-fill so bytes NOT covered by any palette range must also match
    // (a stray out-of-range write would differ from the reference).
    run.palette_copy.assign(w.packet.palette_matrix_total(), glm::mat4(-12345.0f));
    if (!run.palette_copy.empty()) {
        SceneRenderer::prep_copy_palettes(js, w.packet.palettes(), run.palette_copy.data());
    }
}

void expect_runs_identical(const PrepRun& a, const PrepRun& b, const char* label) {
    SCOPED_TRACE(label);

    // Main-view bins: identical dense sizes, touched order, contents + order.
    ASSERT_EQ(a.bins.size(), b.bins.size());
    for (size_t h = 0; h < a.bins.size(); ++h) {
        ASSERT_EQ(a.bins[h].size(), b.bins[h].size()) << "bin " << h;
        if (!a.bins[h].empty()) {
            EXPECT_EQ(0, std::memcmp(a.bins[h].data(), b.bins[h].data(),
                                     a.bins[h].size() * sizeof(gpu::InstanceData)))
                << "bin " << h;
        }
    }
    EXPECT_EQ(a.touched, b.touched);

    for (int c = 0; c < SceneRenderer::MAX_SHADOW_CASCADES; ++c) {
        ASSERT_EQ(a.shadow_bins[c].size(), b.shadow_bins[c].size()) << "cascade " << c;
        for (size_t h = 0; h < a.shadow_bins[c].size(); ++h) {
            ASSERT_EQ(a.shadow_bins[c][h].size(), b.shadow_bins[c][h].size()) << "cascade " << c << " bin " << h;
            if (!a.shadow_bins[c][h].empty()) {
                EXPECT_EQ(0, std::memcmp(a.shadow_bins[c][h].data(), b.shadow_bins[c][h].data(),
                                         a.shadow_bins[c][h].size() * sizeof(gpu::ShadowInstanceData)))
                    << "cascade " << c << " bin " << h;
            }
        }
        EXPECT_EQ(a.shadow_touched[c], b.shadow_touched[c]) << "cascade " << c;
    }

    EXPECT_EQ(a.non_instanced, b.non_instanced); // same packet => identical pointers
    EXPECT_TRUE(stats_equal(a.stats, b.stats));

    // Skinned resolve: field-by-field bit compare (the struct has padding, so
    // whole-struct memcmp would compare indeterminate bytes).
    ASSERT_EQ(a.skinned.size(), b.skinned.size());
    for (size_t i = 0; i < a.skinned.size(); ++i) {
        const auto& x = a.skinned[i];
        const auto& y = b.skinned[i];
        EXPECT_EQ(x.cmd, y.cmd) << i;
        EXPECT_EQ(x.model, y.model) << i;
        EXPECT_TRUE(bits_equal(x.world_center, y.world_center)) << i;
        EXPECT_TRUE(bits_equal(x.world_radius, y.world_radius)) << i;
        EXPECT_EQ(x.palette_offset, y.palette_offset) << i;
        EXPECT_EQ(x.main_visible, y.main_visible) << i;
        EXPECT_TRUE(bits_equal(x.prev_model, y.prev_model)) << i;
        EXPECT_EQ(x.prev_palette_offset, y.prev_palette_offset) << i;
    }
    EXPECT_TRUE(stats_equal(a.skinned_stats, b.skinned_stats));

    // Palette copy: every byte of the packed stream (including sentinel
    // gaps, of which there should be none) identical.
    ASSERT_EQ(a.palette_copy.size(), b.palette_copy.size());
    if (!a.palette_copy.empty()) {
        EXPECT_EQ(0, std::memcmp(a.palette_copy.data(), b.palette_copy.data(),
                                 a.palette_copy.size() * sizeof(glm::mat4)));
    }
}

// Sanity: the run culled something, kept something, binned instances into
// more than one handle, resolved skinned entries and copied palettes — i.e.
// the parity comparison is over non-trivial data.
void expect_run_nontrivial(const PrepRun& run) {
    EXPECT_GT(run.stats.entities_rendered, 0u);
    EXPECT_GT(run.stats.entities_distance_culled, 0u);
    EXPECT_GT(run.stats.entities_frustum_culled, 0u);
    EXPECT_GT(run.touched.size(), 1u);
    EXPECT_FALSE(run.non_instanced.empty());
    bool any_shadow = false;
    for (const auto& t : run.shadow_touched) any_shadow |= !t.empty();
    EXPECT_TRUE(any_shadow);
    EXPECT_FALSE(run.skinned.empty());
    EXPECT_FALSE(run.palette_copy.empty());
    bool any_visible = false;
    bool any_prev_palette = false;
    for (const auto& e : run.skinned) {
        any_visible |= e.main_visible;
        any_prev_palette |= e.prev_palette_offset != gpu::PrepassModelUniforms::NO_PREV_PALETTE;
    }
    EXPECT_TRUE(any_visible);
    EXPECT_TRUE(any_prev_palette);
}

} // namespace

// js(1) vs js(4) vs uninitialized (pure serial path): bit-identical outputs
// over a packet large enough to fan out (several chunks per view, all views
// active). THE determinism contract test (§6.2).
TEST(ScenePrepParity, WorkerCountInvariance) {
    PrepWorld w;
    build_world(w, 700, 300);

    jobs::JobSystem js_serial; // never initialized: every core runs inline
    jobs::JobSystem js1;
    js1.init(1);
    jobs::JobSystem js4;
    js4.init(4);

    SceneRenderer::PrepScratch scratch_serial;
    SceneRenderer::PrepScratch scratch1;
    SceneRenderer::PrepScratch scratch4;
    PrepRun serial;
    PrepRun run1;
    PrepRun run4;
    run_prep(js_serial, w, scratch_serial, serial);
    run_prep(js1, w, scratch1, run1);
    run_prep(js4, w, scratch4, run4);

    expect_run_nontrivial(serial);
    expect_runs_identical(serial, run1, "serial vs js(1)");
    expect_runs_identical(serial, run4, "serial vs js(4)");

    // Palette copy reference: palettes() order == matrix_offset order, so the
    // packed stream is exactly the concatenation of every palette's matrices.
    std::vector<glm::mat4> reference;
    reference.reserve(w.packet.palette_matrix_total());
    for (const BonePalette& rec : w.packet.palettes()) {
        reference.insert(reference.end(), rec.matrices, rec.matrices + rec.joint_count);
    }
    ASSERT_EQ(reference.size(), serial.palette_copy.size());
    EXPECT_EQ(0, std::memcmp(reference.data(), serial.palette_copy.data(), reference.size() * sizeof(glm::mat4)));
}

// The per-chunk scratch pool is reused frame after frame (capacity-stable, no
// clears between frames beyond the chunk jobs' own O(touched) clear): a run
// with a scratch pool dirtied by a DIFFERENT (larger) packet must match a run
// with a fresh pool — stale chunk contents must never leak into a merge.
TEST(ScenePrepParity, ScratchReuseAcrossFrames) {
    PrepWorld big;
    build_world(big, 700, 300);
    PrepWorld small;
    build_world(small, 350, 150); // fewer chunks: high slots of the pool stay stale

    jobs::JobSystem js4;
    js4.init(4);

    SceneRenderer::PrepScratch reused;
    PrepRun big_run;
    run_prep(js4, big, reused, big_run); // dirty the pool

    PrepRun reused_run;
    run_prep(js4, small, reused, reused_run);

    SceneRenderer::PrepScratch fresh;
    PrepRun fresh_run;
    run_prep(js4, small, fresh, fresh_run);

    expect_runs_identical(fresh_run, reused_run, "fresh scratch vs reused scratch");
}

// Below one chunk (the 1-character playground shape) everything must stay on
// the caller-inline path AND still match the serial reference exactly.
TEST(ScenePrepParity, SmallSceneStaysInlineAndIdentical) {
    PrepWorld w;
    build_world(w, 12, 3);
    ASSERT_LE(w.packet.model_commands().size(), SceneRenderer::PREP_COMMAND_GRAIN);

    jobs::JobSystem js_serial;
    jobs::JobSystem js4;
    js4.init(4);

    SceneRenderer::PrepScratch scratch_serial;
    SceneRenderer::PrepScratch scratch4;
    PrepRun serial;
    PrepRun run4;
    run_prep(js_serial, w, scratch_serial, serial);
    run_prep(js4, w, scratch4, run4);

    // Inline path: no chunk scratch may be allocated for the command walks.
    EXPECT_TRUE(scratch4.main.chunks.empty());
    EXPECT_TRUE(scratch4.skinned.chunks.empty());
    for (const auto& v : scratch4.cascades) {
        EXPECT_TRUE(v.chunks.empty());
    }

    expect_runs_identical(serial, run4, "serial vs js(4), small scene");
}

// Stats-collection off must not change any binning/resolve output (only the
// counters), at any worker count.
TEST(ScenePrepParity, StatsOffParity) {
    PrepWorld w;
    build_world(w, 400, 100);

    jobs::JobSystem js4;
    js4.init(4);

    SceneRenderer::PrepScratch s_on;
    SceneRenderer::PrepScratch s_off;
    PrepRun on;
    PrepRun off;
    run_prep(js4, w, s_on, on);

    {
        const SceneRenderer::PrepParams p = w.params(/*collect_stats=*/false);
        const SceneRenderer::InstanceBinOutputs out{off.bins,          off.touched,       off.shadow_bins,
                                                    off.shadow_touched, off.non_instanced, off.stats};
        SceneRenderer::prep_instance_batches(js4, w.packet.model_commands(), p, s_off, out);
        SceneRenderer::prep_resolve_skinned(js4, w.packet.skinned_commands(), w.packet.palettes(), p, s_off.skinned,
                                            off.skinned, off.skinned_stats);
        off.palette_copy.assign(w.packet.palette_matrix_total(), glm::mat4(-12345.0f));
        SceneRenderer::prep_copy_palettes(js4, w.packet.palettes(), off.palette_copy.data());
    }

    // Zeroed counters, identical everything else.
    EXPECT_TRUE(stats_equal(off.stats, FrameStatsPartial{}));
    EXPECT_TRUE(stats_equal(off.skinned_stats, FrameStatsPartial{}));
    off.stats = on.stats;
    off.skinned_stats = on.skinned_stats;
    expect_runs_identical(on, off, "stats on vs off");
}
