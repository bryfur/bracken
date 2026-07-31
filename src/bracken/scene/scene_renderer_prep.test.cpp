// GOLDEN-VALUE tests for the CPU Prepare cores (scene::ScenePrep).
//
// scene_prep_parity.test.cpp next door proves the prep cores are BIT-IDENTICAL
// at any worker count. That is a strong property and a weak test: it compares
// the cores against themselves, so it passes just as happily when every cull
// decision is wrong. Nothing anywhere said what the right answer IS.
//
// This file says it. The scene is small enough to reason about by hand — a
// camera at the origin looking down -Z, an ortho main frustum, axis-aligned
// unit bounds, pure-translation transforms — so every expectation below is a
// number derived from the rules in the source, not a value recorded from a run:
//   * which commands are distance-culled, frustum-culled and rendered;
//   * which handle each instance lands in, and IN WHICH ORDER;
//   * the exact InstanceData bytes (model, normalMatrix, tint, noFog,
//     prevModel from the motion history);
//   * the per-batch base_instance the packing loop and the three draw walks
//     must independently agree on;
//   * the shadow-view cast rules, including the two that are easy to "simplify"
//     away: a command with NO BOUNDS casts into every cascade, and a
//     main-frustum-culled command still casts if the light frustum sees it;
//   * the skinned resolve's joint clamp, drop counters and palette offsets;
//   * the packed palette byte stream.
//
// Two shapes are mandatory here because a later phase splits the renderer
// across translation units (docs/refactor_plan.md Phase 6) and the parity test
// structurally cannot catch either:
//   (a) a command whose ModelHandle resolves to NO MODEL, asserted in the
//       main-view AND shadow-view arms;
//   (b) >= 2 distinct model types AND an unregistered handle at once, with
//       base_instance pinned — the only configuration in which a desync
//       between the instance-buffer packing loop and a draw walk changes a
//       number instead of nothing.
//
// Device-free: models are registered into a ModelManager with no device set,
// exactly as in scene_prep_parity.test.cpp — prep only reads bounds/skeletons.

#include <gtest/gtest.h>

#include "bracken/animation/animation_types.hpp"
#include "bracken/contracts/frustum.hpp"
#include "bracken/testing/frame_test.hpp"
#include "bracken/contracts/render_stats.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/motion_history.hpp"
#include "bracken/scene/scene_prep.hpp"
#include "bracken/scene/transform_math.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <span>
#include <vector>

using bracken::testing::FrameAccess;
using namespace bracken;
using namespace bracken::scene;
namespace jobs = bracken::core::jobs;

namespace {

// ---------------------------------------------------------------------------
// The hand-built world. Every number here is chosen so the cull decisions are
// unambiguous by inspection, with generous margins against float error.
// ---------------------------------------------------------------------------

// Camera sits at the origin looking down -Z. Draw distance is a 100-unit
// radius in the XZ plane (the cull is 2D — see bin_main_chunk).
constexpr float DRAW_DISTANCE = 100.0f;
constexpr float DRAW_DIST_SQ = DRAW_DISTANCE * DRAW_DISTANCE;

// Main view: an ortho box, x/y in [-20, 20], z in [-200, -1] (world == view
// space here). Cascade 0: a box big enough to contain every placement except
// the one deliberately lifted out of it. Cascade 1: a box far off to the side
// that contains NOTHING, so it can prove the "no bounds always casts" rule.
constexpr float MAIN_HALF_EXTENT = 20.0f;
constexpr float CASCADE0_HALF_EXTENT = 500.0f;

// Registration order fixes the handles: ModelManager reserves slot 0 for
// INVALID_MODEL_HANDLE, so the first registered model is handle 1.
constexpr ModelHandle H_CUBE = 1;    // "cube": bounds [-1,1]^3
constexpr ModelHandle H_SLAB = 2;    // "slab": bounds [-2,2] x [0,1] x [-2,2]
constexpr ModelHandle H_SKINNED = 3; // "skinned": cube bounds, 4 joints
// Never registered. NOT INVALID_MODEL_HANDLE (0): FrameBuilder refuses that at
// emit time, so an unresolvable-at-prep command must use a plausible-looking
// handle — which is exactly the shape of a host that unloaded a model.
constexpr ModelHandle H_UNKNOWN = 7;

std::unique_ptr<Model> make_box_model(float hx, float min_y, float max_y, float hz, bool skeleton, uint32_t joints) {
    auto m = std::make_unique<Model>();
    m->min_x = -hx;
    m->max_x = hx;
    m->min_y = min_y;
    m->max_y = max_y;
    m->min_z = -hz;
    m->max_z = hz;
    m->compute_bounding_sphere();
    if (skeleton) {
        m->has_skeleton = true;
        m->skeleton.joints.resize(joints);
    }
    return m;
}

glm::mat4 at(float x, float y, float z) {
    return glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
}

/// Ortho view-projection from (0, 0, `eye_z`) looking down -Z, half extent
/// `half`, depth range [1, far]. The scene is pure translation, so the
/// resulting frustum is exactly the world box
/// x, y in [-half, half] and z in [eye_z - far, eye_z - 1].
glm::mat4 ortho_vp(float half, float far_plane, float eye_z = 0.0f) {
    const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, eye_z), glm::vec3(0.0f, 0.0f, eye_z - 1.0f),
                                       glm::vec3(0.0f, 1.0f, 0.0f));
    return glm::ortho(-half, half, -half, half, 1.0f, far_plane) * view;
}

// Motion ids used below, and the previous-frame state recorded for them.
constexpr uint32_t MOTION_STATIC_MODEL = 5;
constexpr uint32_t MOTION_SKINNED = 9;
constexpr uint32_t PREV_PALETTE_OFFSET = 42;
const glm::mat4 PREV_MODEL_TRANSFORM = at(2.0f, 0.0f, -60.0f);
const glm::mat4 PREV_SKINNED_TRANSFORM = at(0.0f, 900.0f, -60.0f);

// Tints, distinct per command so bin membership is checkable by value.
glm::vec4 tint_for(int index) {
    return glm::vec4(static_cast<float>(index) * 0.0625f, 0.25f, 0.5f, 1.0f);
}

// Palette shapes. Offsets are the seal-time prefix sum: 0, 3, 3+5.
constexpr uint32_t PALETTE_JOINTS[3] = {3, 5, 2};
constexpr uint32_t PALETTE_OFFSETS[3] = {0, 3, 8};
constexpr uint32_t PALETTE_MATRIX_TOTAL = 10;

/// The fixed world: models, motion history, frustums, and one sealed packet.
struct GoldenWorld {
    ModelManager models;
    FramePacket packet{1 << 16};
    MotionHistory history;
    Frustum main_frustum{};
    std::array<Frustum, ScenePrep::MAX_SHADOW_CASCADES> cascades{};
    std::array<BonePaletteId, 3> palettes{};

    GoldenWorld() {
        require_handle(models.register_model("cube", make_box_model(1.0f, -1.0f, 1.0f, 1.0f, false, 0)), H_CUBE);
        require_handle(models.register_model("slab", make_box_model(2.0f, 0.0f, 1.0f, 2.0f, false, 0)), H_SLAB);
        require_handle(models.register_model("skinned", make_box_model(1.0f, -1.0f, 1.0f, 1.0f, true, 4)), H_SKINNED);

        main_frustum.extract_from_matrix(ortho_vp(MAIN_HALF_EXTENT, 200.0f));
        // Cascade 0 is pulled back to +600 so its box spans z in [-400, 599]:
        // it must contain the command BEHIND the main camera (z = +50), which
        // is the whole point of the "main-culled commands still cast" rule.
        cascades[0].extract_from_matrix(ortho_vp(CASCADE0_HALF_EXTENT, 1000.0f, 600.0f));
        // Cascade 1: a box 5000 units to the +X side — contains no placement.
        const glm::mat4 far_view =
            glm::lookAt(glm::vec3(5000.0f, 0.0f, 0.0f), glm::vec3(5000.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        cascades[1].extract_from_matrix(glm::ortho(-50.0f, 50.0f, -50.0f, 50.0f, 1.0f, 200.0f) * far_view);

        // Previous frame: one static-model id and one skinned id (the skinned
        // one also carries last frame's palette offset).
        history.record(MOTION_STATIC_MODEL, PREV_MODEL_TRANSFORM);
        history.record(MOTION_SKINNED, PREV_SKINNED_TRANSFORM, PREV_PALETTE_OFFSET);
        history.end_frame();

        build_packet();
    }

    ScenePrep::PrepParams params(int active_cascades = 2) const {
        ScenePrep::PrepParams p;
        p.models = &models;
        p.motion_history = &history;
        p.camera_pos = glm::vec3(0.0f);
        p.main_frustum = &main_frustum;
        p.cascade_frustums = cascades.data();
        p.active_cascades = active_cascades;
        p.draw_dist_sq = DRAW_DIST_SQ;
        p.do_frustum_cull = true;
        p.collect_stats = true;
        p.prev_palette_valid = true;
        return p;
    }

private:
    static void require_handle(ModelHandle got, ModelHandle want) {
        // Not a gtest assertion (runs in a constructor): the handles are a
        // precondition of every expectation below, so a mismatch must be
        // impossible rather than merely reported.
        BRACKEN_ASSERT(got == want, "registration order changed: got handle {} expected {}", got, want);
        (void)got;
        (void)want;
    }

    void build_packet();
};

// ---------------------------------------------------------------------------
// The packet. Command indices are referenced by the expectations below, so the
// order here is part of the contract.
// ---------------------------------------------------------------------------
enum CommandIndex : std::size_t {
    CMD_CUBE_NEAR = 0,      // rendered, bin[cube] slot 0
    CMD_SLAB_RIGHT = 1,     // rendered, bin[slab] slot 0
    CMD_UNKNOWN = 2,        // rendered (NO BOUNDS), bin[unknown] slot 0, + diagnostic
    CMD_TOO_FAR = 3,        // distance-culled
    CMD_BEHIND = 4,         // main-frustum-culled; still a shadow caster
    CMD_NON_INSTANCED = 5,  // rendered, non-instanced list (never a shadow instance)
    CMD_SLAB_LEFT = 6,      // rendered, bin[slab] slot 1
    CMD_CUBE_MOVING = 7,    // rendered, bin[cube] slot 1, prevModel from history
    CMD_CUBE_NO_FOG = 8,    // rendered, bin[cube] slot 2, noFog = 1
    CMD_HIGH_ABOVE = 9,     // main-frustum-culled AND outside cascade 0
    CMD_COUNT = 10,
};

void GoldenWorld::build_packet() {
    FrameBuilder fb(packet);

    fb.draw_model(H_CUBE, at(0.0f, 0.0f, -50.0f), tint_for(CMD_CUBE_NEAR));
    fb.draw_model(H_SLAB, at(10.0f, 0.0f, -50.0f), tint_for(CMD_SLAB_RIGHT));
    fb.draw_model(H_UNKNOWN, at(0.0f, 0.0f, -50.0f), tint_for(CMD_UNKNOWN));
    fb.draw_model(H_CUBE, at(0.0f, 0.0f, -300.0f), tint_for(CMD_TOO_FAR));
    fb.draw_model(H_CUBE, at(0.0f, 0.0f, 50.0f), tint_for(CMD_BEHIND));
    fb.draw_model(H_CUBE, at(5.0f, 0.0f, -50.0f), tint_for(CMD_NON_INSTANCED), DrawFlags::ForceNonInstanced);
    fb.draw_model(H_SLAB, at(-10.0f, 0.0f, -50.0f), tint_for(CMD_SLAB_LEFT));
    fb.draw_model(H_CUBE, at(2.0f, 0.0f, -50.0f), tint_for(CMD_CUBE_MOVING), DrawFlags::None, MOTION_STATIC_MODEL);
    fb.draw_model(H_CUBE, at(3.0f, 0.0f, -50.0f), tint_for(CMD_CUBE_NO_FOG), DrawFlags::NoFog);
    fb.draw_model(H_CUBE, at(0.0f, 900.0f, -50.0f), tint_for(CMD_HIGH_ABOVE));

    // Palettes: three allocations, prefix-summed offsets 0 / 3 / 8. Every
    // matrix is filled with a value that identifies (palette, joint), so the
    // packed stream can be checked byte for byte.
    for (std::size_t p = 0; p < palettes.size(); ++p) {
        FrameBuilder::BonePaletteAllocation alloc = fb.alloc_bone_palette(PALETTE_JOINTS[p]);
        BRACKEN_ASSERT(alloc.valid(), "golden palette allocation failed");
        for (std::size_t j = 0; j < alloc.matrices.size(); ++j) {
            alloc.matrices[j] = glm::mat4(static_cast<float>(p * 100 + j + 1));
        }
        palettes[p] = alloc.id;
    }

    // Skinned commands, in resolve order.
    fb.draw_skinned(H_SKINNED, at(0.0f, 0.0f, -50.0f), palettes[0]);         // 0: rendered
    fb.draw_skinned(H_CUBE, at(0.0f, 0.0f, -50.0f), palettes[1]);            // 1: no skeleton -> dropped
    fb.draw_skinned(H_UNKNOWN, at(0.0f, 0.0f, -50.0f), palettes[1]);         // 2: no model    -> dropped
    fb.draw_skinned(H_SKINNED, at(0.0f, 0.0f, -300.0f), palettes[1]);        // 3: distance-culled
    fb.draw_skinned(H_SKINNED, at(0.0f, 900.0f, -50.0f), palettes[2], glm::vec4(1.0f), DrawFlags::None,
                    MOTION_SKINNED);                                        // 4: frustum-culled, has history

    FrameAccess::seal(fb);
}

enum SkinnedIndex : std::size_t {
    SK_VISIBLE = 0,
    SK_NO_SKELETON = 1,
    SK_NO_MODEL = 2,
    SK_TOO_FAR = 3,
    SK_HIGH_ABOVE = 4,
    SK_COUNT = 5,
};

// ---------------------------------------------------------------------------
// One prep run against caller-owned outputs.
// ---------------------------------------------------------------------------
struct PrepOutputs {
    std::vector<std::vector<gpu::InstanceData>> bins;
    std::vector<ModelHandle> touched;
    std::array<std::vector<std::vector<gpu::ShadowInstanceData>>, ScenePrep::MAX_SHADOW_CASCADES> shadow_bins;
    std::array<std::vector<ModelHandle>, ScenePrep::MAX_SHADOW_CASCADES> shadow_touched;
    std::vector<const ModelDrawCommand*> non_instanced;
    FrameStatsPartial stats;

    ScenePrep::InstanceBinOutputs bindings() {
        return ScenePrep::InstanceBinOutputs{bins, touched, shadow_bins, shadow_touched, non_instanced, stats};
    }
};

void run_binning(jobs::JobSystem& js, const GoldenWorld& world, PrepOutputs& out, ScenePrep::PrepScratch& scratch) {
    const ScenePrep::InstanceBinOutputs bindings = out.bindings();
    ScenePrep::prep_instance_batches(js, world.packet.model_commands(), world.params(), scratch, bindings);
}

/// The base_instance walk, expressed exactly as upload_instance_buffers packs
/// and as every draw walk re-derives it: ascending handle order over the dense
/// bin array, skipping empty bins, ADVANCING PAST bins whose handle resolves to
/// no model. Returns (handle, base_instance, count) per non-empty bin.
struct PackedBatch {
    ModelHandle handle;
    uint32_t base_instance;
    uint32_t count;
    bool operator==(const PackedBatch&) const = default;
};

std::vector<PackedBatch> packed_batches(const std::vector<std::vector<gpu::InstanceData>>& bins) {
    std::vector<PackedBatch> out;
    uint32_t base = 0;
    for (ModelHandle h = 0; h < bins.size(); ++h) {
        if (bins[h].empty()) {
            continue;
        }
        const uint32_t count = static_cast<uint32_t>(bins[h].size());
        out.push_back(PackedBatch{h, base, count});
        base += count;
    }
    return out;
}

} // namespace

// ============================================================================
// Main view: cull decisions, bin membership + order, instance bytes
// ============================================================================

TEST(ScenePrepGolden, MainViewCullCountsAreExact) {
    GoldenWorld world;
    jobs::JobSystem js; // uninitialized: pure serial reference path
    ScenePrep::PrepScratch scratch;
    PrepOutputs out;
    run_binning(js, world, out, scratch);

    ASSERT_EQ(world.packet.model_commands().size(), CMD_COUNT);
    // 7 survive, 1 is beyond the 100-unit draw radius, 2 are outside the
    // ortho box (one behind the camera, one 900 units up).
    EXPECT_EQ(out.stats.entities_rendered, 7u);
    EXPECT_EQ(out.stats.entities_distance_culled, 1u);
    EXPECT_EQ(out.stats.entities_frustum_culled, 2u);
    EXPECT_EQ(out.stats.entities_rendered + out.stats.entities_distance_culled + out.stats.entities_frustum_culled,
              CMD_COUNT);
    // Exactly one command names a handle no model answers to. NOT gated on
    // collect_stats — a drop is not opt-in instrumentation.
    EXPECT_EQ(out.stats.diagnostics.draws_dropped_unknown_model, 1u);
}

TEST(ScenePrepGolden, MainViewBinMembershipAndOrder) {
    GoldenWorld world;
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;
    PrepOutputs out;
    run_binning(js, world, out, scratch);

    // First-touch order, which is also the order the touched list is walked
    // when clearing: cube (cmd 0), slab (cmd 1), unknown (cmd 2).
    EXPECT_EQ(out.touched, (std::vector<ModelHandle>{H_CUBE, H_SLAB, H_UNKNOWN}));

    ASSERT_GT(out.bins.size(), H_UNKNOWN);
    ASSERT_EQ(out.bins[H_CUBE].size(), 3u);
    ASSERT_EQ(out.bins[H_SLAB].size(), 2u);
    ASSERT_EQ(out.bins[H_UNKNOWN].size(), 1u);
    // Every other bin (including the reserved handle 0) must be empty.
    for (ModelHandle h = 0; h < out.bins.size(); ++h) {
        if (h == H_CUBE || h == H_SLAB || h == H_UNKNOWN) {
            continue;
        }
        EXPECT_TRUE(out.bins[h].empty()) << "bin " << h;
    }

    // Within a bin, command order is preserved.
    EXPECT_EQ(out.bins[H_CUBE][0].tint, tint_for(CMD_CUBE_NEAR));
    EXPECT_EQ(out.bins[H_CUBE][1].tint, tint_for(CMD_CUBE_MOVING));
    EXPECT_EQ(out.bins[H_CUBE][2].tint, tint_for(CMD_CUBE_NO_FOG));
    EXPECT_EQ(out.bins[H_SLAB][0].tint, tint_for(CMD_SLAB_RIGHT));
    EXPECT_EQ(out.bins[H_SLAB][1].tint, tint_for(CMD_SLAB_LEFT));
    EXPECT_EQ(out.bins[H_UNKNOWN][0].tint, tint_for(CMD_UNKNOWN));

    // ForceNonInstanced bypasses the bins entirely and keeps its command
    // pointer (into the sealed packet) rather than being copied.
    ASSERT_EQ(out.non_instanced.size(), 1u);
    EXPECT_EQ(out.non_instanced[0], &world.packet.model_commands()[CMD_NON_INSTANCED]);
}

TEST(ScenePrepGolden, InstanceDataBytesAreDerivedNotCopied) {
    GoldenWorld world;
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;
    PrepOutputs out;
    run_binning(js, world, out, scratch);

    const gpu::InstanceData& still = out.bins[H_CUBE][0];
    EXPECT_EQ(still.model, at(0.0f, 0.0f, -50.0f));
    // Pure translation is uniform scale, so the normal matrix is the model
    // matrix itself (compute_normal_matrix's fast path).
    EXPECT_EQ(still.normalMatrix, still.model);
    EXPECT_FLOAT_EQ(still.noFog, 0.0f);
    // No motion id: previous == current, i.e. camera-only velocity.
    EXPECT_EQ(still.prevModel, still.model);

    // motion_id 5 was recorded last frame at a different position.
    const gpu::InstanceData& moving = out.bins[H_CUBE][1];
    EXPECT_EQ(moving.model, at(2.0f, 0.0f, -50.0f));
    EXPECT_EQ(moving.prevModel, PREV_MODEL_TRANSFORM);
    EXPECT_NE(moving.prevModel, moving.model);

    EXPECT_FLOAT_EQ(out.bins[H_CUBE][2].noFog, 1.0f);

    // The unregistered handle has no model and therefore no bounds — but its
    // instance data is built from the command alone and is complete.
    const gpu::InstanceData& unknown = out.bins[H_UNKNOWN][0];
    EXPECT_EQ(unknown.model, at(0.0f, 0.0f, -50.0f));
    EXPECT_EQ(unknown.normalMatrix, unknown.model);
    EXPECT_EQ(unknown.prevModel, unknown.model);
}

// (b) from the file header. The packing loop in upload_instance_buffers and
// the three draw walks (forward instanced, depth prepass instanced, instanced
// shadow) each re-derive base_instance by walking the dense bin array in
// ascending handle order; Phase 6 puts them in three different translation
// units. With two model types AND a bin whose handle resolves to nothing, any
// walk that skips the unresolvable bin without advancing base_instance draws
// every later batch with the wrong instances.
TEST(ScenePrepGolden, BaseInstanceOffsetsSurviveAnUnresolvableBin) {
    GoldenWorld world;
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;
    PrepOutputs out;
    run_binning(js, world, out, scratch);

    const std::vector<PackedBatch> batches = packed_batches(out.bins);
    ASSERT_EQ(batches.size(), 3u);
    EXPECT_EQ(batches[0], (PackedBatch{H_CUBE, 0, 3}));
    EXPECT_EQ(batches[1], (PackedBatch{H_SLAB, 3, 2}));
    EXPECT_EQ(batches[2], (PackedBatch{H_UNKNOWN, 5, 1}));

    // The unresolvable bin still occupies its slots in the packed buffer:
    // total instances == every bin's size, not just the drawable ones.
    uint32_t total = 0;
    for (ModelHandle h : out.touched) {
        total += static_cast<uint32_t>(out.bins[h].size());
    }
    EXPECT_EQ(total, 6u);
    EXPECT_EQ(batches.back().base_instance + batches.back().count, total);

    // Sanity: the model manager really does not resolve that handle, and does
    // resolve the other two — otherwise this test proves nothing.
    EXPECT_EQ(world.models.get_model(H_UNKNOWN), nullptr);
    EXPECT_NE(world.models.get_model(H_CUBE), nullptr);
    EXPECT_NE(world.models.get_model(H_SLAB), nullptr);
}

// ============================================================================
// Shadow views
// ============================================================================

TEST(ScenePrepGolden, ShadowCascadeZeroCastRules) {
    GoldenWorld world;
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;
    PrepOutputs out;
    run_binning(js, world, out, scratch);

    // Same first-touch order as the main view.
    EXPECT_EQ(out.shadow_touched[0], (std::vector<ModelHandle>{H_CUBE, H_SLAB, H_UNKNOWN}));
    ASSERT_GT(out.shadow_bins[0].size(), H_UNKNOWN);
    // cube: the three main-visible instanced commands PLUS the one behind the
    // camera (main-culled, but inside the light frustum, so it still casts).
    EXPECT_EQ(out.shadow_bins[0][H_CUBE].size(), 4u);
    EXPECT_EQ(out.shadow_bins[0][H_SLAB].size(), 2u);
    // (a) from the file header, shadow arm: no bounds is not "cull it".
    EXPECT_EQ(out.shadow_bins[0][H_UNKNOWN].size(), 1u);

    // The behind-the-camera caster is present, in command order (it follows
    // the near cube and precedes the moving one).
    EXPECT_EQ(out.shadow_bins[0][H_CUBE][0].model, at(0.0f, 0.0f, -50.0f));
    EXPECT_EQ(out.shadow_bins[0][H_CUBE][1].model, at(0.0f, 0.0f, 50.0f));
    EXPECT_EQ(out.shadow_bins[0][H_CUBE][2].model, at(2.0f, 0.0f, -50.0f));
    EXPECT_EQ(out.shadow_bins[0][H_CUBE][3].model, at(3.0f, 0.0f, -50.0f));

    // Never a shadow instance: the distance-culled command, the
    // ForceNonInstanced command (it draws through the per-draw shadow path),
    // and the one placed outside the cascade volume.
    for (const gpu::ShadowInstanceData& inst : out.shadow_bins[0][H_CUBE]) {
        EXPECT_NE(inst.model, at(0.0f, 0.0f, -300.0f));
        EXPECT_NE(inst.model, at(5.0f, 0.0f, -50.0f));
        EXPECT_NE(inst.model, at(0.0f, 900.0f, -50.0f));
    }
}

// Cascade 1's light frustum contains nothing at all — so it must bin exactly
// one thing: the command with no bounds, which casts unconditionally.
TEST(ScenePrepGolden, BoundlessCommandCastsIntoEveryCascade) {
    GoldenWorld world;
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;
    PrepOutputs out;
    run_binning(js, world, out, scratch);

    EXPECT_EQ(out.shadow_touched[1], (std::vector<ModelHandle>{H_UNKNOWN}));
    ASSERT_GT(out.shadow_bins[1].size(), H_UNKNOWN);
    EXPECT_EQ(out.shadow_bins[1][H_UNKNOWN].size(), 1u);
    EXPECT_EQ(out.shadow_bins[1][H_UNKNOWN][0].model, at(0.0f, 0.0f, -50.0f));
    EXPECT_TRUE(out.shadow_bins[1][H_CUBE].empty());
    EXPECT_TRUE(out.shadow_bins[1][H_SLAB].empty());

    // Inactive cascades are untouched.
    EXPECT_TRUE(out.shadow_touched[2].empty());
    EXPECT_TRUE(out.shadow_touched[3].empty());
}

// The shadow views recompute the main-view predicates but must count NOTHING:
// the main view owns every model-command statistic. Two cascades binning the
// same commands would otherwise triple the cull counters.
TEST(ScenePrepGolden, ShadowViewsContributeNoStats) {
    GoldenWorld world;
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;

    PrepOutputs with_shadows;
    {
        const ScenePrep::InstanceBinOutputs b = with_shadows.bindings();
        ScenePrep::prep_instance_batches(js, world.packet.model_commands(), world.params(2), scratch, b);
    }
    PrepOutputs no_shadows;
    {
        ScenePrep::PrepScratch scratch2;
        const ScenePrep::InstanceBinOutputs b = no_shadows.bindings();
        ScenePrep::prep_instance_batches(js, world.packet.model_commands(), world.params(0), scratch2, b);
    }

    EXPECT_EQ(with_shadows.stats.entities_rendered, no_shadows.stats.entities_rendered);
    EXPECT_EQ(with_shadows.stats.entities_distance_culled, no_shadows.stats.entities_distance_culled);
    EXPECT_EQ(with_shadows.stats.entities_frustum_culled, no_shadows.stats.entities_frustum_culled);
    EXPECT_EQ(with_shadows.stats.diagnostics.draws_dropped_unknown_model,
              no_shadows.stats.diagnostics.draws_dropped_unknown_model);
    // With no active cascades nothing is binned for shadows at all.
    for (const std::vector<ModelHandle>& touched : no_shadows.shadow_touched) {
        EXPECT_TRUE(touched.empty());
    }
}

// ============================================================================
// Skinned resolve
// ============================================================================

TEST(ScenePrepGolden, SkinnedResolveDropsAndOffsets) {
    GoldenWorld world;
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;
    std::vector<ScenePrep::ResolvedSkinnedEntry> resolved;
    FrameStatsPartial stats;
    ScenePrep::prep_resolve_skinned(js, world.packet.skinned_commands(), world.packet.palettes(), world.params(),
                                        scratch.skinned, resolved, stats);

    ASSERT_EQ(world.packet.skinned_commands().size(), SK_COUNT);
    // Two commands name something with no usable skeleton (one model without
    // one, one handle without a model) and are dropped here — the only place
    // every skinned command is visited exactly once.
    EXPECT_EQ(stats.diagnostics.skinned_dropped_no_skeleton, 2u);
    ASSERT_EQ(resolved.size(), 3u);

    // Entry 0: inside the frustum and the draw radius.
    EXPECT_EQ(resolved[0].cmd, &world.packet.skinned_commands()[SK_VISIBLE]);
    EXPECT_TRUE(resolved[0].main_visible);
    EXPECT_EQ(resolved[0].palette_offset, PALETTE_OFFSETS[0]);
    EXPECT_EQ(resolved[0].world_center, glm::vec3(0.0f, 0.0f, -50.0f)); // cube bounds centre + translation
    EXPECT_FLOAT_EQ(resolved[0].world_radius, glm::length(glm::vec3(2.0f)) * 0.5f);
    EXPECT_EQ(resolved[0].prev_model, resolved[0].cmd->transform); // motion_id 0
    EXPECT_EQ(resolved[0].prev_palette_offset, gpu::PrepassModelUniforms::NO_PREV_PALETTE);

    // Entry 1: past the draw radius. Still resolved (the shadow cascades and
    // the local-shadow selector consume it) but not main-visible.
    EXPECT_EQ(resolved[1].cmd, &world.packet.skinned_commands()[SK_TOO_FAR]);
    EXPECT_FALSE(resolved[1].main_visible);
    EXPECT_EQ(resolved[1].palette_offset, PALETTE_OFFSETS[1]);

    // Entry 2: inside the radius, outside the frustum, and it has history.
    EXPECT_EQ(resolved[2].cmd, &world.packet.skinned_commands()[SK_HIGH_ABOVE]);
    EXPECT_FALSE(resolved[2].main_visible);
    EXPECT_EQ(resolved[2].palette_offset, PALETTE_OFFSETS[2]);
    EXPECT_EQ(resolved[2].prev_model, PREV_SKINNED_TRANSFORM);
    EXPECT_EQ(resolved[2].prev_palette_offset, PREV_PALETTE_OFFSET);

    EXPECT_EQ(stats.entities_rendered, 1u);
    EXPECT_EQ(stats.entities_distance_culled, 1u);
    EXPECT_EQ(stats.entities_frustum_culled, 1u);
}

// The previous palette buffer is double-buffered: with prev_palette_valid
// false (first frame, or after a resize) the recorded offset must NOT be used,
// or the prepass reads last-run garbage as last frame's pose.
TEST(ScenePrepGolden, PrevPaletteOffsetIsGatedOnBufferValidity) {
    GoldenWorld world;
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;
    ScenePrep::PrepParams p = world.params();
    p.prev_palette_valid = false;
    std::vector<ScenePrep::ResolvedSkinnedEntry> resolved;
    FrameStatsPartial stats;
    ScenePrep::prep_resolve_skinned(js, world.packet.skinned_commands(), world.packet.palettes(), p,
                                        scratch.skinned, resolved, stats);
    ASSERT_EQ(resolved.size(), 3u);
    EXPECT_EQ(resolved[2].prev_palette_offset, gpu::PrepassModelUniforms::NO_PREV_PALETTE);
    // The prev TRANSFORM is still used — only the palette offset is gated.
    EXPECT_EQ(resolved[2].prev_model, PREV_SKINNED_TRANSFORM);
}

// The joint count drawn is min(model joints, palette joints, MAX_BONES). The
// golden skinned model has 4 joints and palette 0 has 3, so the clamp binds —
// and a palette with FEWER joints than the model must not read past its end.
TEST(ScenePrepGolden, PaletteOffsetsAreTheSealTimePrefixSum) {
    GoldenWorld world;
    const std::span<const BonePalette> palettes = world.packet.palettes();
    ASSERT_EQ(palettes.size(), 3u);
    for (std::size_t i = 0; i < palettes.size(); ++i) {
        EXPECT_EQ(palettes[i].joint_count, PALETTE_JOINTS[i]) << i;
        EXPECT_EQ(palettes[i].matrix_offset, PALETTE_OFFSETS[i]) << i;
    }
    EXPECT_EQ(world.packet.palette_matrix_total(), PALETTE_MATRIX_TOTAL);
    // The model has 4 joints, palette 0 has 3: the clamp is real, not vacuous.
    const Model* skinned = world.models.get_model(H_SKINNED);
    ASSERT_NE(skinned, nullptr);
    EXPECT_EQ(skinned->skeleton.joints.size(), 4u);
    EXPECT_LT(palettes[0].joint_count, skinned->skeleton.joints.size());
    EXPECT_LT(PALETTE_JOINTS[0], static_cast<uint32_t>(bracken::animation::MAX_BONES));
}

// ============================================================================
// Palette copy — the exact packed byte stream
// ============================================================================

TEST(ScenePrepGolden, PaletteCopyProducesTheConcatenatedStream) {
    GoldenWorld world;
    jobs::JobSystem js;

    // Sentinel-fill so any byte NOT covered by a palette range is detectable.
    const glm::mat4 sentinel(-98765.0f);
    std::vector<glm::mat4> packed(PALETTE_MATRIX_TOTAL, sentinel);
    ScenePrep::prep_copy_palettes(js, world.packet.palettes(), packed.data());

    // Expected: palette p's joint j holds mat4(p * 100 + j + 1), written at
    // PALETTE_OFFSETS[p] + j. Every slot is covered exactly once.
    for (std::size_t p = 0; p < 3; ++p) {
        for (std::size_t j = 0; j < PALETTE_JOINTS[p]; ++j) {
            const glm::mat4 expected(static_cast<float>(p * 100 + j + 1));
            EXPECT_EQ(packed[PALETTE_OFFSETS[p] + j], expected) << "palette " << p << " joint " << j;
        }
    }
    for (const glm::mat4& m : packed) {
        EXPECT_NE(m, sentinel);
    }
}

// ============================================================================
// The goldens must hold at every worker count too — otherwise a fan-out
// regression could satisfy the parity test (identical everywhere) while being
// identically WRONG.
// ============================================================================

TEST(ScenePrepGolden, GoldensHoldUnderTheFanOut) {
    GoldenWorld world;
    jobs::JobSystem js4;
    js4.init(4);
    ScenePrep::PrepScratch scratch;
    PrepOutputs out;
    run_binning(js4, world, out, scratch);

    EXPECT_EQ(out.stats.entities_rendered, 7u);
    EXPECT_EQ(out.stats.entities_distance_culled, 1u);
    EXPECT_EQ(out.stats.entities_frustum_culled, 2u);
    EXPECT_EQ(out.stats.diagnostics.draws_dropped_unknown_model, 1u);
    EXPECT_EQ(out.touched, (std::vector<ModelHandle>{H_CUBE, H_SLAB, H_UNKNOWN}));
    EXPECT_EQ(packed_batches(out.bins),
              (std::vector<PackedBatch>{{H_CUBE, 0, 3}, {H_SLAB, 3, 2}, {H_UNKNOWN, 5, 1}}));
    EXPECT_EQ(out.shadow_touched[1], (std::vector<ModelHandle>{H_UNKNOWN}));

    std::vector<ScenePrep::ResolvedSkinnedEntry> resolved;
    FrameStatsPartial stats;
    ScenePrep::prep_resolve_skinned(js4, world.packet.skinned_commands(), world.packet.palettes(), world.params(),
                                        scratch.skinned, resolved, stats);
    ASSERT_EQ(resolved.size(), 3u);
    EXPECT_EQ(stats.diagnostics.skinned_dropped_no_skeleton, 2u);
    EXPECT_EQ(resolved[2].prev_palette_offset, PREV_PALETTE_OFFSET);
}
