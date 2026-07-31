#include <gtest/gtest.h>

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/task_group.hpp"
#include "bracken/effects/effect_definition.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace bracken::scene;
using bracken::render::lighting::PointLight;
using bracken::render::lighting::SpotLight;

namespace {

// ---------------------------------------------------------------------------
// Canonical logical serialization of a sealed packet. Field layouts are
// padding-free with zeroed explicit pads, so POD spans append raw; text and
// palettes append by CONTENT (their pointers differ between packets).
// ---------------------------------------------------------------------------

void append_bytes(std::vector<std::byte>& out, const void* p, std::size_t n) {
    const auto* b = static_cast<const std::byte*>(p);
    out.insert(out.end(), b, b + n);
}

template<typename T> void append_pod(std::vector<std::byte>& out, const T& v) {
    append_bytes(out, &v, sizeof(T));
}

template<typename T> void append_span(std::vector<std::byte>& out, std::span<const T> s) {
    append_pod(out, static_cast<uint64_t>(s.size()));
    if (!s.empty()) {
        append_bytes(out, s.data(), s.size_bytes());
    }
}

std::vector<std::byte> canonical_bytes(const FramePacket& p) {
    std::vector<std::byte> out;
    append_span(out, p.model_commands());
    append_span(out, p.skinned_commands());
    append_span(out, p.point_lights());
    append_span(out, p.spot_lights());
    append_span(out, p.billboards());
    append_span(out, p.debug_lines());
    append_span(out, p.effect_spawns()); // definition pointers are process-stable

    append_pod(out, static_cast<uint64_t>(p.palettes().size()));
    append_pod(out, p.palette_matrix_total());
    for (const BonePalette& pal : p.palettes()) {
        append_pod(out, pal.joint_count);
        append_pod(out, pal.matrix_offset);
        append_bytes(out, pal.matrices, std::size_t{pal.joint_count} * sizeof(glm::mat4));
    }

    append_pod(out, static_cast<uint64_t>(p.ui_commands().size()));
    for (const UiCommand& cmd : p.ui_commands()) {
        append_pod(out, static_cast<uint32_t>(cmd.data.index()));
        std::visit(
            [&out](const auto& c) {
                using C = std::decay_t<decltype(c)>;
                if constexpr (std::is_same_v<C, UiTextCommand>) {
                    append_pod(out, c.x);
                    append_pod(out, c.y);
                    append_pod(out, c.scale);
                    append_pod(out, c.color);
                    append_pod(out, static_cast<uint64_t>(c.text.size()));
                    append_bytes(out, c.text.data(), c.text.size());
                } else if constexpr (std::is_same_v<C, UiButtonCommand>) {
                    append_pod(out, c.x);
                    append_pod(out, c.y);
                    append_pod(out, c.w);
                    append_pod(out, c.h);
                    append_pod(out, c.color);
                    append_pod(out, c.selected);
                    append_pod(out, static_cast<uint64_t>(c.label.size()));
                    append_bytes(out, c.label.data(), c.label.size());
                } else {
                    append_pod(out, c);
                }
            },
            cmd.data);
    }

    append_pod(out, p.has_camera());
    append_pod(out, p.camera().view);
    append_pod(out, p.camera().projection);
    append_pod(out, p.camera().view_projection);
    append_pod(out, p.camera().position);
    append_pod(out, p.has_environment());
    append_pod(out, p.should_draw_skybox());
    append_pod(out, p.should_draw_ground());
    append_pod(out, p.should_draw_grass());
    append_pod(out, p.dt());
    append_pod(out, p.time());
    append_pod(out, p.frame_index());
    return out;
}

glm::mat4 translation(float x, float y, float z) {
    glm::mat4 m(1.0f);
    m[3] = glm::vec4(x, y, z, 1.0f);
    return m;
}

} // namespace

// ---------------------------------------------------------------------------
// Single-lane emission: RenderScene-parity content
// ---------------------------------------------------------------------------

TEST(FrameBuilder, EmitsEveryCommandTypeWithExactValues) {
    static const bracken::EffectDefinition effect_def{};

    FramePacket p;
    FrameBuilder fb(p);

    fb.draw_model(42, translation(1, 2, 3), {0.5f, 0.6f, 0.7f, 0.8f},
                  DrawFlags::ForceNonInstanced | DrawFlags::NoFog, 77);
    auto pal = fb.alloc_bone_palette(2);
    ASSERT_TRUE(pal.valid());
    pal.matrices[0] = glm::mat4(2.0f);
    pal.matrices[1] = glm::mat4(3.0f);
    fb.draw_skinned(43, translation(4, 5, 6), pal.id, {0.1f, 0.2f, 0.3f, 0.4f}, DrawFlags::NoFog, 78);
    fb.add_point_light(PointLight{{1, 2, 3}, 320.0f, {1.0f, 0.7f, 0.3f}, 12000.0f, true});
    fb.add_spot_light(SpotLight{{4, 5, 6}, 1400.0f, {0, -1, 0}, 0.95f, {1, 1, 1}, 0.9f, 5000.0f, false});
    fb.add_billboard_3d(7, 8, 9, 46.0f, 0.5f, 0xFFFFC040, 0x60000000, 0xFFFFD080);
    fb.add_debug_line({0, 0, 0}, {1, 1, 1}, 0xFF30FF60);
    fb.spawn_effect(&effect_def, {10, 11, 12}, {0, 1, 0});
    fb.seal();

    ASSERT_EQ(p.model_commands().size(), 1u);
    const ModelDrawCommand& mc = p.model_commands()[0];
    EXPECT_EQ(mc.model_handle, 42u);
    EXPECT_EQ(mc.transform, translation(1, 2, 3));
    EXPECT_EQ(mc.tint, glm::vec4(0.5f, 0.6f, 0.7f, 0.8f));
    EXPECT_TRUE(has(mc.flags, DrawFlags::ForceNonInstanced));
    EXPECT_TRUE(has(mc.flags, DrawFlags::NoFog));
    EXPECT_EQ(mc.motion_id, 77u);

    ASSERT_EQ(p.skinned_commands().size(), 1u);
    const SkinnedDrawCommand& sc = p.skinned_commands()[0];
    EXPECT_EQ(sc.model_handle, 43u);
    EXPECT_EQ(sc.transform, translation(4, 5, 6));
    EXPECT_EQ(sc.tint, glm::vec4(0.1f, 0.2f, 0.3f, 0.4f));
    EXPECT_TRUE(has(sc.flags, DrawFlags::NoFog));
    EXPECT_EQ(sc.motion_id, 78u);
    ASSERT_TRUE(sc.palette.valid());
    EXPECT_EQ(sc.palette.value, 0u); // first (only) merged palette

    ASSERT_EQ(p.palettes().size(), 1u);
    EXPECT_EQ(p.palettes()[0].joint_count, 2u);
    EXPECT_EQ(p.palettes()[0].matrix_offset, 0u);
    EXPECT_EQ(p.palette_matrix_total(), 2u);
    auto mats = p.palette_matrices(sc.palette);
    ASSERT_EQ(mats.size(), 2u);
    EXPECT_EQ(mats[0], glm::mat4(2.0f));
    EXPECT_EQ(mats[1], glm::mat4(3.0f));

    ASSERT_EQ(p.point_lights().size(), 1u);
    EXPECT_EQ(p.point_lights()[0].radius, 320.0f);
    EXPECT_TRUE(p.point_lights()[0].casts_shadows);
    ASSERT_EQ(p.spot_lights().size(), 1u);
    EXPECT_EQ(p.spot_lights()[0].inner_cos, 0.95f);

    ASSERT_EQ(p.billboards().size(), 1u);
    EXPECT_EQ(p.billboards()[0].width, 46.0f);
    EXPECT_EQ(p.billboards()[0].fill_color, 0xFFFFC040u);

    ASSERT_EQ(p.debug_lines().size(), 1u);
    EXPECT_EQ(p.debug_lines()[0].color, 0xFF30FF60u);

    ASSERT_EQ(p.effect_spawns().size(), 1u);
    EXPECT_EQ(p.effect_spawns()[0].definition, &effect_def);
    EXPECT_EQ(p.effect_spawns()[0].position, glm::vec3(10, 11, 12));

    EXPECT_TRUE(p.has_3d_content());
}

TEST(FrameBuilder, DebugSphereAndBoxExpandLikeRenderScene) {
    FramePacket p;
    FrameBuilder fb(p);
    fb.add_debug_sphere({0, 0, 0}, 1.0f, 0xFF0000FF, 16); // 3 circles * 16 segments
    fb.add_debug_box({0, 0, 0}, {1, 1, 1}, 0xFF00FF00);   // 12 edges
    fb.seal();
    EXPECT_EQ(p.debug_lines().size(), 3u * 16u + 12u);
}

TEST(FrameBuilder, SingularStateAndToggles) {
    FramePacket p;
    FrameBuilder fb(p);
    const CameraState cam{glm::mat4(2.0f), glm::mat4(3.0f), glm::mat4(4.0f), glm::vec3(1, 2, 3)};
    fb.set_camera(cam);
    bracken::render::EnvironmentState env;
    env.fog_start = 123.0f;
    fb.set_environment(env);
    fb.set_draw_skybox(false);
    fb.set_draw_ground(false);
    fb.set_draw_grass(false);
    fb.set_frame_timing(0.016f, 3.5, 210);
    fb.seal();

    EXPECT_TRUE(p.has_camera());
    EXPECT_EQ(p.camera().position, glm::vec3(1, 2, 3));
    EXPECT_EQ(p.camera().view, glm::mat4(2.0f));
    EXPECT_TRUE(p.has_environment());
    EXPECT_EQ(p.environment().fog_start, 123.0f);
    EXPECT_FALSE(p.should_draw_skybox());
    EXPECT_FALSE(p.should_draw_ground());
    EXPECT_FALSE(p.should_draw_grass());
    EXPECT_EQ(p.dt(), 0.016f);
    EXPECT_EQ(p.time(), 3.5);
    EXPECT_EQ(p.frame_index(), 210u);
    EXPECT_FALSE(p.has_3d_content()); // everything toggled off, no commands
}

TEST(FrameBuilder, EmptySealedFrameHasDefaults) {
    FramePacket p;
    FrameBuilder fb(p);
    fb.seal();
    EXPECT_TRUE(p.model_commands().empty());
    EXPECT_TRUE(p.skinned_commands().empty());
    EXPECT_TRUE(p.ui_commands().empty());
    EXPECT_TRUE(p.palettes().empty());
    EXPECT_FALSE(p.has_camera());
    EXPECT_FALSE(p.has_environment());
    EXPECT_TRUE(p.has_3d_content()); // world-element toggles default true
}

// ---------------------------------------------------------------------------
// Chunk spill: chains larger than one chunk keep count + order
// ---------------------------------------------------------------------------

TEST(FrameBuilder, ChunkSpillPreservesOrderAndCount) {
    constexpr uint32_t N = 300; // sizeof(ModelDrawCommand) puts ~22 per chunk
    FramePacket p;
    FrameBuilder fb(p);
    for (uint32_t i = 0; i < N; ++i) {
        fb.draw_model(i + 1, translation(float(i), 0, 0), {1, 1, 1, 1}, DrawFlags::None, i);
    }
    fb.seal();
    ASSERT_EQ(p.model_commands().size(), N);
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(p.model_commands()[i].motion_id, i);
        EXPECT_EQ(p.model_commands()[i].model_handle, i + 1);
    }
}

// ---------------------------------------------------------------------------
// Lane merge order: lane-major, emission order within a lane
// ---------------------------------------------------------------------------

TEST(FrameBuilder, LaneMergeOrderIsLaneMajor) {
    FramePacket p;
    FrameBuilder fb(p, 3);
    // Deliberately interleaved emission across lanes.
    fb.lane(2).add_debug_line({2, 0, 0}, {2, 1, 0}, 200);
    fb.lane(0).add_debug_line({0, 0, 0}, {0, 1, 0}, 0);
    fb.lane(1).add_debug_line({1, 0, 0}, {1, 1, 0}, 100);
    fb.lane(0).add_debug_line({0, 0, 1}, {0, 1, 1}, 1);
    fb.lane(2).add_debug_line({2, 0, 1}, {2, 1, 1}, 201);
    fb.seal();

    ASSERT_EQ(p.debug_lines().size(), 5u);
    EXPECT_EQ(p.debug_lines()[0].color, 0u);   // lane 0, first
    EXPECT_EQ(p.debug_lines()[1].color, 1u);   // lane 0, second
    EXPECT_EQ(p.debug_lines()[2].color, 100u); // lane 1
    EXPECT_EQ(p.debug_lines()[3].color, 200u); // lane 2, first
    EXPECT_EQ(p.debug_lines()[4].color, 201u); // lane 2, second
}

// ---------------------------------------------------------------------------
// Bone palettes: alignment, offsets, id stability, cross-lane use, budget
// ---------------------------------------------------------------------------

TEST(FrameBuilder, PaletteAllocationIsAlignedAndOffsetsPrefixSum) {
    FramePacket p;
    FrameBuilder fb(p, 2);
    const uint32_t joints[] = {3, 5, 2};
    std::vector<FrameBuilder::BonePaletteAllocation> allocs;
    allocs.push_back(fb.lane(0).alloc_bone_palette(joints[0]));
    allocs.push_back(fb.lane(0).alloc_bone_palette(joints[1]));
    allocs.push_back(fb.lane(1).alloc_bone_palette(joints[2]));
    for (std::size_t a = 0; a < allocs.size(); ++a) {
        ASSERT_TRUE(allocs[a].valid());
        ASSERT_EQ(allocs[a].matrices.size(), joints[a]);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(allocs[a].matrices.data()) % 16, 0u);
        for (uint32_t m = 0; m < joints[a]; ++m) {
            allocs[a].matrices[m] = glm::mat4(float(a * 100 + m));
        }
        fb.draw_skinned(uint32_t(a + 1), glm::mat4(1.0f), allocs[a].id);
    }
    fb.seal();

    // Merged in (lane, allocation) order == allocation order here.
    ASSERT_EQ(p.palettes().size(), 3u);
    EXPECT_EQ(p.palettes()[0].joint_count, 3u);
    EXPECT_EQ(p.palettes()[0].matrix_offset, 0u);
    EXPECT_EQ(p.palettes()[1].joint_count, 5u);
    EXPECT_EQ(p.palettes()[1].matrix_offset, 3u);
    EXPECT_EQ(p.palettes()[2].joint_count, 2u);
    EXPECT_EQ(p.palettes()[2].matrix_offset, 8u);
    EXPECT_EQ(p.palette_matrix_total(), 10u);

    // Id stability: each sealed command's palette id resolves to the exact
    // matrices its allocation was filled with.
    ASSERT_EQ(p.skinned_commands().size(), 3u);
    for (const SkinnedDrawCommand& cmd : p.skinned_commands()) {
        const std::size_t a = cmd.model_handle - 1;
        auto mats = p.palette_matrices(cmd.palette);
        ASSERT_EQ(mats.size(), joints[a]);
        for (uint32_t m = 0; m < joints[a]; ++m) {
            EXPECT_EQ(mats[m], glm::mat4(float(a * 100 + m)));
        }
    }
}

TEST(FrameBuilder, PaletteIdFromAnotherLaneIsUsable) {
    FramePacket p;
    FrameBuilder fb(p, 2);
    auto pal = fb.lane(0).alloc_bone_palette(2); // shared palette (e.g. identity)
    ASSERT_TRUE(pal.valid());
    pal.matrices[0] = glm::mat4(1.0f);
    pal.matrices[1] = glm::mat4(1.0f);
    fb.lane(1).draw_skinned(5, glm::mat4(1.0f), pal.id); // used from lane 1
    fb.lane(0).draw_skinned(6, glm::mat4(1.0f), pal.id);
    fb.seal();

    ASSERT_EQ(p.palettes().size(), 1u);
    ASSERT_EQ(p.skinned_commands().size(), 2u);
    EXPECT_EQ(p.skinned_commands()[0].palette.value, 0u); // lane 0 command
    EXPECT_EQ(p.skinned_commands()[1].palette.value, 0u); // lane 1 command
}

TEST(FrameBuilder, PaletteBudgetFailsVisiblyAtBuildTime) {
    FramePacket p;
    FrameBuilder fb(p);
    for (uint32_t i = 0; i < FramePacket::MAX_BONE_PALETTES; ++i) {
        auto a = fb.alloc_bone_palette(1);
        ASSERT_TRUE(a.valid()) << "allocation " << i << " under budget must succeed";
        a.matrices[0] = glm::mat4(1.0f);
        fb.draw_skinned(1, glm::mat4(1.0f), a.id);
    }
    // Over budget: invalid allocation, and the dependent draw is dropped.
    auto over = fb.alloc_bone_palette(1);
    EXPECT_FALSE(over.valid());
    EXPECT_TRUE(over.matrices.empty());
    fb.draw_skinned(2, glm::mat4(1.0f), over.id);
    fb.seal();

    EXPECT_EQ(p.palettes().size(), FramePacket::MAX_BONE_PALETTES);
    EXPECT_EQ(p.skinned_commands().size(), FramePacket::MAX_BONE_PALETTES);
}

// ---------------------------------------------------------------------------
// UI commands
// ---------------------------------------------------------------------------

TEST(FrameBuilder, UiCommandsMirrorUiScene) {
    FramePacket p;
    FrameBuilder fb(p);
    auto ui = fb.ui();
    ui.add_filled_rect(1, 2, 3, 4, 0xB0101418);
    ui.add_rect_outline(5, 6, 7, 8, 0xFF00FF00, 3.0f);
    ui.add_circle(9, 10, 11, 0xFF0000FF, 32);
    ui.add_circle_outline(12, 13, 14, 0xFFFF00FF, 1.5f, 12);
    ui.add_line(1, 2, 3, 4, 0xFFFFFFFF, 4.0f);
    ui.add_text("bracken playground", 22, 20, 1.0f, 0xFFFFD080);
    ui.add_button(15, 16, 17, 18, "OK", 0xFF808080, true);
    fb.seal();

    ASSERT_EQ(p.ui_commands().size(), 7u);
    ASSERT_TRUE(p.ui_commands()[0].is<UiFilledRectCommand>());
    EXPECT_EQ(p.ui_commands()[0].get<UiFilledRectCommand>().color, 0xB0101418u);
    ASSERT_TRUE(p.ui_commands()[1].is<UiRectOutlineCommand>());
    EXPECT_EQ(p.ui_commands()[1].get<UiRectOutlineCommand>().line_width, 3.0f);
    ASSERT_TRUE(p.ui_commands()[2].is<UiCircleCommand>());
    EXPECT_EQ(p.ui_commands()[2].get<UiCircleCommand>().segments, 32);
    ASSERT_TRUE(p.ui_commands()[3].is<UiCircleOutlineCommand>());
    EXPECT_EQ(p.ui_commands()[3].get<UiCircleOutlineCommand>().segments, 12);
    ASSERT_TRUE(p.ui_commands()[4].is<UiLineCommand>());
    EXPECT_EQ(p.ui_commands()[4].get<UiLineCommand>().line_width, 4.0f);
    ASSERT_TRUE(p.ui_commands()[5].is<UiTextCommand>());
    EXPECT_EQ(p.ui_commands()[5].get<UiTextCommand>().text, "bracken playground");
    ASSERT_TRUE(p.ui_commands()[6].is<UiButtonCommand>());
    EXPECT_EQ(p.ui_commands()[6].get<UiButtonCommand>().label, "OK");
    EXPECT_TRUE(p.ui_commands()[6].get<UiButtonCommand>().selected);
}

TEST(FrameBuilder, UiTextIsCopiedIntoPacketMemory) {
    std::string transient = "short lived";
    std::string long_text(4000, 'x'); // larger than a text block
    FramePacket p;
    FrameBuilder fb(p);
    fb.ui().add_text(transient, 0, 0);
    fb.ui().add_text(long_text, 0, 20);
    transient.assign("clobbered!!");
    long_text.assign("gone");
    fb.seal();

    ASSERT_EQ(p.ui_commands().size(), 2u);
    EXPECT_EQ(p.ui_commands()[0].get<UiTextCommand>().text, "short lived");
    const std::string_view stored = p.ui_commands()[1].get<UiTextCommand>().text;
    ASSERT_EQ(stored.size(), 4000u);
    EXPECT_EQ(stored, std::string(4000, 'x'));
}

TEST(FrameBuilder, UiTextWrappedMatchesUiSceneAlgorithm) {
    FramePacket p;
    FrameBuilder fb(p);
    // max_width 64 at scale 1 => 8 chars per line; "aaa bbb ccc" wraps at the
    // second space: ["aaa bbb", "ccc"], height 2 * 18.
    const float height = fb.ui().add_text_wrapped("aaa bbb ccc", 10, 10, 64.0f);
    EXPECT_EQ(height, 36.0f);
    fb.seal();
    ASSERT_EQ(p.ui_commands().size(), 2u);
    EXPECT_EQ(p.ui_commands()[0].get<UiTextCommand>().text, "aaa bbb");
    EXPECT_EQ(p.ui_commands()[0].get<UiTextCommand>().y, 10.0f);
    EXPECT_EQ(p.ui_commands()[1].get<UiTextCommand>().text, "ccc");
    EXPECT_EQ(p.ui_commands()[1].get<UiTextCommand>().y, 28.0f);
}

// ---------------------------------------------------------------------------
// Multi-lane determinism: emit from N TaskGroup jobs (lane = job index),
// merged output must be lane/chunk deterministic, content-complete, and
// byte-identical across independent runs.
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t k_det_lanes = 8;
constexpr uint32_t k_det_per_lane = 97; // spills several chunks per chain
constexpr uint32_t k_det_palette_stride = 7;

void build_deterministic_frame(bracken::core::jobs::JobSystem& js, FramePacket& pkt) {
    FrameBuilder fb(pkt, k_det_lanes);
    bracken::core::jobs::TaskGroup group;
    for (uint32_t i = 0; i < k_det_lanes; ++i) {
        js.submit(group, [&fb, i] {
            FrameBuilder::Lane& lane = fb.lane(i);
            for (uint32_t j = 0; j < k_det_per_lane; ++j) {
                const uint32_t tag = i * 100000 + j;
                lane.draw_model(i * 1000 + j + 1, glm::mat4(float(j + 1)), {1, 1, 1, 1}, DrawFlags::None, tag);
                lane.add_debug_line({float(i), 0, 0}, {0, float(j), 0}, tag);
                if (j % k_det_palette_stride == 0) {
                    auto pal = lane.alloc_bone_palette(4);
                    if (pal.valid()) {
                        for (uint32_t m = 0; m < 4; ++m) {
                            pal.matrices[m] = glm::mat4(float(i + j + m));
                        }
                        lane.draw_skinned(i + 1, glm::mat4(float(j + 1)), pal.id, {1, 1, 1, 1}, DrawFlags::None, tag);
                    }
                }
            }
        });
    }
    js.wait(group); // all lane writes happen-before the seal
    fb.set_frame_timing(0.016f, 1.0, 7);
    fb.seal();
}

} // namespace

TEST(FrameBuilder, MultiLaneMergeIsDeterministicAndContentComplete) {
    bracken::core::jobs::JobSystem js;
    js.init(4);

    FramePacket p1;
    build_deterministic_frame(js, p1);
    const std::vector<std::byte> run1 = canonical_bytes(p1);

    // Content-complete + lane-major deterministic order.
    constexpr uint32_t palettes_per_lane = (k_det_per_lane + k_det_palette_stride - 1) / k_det_palette_stride;
    ASSERT_EQ(p1.model_commands().size(), std::size_t{k_det_lanes} * k_det_per_lane);
    ASSERT_EQ(p1.debug_lines().size(), std::size_t{k_det_lanes} * k_det_per_lane);
    ASSERT_EQ(p1.skinned_commands().size(), std::size_t{k_det_lanes} * palettes_per_lane);
    ASSERT_EQ(p1.palettes().size(), std::size_t{k_det_lanes} * palettes_per_lane);
    EXPECT_EQ(p1.palette_matrix_total(), k_det_lanes * palettes_per_lane * 4u);
    std::size_t idx = 0;
    for (uint32_t i = 0; i < k_det_lanes; ++i) {
        for (uint32_t j = 0; j < k_det_per_lane; ++j, ++idx) {
            ASSERT_EQ(p1.model_commands()[idx].motion_id, i * 100000 + j) << "merge order broke at " << idx;
            ASSERT_EQ(p1.debug_lines()[idx].color, i * 100000 + j);
        }
    }
    // Skinned palette ids: dense lane-major indices, resolving to the exact
    // matrices their lane wrote.
    idx = 0;
    for (uint32_t i = 0; i < k_det_lanes; ++i) {
        for (uint32_t k = 0; k < palettes_per_lane; ++k, ++idx) {
            const SkinnedDrawCommand& cmd = p1.skinned_commands()[idx];
            const uint32_t j = k * k_det_palette_stride;
            ASSERT_EQ(cmd.motion_id, i * 100000 + j);
            ASSERT_EQ(cmd.palette.value, i * palettes_per_lane + k);
            const auto mats = p1.palette_matrices(cmd.palette);
            ASSERT_EQ(mats.size(), 4u);
            for (uint32_t m = 0; m < 4; ++m) {
                ASSERT_EQ(mats[m], glm::mat4(float(i + j + m)));
            }
            ASSERT_EQ(p1.palettes()[cmd.palette.value].matrix_offset, cmd.palette.value * 4u);
        }
    }

    // Run 2: a different packet instance — byte-identical logical content.
    FramePacket p2;
    build_deterministic_frame(js, p2);
    EXPECT_EQ(run1, canonical_bytes(p2));

    // Run 3: ring-style reuse of packet 1 (reset + rebuild) — still identical.
    p1.reset();
    build_deterministic_frame(js, p1);
    EXPECT_EQ(run1, canonical_bytes(p1));

    js.shutdown();
}

// ---------------------------------------------------------------------------
// Debug contract asserts. Suite name ends in "DeathTest" so gtest schedules
// these before threaded tests.
// ---------------------------------------------------------------------------

#ifndef NDEBUG

TEST(FrameBuilderDeathTest, EmitAfterSealAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            FrameBuilder fb(p);
            fb.seal();
            fb.draw_model(1, glm::mat4(1.0f));
        },
        "outside the Filling state");
}

TEST(FrameBuilderDeathTest, UiEmitAfterSealAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            FrameBuilder fb(p);
            auto ui = fb.ui();
            fb.seal();
            ui.add_text("late", 0, 0);
        },
        "outside the Filling state");
}

TEST(FrameBuilderDeathTest, SetCameraAfterSealAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            FrameBuilder fb(p);
            fb.seal();
            fb.set_camera(CameraState{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(0)});
        },
        "outside the Filling state");
}

TEST(FrameBuilderDeathTest, SealTwiceAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            FrameBuilder fb(p);
            fb.seal();
            fb.seal();
        },
        "seal called twice");
}

TEST(FrameBuilderDeathTest, LaneIndexOutOfRangeAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            FrameBuilder fb(p, 2);
            fb.lane(2).add_debug_line({0, 0, 0}, {1, 1, 1}, 0);
        },
        "out of range");
}

#endif // NDEBUG
