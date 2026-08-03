#include "bracken/scene/frame_builder.hpp"

#include "bracken/animation/animation_types.hpp" // MAX_BONES (the one authoritative definition)
#include "bracken/core/assert.hpp"
#include "bracken/core/log_throttle.hpp"
#include "bracken/core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <type_traits>

namespace bracken::scene {

namespace {

// Lane-stored payloads are memcpy-merged at seal and reclaimed by a raw arena
// reset — they must stay trivially copyable and trivially destructible.
template<typename T>
constexpr bool k_lane_storable = std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

static_assert(k_lane_storable<ModelDrawCommand>);
static_assert(k_lane_storable<SkinnedDrawCommand>);
static_assert(k_lane_storable<bracken::render::lighting::PointLight>);
static_assert(k_lane_storable<bracken::render::lighting::SpotLight>);
static_assert(k_lane_storable<BillboardDrawCommand>);
static_assert(k_lane_storable<DebugLineDrawCommand>);
static_assert(k_lane_storable<EffectSpawnCommand>);
static_assert(k_lane_storable<UiCommand>); // variant of trivially copyable alternatives
static_assert(k_lane_storable<BonePalette>);

} // namespace

// Lanes live in the packet arena and are never destroyed — only reclaimed
// wholesale by the arena reset.
static_assert(std::is_trivially_destructible_v<FrameBuilder::Lane>);

// ============================================================================
// FrameBuilder
// ============================================================================

FrameBuilder::FrameBuilder(FramePacket& packet, uint32_t lane_count) : packet_(&packet), lane_count_(lane_count) {
    BRACKEN_VERIFY(lane_count >= 1 && lane_count <= k_max_lanes, "FrameBuilder: lane_count {} out of range [1, {}]",
                   lane_count, k_max_lanes);
    packet.begin_fill();
    void* mem = packet.reserve_block(sizeof(Lane) * lane_count, alignof(Lane));
    lanes_ = static_cast<Lane*>(mem);
    for (uint32_t i = 0; i < lane_count; ++i) {
        Lane* lane = new (&lanes_[i]) Lane();
        lane->packet_ = &packet;
        lane->index_ = i;
    }
}

// ---- Singular frame state ------------------------------------------------

void FrameBuilder::set_camera(const CameraState& camera) {
    BRACKEN_ASSERT(packet_->state() == FramePacket::State::Filling,
                   "FrameBuilder write outside the Filling state (set_camera after seal?)");
    packet_->camera_ = camera;
    packet_->has_camera_ = true;
}

void FrameBuilder::set_environment(const bracken::render::EnvironmentState& environment) {
    BRACKEN_ASSERT(packet_->state() == FramePacket::State::Filling,
                   "FrameBuilder write outside the Filling state (set_environment after seal?)");
    packet_->environment_ = environment;
    packet_->has_environment_ = true;
}

void FrameBuilder::set_draw_skybox(bool draw) {
    BRACKEN_ASSERT(packet_->state() == FramePacket::State::Filling,
                   "FrameBuilder write outside the Filling state (set_draw_skybox after seal?)");
    packet_->draw_skybox_ = draw;
}

void FrameBuilder::set_draw_ground(bool draw) {
    BRACKEN_ASSERT(packet_->state() == FramePacket::State::Filling,
                   "FrameBuilder write outside the Filling state (set_draw_ground after seal?)");
    packet_->draw_ground_ = draw;
}

void FrameBuilder::set_draw_grass(bool draw) {
    BRACKEN_ASSERT(packet_->state() == FramePacket::State::Filling,
                   "FrameBuilder write outside the Filling state (set_draw_grass after seal?)");
    packet_->draw_grass_ = draw;
}

void FrameBuilder::set_frame_timing(float dt, double time, uint64_t frame_index) {
    BRACKEN_ASSERT(packet_->state() == FramePacket::State::Filling,
                   "FrameBuilder write outside the Filling state (set_frame_timing after seal?)");
    packet_->dt_ = dt;
    packet_->time_ = time;
    packet_->frame_index_ = frame_index;
}

// ---- Engine-internal sim-stage API ---------------------------------------

FrameBuilder::EffectSnapshotAllocation FrameBuilder::alloc_effect_snapshot(uint32_t emitter_count,
                                                                           uint32_t particle_count) {
    BRACKEN_ASSERT(packet_->state() == FramePacket::State::Filling,
                   "FrameBuilder write outside the Filling state (alloc_effect_snapshot after seal?)");
    BRACKEN_ASSERT(packet_->effect_emitters_.empty() && packet_->effect_particles_.empty(),
                   "alloc_effect_snapshot called twice in one fill");
    EffectSnapshotAllocation alloc;
    if (emitter_count > 0) {
        auto* emitters = static_cast<EffectEmitterSnapshot*>(
            packet_->reserve_block(sizeof(EffectEmitterSnapshot) * emitter_count, alignof(EffectEmitterSnapshot)));
        alloc.emitters = {emitters, emitter_count};
        packet_->effect_emitters_ = {emitters, emitter_count};
    }
    if (particle_count > 0) {
        auto* particles = static_cast<EffectParticleSnapshot*>(
            packet_->reserve_block(sizeof(EffectParticleSnapshot) * particle_count, alignof(EffectParticleSnapshot)));
        alloc.particles = {particles, particle_count};
        packet_->effect_particles_ = {particles, particle_count};
    }
    return alloc;
}

void FrameBuilder::set_engine_frame(float scene_time, float sky_time, uint32_t effect_instance_count) {
    BRACKEN_ASSERT(packet_->state() == FramePacket::State::Filling,
                   "FrameBuilder write outside the Filling state (set_engine_frame after seal?)");
    packet_->scene_time_ = scene_time;
    packet_->sky_time_ = sky_time;
    packet_->effect_instance_count_ = effect_instance_count;
    packet_->has_engine_frame_ = true;
}

// ---- Seal / merge --------------------------------------------------------

template<typename T> std::span<const T> FrameBuilder::merge_chain(Chain<T> Lane::* member) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < lane_count_; ++i) {
        total += (lanes_[i].*member).total;
    }
    if (total == 0) {
        return {};
    }
    auto* dst = static_cast<T*>(packet_->reserve_block(sizeof(T) * total, alignof(T)));
    T* out = dst;
    for (uint32_t i = 0; i < lane_count_; ++i) {
        for (ChunkHeader* c = (lanes_[i].*member).head; c != nullptr; c = c->next) {
            std::memcpy(out, chunk_items<T>(c), c->count * sizeof(T));
            out += c->count;
        }
    }
    return {dst, total};
}

std::span<const SkinnedDrawCommand> FrameBuilder::merge_skinned(const uint32_t* palette_base) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < lane_count_; ++i) {
        total += lanes_[i].skinned_.total;
    }
    if (total == 0) {
        return {};
    }
    auto* dst = static_cast<SkinnedDrawCommand*>(
        packet_->reserve_block(sizeof(SkinnedDrawCommand) * total, alignof(SkinnedDrawCommand)));
    SkinnedDrawCommand* out = dst;
    for (uint32_t i = 0; i < lane_count_; ++i) {
        for (ChunkHeader* c = lanes_[i].skinned_.head; c != nullptr; c = c->next) {
            const SkinnedDrawCommand* items = chunk_items<SkinnedDrawCommand>(c);
            for (uint32_t j = 0; j < c->count; ++j) {
                SkinnedDrawCommand cmd = items[j];
                // Remap the fill-time (lane, local) palette id to its dense
                // merged index (lane-major allocation order — deterministic).
                const uint32_t lane_bits = cmd.palette.value >> k_palette_lane_shift;
                const uint32_t local = cmd.palette.value & k_palette_local_mask;
                BRACKEN_ASSERT(lane_bits < lane_count_ && local < lanes_[lane_bits].palettes_.total,
                               "draw_skinned: bone palette id {:#x} does not belong to this frame", cmd.palette.value);
                if (lane_bits >= lane_count_ || local >= lanes_[lane_bits].palettes_.total) {
                    continue; // release-mode safety: drop rather than mis-index
                }
                cmd.palette.value = palette_base[lane_bits] + local;
                new (out++) SkinnedDrawCommand{cmd};
            }
        }
    }
    return {dst, static_cast<std::size_t>(out - dst)};
}

void FrameBuilder::seal() {
    // Contracts: a second seal re-merges lanes into fresh arena blocks and
    // republishes them under a packet the renderer may already be consuming.
    BRACKEN_CONTRACT(!sealed_, "FrameBuilder::seal called twice");
    FramePacket& pkt = *packet_;
    BRACKEN_CONTRACT(pkt.state() == FramePacket::State::Filling, "FrameBuilder::seal: packet not Filling (state {})",
                     static_cast<uint32_t>(pkt.state()));

    // Palette records first: merged in (lane, allocation) order with matrix
    // prefix sums; palette_base[lane] is the remap base for the skinned merge.
    auto* palette_base = static_cast<uint32_t*>(pkt.reserve_block(sizeof(uint32_t) * lane_count_, alignof(uint32_t)));
    uint32_t palette_total = 0;
    for (uint32_t i = 0; i < lane_count_; ++i) {
        palette_base[i] = palette_total;
        palette_total += lanes_[i].palettes_.total;
    }
    uint32_t matrix_total = 0;
    if (palette_total > 0) {
        auto* records =
            static_cast<BonePalette*>(pkt.reserve_block(sizeof(BonePalette) * palette_total, alignof(BonePalette)));
        uint32_t idx = 0;
        for (uint32_t i = 0; i < lane_count_; ++i) {
            for (ChunkHeader* c = lanes_[i].palettes_.head; c != nullptr; c = c->next) {
                const PaletteFill* fills = chunk_items<PaletteFill>(c);
                for (uint32_t j = 0; j < c->count; ++j) {
                    new (&records[idx++]) BonePalette{fills[j].matrices, fills[j].joint_count, matrix_total};
                    matrix_total += fills[j].joint_count;
                }
            }
        }
        pkt.palettes_ = {records, palette_total};
    } else {
        pkt.palettes_ = {};
    }
    pkt.palette_matrix_total_ = matrix_total;

    pkt.models_ = merge_chain(&Lane::models_);
    pkt.skinned_ = merge_skinned(palette_base);
    pkt.point_lights_ = merge_chain(&Lane::point_lights_);
    pkt.spot_lights_ = merge_chain(&Lane::spot_lights_);
    pkt.billboards_ = merge_chain(&Lane::billboards_);
    pkt.debug_lines_ = merge_chain(&Lane::debug_lines_);
    pkt.effect_spawns_ = merge_chain(&Lane::effect_spawns_);
    pkt.ui_commands_ = merge_chain(&Lane::ui_);

    // One summary line per frame for everything the fill refused, throttled
    // so a host with a standing bug does not drown its own log at 60 Hz. The
    // counts are aggregated (FramePacket::BuildDiagnostics) precisely because
    // the per-occurrence form was N identical lines per frame.
    const FramePacket::BuildDiagnostics build_diag = pkt.build_diagnostics();
    if (build_diag.any()) {
        BRACKEN_LOG_WARN_THROTTLED(
            5.0, "scene",
            "frame {}: dropped {} draw(s) with an invalid ModelHandle, {} skinned draw(s) with an invalid bone "
            "palette id, and refused {} palette allocation(s) over the {}-palette frame budget",
            pkt.frame_index_, build_diag.draws_dropped_invalid_handle, build_diag.skinned_dropped_invalid_palette,
            build_diag.palettes_over_budget, FramePacket::MAX_BONE_PALETTES);
    }

    // Filling → Sealed: THE payload publish edge (see FramePacket::state_).
    pkt.state_.store(FramePacket::State::Sealed, std::memory_order_release);
    sealed_ = true;
}

// ============================================================================
// Lane
// ============================================================================

FrameBuilder::Lane::EmitGuard::EmitGuard(const Lane& lane) noexcept : lane_(&lane) {
    const std::thread::id self = std::this_thread::get_id();
    std::thread::id expected{};
    if (lane.emitting_.compare_exchange_strong(expected, self, std::memory_order_relaxed)) {
        return;
    }
    BRACKEN_CONTRACT(false,
                     "FrameBuilder::lane({}) is being emitted into by two threads AT ONCE (holder {:#x}, caller "
                     "{:#x}). A lane is thread-safety tier `owned`: key lanes by JOB index so each has one "
                     "emitter, or join between emitters",
                     lane.index_, std::hash<std::thread::id>{}(expected), std::hash<std::thread::id>{}(self));
}

FrameBuilder::Lane::EmitGuard::~EmitGuard() noexcept {
    lane_->emitting_.store(std::thread::id{}, std::memory_order_relaxed);
}

template<typename T> void* FrameBuilder::Lane::push_slot(Chain<T>& chain) {
    const EmitGuard guard(*this); // concurrent-emitter detector
    if (chain.tail == nullptr || chain.tail->count == chunk_capacity<T>()) {
        const std::size_t bytes = chunk_payload_offset<T>() + std::size_t{chunk_capacity<T>()} * sizeof(T);
        const std::size_t align = alignof(T) > alignof(ChunkHeader) ? alignof(T) : alignof(ChunkHeader);
        auto* chunk = new (packet_->reserve_block(bytes, align)) ChunkHeader{nullptr, 0};
        if (chain.tail != nullptr) {
            chain.tail->next = chunk;
        } else {
            chain.head = chunk;
        }
        chain.tail = chunk;
    }
    ++chain.total;
    return chunk_items<T>(chain.tail) + chain.tail->count++;
}

std::string_view FrameBuilder::Lane::store_text(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    const EmitGuard guard(*this); // same bump-pointer race as push_slot
    if (static_cast<std::size_t>(text_end_ - text_cur_) < text.size()) {
        const std::size_t block = text.size() > k_text_block_bytes ? text.size() : k_text_block_bytes;
        text_cur_ = static_cast<std::byte*>(packet_->reserve_block(block, 1));
        text_end_ = text_cur_ + block;
    }
    char* dst = reinterpret_cast<char*>(text_cur_);
    std::memcpy(dst, text.data(), text.size());
    text_cur_ += text.size();
    return {dst, text.size()};
}

void FrameBuilder::Lane::draw_model(bracken::ModelHandle handle, const glm::mat4& transform, const glm::vec4& tint,
                                    DrawFlags flags, uint32_t motion_id) {
    assert_filling();
    // Validate at EMIT time, the same treatment draw_skinned gives palettes:
    // a command carrying INVALID_MODEL_HANDLE can never draw, and catching it
    // here attributes the mistake to the frame that made it instead of to a
    // render pass three stages downstream. Counted, not logged per call — see
    // FramePacket::BuildDiagnostics; seal() prints the summary.
    if (handle == bracken::INVALID_MODEL_HANDLE) {
        packet_->count_drop_invalid_handle();
        return;
    }
    new (push_slot(models_)) ModelDrawCommand{handle, motion_id, transform, tint, flags};
}

void FrameBuilder::Lane::draw_skinned(bracken::ModelHandle handle, const glm::mat4& transform, BonePaletteId palette,
                                      const glm::vec4& tint, DrawFlags flags, uint32_t motion_id) {
    assert_filling();
    if (handle == bracken::INVALID_MODEL_HANDLE) {
        packet_->count_drop_invalid_handle();
        return;
    }
    if (!palette.valid()) {
        packet_->count_drop_invalid_palette();
        return;
    }
    new (push_slot(skinned_)) SkinnedDrawCommand{handle, motion_id, transform, tint, palette, flags};
}

FrameBuilder::BonePaletteAllocation FrameBuilder::Lane::alloc_bone_palette(uint32_t joint_count) {
    assert_filling();
    BRACKEN_ASSERT(joint_count >= 1 && joint_count <= static_cast<uint32_t>(bracken::animation::MAX_BONES),
                   "alloc_bone_palette: joint_count {} out of range [1, {}]", joint_count,
                   bracken::animation::MAX_BONES);
    if (!packet_->take_palette_slot()) {
        // Build-time budget failure — visible here instead of a silent
        // render-time drop (docs/multithreading_plan.md §4.2). Counted rather
        // than logged per call: a host that overruns the budget overruns it
        // for hundreds of draws per frame, and N identical lines at 60 Hz
        // bury the one line that says what the budget is. seal() summarizes.
        packet_->count_drop_palette_budget();
        return {};
    }
    auto* matrices =
        static_cast<glm::mat4*>(packet_->reserve_block(std::size_t{joint_count} * sizeof(glm::mat4), 16));
    new (push_slot(palettes_)) PaletteFill{matrices, joint_count};
    const BonePaletteId id{(index_ << k_palette_lane_shift) | palette_count_};
    ++palette_count_;
    return {std::span<glm::mat4>(matrices, joint_count), id};
}

void FrameBuilder::Lane::add_point_light(const bracken::render::lighting::PointLightDesc& light) {
    assert_filling();
    // Desc -> GPU POD. The engine owns _pad0/_pad1 and shadow_slot; a host
    // never names them, which is the whole reason the descriptor exists.
    // stable_id is host DATA (0 = "engine assigns") and is assigned by
    // position, not name, only because it sits past the padding.
    auto* l = new (push_slot(point_lights_)) bracken::render::lighting::PointLight{
        light.position, light.radius, light.color, light.intensity, light.casts_shadows};
    l->stable_id = light.stable_id;
    l->source_radius = light.source_radius;
}

void FrameBuilder::Lane::add_spot_light(const bracken::render::lighting::SpotLightDesc& light) {
    assert_filling();
    // The cone cosines are taken HERE — every host used to write
    // std::cos(glm::radians(x)) itself, and the shader wants the cosine.
    auto* l = new (push_slot(spot_lights_)) bracken::render::lighting::SpotLight{
        light.position,   light.radius,          light.direction, std::cos(light.inner_angle.radians()),
        light.color,      std::cos(light.outer_angle.radians()),  light.intensity, light.casts_shadows};
    l->stable_id = light.stable_id;
    l->source_radius = light.source_radius;
}

void FrameBuilder::Lane::add_billboard_3d(float world_x, float world_y, float world_z, float width, float fill_ratio,
                                          bracken::core::Color fill_color, bracken::core::Color bg_color,
                                          bracken::core::Color frame_color) {
    assert_filling();
    new (push_slot(billboards_))
        BillboardDrawCommand{world_x, world_y, world_z, width, fill_ratio, fill_color, bg_color, frame_color};
}

void FrameBuilder::Lane::add_debug_line(const glm::vec3& start, const glm::vec3& end, bracken::core::Color color) {
    assert_filling();
    new (push_slot(debug_lines_)) DebugLineDrawCommand{start, end, color};
}

void FrameBuilder::Lane::add_debug_sphere(const glm::vec3& center, float radius, bracken::core::Color color,
                                          int segments) {
    // Mirrors RenderScene::add_debug_sphere: three circle outlines (XZ, XY, YZ).
    const float step = 2.0f * 3.14159265358979f / static_cast<float>(segments);

    for (int i = 0; i < segments; ++i) {
        const float a0 = step * static_cast<float>(i);
        const float a1 = step * static_cast<float>(i + 1);
        const glm::vec3 p0 = center + glm::vec3(std::cos(a0) * radius, 0.0f, std::sin(a0) * radius);
        const glm::vec3 p1 = center + glm::vec3(std::cos(a1) * radius, 0.0f, std::sin(a1) * radius);
        add_debug_line(p0, p1, color);
    }
    for (int i = 0; i < segments; ++i) {
        const float a0 = step * static_cast<float>(i);
        const float a1 = step * static_cast<float>(i + 1);
        const glm::vec3 p0 = center + glm::vec3(std::cos(a0) * radius, std::sin(a0) * radius, 0.0f);
        const glm::vec3 p1 = center + glm::vec3(std::cos(a1) * radius, std::sin(a1) * radius, 0.0f);
        add_debug_line(p0, p1, color);
    }
    for (int i = 0; i < segments; ++i) {
        const float a0 = step * static_cast<float>(i);
        const float a1 = step * static_cast<float>(i + 1);
        const glm::vec3 p0 = center + glm::vec3(0.0f, std::cos(a0) * radius, std::sin(a0) * radius);
        const glm::vec3 p1 = center + glm::vec3(0.0f, std::cos(a1) * radius, std::sin(a1) * radius);
        add_debug_line(p0, p1, color);
    }
}

void FrameBuilder::Lane::add_debug_box(const glm::vec3& min, const glm::vec3& max, bracken::core::Color color) {
    // Mirrors RenderScene::add_debug_box: 12 AABB edges.
    const glm::vec3 c[8] = {
        {min.x, min.y, min.z}, {max.x, min.y, min.z}, {max.x, min.y, max.z}, {min.x, min.y, max.z},
        {min.x, max.y, min.z}, {max.x, max.y, min.z}, {max.x, max.y, max.z}, {min.x, max.y, max.z},
    };
    add_debug_line(c[0], c[1], color);
    add_debug_line(c[1], c[2], color);
    add_debug_line(c[2], c[3], color);
    add_debug_line(c[3], c[0], color);
    add_debug_line(c[4], c[5], color);
    add_debug_line(c[5], c[6], color);
    add_debug_line(c[6], c[7], color);
    add_debug_line(c[7], c[4], color);
    add_debug_line(c[0], c[4], color);
    add_debug_line(c[1], c[5], color);
    add_debug_line(c[2], c[6], color);
    add_debug_line(c[3], c[7], color);
}

void FrameBuilder::Lane::spawn_effect(const bracken::EffectDefinition* definition, const glm::vec3& position,
                                      const glm::vec3& direction) {
    assert_filling();
    // One of the packet's two host-lifetime exceptions (see the frame-payload
    // comment in frame_packet.hpp). A contract, not an assert: this is the
    // host boundary, the pointer is retained past the frame that emitted it,
    // and a null here used to travel all the way to the sim tail before being
    // silently dropped — attributing the mistake to nothing.
    BRACKEN_CONTRACT(definition != nullptr,
                     "FrameBuilder::spawn_effect: null EffectDefinition. The definition is host-owned and must "
                     "outlive every instance spawned from it");
    new (push_slot(effect_spawns_)) EffectSpawnCommand{definition, position, direction};
}

// ============================================================================
// UiLane
// ============================================================================

void FrameBuilder::UiLane::add_filled_rect(float x, float y, float w, float h, bracken::core::Color color) {
    lane_->assert_filling();
    new (lane_->push_slot(lane_->ui_)) UiCommand{UiFilledRectCommand{x, y, w, h, color}};
}

void FrameBuilder::UiLane::add_rect_outline(float x, float y, float w, float h, bracken::core::Color color,
                                            float line_width) {
    lane_->assert_filling();
    new (lane_->push_slot(lane_->ui_)) UiCommand{UiRectOutlineCommand{x, y, w, h, color, line_width}};
}

void FrameBuilder::UiLane::add_circle(float x, float y, float radius, bracken::core::Color color, int segments) {
    lane_->assert_filling();
    new (lane_->push_slot(lane_->ui_)) UiCommand{UiCircleCommand{x, y, radius, color, segments}};
}

void FrameBuilder::UiLane::add_circle_outline(float x, float y, float radius, bracken::core::Color color,
                                              float line_width, int segments) {
    lane_->assert_filling();
    new (lane_->push_slot(lane_->ui_)) UiCommand{UiCircleOutlineCommand{x, y, radius, color, line_width, segments}};
}

void FrameBuilder::UiLane::add_line(float x1, float y1, float x2, float y2, bracken::core::Color color,
                                    float line_width) {
    lane_->assert_filling();
    new (lane_->push_slot(lane_->ui_)) UiCommand{UiLineCommand{x1, y1, x2, y2, color, line_width}};
}

void FrameBuilder::UiLane::add_text(std::string_view text, float x, float y, float scale,
                                    bracken::core::Color color) {
    lane_->assert_filling();
    new (lane_->push_slot(lane_->ui_)) UiCommand{UiTextCommand{lane_->store_text(text), x, y, scale, color}};
}

float FrameBuilder::UiLane::add_text_wrapped(std::string_view text, float x, float y, float max_width, float scale,
                                             bracken::core::Color color, float line_height) {
    // Mirrors UIScene::add_text_wrapped (approximate 8px * scale char width).
    constexpr float CHAR_WIDTH_BASE = 8.0f;
    const float char_w = CHAR_WIDTH_BASE * scale;
    if (line_height <= 0.0f) {
        line_height = 18.0f * scale;
    }
    int max_chars = static_cast<int>(max_width / char_w);
    max_chars = std::max(max_chars, 1);

    float cur_y = y;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t remaining = text.size() - pos;
        if (static_cast<int>(remaining) <= max_chars) {
            add_text(text.substr(pos), x, cur_y, scale, color);
            cur_y += line_height;
            break;
        }
        const std::size_t end = pos + static_cast<std::size_t>(max_chars);
        std::size_t wrap = text.rfind(' ', end);
        if (wrap == std::string_view::npos || wrap <= pos) {
            wrap = end; // no space found — hard break
        }
        add_text(text.substr(pos, wrap - pos), x, cur_y, scale, color);
        cur_y += line_height;
        pos = wrap;
        if (pos < text.size() && text[pos] == ' ') {
            ++pos; // skip the space we broke at
        }
    }
    return cur_y - y;
}

void FrameBuilder::UiLane::add_button(float x, float y, float w, float h, std::string_view label,
                                      bracken::core::Color color, bool selected) {
    lane_->assert_filling();
    new (lane_->push_slot(lane_->ui_)) UiCommand{UiButtonCommand{x, y, w, h, lane_->store_text(label), color, selected}};
}

} // namespace bracken::scene
