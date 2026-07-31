#pragma once

#include "bracken/core/jobs/cache_line.hpp"
#include "bracken/scene/frame_packet.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string_view>

namespace bracken::scene {

/**
 * FrameBuilder — the ONLY write interface to a FramePacket
 * (docs/multithreading_plan.md §3.3, §4.2). The host describes the frame
 * through it; seal() turns the description into the packet's immutable views
 * and publishes them. Construct one per frame over the packet returned by
 * PacketRing::acquire_for_fill (construction performs Retired → Filling);
 * after seal() the builder is spent — discard it. Construction is cheap
 * (lanes live in the packet arena).
 *
 * Thread-safety tier: `sim-stage`. Emission is multi-lane: every emit method
 * appends to a lane — thread-owned chunked storage carved from the packet
 * arena in block reservations (one atomic fetch_add per BLOCK, private bump
 * per item; lane headers are cache-line padded, §7). The builder methods on
 * FrameBuilder itself forward to lane(0) — today's single-threaded hosts use
 * only those and never see a lane. Concurrent emission fans out by lane:
 *
 *   FrameBuilder fb(packet, job_count);
 *   for (uint32_t i = 0; i < job_count; ++i)
 *       jobs.submit(group, [&fb, i] { fb.lane(i).draw_model(...); });
 *   js.wait(group);   // ALL emitting jobs must be joined before seal()
 *   fb.seal();
 *
 * Each lane is tier `owned`: at most one thread may emit into a given lane at
 * a time. Determinism (§6.2): seal() merges lanes in (lane-index,
 * emission-order) order — key lanes by chunk/job index, never by thread id,
 * and the merged frame is byte-identical at any worker count.
 *
 * Singular frame state (set_camera/set_environment/toggles/timing) is
 * single-writer: call it from the frame's root code, not from fanned-out
 * jobs (last write wins; unsynchronized concurrent calls are a data race).
 *
 * Mirrors every RenderScene/UIScene emit API so ports are mechanical:
 *   scene.add_model(...)                 -> fb.draw_model(...)
 *   scene.add_skinned_model(h, t, bones) -> fb.alloc_bone_palette + write +
 *                                           fb.draw_skinned(h, t, palette.id)
 *   scene.add_point_light / add_spot_light / add_billboard_3d /
 *   add_debug_line / add_debug_sphere / add_debug_box
 *                                        -> same names on fb
 *   scene.add_particle_effect_spawn(...) -> fb.spawn_effect(...)
 *   scene.set_draw_skybox(...) etc.      -> fb.set_draw_skybox(...) etc.
 *   ui.add_text(...)                     -> fb.ui().add_text(...)
 */
class FrameBuilder {
public:
    /**
     * Result of alloc_bone_palette: `matrices` is UNINITIALIZED packet memory
     * (16-byte aligned) the host writes joint matrices into — e.g.
     * AnimationPlayer output copied/computed straight into frame memory; `id`
     * is what draw_skinned takes. Invalid (over-budget) allocations return an
     * empty span + invalid id.
     */
    struct BonePaletteAllocation {
        std::span<glm::mat4> matrices;
        BonePaletteId id;
        [[nodiscard]] bool valid() const { return id.valid(); }
    };

    class UiLane;

    /** Lane index limit (the palette-id encoding reserves 12 bits). */
    static constexpr uint32_t k_max_lanes = 1u << 12;

private:
    // Chunked per-lane, per-type storage. Each chunk is a single packet block
    // reservation (header + a fixed item slab) linked in emission order; the
    // seal merge walks head→tail per lane.
    struct ChunkHeader {
        ChunkHeader* next;
        uint32_t count;
    };

    template<typename T> struct Chain {
        ChunkHeader* head = nullptr;
        ChunkHeader* tail = nullptr;
        uint32_t total = 0;
    };

    // Palette allocation as recorded during fill; seal() turns these into the
    // packet's BonePalette records (lane-major order + matrix prefix sums).
    struct PaletteFill {
        glm::mat4* matrices;
        uint32_t joint_count;
    };

    static constexpr std::size_t k_chunk_payload_bytes = 2048;
    static constexpr std::size_t k_text_block_bytes = 1024;
    // Fill-time BonePaletteId encoding: (lane << shift) | lane-local index.
    // seal() remaps every stored id to its dense merged index.
    static constexpr uint32_t k_palette_lane_shift = 20;
    static constexpr uint32_t k_palette_local_mask = (1u << k_palette_lane_shift) - 1;

    template<typename T> static constexpr uint32_t chunk_capacity() {
        const auto n = static_cast<uint32_t>(k_chunk_payload_bytes / sizeof(T));
        return n < 4 ? 4u : n;
    }
    template<typename T> static constexpr std::size_t chunk_payload_offset() {
        return (sizeof(ChunkHeader) + alignof(T) - 1) & ~(alignof(T) - 1);
    }
    template<typename T> static T* chunk_items(ChunkHeader* c) {
        return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(c) + chunk_payload_offset<T>());
    }

public:
    /**
     * One writer's private emission stream (thread-safety tier: `owned` — one
     * thread at a time). Obtained from FrameBuilder::lane(i); lives in the
     * packet arena and dies with the packet. The header is cache-line padded
     * (core/jobs/cache_line.hpp) so concurrent lanes never false-share.
     */
    class alignas(bracken::core::jobs::k_cache_line) Lane {
    public:
        /** Mirror of RenderScene::add_model — static model draw. */
        void draw_model(bracken::ModelHandle handle, const glm::mat4& transform,
                        const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f}, bool force_non_instanced = false,
                        bool no_fog = false, uint32_t motion_id = 0);

        /**
         * Skinned model draw referencing a packet palette (see
         * alloc_bone_palette). The id may come from any lane, provided its
         * allocation happened-before this call. Invalid ids (over-budget
         * allocations) drop the draw with a warning.
         */
        void draw_skinned(bracken::ModelHandle handle, const glm::mat4& transform, BonePaletteId palette,
                          const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f}, bool no_fog = false,
                          uint32_t motion_id = 0);

        /**
         * Allocate a bone palette in packet memory: joint_count matrices
         * (1..=animation::MAX_BONES), 16-byte aligned, uninitialized — write
         * every matrix before seal. Budget: FramePacket::MAX_BONE_PALETTES
         * per frame; over-budget allocations log an error and return an
         * invalid allocation (empty span) — this is the visible build-time
         * failure that replaces the old render-time drop. Share one palette
         * across many draws by reusing the id (replaces the renderer's
         * pointer-dedupe of identity palettes).
         */
        [[nodiscard]] BonePaletteAllocation alloc_bone_palette(uint32_t joint_count);

        /** Mirrors of the RenderScene light/billboard/debug emits. Colors are
         *  packed ARGB (0xAARRGGBB) — see bracken/core/color.hpp. */
        void add_point_light(const bracken::render::lighting::PointLight& light);
        void add_spot_light(const bracken::render::lighting::SpotLight& light);
        void add_billboard_3d(float world_x, float world_y, float world_z, float width, float fill_ratio,
                              uint32_t fill_color, uint32_t bg_color, uint32_t frame_color);
        void add_debug_line(const glm::vec3& start, const glm::vec3& end, uint32_t color);
        void add_debug_sphere(const glm::vec3& center, float radius, uint32_t color, int segments = 16);
        void add_debug_box(const glm::vec3& min, const glm::vec3& max, uint32_t color);

        /** Queue a particle effect spawn (mirror of add_particle_effect_spawn;
         *  consumed exactly once from the sealed packet). */
        void spawn_effect(const bracken::EffectDefinition* definition, const glm::vec3& position,
                          const glm::vec3& direction = {1, 0, 0});

        /** 2D UI emission for this lane (mirror of UIScene). */
        [[nodiscard]] UiLane ui();

    private:
        friend class FrameBuilder;
        friend class UiLane;

        Lane() = default;

        // Returns the next raw item slot of `chain`, growing it by one block
        // reservation when the tail chunk is full.
        template<typename T> [[nodiscard]] void* push_slot(Chain<T>& chain);

        // Copy `text` into lane-local packet memory (bump within reserved
        // text blocks) and return a packet-lifetime view of it.
        [[nodiscard]] std::string_view store_text(std::string_view text);

        void assert_filling() const {
            BRACKEN_ASSERT(packet_ != nullptr && packet_->state() == FramePacket::State::Filling,
                           "FrameBuilder write outside the Filling state (emit after seal?)");
        }

        FramePacket* packet_ = nullptr;
        uint32_t index_ = 0;
        uint32_t palette_count_ = 0; // lane-local allocations (low bits of fill-time ids)
        Chain<ModelDrawCommand> models_;
        Chain<SkinnedDrawCommand> skinned_;
        Chain<bracken::render::lighting::PointLight> point_lights_;
        Chain<bracken::render::lighting::SpotLight> spot_lights_;
        Chain<BillboardDrawCommand> billboards_;
        Chain<DebugLineDrawCommand> debug_lines_;
        Chain<EffectSpawnCommand> effect_spawns_;
        Chain<UiCommand> ui_;
        Chain<PaletteFill> palettes_;
        std::byte* text_cur_ = nullptr;
        std::byte* text_end_ = nullptr;
    };

    /**
     * 2D UI emission handle (mirror of UIScene's API), bound to one lane —
     * same ownership rules as the lane itself. A small value handle: copy it
     * freely, or chain calls off ui() directly. Text/label parameters are
     * copied into packet memory at emit time.
     */
    class UiLane {
    public:
        void add_filled_rect(float x, float y, float w, float h, uint32_t color);
        void add_rect_outline(float x, float y, float w, float h, uint32_t color, float line_width = 2.0f);
        void add_circle(float x, float y, float radius, uint32_t color, int segments = 24);
        void add_circle_outline(float x, float y, float radius, uint32_t color, float line_width = 2.0f,
                                int segments = 24);
        void add_line(float x1, float y1, float x2, float y2, uint32_t color, float line_width = 2.0f);
        void add_text(std::string_view text, float x, float y, float scale = 1.0f, uint32_t color = 0xFFFFFFFF);

        /**
         * Word-wrapped text (mirror of UIScene::add_text_wrapped). Returns
         * the total height consumed. Approximate char width: 8px * scale;
         * line_height defaults to 18 * scale.
         */
        float add_text_wrapped(std::string_view text, float x, float y, float max_width, float scale = 1.0f,
                               uint32_t color = 0xFFFFFFFF, float line_height = 0.0f);

        void add_button(float x, float y, float w, float h, std::string_view label, uint32_t color,
                        bool selected = false);

    private:
        friend class Lane;
        explicit UiLane(Lane& lane) : lane_(&lane) {}
        Lane* lane_;
    };

    /**
     * Bind a builder to `packet` (performs Retired → Filling) with
     * `lane_count` emission lanes (1..k_max_lanes). Lane storage lives in the
     * packet arena.
     */
    explicit FrameBuilder(FramePacket& packet, uint32_t lane_count = 1);
    ~FrameBuilder() = default;

    FrameBuilder(const FrameBuilder&) = delete;
    FrameBuilder& operator=(const FrameBuilder&) = delete;
    FrameBuilder(FrameBuilder&&) = delete;
    FrameBuilder& operator=(FrameBuilder&&) = delete;

    [[nodiscard]] uint32_t lane_count() const { return lane_count_; }

    [[nodiscard]] Lane& lane(uint32_t index) {
        BRACKEN_ASSERT(index < lane_count_, "FrameBuilder::lane: index {} out of range (lane_count {})", index,
                       lane_count_);
        return lanes_[index];
    }

    // ---- Single-threaded conveniences: forward to lane(0) ----

    void draw_model(bracken::ModelHandle handle, const glm::mat4& transform,
                    const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f}, bool force_non_instanced = false,
                    bool no_fog = false, uint32_t motion_id = 0) {
        lane(0).draw_model(handle, transform, tint, force_non_instanced, no_fog, motion_id);
    }
    void draw_skinned(bracken::ModelHandle handle, const glm::mat4& transform, BonePaletteId palette,
                      const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f}, bool no_fog = false, uint32_t motion_id = 0) {
        lane(0).draw_skinned(handle, transform, palette, tint, no_fog, motion_id);
    }
    [[nodiscard]] BonePaletteAllocation alloc_bone_palette(uint32_t joint_count) {
        return lane(0).alloc_bone_palette(joint_count);
    }
    void add_point_light(const bracken::render::lighting::PointLight& light) { lane(0).add_point_light(light); }
    void add_spot_light(const bracken::render::lighting::SpotLight& light) { lane(0).add_spot_light(light); }
    void add_billboard_3d(float world_x, float world_y, float world_z, float width, float fill_ratio,
                          uint32_t fill_color, uint32_t bg_color, uint32_t frame_color) {
        lane(0).add_billboard_3d(world_x, world_y, world_z, width, fill_ratio, fill_color, bg_color, frame_color);
    }
    void add_debug_line(const glm::vec3& start, const glm::vec3& end, uint32_t color) {
        lane(0).add_debug_line(start, end, color);
    }
    void add_debug_sphere(const glm::vec3& center, float radius, uint32_t color, int segments = 16) {
        lane(0).add_debug_sphere(center, radius, color, segments);
    }
    void add_debug_box(const glm::vec3& min, const glm::vec3& max, uint32_t color) {
        lane(0).add_debug_box(min, max, color);
    }
    void spawn_effect(const bracken::EffectDefinition* definition, const glm::vec3& position,
                      const glm::vec3& direction = {1, 0, 0}) {
        lane(0).spawn_effect(definition, position, direction);
    }
    [[nodiscard]] UiLane ui() { return lane(0).ui(); }

    // ---- Singular frame state (single writer; last write wins) ----

    void set_camera(const CameraState& camera);
    void set_environment(const bracken::render::EnvironmentState& environment);
    void set_draw_skybox(bool draw);
    void set_draw_ground(bool draw);
    void set_draw_grass(bool draw);
    /** Frame timing carried by the packet (dt for effects/animation advance,
     *  accumulated time, monotonically increasing frame index). */
    void set_frame_timing(float dt, double time, uint64_t frame_index);

    // ---- Engine-internal sim-stage API (Application's sim-job tail; not
    // ---- for host code) --------------------------------------------------
    //
    // All of it runs single-threaded on the sim job AFTER on_simulate
    // returned and every host emission job was joined, and BEFORE seal().

    /**
     * Walk every effect spawn emitted this fill in deterministic (lane-index,
     * emission-order) order — the same order seal() will merge them — so the
     * sim stage can apply them to the EffectSystem before its update.
     * PRECONDITION: all emitting jobs joined; packet still Filling.
     */
    template<typename Fn> void for_each_effect_spawn(Fn&& fn) const {
        for (uint32_t i = 0; i < lane_count_; ++i) {
            for (ChunkHeader* c = lanes_[i].effect_spawns_.head; c != nullptr; c = c->next) {
                const EffectSpawnCommand* items = chunk_items<EffectSpawnCommand>(c);
                for (uint32_t j = 0; j < c->count; ++j) {
                    fn(items[j]);
                }
            }
        }
    }

    /** Result of alloc_effect_snapshot: UNINITIALIZED packet memory the sim
     *  stage fills (EffectSystem::snapshot_into); the spans are already
     *  registered as the packet's effect_emitters()/effect_particles(). */
    struct EffectSnapshotAllocation {
        std::span<EffectEmitterSnapshot> emitters;
        std::span<EffectParticleSnapshot> particles;
    };

    /**
     * Allocate and register the frame's effect-instance snapshot in packet
     * memory. Call at most once per fill; zero counts register empty spans.
     */
    [[nodiscard]] EffectSnapshotAllocation alloc_effect_snapshot(uint32_t emitter_count, uint32_t particle_count);

    /**
     * Record the engine frame state: the sim-advanced scene/sky clocks and
     * the live effect-instance count (see the packet's engine-frame reads).
     * Sets has_engine_frame().
     */
    void set_engine_frame(float scene_time, float sky_time, uint32_t effect_instance_count);

    /** Staged singular-state reads for the sim-stage tail (e.g. the sky-time
     *  advance needs the environment's cloud_speed). Valid during Filling. */
    [[nodiscard]] bool has_staged_environment() const { return packet_->has_environment_; }
    [[nodiscard]] const bracken::render::EnvironmentState& staged_environment() const {
        return packet_->environment_;
    }

    // ---- Seal ------------------------------------------------------------

    /**
     * Merge all lanes into the packet's contiguous views — deterministic
     * (lane-index, emission-order) order, palette ids remapped to their dense
     * merged indices — then publish Filling → Sealed (the packet's payload
     * release edge). PRECONDITION: every job that emitted into a lane has
     * been joined (TaskGroup wait) — their writes must happen-before this
     * call. seal() runs on one thread and may be called exactly once; the
     * builder is spent afterwards.
     */
    void seal();

    [[nodiscard]] bool sealed() const { return sealed_; }
    [[nodiscard]] FramePacket& packet() { return *packet_; }

private:
    template<typename T> [[nodiscard]] std::span<const T> merge_chain(Chain<T> Lane::* member);
    [[nodiscard]] std::span<const SkinnedDrawCommand> merge_skinned(const uint32_t* palette_base);

    FramePacket* packet_ = nullptr;
    Lane* lanes_ = nullptr; // packet-arena array, one cache-line-padded Lane per index
    uint32_t lane_count_ = 0;
    bool sealed_ = false;
};

inline FrameBuilder::UiLane FrameBuilder::Lane::ui() {
    return UiLane(*this);
}

} // namespace bracken::scene
