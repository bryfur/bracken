#pragma once

#include "bracken/contracts/camera_state.hpp"
#include "bracken/contracts/environment.hpp"
#include "bracken/contracts/model_handle.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/memory/arena.hpp"
#include "bracken/render/lighting/light.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <mutex>
#include <span>
#include <string_view>
#include <variant>

namespace bracken {
struct EffectDefinition;
struct EmitterDefinition;
}

namespace bracken::scene {

class FrameBuilder;

// ============================================================================
// Frame command payloads
//
// Everything below is by value or packet-allocated: no field references host
// storage, so a sealed FramePacket carries no lifetime contract whatsoever
// (docs/multithreading_plan.md §3.3). All structs are trivially copyable and
// laid out padding-free (explicit `_pad` bytes, zeroed by their initializers)
// so lane merges are memcpys and a merged frame is byte-deterministic.
// ============================================================================

/**
 * Per-frame bone-palette handle returned by FrameBuilder's
 * alloc_bone_palette() and stored in SkinnedDrawCommand.
 *
 * Opaque while the frame is FILLING: the value is an internal (lane,
 * allocation-order) encoding — pass it to draw_skinned (any lane may use an
 * id allocated on another lane, provided the allocation happened-before the
 * use), never interpret it. At seal every id stored in a command is remapped
 * to a dense index into FramePacket::palettes(), assigned in deterministic
 * (lane-index, allocation-order) order — ids read from a SEALED packet's
 * commands index palettes() directly and are stable for the packet's
 * lifetime. Ids held by the host across seal() are stale; only re-read them
 * from sealed commands.
 */
struct BonePaletteId {
    static constexpr uint32_t k_invalid = 0xFFFFFFFFu;
    uint32_t value = k_invalid;
    [[nodiscard]] bool valid() const { return value != k_invalid; }
};

/**
 * Static model draw command (mirrors the retired RenderScene ModelCommand).
 * Handle-only: names are resolved to ModelHandle at spawn/placement time by
 * the submitter. motion_id: host-unique nonzero id for MOVING objects so the
 * velocity prepass can pair this frame's transform with last frame's; 0
 * (default) = static, velocity is camera-only (scene/motion_history.hpp).
 */
struct ModelDrawCommand {
    bracken::ModelHandle model_handle = bracken::INVALID_MODEL_HANDLE;
    uint32_t motion_id = 0;
    glm::mat4 transform;
    glm::vec4 tint = {1.0f, 1.0f, 1.0f, 1.0f};
    bool force_non_instanced = false;
    bool no_fog = false;
    uint8_t _pad0[2] = {0, 0};
};

/**
 * Skinned/animated model draw command. The bone palette lives in the packet
 * (allocated via alloc_bone_palette, written by the host during fill) and is
 * referenced by `palette` — after seal a dense index into
 * FramePacket::palettes(). No non-owning spans over host storage anymore.
 * motion_id: see ModelDrawCommand; for skinned commands the history
 * additionally pairs this frame's palette offset with last frame's
 * (per-bone motion vectors).
 */
struct SkinnedDrawCommand {
    bracken::ModelHandle model_handle = bracken::INVALID_MODEL_HANDLE;
    uint32_t motion_id = 0;
    glm::mat4 transform;
    glm::vec4 tint = {1.0f, 1.0f, 1.0f, 1.0f};
    BonePaletteId palette{};
    bool no_fog = false;
    uint8_t _pad0[3] = {0, 0, 0};
};

/**
 * One packed bone palette of a sealed frame. `matrices` points at packet
 * memory: joint_count matrices written by the host through the span returned
 * from alloc_bone_palette. `matrix_offset` is the palette's first-matrix
 * index in the frame's packed palette stream (running sum of joint_count over
 * palettes() in order) — exactly the per-draw palette offset the skinned
 * shaders consume once the renderer uploads palettes in palettes() order, and
 * the value MotionHistory records for N/N−1 palette pairing.
 */
struct BonePalette {
    const glm::mat4* matrices = nullptr;
    uint32_t joint_count = 0;
    uint32_t matrix_offset = 0;
};

/**
 * Billboard UI element positioned in 3D world space (projected + distance-
 * scaled by the renderer). Colors are packed ARGB (0xAARRGGBB) — see
 * bracken/core/color.hpp.
 */
struct BillboardDrawCommand {
    float world_x, world_y, world_z;
    float width;
    float fill_ratio;
    uint32_t fill_color;  // packed ARGB
    uint32_t bg_color;    // packed ARGB
    uint32_t frame_color; // packed ARGB
};

/**
 * Debug line command: a single colored line segment in world space.
 * color is packed ARGB (0xAARRGGBB).
 */
struct DebugLineDrawCommand {
    glm::vec3 start;
    glm::vec3 end;
    uint32_t color;
};

/**
 * Particle effect spawn command. The definition pointer is host-owned
 * configuration data (effect definitions are long-lived assets, not per-frame
 * state); the engine consumes each sealed packet's spawns exactly once.
 */
struct EffectSpawnCommand {
    const bracken::EffectDefinition* definition = nullptr;
    glm::vec3 position = {0, 0, 0};
    glm::vec3 direction = {1, 0, 0};
};

// ---------------------------------------------------------------------------
// Engine frame section (sim-stage-written; docs/multithreading_plan.md §4.2:
// "the renderer consumes a particle instance snapshot from the packet").
// Filled by Application's sim-job tail through FrameBuilder's engine-internal
// API AFTER the host's on_simulate returns and the EffectSystem update ran —
// never by host code. The renderer reads it from the SEALED packet; the live
// EffectSystem is sim-stage-owned and mutates concurrently with rendering.
// ---------------------------------------------------------------------------

/**
 * One live emitter of the frame's effect-instance snapshot. `definition` is a
 * host-owned, long-lived asset pointer (same contract as EffectSpawnCommand)
 * — the effect renderer resolves the emitter's model from it. The particle
 * range indexes FramePacket::effect_particles().
 */
struct EffectEmitterSnapshot {
    const bracken::EmitterDefinition* definition = nullptr;
    uint32_t first_particle = 0;
    uint32_t particle_count = 0;
};

/**
 * Render-relevant state of one live particle: exactly the fields the effect
 * renderer's transform build + instance packing consume (position/rotation/
 * scale for the model matrix, color+opacity for the tint record).
 */
struct EffectParticleSnapshot {
    glm::vec3 position = {0, 0, 0};
    float scale = 1.0f;
    glm::vec3 rotation = {0, 0, 0};
    float opacity = 1.0f;
    glm::vec4 color = {1, 1, 1, 1};
};

// ---------------------------------------------------------------------------
// UI commands (mirror the retired UIScene commands). All `uint32_t color`
// fields are packed ARGB (0xAARRGGBB) — see bracken/core/color.hpp. Text is
// stored as string_views over packet-arena copies: no std::string, no heap,
// no lifetime contract beyond the packet itself.
// ---------------------------------------------------------------------------

struct UiFilledRectCommand {
    float x, y, w, h;
    uint32_t color;
};

struct UiRectOutlineCommand {
    float x, y, w, h;
    uint32_t color;
    float line_width = 2.0f;
};

struct UiCircleCommand {
    float x, y, radius;
    uint32_t color;
    int segments = 24;
};

struct UiCircleOutlineCommand {
    float x, y, radius;
    uint32_t color;
    float line_width = 2.0f;
    int segments = 24;
};

struct UiLineCommand {
    float x1, y1, x2, y2;
    uint32_t color;
    float line_width = 2.0f;
};

/** Text render command; `text` points into the packet arena. */
struct UiTextCommand {
    std::string_view text;
    float x{}, y{};
    float scale = 1.0f;
    uint32_t color = 0xFFFFFFFF;
};

/** Button render command; `label` points into the packet arena. */
struct UiButtonCommand {
    float x{}, y{}, w{}, h{};
    std::string_view label;
    uint32_t color{};
    bool selected = false;
    uint8_t _pad0[3] = {0, 0, 0};
};

using UiCommandData = std::variant<UiFilledRectCommand, UiRectOutlineCommand, UiCircleCommand, UiCircleOutlineCommand,
                                   UiLineCommand, UiTextCommand, UiButtonCommand>;

struct UiCommand {
    UiCommandData data;

    template<typename T> [[nodiscard]] bool is() const { return std::holds_alternative<T>(data); }
    template<typename T> [[nodiscard]] const T& get() const { return std::get<T>(data); }
};

// ============================================================================
// FramePacket
// ============================================================================

/**
 * FramePacket — one frame's complete, immutable description of everything the
 * host wants drawn: static/skinned model commands, bone palettes, lights,
 * billboards, debug lines, particle-effect spawns, UI commands, camera,
 * environment snapshot, world-element draw toggles and frame timing. Replaces
 * the RenderScene/UIScene contract (docs/multithreading_plan.md §4.2).
 *
 * All storage is carved from the packet's own memory::Arena — commands,
 * palette matrices and UI text live in packet memory, so nothing the host
 * hands in carries a lifetime contract: everything is by value or
 * packet-allocated, and the packet outlives its own consumption by
 * construction. The arena is reset when the packet is retired (ring reuse);
 * steady-state frames allocate no heap memory.
 *
 * Lifecycle state machine: Retired → Filling → Sealed → Consuming → Retired.
 * Writes happen only during Filling and only through FrameBuilder; reads are
 * legal only in Sealed/Consuming. Both directions are asserted in debug:
 * every read accessor asserts the packet is sealed, and every builder write
 * (and reserve_block) asserts it is not.
 *
 * Thread-safety tiers (docs/multithreading_plan.md §6.1):
 * - during fill: `owned` per builder lane; reserve_block() itself is safe
 *   from any fill-stage thread (atomic block carve, see below);
 * - after seal: `worker-safe` — const reads from any thread, plain loads,
 *   no atomics on the read path;
 * - begin_fill()/begin_consume()/reset(): stage-transition points with one
 *   caller each per frame, ordered by the ring's publish/acquire edges (or
 *   the caller's own sequencing in ring-less/standalone use).
 */
class FramePacket {
public:
    enum class State : uint32_t {
        Retired = 0,  // empty, arena reset, ready for the next fill
        Filling = 1,  // FrameBuilder lanes are writing
        Sealed = 2,   // immutable; readable by anyone
        Consuming = 3 // acquired by the consuming stage (still immutable)
    };

    /** Initial arena chunk size. Frames that outgrow it grow the arena once
     *  and the memory is retained across resets (steady-state: zero allocs). */
    static constexpr std::size_t k_default_arena_bytes = 256 * 1024;

    /** Per-frame skinned-draw palette budget. Accounting happens in
     *  alloc_bone_palette at BUILD time — over-budget allocations fail
     *  visibly there (log + invalid id), replacing the old render-time
     *  silent drop (docs/multithreading_plan.md §4.2). */
    static constexpr uint32_t MAX_BONE_PALETTES = 1024;

    explicit FramePacket(std::size_t initial_arena_bytes = k_default_arena_bytes);
    ~FramePacket() = default;

    FramePacket(const FramePacket&) = delete;
    FramePacket& operator=(const FramePacket&) = delete;
    FramePacket(FramePacket&&) = delete;
    FramePacket& operator=(FramePacket&&) = delete;

    [[nodiscard]] State state() const noexcept { return state_.load(std::memory_order_acquire); }

    // ========== Sealed-frame reads (worker-safe; assert Sealed/Consuming) ==========

    [[nodiscard]] std::span<const ModelDrawCommand> model_commands() const {
        assert_readable();
        return models_;
    }
    [[nodiscard]] std::span<const SkinnedDrawCommand> skinned_commands() const {
        assert_readable();
        return skinned_;
    }
    [[nodiscard]] std::span<const BillboardDrawCommand> billboards() const {
        assert_readable();
        return billboards_;
    }
    [[nodiscard]] std::span<const DebugLineDrawCommand> debug_lines() const {
        assert_readable();
        return debug_lines_;
    }
    [[nodiscard]] std::span<const EffectSpawnCommand> effect_spawns() const {
        assert_readable();
        return effect_spawns_;
    }
    [[nodiscard]] std::span<const bracken::render::lighting::PointLight> point_lights() const {
        assert_readable();
        return point_lights_;
    }
    [[nodiscard]] std::span<const bracken::render::lighting::SpotLight> spot_lights() const {
        assert_readable();
        return spot_lights_;
    }
    [[nodiscard]] std::span<const UiCommand> ui_commands() const {
        assert_readable();
        return ui_commands_;
    }

    /** All bone palettes of the frame in deterministic (lane, allocation)
     *  order; SkinnedDrawCommand::palette.value indexes this span. */
    [[nodiscard]] std::span<const BonePalette> palettes() const {
        assert_readable();
        return palettes_;
    }

    /** Matrices of one palette, by a SEALED command's palette id. */
    [[nodiscard]] std::span<const glm::mat4> palette_matrices(BonePaletteId id) const;

    /**
     * What the FILL threw away before it ever became a command.
     *
     * These are host-side mistakes caught at emit time — an invalid handle, a
     * palette id from a failed allocation, an allocation over the frame's
     * MAX_BONE_PALETTES budget — and they are counted rather than logged per
     * occurrence, because a host that gets one of these wrong gets it wrong
     * for every instance, every frame. FrameBuilder::seal() emits ONE summary
     * line per frame from these counts; the renderer folds them into
     * FrameDiagnostics. Readable while Filling too (seal() reads them itself).
     */
    struct BuildDiagnostics {
        uint32_t draws_dropped_invalid_handle = 0;
        uint32_t skinned_dropped_invalid_palette = 0;
        uint32_t palettes_over_budget = 0;

        [[nodiscard]] bool any() const {
            return draws_dropped_invalid_handle != 0 || skinned_dropped_invalid_palette != 0 ||
                   palettes_over_budget != 0;
        }
    };

    [[nodiscard]] BuildDiagnostics build_diagnostics() const noexcept {
        return {drops_invalid_handle_.load(std::memory_order_relaxed),
                drops_invalid_palette_.load(std::memory_order_relaxed),
                drops_palette_budget_.load(std::memory_order_relaxed)};
    }

    /** Total matrix count across palettes() — the size of the packed palette
     *  stream a renderer uploads (sum of joint_count, == last palette's
     *  matrix_offset + joint_count). */
    [[nodiscard]] uint32_t palette_matrix_total() const {
        assert_readable();
        return palette_matrix_total_;
    }

    [[nodiscard]] const CameraState& camera() const {
        assert_readable();
        return camera_;
    }
    /** False when the host never called set_camera this frame (consumer keeps
     *  its previous camera / skips 3D). */
    [[nodiscard]] bool has_camera() const {
        assert_readable();
        return has_camera_;
    }

    [[nodiscard]] const bracken::render::EnvironmentState& environment() const {
        assert_readable();
        return environment_;
    }
    /** False when the host never called set_environment this frame (consumer
     *  keeps its current environment). */
    [[nodiscard]] bool has_environment() const {
        assert_readable();
        return has_environment_;
    }

    // World-element draw toggles (default true each frame, like RenderScene).
    [[nodiscard]] bool should_draw_skybox() const {
        assert_readable();
        return draw_skybox_;
    }
    [[nodiscard]] bool should_draw_ground() const {
        assert_readable();
        return draw_ground_;
    }
    [[nodiscard]] bool should_draw_grass() const {
        assert_readable();
        return draw_grass_;
    }

    // ---- Engine frame section reads (sim-stage-written; see the section
    // ---- comment above EffectEmitterSnapshot) ----

    /** True when the sim-stage tail wrote the engine section this frame
     *  (always the case under Application::run; false for hand-built test
     *  packets — consumers then keep their previous clocks). */
    [[nodiscard]] bool has_engine_frame() const {
        assert_readable();
        return has_engine_frame_;
    }

    /** Accumulated scene clock (grass wind / water shading), advanced by the
     *  sim stage — the renderer adopts it instead of integrating dt. */
    [[nodiscard]] float scene_time() const {
        assert_readable();
        return scene_time_;
    }

    /** Accumulated sky-cloud scroll phase (dt integrated scaled by
     *  EnvironmentState::cloud_speed), advanced by the sim stage. */
    [[nodiscard]] float sky_time() const {
        assert_readable();
        return sky_time_;
    }

    /** The frame's effect-instance snapshot: live emitters in deterministic
     *  (effect, emitter) order; particle ranges index effect_particles(). */
    [[nodiscard]] std::span<const EffectEmitterSnapshot> effect_emitters() const {
        assert_readable();
        return effect_emitters_;
    }
    [[nodiscard]] std::span<const EffectParticleSnapshot> effect_particles() const {
        assert_readable();
        return effect_particles_;
    }

    /** Number of live EffectInstances at snapshot time (dev-overlay stats). */
    [[nodiscard]] uint32_t effect_instance_count() const {
        assert_readable();
        return effect_instance_count_;
    }

    [[nodiscard]] float dt() const {
        assert_readable();
        return dt_;
    }
    [[nodiscard]] double time() const {
        assert_readable();
        return time_;
    }
    [[nodiscard]] uint64_t frame_index() const {
        assert_readable();
        return frame_index_;
    }

    /**
     * True if this frame has anything for the 3D pass: world elements
     * (skybox/ground/grass), model or skinned commands, billboards, debug
     * lines, or dynamic lights (mirrors RenderScene::has_3d_content).
     */
    [[nodiscard]] bool has_3d_content() const {
        assert_readable();
        return draw_skybox_ || draw_ground_ || draw_grass_ || !models_.empty() || !skinned_.empty() ||
               !billboards_.empty() || !debug_lines_.empty() || !point_lights_.empty() || !spot_lights_.empty();
    }

    // Arena introspection (tests / stats).
    [[nodiscard]] std::size_t bytes_used() const noexcept { return arena_.bytes_used(); }
    [[nodiscard]] std::size_t bytes_capacity() const noexcept { return arena_.bytes_capacity(); }

    // ---- Internal API (FrameBuilder / PacketRing / stage coordinator; not for host code) ----

    /** Retired → Filling. Called by the FrameBuilder constructor. */
    void begin_fill();

    /** Sealed → Consuming. Called by PacketRing::try_acquire_sealed (or the
     *  coordinator in ring-less use). Reads are legal in either state. */
    void begin_consume();

    /** Any state → Retired: clears all sealed views and resets the arena
     *  (memory retained). The caller provides the synchronization that makes
     *  this safe (frame fence / ring retire ordering); the arena owner
     *  binding is explicitly handed to the calling thread. */
    void reset();

    /**
     * Carve a block of frame memory. Safe from ANY fill-stage thread: one
     * relaxed fetch_add on the active slab per call (this is the "one
     * fetch_add per block" of docs/multithreading_plan.md §7 — callers bump
     * privately within their block), with a mutex-guarded arena refill on the
     * cold slab-exhausted path. `align` must be a power of two ≤ 128.
     * Returned memory is uninitialized and thread-private until the seal (or
     * the caller's own join) publishes its contents. Aborts on arena OOM.
     * Asserts the packet is Filling.
     */
    [[nodiscard]] void* reserve_block(std::size_t size, std::size_t align);

    /** Claim one of the frame's MAX_BONE_PALETTES palette slots. False =
     *  budget exhausted; the caller must fail the allocation visibly. Any
     *  fill-stage thread. */
    [[nodiscard]] bool take_palette_slot() noexcept {
        return palette_slots_.fetch_add(1, std::memory_order_relaxed) < MAX_BONE_PALETTES;
    }

    /** Palette slots claimed this fill (may exceed MAX_BONE_PALETTES on an
     *  over-budget frame — that overshoot IS the diagnostic). */
    [[nodiscard]] uint32_t palette_slots_taken() const noexcept {
        return palette_slots_.load(std::memory_order_relaxed);
    }

    /** Emit-time drop counters, bumped from any fill-stage lane thread.
     *  Relaxed: each is a self-contained count on an already-failing path,
     *  published to readers by the Filling → Sealed release edge. */
    void count_drop_invalid_handle() noexcept { drops_invalid_handle_.fetch_add(1, std::memory_order_relaxed); }
    void count_drop_invalid_palette() noexcept { drops_invalid_palette_.fetch_add(1, std::memory_order_relaxed); }
    void count_drop_palette_budget() noexcept { drops_palette_budget_.fetch_add(1, std::memory_order_relaxed); }

private:
    friend class FrameBuilder; // writes the sealed views + performs the Filling→Sealed publish

    struct SlabState {
        std::byte* base = nullptr;
        std::size_t capacity = 0;
        // cursor — private bump offset within this slab, advanced by relaxed
        // fetch_add in reserve_block. Publishes nothing: each reserved range
        // is written only by its reserving thread and its contents ride the
        // packet's Filling→Sealed release edge (or the fill stage's own
        // TaskGroup join). No release/acquire pairing partner. Kept 16-byte
        // granular so every reservation stays 16-aligned.
        std::atomic<std::size_t> cursor{0};
    };

    void refill_slab(SlabState* observed, std::size_t min_bytes);

    // Contract, not assert: reading an unsealed packet hands the caller spans
    // that are empty or mid-write, and the corruption surfaces in a renderer
    // far from the mistake. Host-facing and off every per-item loop.
    void assert_readable() const {
        BRACKEN_CONTRACT(state() == State::Sealed || state() == State::Consuming,
                         "FramePacket read before seal (state {})", static_cast<uint32_t>(state()));
    }

    // ---- Arena + shared block allocator ----
    memory::Arena arena_;
    // active_slab_ — publishes a fully initialized SlabState (base/capacity/
    // cursor) to reserving threads: store(release) under slab_refill_mutex_ in
    // refill_slab ↔ load(acquire) at the top of reserve_block. Slabs are
    // never recycled within a frame (reclaimed only by reset()), so a stale
    // load is merely a full slab that sends the caller back to the refill
    // path — no ABA. reset() stores nullptr (relaxed; the retire fence orders
    // it).
    std::atomic<SlabState*> active_slab_{nullptr};
    // Cold path only (docs/multithreading_plan.md §7 rule 5): serializes slab
    // refills, i.e. arena growth. Never taken by an in-slab reservation.
    std::mutex slab_refill_mutex_;
    // palette_slots_ — MAX_BONE_PALETTES accounting; relaxed fetch_add,
    // publishes nothing (palette data rides the seal edge). May over-count
    // past the cap on over-budget frames; reset() zeroes it. No pairing
    // partner.
    std::atomic<uint32_t> palette_slots_{0};
    // Emit-time drop counts (see count_drop_*): relaxed fetch_add on failing
    // paths only, zeroed by reset(). No pairing partner — their values ride
    // the seal edge like the rest of the payload.
    std::atomic<uint32_t> drops_invalid_handle_{0};
    std::atomic<uint32_t> drops_invalid_palette_{0};
    std::atomic<uint32_t> drops_palette_budget_{0};

    // state_ — the packet lifecycle word (Retired → Filling → Sealed →
    // Consuming → Retired). Every transition is a release store by the stage
    // that owns the packet in the source state; observers acquire-load via
    // state(). THE payload publish edge is the Filling→Sealed store in
    // FrameBuilder::seal(): all packet content written during fill (by any
    // lane thread, joined by the sealing caller first) happens-before any
    // reader that acquire-observes Sealed/Consuming — house style: non-atomic
    // payload + one atomic publish edge. Cross-stage handoff in the pipelined
    // engine additionally rides PacketRing::sealed_slot_ (the single
    // release-publish / acquire-take of §7); state_ makes standalone
    // (ring-less) use and the debug read/write asserts self-sufficient.
    // Pairing: store(release) in begin_fill/seal/begin_consume/reset ↔
    // load(acquire) in state().
    std::atomic<State> state_{State::Retired};

    // ---- Sealed content ----
    // Non-atomic payload, written only by FrameBuilder during Filling (merged
    // views point into the packet arena) and published by the Filling→Sealed
    // release edge above.
    std::span<const ModelDrawCommand> models_;
    std::span<const SkinnedDrawCommand> skinned_;
    std::span<const BillboardDrawCommand> billboards_;
    std::span<const DebugLineDrawCommand> debug_lines_;
    std::span<const EffectSpawnCommand> effect_spawns_;
    std::span<const bracken::render::lighting::PointLight> point_lights_;
    std::span<const bracken::render::lighting::SpotLight> spot_lights_;
    std::span<const UiCommand> ui_commands_;
    std::span<const BonePalette> palettes_;
    uint32_t palette_matrix_total_ = 0;

    // Engine frame section (sim-stage tail writes via FrameBuilder's
    // engine-internal API; published by the same Filling→Sealed edge).
    std::span<const EffectEmitterSnapshot> effect_emitters_;
    std::span<const EffectParticleSnapshot> effect_particles_;
    uint32_t effect_instance_count_ = 0;
    float scene_time_ = 0.0f;
    float sky_time_ = 0.0f;
    bool has_engine_frame_ = false;

    CameraState camera_{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(0.0f)};
    bracken::render::EnvironmentState environment_{};
    bool has_camera_ = false;
    bool has_environment_ = false;
    bool draw_skybox_ = true;
    bool draw_ground_ = true;
    bool draw_grass_ = true;
    float dt_ = 0.0f;
    double time_ = 0.0;
    uint64_t frame_index_ = 0;
};

} // namespace bracken::scene
