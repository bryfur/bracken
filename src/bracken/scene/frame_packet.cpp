#include "bracken/scene/frame_packet.hpp"

#include "bracken/core/assert.hpp"

#include <cstdint>
#include <mutex>
#include <new>

namespace bracken::scene {

namespace {

// Every reservation is rounded so slab cursors only ever hold multiples of
// this; combined with 16-aligned slab bases, plain bumps are 16-aligned and
// only over-aligned requests (> 16) need headroom.
constexpr std::size_t k_cursor_align = 16;
constexpr std::size_t k_max_block_align = 128; // covers cache-line-aligned lane arrays
constexpr std::size_t k_slab_bytes = 64 * 1024;

constexpr bool is_pow2(std::size_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

// align_up() below is a mask-based round-up, which is only correct for
// power-of-two alignments — so pin that here rather than trusting the
// constants to stay well-formed. This also keeps is_pow2 referenced in
// NDEBUG builds, where its only other use (a BRACKEN_ASSERT) compiles away.
static_assert(is_pow2(k_cursor_align));
static_assert(is_pow2(k_max_block_align));

constexpr std::size_t align_up(std::size_t v, std::size_t a) {
    return (v + a - 1) & ~(a - 1);
}

inline std::byte* align_up_ptr(std::byte* p, std::size_t a) {
    return reinterpret_cast<std::byte*>(align_up(reinterpret_cast<std::uintptr_t>(p), a));
}

} // namespace

FramePacket::FramePacket(std::size_t initial_arena_bytes) : arena_(initial_arena_bytes) {}

void FramePacket::begin_fill() {
    BRACKEN_ASSERT(state() == State::Retired, "FramePacket::begin_fill: packet not Retired (state {})",
                   static_cast<uint32_t>(state()));
    state_.store(State::Filling, std::memory_order_release);
}

void FramePacket::begin_consume() {
    BRACKEN_ASSERT(state() == State::Sealed, "FramePacket::begin_consume: packet not Sealed (state {})",
                   static_cast<uint32_t>(state()));
    state_.store(State::Consuming, std::memory_order_release);
}

void FramePacket::reset() {
    // Clear the sealed views BEFORE the arena reset reclaims their storage so
    // a stale span can never survive into the next cycle.
    models_ = {};
    skinned_ = {};
    billboards_ = {};
    debug_lines_ = {};
    effect_spawns_ = {};
    point_lights_ = {};
    spot_lights_ = {};
    ui_commands_ = {};
    palettes_ = {};
    palette_matrix_total_ = 0;
    effect_emitters_ = {};
    effect_particles_ = {};
    effect_instance_count_ = 0;
    scene_time_ = 0.0f;
    sky_time_ = 0.0f;
    has_engine_frame_ = false;
    camera_ = CameraState{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(0.0f)};
    environment_ = bracken::render::EnvironmentState{};
    has_camera_ = false;
    has_environment_ = false;
    draw_skybox_ = true;
    draw_ground_ = true;
    draw_grass_ = true;
    dt_ = 0.0f;
    time_ = 0.0;
    frame_index_ = 0;

    active_slab_.store(nullptr, std::memory_order_relaxed);
    palette_slots_.store(0, std::memory_order_relaxed);

    // Explicit cross-thread handoff: the retiring thread takes the arena
    // owner binding for the reset. The caller's frame fence (ring retire
    // ordering / stage join) is the synchronization that makes this safe.
    arena_.rebind_owner();
    arena_.reset();

    state_.store(State::Retired, std::memory_order_release);
}

void* FramePacket::reserve_block(std::size_t size, std::size_t align) {
    BRACKEN_ASSERT(state() == State::Filling, "FramePacket::reserve_block outside the Filling state (state {})",
                   static_cast<uint32_t>(state()));
    BRACKEN_ASSERT(size > 0, "FramePacket::reserve_block: zero-size reservation");
    BRACKEN_ASSERT(is_pow2(align) && align <= k_max_block_align,
                   "FramePacket::reserve_block: align {} must be a power of two <= {}", align, k_max_block_align);

    // Cursor offsets are multiples of k_cursor_align over a 16-aligned base;
    // over-aligned requests carry headroom so the pointer can be aligned
    // forward inside the reserved range.
    const std::size_t headroom = align > k_cursor_align ? align - k_cursor_align : 0;
    const std::size_t reserve = align_up(size + headroom, k_cursor_align);

    for (;;) {
        SlabState* slab = active_slab_.load(std::memory_order_acquire);
        if (slab != nullptr) {
            const std::size_t off = slab->cursor.fetch_add(reserve, std::memory_order_relaxed);
            if (off + reserve <= slab->capacity) {
                return align_up_ptr(slab->base + off, align);
            }
            // Slab exhausted; the overshot cursor is harmless — the slab is
            // abandoned once a fresh one is published.
        }
        refill_slab(slab, reserve);
    }
}

void FramePacket::refill_slab(SlabState* observed, std::size_t min_bytes) {
    std::scoped_lock lock(slab_refill_mutex_);
    // Someone else refilled while we waited: retry against the fresh slab.
    if (active_slab_.load(std::memory_order_relaxed) != observed) {
        return;
    }

    // The mutex serializes refills, so this cross-thread arena handoff is the
    // sanctioned rebind-under-synchronization pattern (memory/arena.hpp).
    arena_.rebind_owner();

    const std::size_t slab_bytes = min_bytes > k_slab_bytes ? min_bytes : k_slab_bytes;
    void* state_mem = arena_.allocate(sizeof(SlabState), alignof(SlabState));
    void* base = arena_.allocate(slab_bytes, k_cursor_align);
    BRACKEN_VERIFY(state_mem != nullptr && base != nullptr,
                   "FramePacket: arena out of memory reserving a {}-byte slab", slab_bytes);

    auto* slab = new (state_mem) SlabState;
    slab->base = static_cast<std::byte*>(base);
    slab->capacity = slab_bytes;
    // Publish the fully initialized slab (see active_slab_ protocol note).
    active_slab_.store(slab, std::memory_order_release);
}

std::span<const glm::mat4> FramePacket::palette_matrices(BonePaletteId id) const {
    assert_readable();
    BRACKEN_ASSERT(id.valid() && id.value < palettes_.size(),
                   "FramePacket::palette_matrices: palette id {} out of range (use ids from sealed commands)",
                   id.value);
    if (!id.valid() || id.value >= palettes_.size()) {
        return {};
    }
    const BonePalette& p = palettes_[id.value];
    return {p.matrices, p.joint_count};
}

} // namespace bracken::scene
