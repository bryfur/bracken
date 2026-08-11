#pragma once

#include "bracken/contracts/camera_state.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/handle.hpp"
#include "bracken/gpu/pipeline_registry.hpp"

#include <SDL3/SDL_gpu.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <vector>

/**
 * @file render_hook.hpp
 * @brief Host render passes: an ORDERED, MULTI-SLOT registry of callbacks
 *        SceneRenderer runs at named points inside a frame.
 *
 * This replaces `SceneRenderer::set_post_ui_callback` as the way a game adds a
 * pass, and the reason is a bug rather than a preference: that setter is ONE
 * slot, `Application::init_dev_ui` already claims it, and a game that called it
 * silently turned the developer overlay off. A single-assignment extension
 * point cannot be shared, so it was never an extension point.
 *
 * What this is NOT: a render graph. There are no declared reads or writes, no
 * transient resource pool and no reordering — hooks run in the order the host
 * asked for, at the point the host named. `docs/engine_design_notes.md` is
 * explicit that the renderer should not be routed through `render_graph/`
 * without a measured reason, and adding resource declarations here would be
 * exactly that rewrite by the back door.
 */

namespace bracken::scene {

/**
 * @brief Where in `SceneRenderer::render_frame` a host pass runs.
 *
 * EVERY slot fires with a live command buffer and NO render/compute/copy pass
 * open. A callback that begins one MUST end it before returning: SDL3 GPU
 * offers no way to ask whether a pass is open, so a leaked pass corrupts every
 * later pass in the frame and fails inside the driver, far from the cause.
 *
 * The first four are HDR-PATH ONLY: they need the offscreen scene targets and a
 * 3D pass to sit between. They do not run on the fallback direct-to-swapchain
 * path (no HDR offscreen target), on a frame with no 3D content or no camera
 * ever set, or on a minimized window; `AfterUi` additionally does not run when
 * the swapchain image could not be acquired. EVERY one of those cases is
 * counted in `FrameDiagnostics::render_hooks_skipped` rather than dropped
 * silently, which is what lets a host tell "my pass never ran" apart from "my
 * pass ran and drew nothing".
 */
enum class RenderHookSlot : uint8_t {
    /// After GTAO / GI / ray tracing and BEFORE the forward pass. Scene colour
    /// is about to be CLEARED, so this is the slot for compute work writing a
    /// host-owned target the forward pass will sample — not for painting into
    /// scene colour. Scene DEPTH is READ-ONLY here: the opaque forward draws
    /// test `EQUAL` against exactly what the prepass wrote, so writing depth in
    /// this slot makes the forward pass drop geometry.
    PreForward,
    /// Immediately after the forward pass ends, before the ray-traced
    /// reflection resolve. Scene colour (RGBA16F, linear HDR) and scene depth
    /// (D32, REVERSED-Z) both hold this frame's result and neither is bound.
    /// THE DEFAULT SLOT for "draw something into the world".
    AfterOpaque,
    /// After water, before the TAA resolve. Anything drawn here is temporally
    /// stabilized along with everything else — which is what you want for
    /// world-space geometry and emphatically not what you want for an overlay.
    BeforeTaa,
    /// After the composite (and FXAA when `AAMode::FXAA`), before the UI pass.
    /// `swapchain` is live and LDR/gamma-encoded; `scene_color` still holds the
    /// pre-tonemap HDR image.
    AfterComposite,
    /// After the engine's UI pass, on the swapchain. Runs BEFORE the legacy
    /// `set_post_ui_callback`, so the developer overlay still draws on top of
    /// a host's own overlay. The only slot that also runs on the fallback path.
    AfterUi,
};

/// Number of slots. Not an enumerator inside `RenderHookSlot` so that a
/// `switch` over the enum stays exhaustive under `-Wswitch`.
inline constexpr size_t k_render_hook_slot_count = 5;

/// Debug name of a slot ("Unknown" out of range).
[[nodiscard]] const char* render_hook_slot_to_string(RenderHookSlot slot);

/**
 * @brief Everything a host pass is given.
 *
 * Every pointer here is valid ONLY for the duration of the call. Nothing may
 * be cached across frames: the offscreen targets are recreated on resize and
 * the swapchain texture is a different image every frame.
 */
struct RenderHookContext {
    /// The frame's command buffer. Never null when a hook is invoked.
    SDL_GPUCommandBuffer* cmd = nullptr;
    /// For `get_pipeline(HostPipelineHandle)`. Re-fetch by handle here rather
    /// than caching a `GPUPipeline*` at init: a hot reload rebuilds every
    /// pipeline and the old pointer dies with it.
    gpu::PipelineRegistry* pipelines = nullptr;
    /// Which slot is running. Lets one callback serve several slots.
    RenderHookSlot slot = RenderHookSlot::AfterOpaque;

    /// RGBA16F linear HDR. Null whenever the offscreen pair does not hold THIS
    /// frame's image — the fallback direct-to-swapchain path, or a frame with
    /// no 3D content, both of which can reach `AfterUi`. Null rather than a
    /// stale texture is deliberate: the textures often still exist there,
    /// holding the last frame that did render into them.
    SDL_GPUTexture* scene_color = nullptr;
    /// D32_FLOAT, REVERSED-Z (near→1, far→0, clear 0.0). Same nulling rule as
    /// `scene_color`.
    SDL_GPUTexture* scene_depth = nullptr;
    SDL_GPUTexture* velocity = nullptr;    ///< RG16F motion vectors; null when the prepass did not run
    SDL_GPUTexture* normal_roughness = nullptr; ///< RGBA8 G-buffer; null unless RT / ProbesScreen is on
    /// RGBA8 (AO, bent normal, cone aperture). NOT null when AO is off — it is
    /// the renderer's 1x1 opaque-white dummy then, so a host pass can sample it
    /// unconditionally and get "fully unoccluded" instead of having to branch.
    SDL_GPUTexture* ao = nullptr;
    SDL_GPUTexture* swapchain = nullptr;        ///< AfterComposite / AfterUi only; null elsewhere

    SDL_GPUTextureFormat scene_color_format = SDL_GPU_TEXTUREFORMAT_INVALID;
    SDL_GPUTextureFormat depth_format = SDL_GPU_TEXTUREFORMAT_INVALID;
    SDL_GPUTextureFormat swapchain_format = SDL_GPU_TEXTUREFORMAT_INVALID;

    int width = 0;  ///< Offscreen/swapchain width in pixels
    int height = 0; ///< …and height

    /// The DRAW camera — its projection carries this frame's TAA sub-pixel
    /// jitter when jitter is on. Use the two unjittered matrices below for any
    /// reprojection or velocity math, never these.
    CameraState camera{};
    glm::mat4 view_projection_unjittered{1.0f};
    glm::mat4 prev_view_projection_unjittered{1.0f};

    bool depth_prepass_active = false; ///< false ⇒ `velocity` is stale/null and depth is forward-written
    float scene_time = 0.0f;           ///< Seconds of scene clock
    float dt = 0.0f;                   ///< This frame's delta time
    uint64_t frame_index = 0;          ///< Rendered-frame counter (drives the jitter sequence)
};

using RenderHookFn = std::function<void(const RenderHookContext&)>;

struct RenderHookTag {};
/// Opaque, generation-checked. Default-constructed == invalid.
using RenderHookHandle = core::Handle<RenderHookTag>;

/**
 * @brief The ordered registry of host passes.
 *
 * Pure and device-free — no GPU, no SDL calls, no SceneRenderer — which is what
 * makes slot dispatch and ordering testable headless
 * (`scene/render_hook.test.cpp`).
 *
 * Tier: main-thread (asserted). Register from `on_init` after
 * `init_renderer()`, or at the frame fence in `on_main_frame`. NEVER from
 * `on_simulate`: the sim job runs while the previous frame is still recording,
 * so mutating the list there would be a data race with dispatch.
 */
class RenderHookRegistry {
public:
    /// One registration. Held by pointer with a stable address for its whole
    /// life, because `name.c_str()` is handed to the GPU debug-label API, which
    /// stores the raw pointer and requires it to outlive the CALL.
    struct Entry {
        std::string name;
        /// The same text, interned in a pool this registry NEVER frees, so the
        /// pointer stays valid after the entry is removed.
        ///
        /// That distinction is the whole reason this field exists.
        /// `GPUTimestampPool::reserve_pair` stores its `pass_name` as a raw
        /// pointer and `PassTiming::name` is a `std::string_view` over it, and
        /// that view lives in `RenderStats::pass_times_ms` until the NEXT
        /// frame's `read_results`. A host may legally `remove()` a hook at the
        /// frame fence — between the frame that recorded the timing and the
        /// read of it — so handing `name.c_str()` to the timer pool is a
        /// use-after-free. Handing `stable_name` is not.
        const char* stable_name = "";
        int order = 0;
        uint64_t sequence = 0; ///< Registration counter; the tie-break within an `order`
        RenderHookSlot slot = RenderHookSlot::AfterOpaque;
        RenderHookFn fn;
    };

    /**
     * @brief RAII marker: while one of these is alive the registry refuses
     * mutation.
     *
     * Held by `SceneRenderer` around each slot's dispatch. Adding or removing a
     * hook from inside a hook would reallocate the very list being walked; that
     * is a host mistake, so it is a `BRACKEN_CONTRACT` (live in release) rather
     * than a debug assert or undefined behaviour.
     */
    class DispatchScope {
    public:
        explicit DispatchScope(RenderHookRegistry& registry) noexcept : registry_(&registry) {
            ++registry_->dispatch_depth_;
        }
        ~DispatchScope() { --registry_->dispatch_depth_; }
        DispatchScope(const DispatchScope&) = delete;
        DispatchScope& operator=(const DispatchScope&) = delete;
        DispatchScope(DispatchScope&&) = delete;
        DispatchScope& operator=(DispatchScope&&) = delete;

    private:
        RenderHookRegistry* registry_;
    };

    /**
     * @brief Register `fn` to run in `slot`.
     *
     * Lower `order` runs first; ties break by registration order. That makes
     * the set of hooks in a slot a TOTAL order which depends on nothing but
     * what the host asked for — no hash, no pointer value — the same
     * determinism rule the Prepare fan-out's chunk-index merge follows.
     *
     * `name` is used for the profiler zone and the GPU debug label. Returns an
     * invalid handle (logging an ERROR) if `fn` is empty.
     */
    [[nodiscard]] RenderHookHandle add(RenderHookSlot slot, std::string name, int order, RenderHookFn fn);

    /// Convenience: `order == 0`.
    [[nodiscard]] RenderHookHandle add(RenderHookSlot slot, std::string name, RenderHookFn fn) {
        return add(slot, std::move(name), 0, std::move(fn));
    }

    /// Remove a registration. Bumps the slot's generation, so a stale copy of
    /// the handle is a no-op rather than a removal of somebody else's hook.
    /// @return false if the handle was already invalid or stale.
    bool remove(RenderHookHandle handle);

    /// Remove everything.
    void clear();

    /// True when no hook is registered in ANY slot — the frame path's early-out.
    [[nodiscard]] bool empty() const noexcept { return live_ == 0; }
    /// True when no hook is registered in `slot`.
    ///
    /// `slot` is bounds-checked with a BRACKEN_CONTRACT (live in release): the
    /// backing store is a fixed array and `RenderHookSlot` is a host-supplied
    /// scoped enum, so a bad cast would otherwise be an out-of-bounds read. The
    /// engine calls this five times per frame, which is a per-frame path, not a
    /// per-item loop.
    [[nodiscard]] bool empty(RenderHookSlot slot) const { return entries(slot).empty(); }
    /// Live registrations across all slots.
    [[nodiscard]] size_t size() const noexcept { return live_; }

    /// The hooks of `slot`, IN EXECUTION ORDER. The span and the entries it
    /// points at stay valid until the next `add`/`remove`/`clear`.
    /// Bounds-checked — see `empty(RenderHookSlot)`.
    [[nodiscard]] std::span<Entry* const> entries(RenderHookSlot slot) const {
        const size_t slot_index = index_of(slot);
        BRACKEN_CONTRACT(slot_index < k_render_hook_slot_count, "RenderHookSlot out of range");
        return order_[slot_index];
    }

    /// True while a `DispatchScope` is alive.
    [[nodiscard]] bool is_dispatching() const noexcept { return dispatch_depth_ > 0; }

    [[nodiscard]] static constexpr size_t index_of(RenderHookSlot slot) noexcept {
        return static_cast<size_t>(slot);
    }

private:
    struct Slot {
        std::unique_ptr<Entry> entry; ///< null ⇒ free slot
        uint32_t generation = 1;
    };

    // Owning storage, indexed by handle index; slots are reused after removal.
    std::vector<Slot> slots_;
    // Per-slot execution order: raw pointers into slots_' heap entries, kept
    // sorted by (order, sequence). Rebuilding the sort on every add is fine —
    // this list has single digits of elements and is touched at init, not per
    // frame.
    std::array<std::vector<Entry*>, k_render_hook_slot_count> order_{};
    // Interned hook names, deduplicated by value and NEVER erased — a node-based
    // container, so every `c_str()` handed out stays valid for the registry's
    // lifetime. See Entry::stable_name for why "for the registration's lifetime"
    // is not long enough. Dedup is what bounds it: a host that adds and removes
    // the same named hook every frame reuses one node forever.
    std::set<std::string> name_pool_;
    uint64_t next_sequence_ = 0;
    size_t live_ = 0;
    int dispatch_depth_ = 0;
};

} // namespace bracken::scene
