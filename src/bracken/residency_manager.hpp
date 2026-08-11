#pragma once

#include "bracken/async_model_loader.hpp"
#include "bracken/contracts/model_handle.hpp"
#include "bracken/model_loader.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vector>

namespace bracken {

/**
 * Distance-driven asset residency over the pieces that already existed.
 *
 * WHAT WAS MISSING. `AsyncModelLoader` already does the hard half of streaming
 * — decode on the pool, budgeted upload on the main thread at the frame fence
 * — but nothing above it ever gave anything back: `TextureCache` had only
 * `clear()`, `ModelManager` only `unload_all()`, and a draw whose model was
 * absent was counted as `draws_dropped_unknown_model`, which is a *bug*
 * diagnosis for what is really a *budget* condition. This class is the missing
 * half: a byte budget, a distance band, and per-item residency state.
 *
 * THE BAND, NOT A RADIUS. One radius is a thrash generator: an item sitting
 * exactly on it loads and evicts on alternate frames forever, and each cycle
 * costs a decode plus an upload. So there are two radii and the space between
 * them is dead: inside `load_radius` an item is requested, outside
 * `evict_radius` it becomes ELIGIBLE for eviction, and in between nothing ever
 * happens. `set_config` REFUSES `evict_radius <= load_radius` rather than
 * clamping it, because a silently-clamped band is a band that looks configured
 * and thrashes anyway.
 *
 * ELIGIBLE IS NOT EVICTED. Being outside the band only makes an item a
 * candidate; eviction fires only while the budget is exceeded, only for items
 * resident at least `min_frames_resident` frames, and at most
 * `max_evicts_per_pump` per call. A budget of 0 means "never evict", which is
 * what every existing host gets by not constructing this class at all.
 *
 * HANDLE STABILITY IS THE LOAD-BEARING INVARIANT. A `ModelHandle` is a dense
 * index baked into draw commands and into the ray tracer's instance list, and
 * `async_model_loader.cpp` records that assigning them out of order once made
 * three smoke runs render three different frames. Nothing here renumbers,
 * compacts or reuses a slot: an evicted item keeps its handle,
 * `ModelManager::unload_model` keeps the slot, and the reload goes back into
 * that same slot through `Request::reupload_into`.
 *
 * THE ACCELERATION STRUCTURE. Evicting a model changes neither the instance
 * count nor the handle sequence, so `RtScene`'s topology hash does NOT fire
 * and the BLAS would keep pointing at geometry `unload_model` just freed. Every
 * eviction and every completed reload therefore calls the invalidator
 * installed with `set_geometry_invalidator` — wire it to
 * `SceneRenderer::invalidate_raytracing_scene()`. Leaving it unset with ray
 * tracing on is a use-after-free, and `pump()` says so, once.
 *
 * Tier: main-thread for `add`, `set_config`, `pump`. `set_viewer` is
 * worker-safe (a relaxed store of three floats) so a game can call it straight
 * from `on_simulate`.
 *
 * `handle()`, `model_handle()`, `resident()` and `state()` are worker-safe
 * reads ONLY against the two main-thread writers, `pump()` and `add()` — and
 * `add()` can reallocate the item table, so a read concurrent with one is a
 * data race, not a stale answer. In the engine's frame that is not a hazard:
 * `pump()` runs at the frame fence with no sim in flight, and `add()` belongs
 * to load time. A host that registers items from a background thread while a
 * sim job resolves handles has to serialize that itself.
 */
class ResidencyManager {
public:
    /** Opaque, stable per-item id. Not a ModelHandle: an item HAS a handle,
     *  and only sometimes. */
    struct StreamId {
        uint32_t value = 0; ///< 0 is the invalid id; items start at 1
        [[nodiscard]] bool valid() const noexcept { return value != 0; }
        friend bool operator==(const StreamId&, const StreamId&) = default;
    };

    /** Where an item is in its lifecycle. */
    enum class State : uint8_t {
        /// Registered here, never loaded. `handle()` gives the proxy.
        Absent,
        /// A decode is in flight; still drawing the proxy.
        Loading,
        /// The real model is in the ModelManager and `Model::loaded` is true.
        Resident,
        /// Was resident, has been unloaded. The HANDLE IS STILL VALID and the
        /// slot still exists — only the payload is gone.
        Evicted,
        /// The decode failed. Never retried: retrying a missing file every
        /// pump is a log flood, and the file will not appear.
        Failed,
    };

    struct Config {
        /// Byte ceiling for everything this manager loaded. 0 = unlimited,
        /// which means eviction never fires at all.
        std::size_t budget_bytes = 512ull << 20;
        /// Inside this distance (plus the item's cull_radius) an absent item
        /// is requested.
        float load_radius = 4000.0f;
        /// Outside this distance an item becomes eligible for eviction.
        /// MUST be greater than load_radius — the gap is the hysteresis band.
        float evict_radius = 6000.0f;
        /// Frames an item must have been resident before it may be evicted.
        /// Time-domain hysteresis on top of the spatial band: it stops a
        /// teleporting or fast-strafing viewer from evicting what it just paid
        /// to load.
        uint32_t min_frames_resident = 120;
        /// Upload budget handed to AsyncModelLoader::pump each call.
        std::size_t max_uploads_per_pump = 2;
        /// Decodes started per call. Requests are cheap on the main thread but
        /// each one occupies a worker.
        std::size_t max_requests_per_pump = 4;
        /// Evictions applied per call. Each frees GPU resources on the main
        /// thread, so a big number is a frame spike.
        std::size_t max_evicts_per_pump = 4;
    };

    struct Item {
        std::string name;   ///< ModelManager registration name
        std::string source; ///< pack entry or filesystem path
        glm::vec3 anchor{0.0f};         ///< where in the world it is needed
        float cull_radius = 0.0f;       ///< added to both radii for this item
        /// Drawn while the item is not resident. INVALID_MODEL_HANDLE means
        /// "draw nothing", which the Prepare stage counts as
        /// `draws_deferred_not_resident`.
        ModelHandle proxy = INVALID_MODEL_HANDLE;
    };

    struct Stats {
        std::size_t resident_bytes = 0;
        std::size_t budget_bytes = 0;
        uint32_t resident = 0;
        uint32_t loading = 0;
        uint32_t absent = 0;
        uint32_t evicted = 0;
        uint32_t failed = 0;
        uint32_t loads_started = 0;   ///< cumulative
        uint32_t loads_completed = 0; ///< cumulative
        uint32_t evictions = 0;       ///< cumulative
        /// Pumps that ended still over budget with nothing evictable. A
        /// non-zero value means the budget cannot be met from the items in
        /// range — raise the budget or shrink the load radius.
        uint32_t budget_overruns = 0;
        /// Evictions the time-domain hysteresis refused. CUMULATIVE, like
        /// loads_started/loads_completed/evictions above it — a per-pump figure
        /// is the difference between two reads.
        uint32_t evictions_blocked_by_hysteresis = 0;
    };

    // ------------------------------------------------------------------
    // The decision core, as PURE functions.
    //
    // Everything below decides; nothing below acts. They are static and take
    // only their arguments so the band, the hysteresis and the eviction order
    // are unit-testable with no ModelManager, no job system and no GPU —
    // residency_manager.test.cpp is entirely built on them.
    // ------------------------------------------------------------------

    /** Inside the load band? `distance` is viewer-to-anchor. */
    [[nodiscard]] static bool wants_load(float distance, float cull_radius, const Config& cfg) noexcept {
        return distance <= cfg.load_radius + cull_radius;
    }

    /** Outside the evict band? Being outside only makes an item a CANDIDATE —
     *  the budget and the frame count still have to agree. */
    [[nodiscard]] static bool outside_evict_band(float distance, float cull_radius, const Config& cfg) noexcept {
        return distance > cfg.evict_radius + cull_radius;
    }

    /** In the dead zone between the two radii, where nothing ever happens.
     *  This is the property the oscillation test asserts. */
    [[nodiscard]] static bool in_hysteresis_band(float distance, float cull_radius, const Config& cfg) noexcept {
        return !wants_load(distance, cull_radius, cfg) && !outside_evict_band(distance, cull_radius, cfg);
    }

    /** A config whose band is empty or inverted, which would thrash. */
    [[nodiscard]] static bool config_is_valid(const Config& cfg) noexcept {
        return cfg.load_radius > 0.0f && cfg.evict_radius > cfg.load_radius;
    }

    /** One eviction candidate, reduced to the numbers the ordering uses. */
    struct EvictCandidate {
        uint32_t index = 0;          ///< caller's item index, echoed back
        float distance = 0.0f;       ///< viewer-to-anchor
        uint32_t frames_resident = 0;
        std::size_t bytes = 0;
    };

    /**
     * Choose which candidates to evict, in order, to get back under budget.
     *
     * PURE and DETERMINISTIC: farthest first, ties broken by ascending index,
     * so the same inputs always produce the same plan regardless of the order
     * they were handed in. Stops as soon as the projected resident total is
     * within `cfg.budget_bytes`, and never exceeds `cfg.max_evicts_per_pump`.
     * Candidates below `cfg.min_frames_resident` are skipped and counted in
     * `blocked_by_hysteresis`.
     *
     * A zero budget returns an empty plan: unlimited means never evict.
     *
     * @param out cleared, then filled with item indices in eviction order.
     */
    static void plan_evictions(std::span<const EvictCandidate> candidates, std::size_t resident_bytes,
                               const Config& cfg, std::vector<uint32_t>& out,
                               uint32_t* blocked_by_hysteresis = nullptr);

    // ------------------------------------------------------------------

    /** @param loader must outlive this manager; it is where decodes happen. */
    ResidencyManager(ModelManager& models, AsyncModelLoader& loader, Config cfg);
    ~ResidencyManager();

    ResidencyManager(const ResidencyManager&) = delete;
    ResidencyManager& operator=(const ResidencyManager&) = delete;

    /**
     * Register a streamable item. Main-thread.
     *
     * If `item.name` already names a loaded model, the item starts Resident
     * against that handle rather than re-requesting it — which is how a host
     * brings already-loaded content under management without reloading it.
     */
    [[nodiscard]] StreamId add(Item item);

    /** Replace the config. Refuses (false, logged) an invalid band; the old
     *  config stays in force. Main-thread. */
    [[nodiscard]] bool set_config(Config cfg);
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /** Viewer position used by the next `pump()`. Worker-safe. */
    void set_viewer(const glm::vec3& position) noexcept;

    /**
     * Called whenever a model's GEOMETRY set changed under the ray tracer's
     * feet — every eviction and every completed reload. Wire it to
     * `SceneRenderer::invalidate_raytracing_scene()`. See the class comment:
     * without it, eviction is a use-after-free rather than a saving.
     */
    void set_geometry_invalidator(std::function<void()> invalidator);

    /** Resolve for a draw: the real handle when resident, the item's proxy
     *  otherwise, INVALID_MODEL_HANDLE when there is no proxy. Worker-safe. */
    [[nodiscard]] ModelHandle handle(StreamId id) const noexcept;
    /** The item's own slot, resident or not — INVALID_MODEL_HANDLE until it
     *  has been loaded at least once. Worker-safe. */
    [[nodiscard]] ModelHandle model_handle(StreamId id) const noexcept;
    [[nodiscard]] bool resident(StreamId id) const noexcept;
    [[nodiscard]] State state(StreamId id) const noexcept;
    [[nodiscard]] std::size_t item_count() const noexcept { return items_.size(); }

    /**
     * Decide and apply, at the frame fence.
     *
     * In order: adopt whatever the loader finished, request what entered the
     * load band, then evict while over budget. Main-thread (asserted) — it
     * creates and destroys GPU resources, and it must run at the pipeline's
     * quiescent point (`Application::frame_fence`), where no packet
     * referencing an evicted model is being built or rendered. Hot-reload
     * applies there for exactly the same reason.
     */
    void pump();

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    struct Entry {
        Item item;
        State state = State::Absent;
        ModelHandle handle = INVALID_MODEL_HANDLE;
        uint32_t frames_resident = 0;
        std::size_t bytes = 0;
    };

    void invalidate_geometry();

    ModelManager& models_;
    AsyncModelLoader& loader_;
    Config config_;
    std::vector<Entry> items_;
    /// Written by pump()'s adoption step, read by nothing else.
    std::vector<AsyncModelLoader::Completed> completed_scratch_;
    std::vector<EvictCandidate> candidate_scratch_;
    /// Load-request order for one pump. Kept apart from the eviction plan
    /// below: one buffer for two plans in the same call is a hazard waiting for
    /// someone to move the over-budget check that reads it.
    std::vector<uint32_t> request_scratch_;
    std::vector<uint32_t> plan_scratch_;
    std::function<void()> invalidate_geometry_;
    /// Viewer position. Relaxed on both ends: it is an advisory input to a
    /// distance comparison, a pump may legitimately use a position one frame
    /// stale, and there is no other datum whose visibility it publishes.
    /// Pairing: none by design.
    std::atomic<float> viewer_x_{0.0f};
    std::atomic<float> viewer_y_{0.0f};
    std::atomic<float> viewer_z_{0.0f};
    std::size_t resident_bytes_ = 0;
    Stats stats_{};
};

} // namespace bracken
