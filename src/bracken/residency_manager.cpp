#include "bracken/residency_manager.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/log_throttle.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

#include <algorithm>

namespace bracken {

namespace {
constexpr const char* LOG_TAG = "residency";
} // namespace

void ResidencyManager::plan_evictions(std::span<const EvictCandidate> candidates, std::size_t resident_bytes,
                                      const Config& cfg, std::vector<uint32_t>& out,
                                      uint32_t* blocked_by_hysteresis) {
    out.clear();
    if (blocked_by_hysteresis != nullptr) {
        *blocked_by_hysteresis = 0;
    }
    // Unlimited means never evict — the configuration every host that does not
    // construct this class effectively has.
    if (cfg.budget_bytes == 0 || resident_bytes <= cfg.budget_bytes) {
        return;
    }

    // Farthest first, ties by ascending index. The tie-break is not cosmetic:
    // two props at the same distance must produce the same plan whatever order
    // the caller enumerated them in, or a scene's eviction sequence depends on
    // container iteration order.
    std::vector<const EvictCandidate*> ordered;
    ordered.reserve(candidates.size());
    for (const EvictCandidate& c : candidates) {
        if (c.frames_resident < cfg.min_frames_resident) {
            if (blocked_by_hysteresis != nullptr) {
                ++*blocked_by_hysteresis;
            }
            continue;
        }
        ordered.push_back(&c);
    }
    std::sort(ordered.begin(), ordered.end(), [](const EvictCandidate* a, const EvictCandidate* b) {
        if (a->distance != b->distance) {
            return a->distance > b->distance;
        }
        return a->index < b->index;
    });

    std::size_t projected = resident_bytes;
    for (const EvictCandidate* c : ordered) {
        if (out.size() >= cfg.max_evicts_per_pump) {
            break;
        }
        if (projected <= cfg.budget_bytes) {
            break;
        }
        out.push_back(c->index);
        projected = c->bytes >= projected ? 0 : projected - c->bytes;
    }
}

ResidencyManager::ResidencyManager(ModelManager& models, AsyncModelLoader& loader, Config cfg)
    : models_(models), loader_(loader), config_(cfg) {
    BRACKEN_ASSERT_MAIN_THREAD();
    if (!config_is_valid(config_)) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "ResidencyManager: load_radius {} / evict_radius {} is not a band "
                          "(evict must exceed load); falling back to the defaults",
                          config_.load_radius, config_.evict_radius);
        const Config defaults;
        config_.load_radius = defaults.load_radius;
        config_.evict_radius = defaults.evict_radius;
    }
    stats_.budget_bytes = config_.budget_bytes;
}

ResidencyManager::~ResidencyManager() = default;

ResidencyManager::StreamId ResidencyManager::add(Item item) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (item.name.empty()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ResidencyManager::add: an item needs a registration name");
        return {};
    }
    for (const Entry& existing : items_) {
        if (existing.item.name == item.name) {
            BRACKEN_LOG_ERROR(LOG_TAG, "ResidencyManager::add: '{}' is already managed", item.name);
            return {};
        }
    }

    Entry entry;
    entry.item = std::move(item);
    // Already loaded? Adopt it rather than re-requesting: this is how a host
    // brings content it loaded at boot under management without paying for it
    // twice, and it is also what makes the manager testable against a
    // ModelManager populated by register_model.
    const ModelHandle existing = models_.get_handle(entry.item.name);
    if (existing != INVALID_MODEL_HANDLE) {
        const Model* model = models_.get_model(existing);
        if (model != nullptr && model->loaded) {
            entry.handle = existing;
            entry.state = State::Resident;
            entry.bytes = models_.gpu_bytes(existing);
            entry.frames_resident = 0;
            resident_bytes_ += entry.bytes;
        } else if (model != nullptr) {
            entry.handle = existing;
            entry.state = State::Evicted;
        }
    }

    items_.push_back(std::move(entry));
    return StreamId{static_cast<uint32_t>(items_.size())};
}

bool ResidencyManager::set_config(Config cfg) {
    BRACKEN_ASSERT_MAIN_THREAD();
    if (!config_is_valid(cfg)) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "ResidencyManager::set_config refused: load_radius {} / evict_radius {} is not a "
                          "band. A single radius (or an inverted one) makes an item on the boundary load and "
                          "evict on alternate frames forever.",
                          cfg.load_radius, cfg.evict_radius);
        return false;
    }
    config_ = cfg;
    stats_.budget_bytes = config_.budget_bytes;
    return true;
}

void ResidencyManager::set_viewer(const glm::vec3& position) noexcept {
    viewer_x_.store(position.x, std::memory_order_relaxed);
    viewer_y_.store(position.y, std::memory_order_relaxed);
    viewer_z_.store(position.z, std::memory_order_relaxed);
}

void ResidencyManager::set_geometry_invalidator(std::function<void()> invalidator) {
    BRACKEN_ASSERT_MAIN_THREAD();
    invalidate_geometry_ = std::move(invalidator);
}

ModelHandle ResidencyManager::handle(StreamId id) const noexcept {
    if (!id.valid() || id.value > items_.size()) {
        return INVALID_MODEL_HANDLE;
    }
    const Entry& e = items_[id.value - 1];
    return e.state == State::Resident ? e.handle : e.item.proxy;
}

ModelHandle ResidencyManager::model_handle(StreamId id) const noexcept {
    if (!id.valid() || id.value > items_.size()) {
        return INVALID_MODEL_HANDLE;
    }
    return items_[id.value - 1].handle;
}

bool ResidencyManager::resident(StreamId id) const noexcept {
    return state(id) == State::Resident;
}

ResidencyManager::State ResidencyManager::state(StreamId id) const noexcept {
    if (!id.valid() || id.value > items_.size()) {
        return State::Absent;
    }
    return items_[id.value - 1].state;
}

void ResidencyManager::invalidate_geometry() {
    if (invalidate_geometry_) {
        invalidate_geometry_();
        return;
    }
    // Not an abort: a host with ray tracing off is entirely correct not to
    // install one. But the failure it causes when ray tracing IS on is a
    // use-after-free inside the BVH walk, which reads as a driver crash, so it
    // gets said out loud exactly once.
    BRACKEN_LOG_WARN_ONCE(LOG_TAG,
                          "ResidencyManager: a model's geometry changed but no geometry invalidator is "
                          "installed. With ray tracing enabled the acceleration structure will keep "
                          "referencing freed geometry — call set_geometry_invalidator().");
}

void ResidencyManager::pump() {
    BRACKEN_PROFILE_ZONE("ResidencyManager::pump");
    // Uploads and frees GPU resources, and must see a quiescent engine: it runs
    // at Application::frame_fence, where no sim job is in flight and no packet
    // referencing an evicted model is being built or rendered.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();

    const glm::vec3 viewer{viewer_x_.load(std::memory_order_relaxed), viewer_y_.load(std::memory_order_relaxed),
                           viewer_z_.load(std::memory_order_relaxed)};

    bool geometry_changed = false;

    // ---- 1. Adopt whatever the loader finished ---------------------------
    completed_scratch_ = loader_.pump(config_.max_uploads_per_pump);
    for (const AsyncModelLoader::Completed& done : completed_scratch_) {
        auto it = std::find_if(items_.begin(), items_.end(),
                               [&done](const Entry& e) { return e.item.name == done.name; });
        if (it == items_.end()) {
            // Somebody else's request went through the same loader. Fine — the
            // loader is shared infrastructure, not ours.
            continue;
        }
        if (!done.ok) {
            // No retry, ever: a decode that failed failed because the source is
            // missing or malformed, and re-requesting it every pump is a log
            // flood plus a worker per pump for a file that will not appear.
            it->state = State::Failed;
            continue;
        }
        it->handle = done.handle;
        it->state = State::Resident;
        it->frames_resident = 0;
        it->bytes = models_.gpu_bytes(done.handle);
        resident_bytes_ += it->bytes;
        ++stats_.loads_completed;
        geometry_changed = true;
    }

    // ---- 2. Age the resident set ----------------------------------------
    for (Entry& e : items_) {
        if (e.state == State::Resident && e.frames_resident != UINT32_MAX) {
            ++e.frames_resident;
        }
    }

    // ---- 3. Request what entered the load band ---------------------------
    // Nearest first, ties by index: deterministic, and it means the thing the
    // player is about to walk into wins the budget over the thing at the far
    // edge of the radius.
    const bool room_to_load = config_.budget_bytes == 0 || resident_bytes_ < config_.budget_bytes;
    if (room_to_load && config_.max_requests_per_pump > 0) {
        plan_scratch_.clear();
        for (uint32_t i = 0; i < items_.size(); ++i) {
            const Entry& e = items_[i];
            if (e.state != State::Absent && e.state != State::Evicted) {
                continue;
            }
            const float distance = glm::length(e.item.anchor - viewer);
            if (!wants_load(distance, e.item.cull_radius, config_)) {
                continue;
            }
            plan_scratch_.push_back(i);
        }
        std::sort(plan_scratch_.begin(), plan_scratch_.end(), [&](uint32_t a, uint32_t b) {
            const float da = glm::length(items_[a].item.anchor - viewer);
            const float db = glm::length(items_[b].item.anchor - viewer);
            if (da != db) {
                return da < db;
            }
            return a < b;
        });
        std::size_t started = 0;
        for (uint32_t index : plan_scratch_) {
            if (started >= config_.max_requests_per_pump) {
                break;
            }
            Entry& e = items_[index];
            AsyncModelLoader::Request req;
            req.name = e.item.name;
            req.source = e.item.source;
            // An evicted item goes back into ITS OWN slot: register_model would
            // refuse the name, and any scheme that answered that by allocating
            // a new handle would break every draw command still carrying the
            // old one.
            req.reupload_into = e.state == State::Evicted ? e.handle : INVALID_MODEL_HANDLE;
            loader_.request(std::move(req));
            e.state = State::Loading;
            ++stats_.loads_started;
            ++started;
        }
    }

    // ---- 4. Evict, while over budget -------------------------------------
    candidate_scratch_.clear();
    for (uint32_t i = 0; i < items_.size(); ++i) {
        const Entry& e = items_[i];
        if (e.state != State::Resident) {
            continue;
        }
        const float distance = glm::length(e.item.anchor - viewer);
        if (!outside_evict_band(distance, e.item.cull_radius, config_)) {
            continue;
        }
        candidate_scratch_.push_back({i, distance, e.frames_resident, e.bytes});
    }
    uint32_t blocked = 0;
    plan_evictions(candidate_scratch_, resident_bytes_, config_, plan_scratch_, &blocked);
    stats_.evictions_blocked_by_hysteresis += blocked;
    for (uint32_t index : plan_scratch_) {
        Entry& e = items_[index];
        const std::size_t released = models_.unload_model(e.handle);
        (void)released; // the manager's own estimate is what the budget tracks
        resident_bytes_ = e.bytes >= resident_bytes_ ? 0 : resident_bytes_ - e.bytes;
        e.bytes = 0;
        e.frames_resident = 0;
        e.state = State::Evicted;
        ++stats_.evictions;
        geometry_changed = true;
    }
    if (config_.budget_bytes != 0 && resident_bytes_ > config_.budget_bytes && plan_scratch_.empty()) {
        // Over budget with nothing evictable — every resident item is either
        // inside the band or too young. The budget is not achievable from here
        // and saying so is the only honest thing to do.
        ++stats_.budget_overruns;
    }

    if (geometry_changed) {
        invalidate_geometry();
    }

    // ---- 5. Republish the frame-level counters ---------------------------
    stats_.resident = stats_.loading = stats_.absent = stats_.evicted = stats_.failed = 0;
    for (const Entry& e : items_) {
        switch (e.state) {
            case State::Resident:
                ++stats_.resident;
                break;
            case State::Loading:
                ++stats_.loading;
                break;
            case State::Absent:
                ++stats_.absent;
                break;
            case State::Evicted:
                ++stats_.evicted;
                break;
            case State::Failed:
                ++stats_.failed;
                break;
        }
    }
    stats_.resident_bytes = resident_bytes_;
    stats_.budget_bytes = config_.budget_bytes;
}

} // namespace bracken
