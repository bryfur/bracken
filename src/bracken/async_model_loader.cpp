#include "bracken/async_model_loader.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

#include <mutex>

namespace bracken {
namespace {

/** One model's journey from bytes to a registered handle. */
struct Job {
    std::string name;
    std::string source;
    std::unique_ptr<Model> model; // filled by the worker
    bool decoded = false;
    bool failed = false;
    /// Set with release by the decoding worker; read with acquire by pump().
    std::atomic<bool> ready{false};
};

} // namespace

struct AsyncModelLoader::Impl {
    ModelManager& models;
    const core::asset::AssetPack* pack = nullptr;

    // Jobs in REQUEST order, and a cursor for the next one to register.
    //
    // Decoding finishes out of order — that is the entire point of doing it on
    // a pool — but REGISTRATION must not, because register_model hands out
    // handles sequentially and those handles end up inside draw commands. If
    // the main thread registered whatever finished first, model handles would
    // differ between runs and between worker counts, and the frame's content
    // would change with them. The smoke run's packet digest caught exactly
    // that: three runs produced three different values until this became
    // ordered.
    //
    // So: workers fill slots in place and flag them done; pump() walks
    // forward from the cursor and stops at the first slot still decoding.
    std::mutex slots_mutex;
    std::vector<std::unique_ptr<Job>> slots;
    std::size_t next_to_register = 0;

    std::atomic<std::size_t> outstanding{0};
    std::atomic<std::size_t> failures{0};
    std::vector<core::jobs::TaskHandle> handles;

    Impl(ModelManager& m, const core::asset::AssetPack* p) : models(m), pack(p) {}
};

AsyncModelLoader::AsyncModelLoader(ModelManager& models, const core::asset::AssetPack* pack)
    : impl_(std::make_unique<Impl>(models, pack)) {
    BRACKEN_ASSERT_MAIN_THREAD();
}

AsyncModelLoader::~AsyncModelLoader() {
    // Outstanding decode jobs capture `impl_`, so they must finish before it
    // dies. Dropping the loader mid-load would otherwise be a use-after-free
    // that only shows up under load.
    for (core::jobs::TaskHandle& h : impl_->handles) {
        core::jobs::JobSystem::instance().wait(h);
    }
}

void AsyncModelLoader::request(Request req) {
    BRACKEN_ASSERT_MAIN_THREAD();

    Impl* impl = impl_.get();

    // Reserve this request's slot now, on the main thread, so slot index ==
    // request order regardless of when the decode finishes.
    std::size_t index = 0;
    {
        std::scoped_lock lock(impl->slots_mutex);
        index = impl->slots.size();
        auto job = std::make_unique<Job>();
        job->name = std::move(req.name);
        job->source = std::move(req.source);
        impl->slots.push_back(std::move(job));
    }

    impl->outstanding.fetch_add(1, std::memory_order_relaxed);
    impl->handles.push_back(core::jobs::JobSystem::instance().submit([impl, index] {
        BRACKEN_PROFILE_ZONE("AsyncModelLoader::decode");
        // The slot vector only grows and only from the main thread (which is
        // not requesting while pumping), so the element address is stable for
        // the job's lifetime; the lock is taken just for the push above.
        Job* owned = nullptr;
        {
            std::scoped_lock lock(impl->slots_mutex);
            owned = impl->slots[index].get();
        }
        owned->model = std::make_unique<Model>();

        if (impl->pack != nullptr) {
            auto bytes = impl->pack->read(owned->source);
            if (!bytes) {
                BRACKEN_LOG_ERROR("asset", "async load '{}': {}", owned->source,
                                  core::asset::to_string(bytes.error()));
                owned->failed = true;
            } else {
                owned->decoded = ModelLoader::load_glb_from_memory(
                    std::move(bytes).value(), owned->source, *owned->model);
            }
        } else {
            owned->decoded = ModelLoader::load_glb(owned->source, *owned->model);
        }

        if (!owned->decoded) {
            owned->failed = true;
        }

        // Release: publishes the decoded Model to the main thread's acquire
        // on this same flag in pump().
        owned->ready.store(true, std::memory_order_release);
        impl->outstanding.fetch_sub(1, std::memory_order_release);
    }));
}

void AsyncModelLoader::request_all(std::span<const Request> requests) {
    for (const Request& r : requests) {
        request(r);
    }
}

std::vector<AsyncModelLoader::Completed> AsyncModelLoader::pump(std::size_t max_uploads) {
    // GPU resource creation is main-thread tier; this is the whole reason the
    // decode/upload split exists.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();

    std::vector<Completed> out;
    std::size_t uploaded = 0;

    for (;;) {
        if (max_uploads != 0 && uploaded >= max_uploads) {
            break;
        }

        Job* job = nullptr;
        {
            std::scoped_lock lock(impl_->slots_mutex);
            if (impl_->next_to_register >= impl_->slots.size()) {
                break;
            }
            Job* candidate = impl_->slots[impl_->next_to_register].get();
            // Stop at the first slot still decoding: a later slot may well be
            // finished, but registering it now would assign handles out of
            // request order.
            if (!candidate->ready.load(std::memory_order_acquire)) {
                break;
            }
            ++impl_->next_to_register;
            job = candidate;
        }

        Completed done;
        done.name = job->name;
        if (job->failed || job->model == nullptr) {
            impl_->failures.fetch_add(1, std::memory_order_relaxed);
            BRACKEN_LOG_ERROR("asset", "async load failed: '{}' (source '{}')", job->name,
                              job->source);
            out.push_back(done);
            continue;
        }

        // register_model derives bounds and uploads through the manager, which
        // is where the main-thread GPU work happens.
        auto result = impl_->models.register_model(job->name, std::move(job->model));
        if (result) {
            done.handle = result.value();
            done.ok = true;
        } else {
            impl_->failures.fetch_add(1, std::memory_order_relaxed);
            BRACKEN_LOG_ERROR("asset", "async load '{}': register_model refused it: {}", job->name,
                              to_string(result.error()));
        }
        out.push_back(done);
        ++uploaded;
    }

    return out;
}

std::vector<AsyncModelLoader::Completed> AsyncModelLoader::drain() {
    BRACKEN_ASSERT_MAIN_THREAD();

    std::vector<Completed> all;
    // Wait on the submitted jobs rather than spinning on the counter: the
    // job system's wait() lets the main thread execute pool work while it
    // blocks, so a loading screen contributes instead of idling.
    for (core::jobs::TaskHandle& h : impl_->handles) {
        core::jobs::JobSystem::instance().wait(h);
    }
    impl_->handles.clear();

    auto batch = pump(0);
    all.insert(all.end(), batch.begin(), batch.end());
    return all;
}

std::size_t AsyncModelLoader::in_flight() const noexcept {
    // Decoding, plus anything decoded but not yet registered.
    const std::size_t decoding = impl_->outstanding.load(std::memory_order_acquire);
    std::scoped_lock lock(impl_->slots_mutex);
    return decoding + (impl_->slots.size() - impl_->next_to_register - decoding);
}

std::size_t AsyncModelLoader::failed_count() const noexcept {
    return impl_->failures.load(std::memory_order_relaxed);
}

} // namespace bracken
