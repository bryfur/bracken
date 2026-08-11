#pragma once

#include "bracken/core/asset/asset_pack.hpp"
#include "bracken/model_loader.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace bracken {

/**
 * Loads models without blocking the main thread, by splitting the work where
 * the engine's thread tiers already split it.
 *
 * THE SPLIT. Decoding a .glb — parsing JSON, de-interleaving accessors,
 * decompressing textures — is pure CPU work over per-call state, so it runs on
 * the job-system pool, N models at once. Creating GPU buffers and textures is
 * main-thread tier, so it happens in `pump()`, which the host calls at the
 * frame fence. That is the same shape as hot-reload: do the expensive part
 * anywhere, apply it at the one point in the frame where it is safe.
 *
 * WHY IT MATTERS. The playground spent 2.14 s of a 3.36 s startup decoding 52
 * models one after another on the main thread, with an idle worker pool
 * sitting next to it. Decoding is embarrassingly parallel and nothing about it
 * needed the main thread.
 *
 * `pump()` takes a budget so a game can stream assets during play without
 * spiking a frame: it uploads until the budget is spent and returns, leaving
 * the rest for the next call. Uploads are not preemptible once begun, so the
 * budget is a floor to stop at, not a hard cap.
 *
 * WHERE THIS LIVES. Beside model_loader.hpp rather than in core/asset/, even
 * though AssetPack itself is core infrastructure: this class knows Model and
 * ModelManager, which are root-level concepts, and core/ is a lower layer that
 * may not include upward. module_layering.test.cpp caught the first attempt at
 * putting it in core/asset/.
 *
 * Tier: construct, request() and pump() on the main thread. The decode jobs it
 * submits are worker-safe by construction.
 */
class AsyncModelLoader {
public:
    struct Request {
        std::string name;        ///< registration name
        std::string source;      ///< pack entry name, or a filesystem path
        /**
         * Slot to REUPLOAD into instead of registering a new model.
         *
         * INVALID_MODEL_HANDLE (the default) is the original behaviour: the
         * finished decode goes through `ModelManager::register_model`, which
         * assigns the next handle in request order.
         *
         * A valid handle instead routes the decode to
         * `ModelManager::reupload_model(handle, ...)`. That exists because
         * `register_model` REFUSES a name it already knows
         * (`ModelRegisterError::NameAlreadyRegistered`) — so streaming a model
         * back in after an eviction has no route through the registering path,
         * and inventing one would mean either orphaning the old handle or
         * renumbering, both of which break handle stability.
         */
        ModelHandle reupload_into = INVALID_MODEL_HANDLE;
    };

    struct Completed {
        std::string name;
        ModelHandle handle = INVALID_MODEL_HANDLE;
        bool ok = false;
    };

    /** @param pack optional; when null, sources are filesystem paths. */
    AsyncModelLoader(ModelManager& models, const core::asset::AssetPack* pack);
    ~AsyncModelLoader();

    AsyncModelLoader(const AsyncModelLoader&) = delete;
    AsyncModelLoader& operator=(const AsyncModelLoader&) = delete;

    /** Queue a model for decoding. Returns immediately. */
    void request(Request request);
    void request_all(std::span<const Request> requests);

    /**
     * Upload whatever finished decoding, on the main thread.
     *
     * @param max_uploads Stop after this many models. 0 = no limit (what you
     *        want during a loading screen); a small number keeps a streaming
     *        frame smooth.
     * @return the models registered by this call.
     */
    [[nodiscard]] std::vector<Completed> pump(std::size_t max_uploads = 0);

    /** Block until every outstanding decode has finished and been uploaded.
     *  For a loading screen that has nothing else to do. Main-thread: it helps
     *  the pool while it waits rather than sleeping. */
    std::vector<Completed> drain();

    [[nodiscard]] std::size_t in_flight() const noexcept;
    [[nodiscard]] bool idle() const noexcept { return in_flight() == 0; }

    /** Models whose decode failed, for a loud report rather than a silent gap
     *  in the world. */
    [[nodiscard]] std::size_t failed_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bracken
