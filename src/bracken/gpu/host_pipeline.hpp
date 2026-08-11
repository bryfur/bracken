#pragma once

#include "bracken/core/handle.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include "bracken/gpu/pipeline_catalog.hpp"

#include <SDL3/SDL_gpu.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file host_pipeline.hpp
 * @brief Pipelines a GAME declares at runtime, next to the catalogue the
 *        engine declares at compile time.
 *
 * The catalogue (`gpu/pipeline_catalog.hpp`) is deliberately CLOSED: one
 * `constexpr` row per `PipelineType`, positionally pinned against a checked-in
 * golden, so a wrong depth compare or a reordered row is a failing test rather
 * than a subtly wrong image. That closure is worth keeping — and it is exactly
 * what makes the catalogue useless to a host, which cannot add an enumerator.
 *
 * So the host path is a SECOND namespace, not a widening of the first:
 *
 * - a host pipeline is named by an opaque generation-checked handle, never by
 *   a `PipelineType`, and the two are not convertible in either direction;
 * - `pipeline_desc()`, `compute_pipeline_desc()`, `pipeline_type_to_string()`
 *   and `pipeline_targets_swapchain()` keep their enum-only meaning, so the
 *   golden walk in `gpu/pipeline_catalog.test.cpp` — which asserts EXACTLY
 *   `PipelineType::Count + ComputePipelineType::Count` blocks — cannot see a
 *   host registration however many a game makes;
 * - what the two paths DO share is the lifecycle, and they share it through
 *   one cache key (`PipelineCacheKey`) so that "registered pipelines get
 *   invalidated / rebuilt / hot-reloaded too" is a property of the data
 *   structure rather than a second code path somebody has to remember to
 *   update.
 */

namespace bracken::gpu {

/// Tag types for the two host handle families. Distinct types on purpose: a
/// graphics handle passed to `get_compute_pipeline` is a compile error, not a
/// silent index into the wrong table.
struct HostGraphicsPipelineTag {};
struct HostComputePipelineTag {};

/// Opaque, generation-checked, default-constructed == invalid
/// (`core/handle.hpp`). NOT convertible to or from `PipelineType`.
using HostPipelineHandle = core::Handle<HostGraphicsPipelineTag>;
using HostComputePipelineHandle = core::Handle<HostComputePipelineTag>;

/**
 * @brief The pipeline cache's key: a catalogue enumerator OR a host slot.
 *
 * The registry's caches used to be keyed on the enums directly. Widening the
 * key rather than adding a third and fourth map is what makes every lifecycle
 * path — `invalidate_all`, `shutdown`, the swapchain-format rebuild,
 * hot-reload — cover host registrations for free instead of by remembering to
 * add a line. `origin` keeps the two ordinal spaces disjoint, so host slot 3
 * and `PipelineType(3)` are different keys.
 *
 * Trivially copyable and small; hashed by the specialization at the bottom of
 * this header.
 */
struct PipelineCacheKey {
    enum class Origin : uint8_t {
        Catalog, ///< `index` is a `PipelineType` / `ComputePipelineType` ordinal
        Host,    ///< `index` is a slot in the registry's host table
    };

    Origin origin = Origin::Catalog;
    uint32_t index = 0;

    [[nodiscard]] static constexpr PipelineCacheKey catalog(PipelineType type) {
        return {Origin::Catalog, static_cast<uint32_t>(type)};
    }
    [[nodiscard]] static constexpr PipelineCacheKey catalog(ComputePipelineType type) {
        return {Origin::Catalog, static_cast<uint32_t>(type)};
    }
    [[nodiscard]] static constexpr PipelineCacheKey host(uint32_t slot) { return {Origin::Host, slot}; }

    friend bool operator==(const PipelineCacheKey&, const PipelineCacheKey&) = default;
};

/**
 * @brief What a host declares to get a graphics pipeline.
 *
 * An owning-string mirror of `PipelineDesc` (`gpu/pipeline_catalog.hpp`) plus
 * the four things a closed catalogue could afford to leave implicit: explicit
 * target formats, an explicit vertex layout escape hatch, per-stage entry
 * points, and whether the resource counts are declared or reflected.
 *
 * EVERY fixed-function default here is byte-identical to `PipelineDesc`'s, and
 * that is load-bearing rather than tidy: the engine renders reversed-Z
 * (near → depth 1, far → depth 0, clear 0.0, compare `GREATER_OR_EQUAL`), so a
 * host that copies the defaults inherits the convention instead of discovering
 * it. `gpu/host_pipeline.test.cpp` pins the equality field by field.
 */
struct HostGraphicsPipelineDesc {
    /// Debug name. Non-empty and unique among LIVE host registrations
    /// (duplicates are refused) — it is what shows up in the registry's logs
    /// and it is a stable `const char*` for the lifetime of the registration,
    /// so it is safe to hand to `BRACKEN_GPU_SCOPE` / `BRACKEN_GPU_TIMER`,
    /// which store the label as a raw pointer.
    std::string name;

    /// SPIR-V FILE NAMES, not paths — resolved against the registry's shader
    /// search roots in registration order (`PipelineRegistry::add_shader_search_path`).
    /// Keeping them relative is what lets hot-reload watch the roots.
    std::string vertex_shader;   ///< e.g. "my_tint.vert.spv"
    std::string fragment_shader; ///< e.g. "my_tint.frag.spv"
    std::string vertex_entry = "VSMain";
    std::string fragment_entry = "PSMain";

    /// When true (the default) the resource counts are taken from SPIR-V
    /// reflection and `vs`/`fs` are ignored. This is what a host normally
    /// wants: a declared/reflected mismatch is an ERROR per load
    /// (`gpu/gpu_shader.hpp`), and hand-counting samplers to silence it is
    /// exactly the maintenance the engine's own catalogue pays for and a game
    /// should not have to.
    bool reflect_resources = true;
    ShaderResources vs{}; ///< Declared vertex-stage counts; used only when `reflect_resources` is false
    ShaderResources fs{}; ///< Declared fragment-stage counts; same

    // ---- Fixed-function state. Defaults == PipelineDesc's defaults. ----
    VertexLayout vertex_layout = VertexLayout::None;
    SDL_GPUPrimitiveType primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    SDL_GPUCullMode cull_mode = SDL_GPU_CULLMODE_BACK;
    BlendMode blend_mode = BlendMode::None;
    bool depth_test = true;
    bool depth_write = true;
    SDL_GPUCompareOp depth_compare = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
    bool depth_clip = true;
    DepthBias depth_bias{};

    TargetFormat color = TargetFormat::SceneColor;
    TargetFormat color2 = TargetFormat::DepthOnly;
    /// false ⇒ no depth attachment at all; test and write are forced off
    /// whatever the two fields above say, exactly as for a catalogue row.
    bool depth_target = true;

    /// Format escape hatches. `TargetFormat` is a closed enum whose resolver
    /// switch is deliberately exhaustive (adding an enumerator is a -Werror
    /// compile error until every case exists), so a host that needs a format
    /// the engine has never named overrides the resolved value here instead of
    /// growing the enum. INVALID (the default) means "use `color` / `color2` /
    /// the engine depth format".
    SDL_GPUTextureFormat color_format_override = SDL_GPU_TEXTUREFORMAT_INVALID;
    SDL_GPUTextureFormat color2_format_override = SDL_GPU_TEXTUREFORMAT_INVALID;
    SDL_GPUTextureFormat depth_format_override = SDL_GPU_TEXTUREFORMAT_INVALID;

    /// Vertex input escape hatch, for the same reason as the format one:
    /// `VertexLayout`'s switch is exhaustive by design. When `buffers` is
    /// non-empty it REPLACES whatever `vertex_layout` produced (`attributes`
    /// is taken with it, empty or not).
    std::vector<SDL_GPUVertexBufferDescription> buffers;
    std::vector<SDL_GPUVertexAttribute> attributes;
};

/**
 * @brief What a host declares to get a compute pipeline.
 *
 * Compute creation REQUIRES SPIR-V reflection to succeed — the threadgroup
 * dimensions come from the module's `[numthreads]` and SDL has no way to take
 * them from the caller (`gpu/gpu_compute.hpp`). So `resources` is only ever a
 * cross-check, and `reflect_resources` defaults to true, meaning "don't
 * bother".
 */
struct HostComputePipelineDesc {
    std::string name;
    std::string shader; ///< SPIR-V file name, e.g. "my_blur.comp.spv"
    std::string entry_point = "CSMain";
    bool reflect_resources = true;
    ComputeResources resources{};
};

/**
 * @brief Turn a host desc + the live formats into a `PipelineConfig`.
 *
 * Pure — no device, no shader loading, no global state — the same contract
 * `build_pipeline_config` carries, and for the same reason: it is what makes
 * the host path testable with no GPU. The returned config has null
 * `vertex_shader`/`fragment_shader`; the registry resolves those.
 *
 * Implemented by projecting the host desc onto a `PipelineDesc` and calling
 * `build_pipeline_config`, so the two paths cannot drift in their handling of
 * reversed-Z, the depth-target force-off, or a target format's resolution.
 * Only the four explicit overrides are applied afterwards.
 */
[[nodiscard]] PipelineConfig build_host_pipeline_config(const HostGraphicsPipelineDesc& desc,
                                                        const PipelineFormats& formats);

/**
 * @brief Slot table behind the host handles.
 *
 * Deliberately not `core::HandlePool`: this table has to be WALKED in
 * registration order — preload, the swapchain rebuild and hot-reload all
 * iterate it — and the pool exposes no iteration. Walking in slot order also
 * makes rebuild order deterministic, which is the same rule the Prepare
 * fan-out's chunk-index merge follows.
 *
 * Entries are heap-allocated (`unique_ptr`), never inline in the vector,
 * because the desc's `name` is handed out as a raw `const char*` to the GPU
 * debug-label API, which stores the pointer and expects it to outlive the
 * frame. A `std::vector<Entry>` would dangle that on the first reallocation —
 * intermittently, in a host's game, never in the suite.
 *
 * @tparam Tag  handle tag type
 * @tparam Desc `HostGraphicsPipelineDesc` or `HostComputePipelineDesc`
 */
template<typename Tag, typename Desc> class HostPipelineTable {
public:
    using HandleT = core::Handle<Tag>;

    struct Entry {
        Desc desc;
        uint32_t generation = 1;
        bool live = false;
    };

    /// Register `desc`, reusing the lowest free slot. Never invalidates a live
    /// handle.
    [[nodiscard]] HandleT add(Desc desc) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(slots_.size()); ++i) {
            if (!slots_[i]->live) {
                slots_[i]->desc = std::move(desc);
                slots_[i]->live = true;
                ++live_;
                return HandleT{i, slots_[i]->generation};
            }
        }
        auto entry = std::make_unique<Entry>();
        entry->desc = std::move(desc);
        entry->live = true;
        slots_.push_back(std::move(entry));
        ++live_;
        const auto index = static_cast<uint32_t>(slots_.size() - 1);
        return HandleT{index, slots_[index]->generation};
    }

    /// Release the slot and bump its generation, so a stale copy of the handle
    /// validates false instead of aliasing the next registration.
    /// @return false if the handle was already invalid/stale (a no-op).
    bool remove(HandleT h) {
        Entry* e = find(h);
        if (e == nullptr) {
            return false;
        }
        e->desc = Desc{};
        e->live = false;
        e->generation += 1;
        --live_;
        return true;
    }

    [[nodiscard]] Entry* find(HandleT h) {
        return const_cast<Entry*>(static_cast<const HostPipelineTable*>(this)->find(h));
    }

    [[nodiscard]] const Entry* find(HandleT h) const {
        if (!h.is_valid() || h.index >= slots_.size()) {
            return nullptr;
        }
        const Entry& e = *slots_[h.index];
        if (!e.live || e.generation != h.generation) {
            return nullptr;
        }
        return &e;
    }

    /// True when a LIVE entry already carries this name. Duplicate names are
    /// refused at registration so a log line names one pipeline.
    [[nodiscard]] bool has_live_name(std::string_view name) const {
        for (const auto& slot : slots_) {
            if (slot->live && slot->desc.name == name) {
                return true;
            }
        }
        return false;
    }

    /// Every slot ever allocated, in slot order. Callers must skip
    /// `!live` entries; the index IS the `PipelineCacheKey::host` ordinal.
    [[nodiscard]] std::span<const std::unique_ptr<Entry>> slots() const noexcept { return slots_; }

    [[nodiscard]] size_t live_count() const noexcept { return live_; }

    /// Release every live slot, bumping generations. Slot storage is kept, so
    /// the host ordinals a cache may still hold stay meaningful.
    void clear() {
        for (auto& slot : slots_) {
            if (slot->live) {
                slot->desc = Desc{};
                slot->live = false;
                slot->generation += 1;
            }
        }
        live_ = 0;
    }

private:
    std::vector<std::unique_ptr<Entry>> slots_;
    size_t live_ = 0;
};

} // namespace bracken::gpu

namespace std {
template<> struct hash<::bracken::gpu::PipelineCacheKey> {
    size_t operator()(const ::bracken::gpu::PipelineCacheKey& k) const noexcept {
        // Combine in a fixed 64-bit type: shifting size_t by 32 is UB on
        // 32-bit targets (same reasoning as core::Handle's hash).
        return static_cast<size_t>((static_cast<uint64_t>(static_cast<uint8_t>(k.origin)) << 32) | k.index);
    }
};
} // namespace std
