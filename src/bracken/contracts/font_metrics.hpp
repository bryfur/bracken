#pragma once

#include "bracken/core/utf8.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace bracken {

/**
 * FontMetrics — the CPU-side mirror of `render::TextRenderer`'s glyph
 * metrics: everything text LAYOUT needs, and nothing that needs SDL_ttf,
 * a GPU or the main thread.
 *
 * WHY IT EXISTS. Until this landed, the only text-extent query in the engine
 * was `TextRenderer::get_text_width`, which needs the live `TTF_Font*` and is
 * therefore `main-thread` tier. The UI is described from `on_simulate`, which
 * is `sim-stage` tier and may touch neither. The visible consequence was
 * `FrameBuilder::UiLane::add_text_wrapped` wrapping proportional text on a
 * hardcoded "8 px per character" guess. Every layout and wrap decision in the
 * engine now goes through this table instead.
 *
 * HOW IT IS PUBLISHED, exactly. One instance is filled ONCE, on the main
 * thread, inside `Application::init_renderer()` — after the renderer (and
 * therefore the font) is up and BEFORE `Application::run()` has ever kicked a
 * sim job. It is never mutated afterwards; `Application` owns it for the
 * process's life and hands out `const FontMetrics&`. The happens-before edge
 * that makes a pool worker's read safe is the sim job's own submit/execute
 * edge (`JobSystem::submit` publishes, the running worker acquires), the same
 * edge that publishes the input snapshot. There is no atomic here and none is
 * needed: after init this is an immutable value.
 *
 * Thread-safety tier after the freeze: `worker-safe`.
 *
 * WHAT IT DELIBERATELY IS NOT. It is a snapshot of ONE font at ONE pixel
 * size, over a bounded codepoint range. A string containing a codepoint
 * outside `[k_first_cp, k_last_cp]` and outside `sparse_advance` measures with
 * `fallback_advance` but DRAWS at its real advance, so it will overrun its
 * box. `has_advance()` is the hook a caller uses to count that (the UI module
 * folds it into `ui::Context::Diagnostics::text_fallback_advances`) — this
 * struct itself stays a pure value with no mutable counters, because a
 * measure that is not a pure function of its inputs is not reproducible at a
 * different worker count.
 */
struct FontMetrics {
    /// Dense range: U+0020 (space) through U+024F (end of Latin Extended-B).
    static constexpr uint32_t k_first_cp = 0x20;
    static constexpr uint32_t k_last_cp = 0x24F;
    static constexpr std::size_t k_dense_count = k_last_cp - k_first_cp + 1; // 560

    /// Sentinel stored in `advance` for a codepoint the font does not have.
    static constexpr int16_t k_absent = INT16_MIN;

    int32_t line_height = 0;      ///< TTF_GetFontHeight — the glyph CELL height.
    int32_t line_skip = 0;        ///< TTF_GetFontLineSkip — baseline to baseline.
    int32_t ascent = 0;
    int32_t descent = 0;
    int32_t fallback_advance = 0; ///< Advance charged to a codepoint outside the table.
    int32_t pixel_size = 0;       ///< The size the atlas was rasterized at (18 today).

    /// Dense advances for [k_first_cp, k_last_cp]; k_absent where the font
    /// has no glyph. 1120 bytes — small enough to copy, big enough to cover
    /// every Latin script the engine's one font actually renders.
    std::array<int16_t, k_dense_count> advance{};

    /// Codepoints outside the dense range, SORTED by codepoint (binary
    /// search). Sorted, not hashed: the whole module must be deterministic and
    /// an unordered_map's iteration order is not.
    std::vector<std::pair<uint32_t, int16_t>> sparse_advance;

    /// NONZERO kerning pairs only, key = (prev << 32) | cur, SORTED. Most
    /// pairs kern by zero; storing them would multiply the table by ~300.
    std::vector<std::pair<uint64_t, int16_t>> kerning;

    /// True once a font has been exported into this table. Every measure
    /// returns 0 while it is false, which is what a headless test (and a
    /// machine with no font installed) sees.
    [[nodiscard]] bool valid() const noexcept { return line_height > 0; }

    /// True when `cp` has a real entry — i.e. `advance_of` will not fall back.
    [[nodiscard]] bool has_advance(uint32_t cp) const noexcept;

    /// Pen advance of one codepoint in unscaled font pixels.
    [[nodiscard]] int32_t advance_of(uint32_t cp) const noexcept;

    /// Kerning applied BEFORE `cur` when it follows `prev`; 0 for an absent
    /// pair (which is the overwhelmingly common case).
    [[nodiscard]] int32_t kerning_of(uint32_t prev, uint32_t cur) const noexcept;

    /**
     * Width of `utf8` at `scale`, in pixels.
     *
     * This reproduces EXACTLY the pen walk `TextRenderer::build_string_quads`
     * performs — kerning added before the glyph, then the glyph's advance —
     * so a measured width and a drawn string agree to the pixel. It is
     * deliberately not `TTF_GetStringSize`, which is what
     * `TextRenderer::get_text_width` used to use and which is a second,
     * independently drifting answer to the same question. (`get_text_width` is
     * implemented on top of this now, so the tree has ONE answer.)
     */
    [[nodiscard]] float measure_width(std::string_view utf8, float scale = 1.0f) const;

    /// Height of one line at `scale` — the glyph cell, matching the quads the
    /// text renderer emits (not `line_skip`).
    [[nodiscard]] float measure_height(float scale = 1.0f) const {
        return static_cast<float>(line_height) * scale;
    }

    /// How many codepoints of `utf8` would be measured with `fallback_advance`
    /// (i.e. how wrong `measure_width` is allowed to be). 0 is the happy case.
    [[nodiscard]] uint32_t count_fallbacks(std::string_view utf8) const;

    /**
     * Longest prefix of `utf8` that fits `max_width` at `scale`, ending on a
     * UTF-8 sequence boundary. Returns the prefix's BYTE length and, when
     * `out_width` is non-null, the width it achieved.
     *
     * Monotone in `max_width`, and never splits a multi-byte sequence — those
     * are the two properties wrapping depends on and the two the test pins.
     */
    [[nodiscard]] std::size_t fit_prefix(std::string_view utf8, float max_width, float scale = 1.0f,
                                         float* out_width = nullptr) const;

    /**
     * Longest prefix of `utf8` that fits `max_width` and breaks at a SPACE
     * when one is available in the prefix, for word wrapping. Returns
     * `{consumed_bytes, emit_bytes}`: `emit_bytes` is what to draw,
     * `consumed_bytes` is what to skip (it swallows the break space).
     * `consumed_bytes` is always > 0 for a non-empty input, so a wrap loop
     * driven by it always terminates.
     */
    struct WrapPoint {
        std::size_t consumed = 0;
        std::size_t emit = 0;
    };
    [[nodiscard]] WrapPoint wrap_point(std::string_view utf8, float max_width, float scale = 1.0f) const;
};

} // namespace bracken
