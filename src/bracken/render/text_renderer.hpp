#pragma once

#include "bracken/contracts/font_metrics.hpp"
#include "bracken/core/utf8.hpp"

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_surface.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <string_view>
#include <unordered_map>
#include <vector>

// Forward declarations
namespace bracken::gpu {
class GPUDevice;
class GPUBuffer;
class GPUTexture;
class PipelineRegistry;
} // namespace bracken::gpu

namespace bracken::render {

namespace gpu = bracken::gpu;

/**
 * The engine's one UTF-8 decoder. It moved to `core/utf8.hpp` when
 * `contracts/font_metrics.hpp` needed it — `contracts/` is a leaf layer and
 * may not include `render/` — and is re-exported here so every existing
 * `render::decode_utf8` call site still names the same function.
 */
using bracken::core::decode_utf8;

/**
 * Left-to-right shelf packer for a fixed-size texture atlas.
 *
 * Rects are placed on horizontal shelves; a new shelf opens below the current
 * one when a rect no longer fits on the row. This is a good fit for font
 * glyphs, which all share (roughly) the same height.
 */
class ShelfPacker {
public:
    ShelfPacker(int width, int height) : width_(width), height_(height) {}

    /**
     * Try to place a w x h rect. On success returns true and writes the
     * top-left corner to out_x/out_y. Returns false when the atlas is full
     * (or the rect can never fit).
     */
    bool pack(int w, int h, int& out_x, int& out_y) {
        if (w <= 0 || h <= 0 || w > width_ || h > height_) {
            return false;
        }
        if (cursor_x_ + w > width_) {
            // Open a new shelf below the current one
            shelf_y_ += shelf_height_;
            cursor_x_ = 0;
            shelf_height_ = 0;
        }
        if (shelf_y_ + h > height_) {
            return false;
        }
        out_x = cursor_x_;
        out_y = shelf_y_;
        cursor_x_ += w;
        shelf_height_ = std::max(shelf_height_, h);
        return true;
    }

    void reset() {
        cursor_x_ = 0;
        shelf_y_ = 0;
        shelf_height_ = 0;
    }

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_ = 0;
    int height_ = 0;
    int cursor_x_ = 0;
    int shelf_y_ = 0;
    int shelf_height_ = 0;
};

/**
 * TextRenderer handles text rendering using SDL_ttf and SDL3 GPU API.
 *
 * Glyphs are rasterized on demand via TTF_RenderGlyph_Blended into a single
 * fixed-size shelf-packed atlas texture (one atlas for the one font size this
 * renderer uses). Strings are drawn as runs of per-glyph quads with atlas UVs
 * through the shared batch vertex buffer, so steady-state text costs pure
 * vertex generation — no per-string textures, no per-frame rasterization, and
 * no one-frame latency for new strings. If the atlas ever fills up, all
 * glyphs are evicted and re-rasterized on demand (logged; may glitch text for
 * a single frame).
 */
class TextRenderer {
public:
    TextRenderer();
    ~TextRenderer();

    /**
     * Initialize text rendering resources.
     * @param device GPU device for creating resources
     * @param pipeline_registry Registry for accessing text pipeline
     */
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry);
    void shutdown();

    // Set projection matrix before drawing
    void set_projection(const glm::mat4& projection);

    /**
     * Queue a text draw for batched rendering (call during recording phase).
     * `color` is packed ARGB (0xAARRGGBB) — the engine-wide convention, see
     * bracken/core/color.hpp. Missing glyphs are rasterized into the
     * CPU-side atlas immediately and uploaded by the next copy-pass flush
     * (create_pending_textures / upload_queued_text), so new strings render
     * the same frame.
     */
    void queue_text_draw(std::string_view text, float x, float y, uint32_t color = 0xFFFFFFFF, float scale = 1.0f,
                         const SDL_Rect* clip = nullptr);

    /**
     * Upload all queued text vertex data (and any new glyphs) to the GPU.
     * Call BEFORE starting the render pass. This performs copy pass operations.
     */
    void upload_queued_text(SDL_GPUCommandBuffer* cmd);

    /**
     * Draw all queued text. Call DURING render pass, after upload_queued_text.
     * Binds the atlas once and issues one draw per color change.
     */
    void draw_queued_text(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* render_pass);

    /**
     * Runs queued since the last draw. `queue_text_draw` appends AT MOST one
     * run per call (none when the string produced no visible glyph), which is
     * what lets UIRenderer map a submission-ordered span of text draws onto a
     * span of runs.
     */
    [[nodiscard]] std::size_t queued_run_count() const { return queued_runs_.size(); }

    /**
     * Draw runs [first_run, first_run + run_count) only, and DO NOT clear the
     * queue — the caller is interleaving text with geometry and will come back
     * (see UIRenderer::execute). `clear_queued_text()` retires the queue when
     * the walk is done. `default_scissor` is restored after any clipped run so
     * the pass's scissor state is the caller's again on return.
     *
     * draw_queued_text() is exactly draw_run_range(0, all) + clear.
     */
    void draw_run_range(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* render_pass, std::size_t first_run,
                        std::size_t run_count, const SDL_Rect& default_scissor);

    /** Drop the queued runs/vertices without drawing them. */
    void clear_queued_text();

    /**
     * Fill `out` from the live font: advances, kerning pairs and the vertical
     * metrics, for [first_cp, last_cp] plus `extra`.
     *
     * Does NOT rasterize and does not touch the atlas or the packer — pure
     * metric queries (TTF_GetGlyphMetrics / TTF_GetGlyphKerning /
     * TTF_GetFontHeight). Returns false when no font is loaded, in which case
     * `out` is left in its invalid state and every measure on it returns 0.
     *
     * Tier: `main-thread`. Call it ONCE, at init; `out` is frozen afterwards
     * and read from sim jobs by const reference (see FontMetrics' publication
     * note).
     */
    [[nodiscard]] bool export_metrics(bracken::FontMetrics& out, uint32_t first_cp = bracken::FontMetrics::k_first_cp,
                                      uint32_t last_cp = bracken::FontMetrics::k_last_cp);

    /**
     * This renderer's own exported table, filled at init(). Invalid when no
     * font loaded. It is what the pen walk consumes, so it is also the table
     * a caller must measure against to agree with what gets drawn.
     */
    [[nodiscard]] const bracken::FontMetrics& metrics() const { return metrics_; }

    /// Rendered width in pixels of `text` at `scale` (0 when no font is loaded).
    int get_text_width(std::string_view text, float scale = 1.0f) const;
    /// Font line height in pixels at `scale` (0 when no font is loaded).
    int get_text_height(float scale = 1.0f) const;

    bool is_ready() const { return initialized_ && font_ != nullptr; }

    /**
     * Frame-boundary housekeeping. Call at the beginning of each frame.
     */
    void release_pending_resources();

    /**
     * Upload any newly rasterized glyphs to the atlas texture.
     * Call outside of render passes (performs a copy pass).
     */
    void create_pending_textures(SDL_GPUCommandBuffer* cmd);

private:
    // A rasterized glyph cell in the atlas. Glyph bitmaps from
    // TTF_RenderGlyph_Blended are full line-height cells whose origin is the
    // pen position at the top of the line box, so layout is: quad at
    // (x + pen * scale, y), then pen += advance (+ kerning).
    struct Glyph {
        uint16_t atlas_x = 0;
        uint16_t atlas_y = 0;
        uint16_t width = 0;      // cell width in pixels
        uint16_t height = 0;     // cell height in pixels (== font line height)
        int advance = 0;         // pen advance in pixels
        bool has_bitmap = false; // false: advance-only (e.g. space)
    };

    // A contiguous range of quads in batch_vertices_ sharing one color AND
    // one clip rectangle. The clip splits the same-color run merge for the
    // same reason a color change does: both are pass state that has to be
    // re-issued before the draw.
    struct TextRun {
        size_t first_vertex = 0;
        uint32_t vertex_count = 0;
        uint32_t color = 0;
        SDL_Rect clip{};
        bool clip_enabled = false;
    };

    /// Look up a glyph, rasterizing it into the CPU atlas on first use.
    /// Returns nullptr only when no font is loaded or rasterization failed.
    const Glyph* ensure_glyph(uint32_t codepoint);

    /// Copy an RGBA32 glyph surface into the CPU atlas pixels at (x, y).
    void blit_glyph_to_atlas(const SDL_Surface* rgba, int x, int y);

    /// Evict every glyph and reset the packer (atlas overflow handling).
    void evict_all_glyphs();

    /// Extend the dirty row band to cover rows [y0, y1).
    void mark_dirty_rows(int y0, int y1);

    /// Upload the dirty atlas row band to the GPU (copy pass). No-op if clean.
    void flush_atlas(SDL_GPUCommandBuffer* cmd);

    /// Append one textured quad per visible glyph of `text` to `out`
    /// (screen-space positions + atlas UVs), stopping before `max_floats` is
    /// exceeded. Rasterizes missing glyphs into the CPU atlas as a side
    /// effect. Returns the number of quads appended.
    size_t build_string_quads(std::string_view text, float x, float y, float scale, std::vector<float>& out,
                              size_t max_floats);

    TTF_Font* font_ = nullptr;
    int font_size_ = 18;
    bool initialized_ = false;

    // Frozen after init(). Consumed by build_string_quads (so measuring and
    // drawing cannot disagree) and copied out to the sim stage by
    // Application. Never mutated after init — see FontMetrics' publication
    // note in contracts/font_metrics.hpp.
    bracken::FontMetrics metrics_;

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipeline_registry_ = nullptr;
    glm::mat4 projection_;

    // Dynamic vertex buffer shared by all text quads
    std::unique_ptr<gpu::GPUBuffer> vertex_buffer_;

    // Sampler for the glyph atlas
    SDL_GPUSampler* sampler_ = nullptr;

    // Glyph atlas: one fixed-size texture for the single font size.
    // atlas_pixels_ is the CPU-side mirror; dirty rows are uploaded through a
    // persistent transfer buffer before the UI render pass.
    static constexpr int ATLAS_SIZE = 512;
    static constexpr int GLYPH_PADDING = 1; // gap between cells (linear filter bleed)
    SDL_GPUTexture* atlas_texture_ = nullptr;
    SDL_GPUTransferBuffer* atlas_transfer_ = nullptr;
    std::vector<uint8_t> atlas_pixels_; // RGBA8, ATLAS_SIZE x ATLAS_SIZE
    ShelfPacker packer_{ATLAS_SIZE, ATLAS_SIZE};
    std::unordered_map<uint32_t, Glyph> glyphs_;
    int dirty_y0_ = 0; // dirty row band [dirty_y0_, dirty_y1_); empty when y0 >= y1
    int dirty_y1_ = 0;

    uint64_t current_frame_ = 0;

    // Batched text rendering data
    std::vector<TextRun> queued_runs_;
    std::vector<float> batch_vertices_; // All vertex data for queued texts
    static constexpr size_t VERTICES_PER_QUAD = 6;
    static constexpr size_t FLOATS_PER_VERTEX = 4; // x, y, u, v
    /// Initial GPU vertex-buffer size, in glyph quads. NOT a cap since the
    /// scroll-view work: the buffer GROWS in upload_queued_text the way
    /// UIRenderer's does. It used to be a hard budget that dropped glyphs with
    /// a throttled warning, which a 150-row scrolling list hits immediately.
    static constexpr size_t INITIAL_TEXT_QUADS = 4096;
    /// Absolute ceiling on one frame's CPU-side text batch. A runaway host
    /// (a label built in a loop with no bound) would otherwise grow the vertex
    /// vector until the process dies; this fails visibly instead.
    static constexpr size_t MAX_TEXT_QUADS = 1u << 20;

    /// Grow the GPU vertex buffer to hold the recorded batch. Returns false
    /// when growth failed, in which case the batch is truncated to what fits.
    bool reserve_gpu_capacity();
};

} // namespace bracken::render
