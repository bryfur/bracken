#pragma once

#include "bracken/contracts/font_metrics.hpp"
#include "bracken/contracts/ui_texture.hpp"
#include "bracken/core/color.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_rect.h>
#include <string>
#include <string_view>
#include <vector>

// Forward declarations
namespace bracken::gpu {
class GPUDevice;
class GPUBuffer;
class GPUPipeline;
class PipelineRegistry;
} // namespace bracken::gpu

namespace bracken::render {

namespace gpu = bracken::gpu;

class TextRenderer;

/**
 * UIRenderer handles all 2D UI rendering using SDL3 GPU API:
 * - Rectangles (filled/outline)
 * - Circles
 * - Lines
 * - Textured quads (image widgets)
 * - Text
 * - Buttons
 *
 * Drawing is two-phase: primitives are recorded CPU-side between begin() and
 * end() (no GPU work), then execute() uploads the batch and renders it. The
 * vertex batch is unbounded — the GPU vertex buffer grows on demand inside
 * execute() when a frame outgrows it.
 */
class UIRenderer {
public:
    UIRenderer();
    ~UIRenderer();

    // Non-copyable
    UIRenderer(const UIRenderer&) = delete;
    UIRenderer& operator=(const UIRenderer&) = delete;
    UIRenderer(UIRenderer&&) = delete;
    UIRenderer& operator=(UIRenderer&&) = delete;

    /**
     * Initialize UI rendering resources.
     * @param device GPU device for creating resources
     * @param pipeline_registry Registry for accessing UI pipeline
     * @param width Screen width
     * @param height Screen height
     */
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, int width, int height);

    /**
     * Clean up resources.
     */
    void shutdown();

    /**
     * Update screen dimensions (call on resize).
     */
    void set_screen_size(int width, int height);

    /**
     * Begin UI recording phase.
     * @param cmd Command buffer for the frame (unused during recording; kept
     *            for signature stability)
     * Note: This only starts recording, not rendering. Call execute() to render.
     */
    void begin(SDL_GPUCommandBuffer* cmd);

    /**
     * End UI recording phase.
     * After this, call execute() to upload data and render.
     */
    void end();

    /**
     * Upload all recorded UI data (copy pass) and render (render pass).
     * Must be called AFTER end() and before frame submission.
     * @param cmd Command buffer for the frame
     * @param swapchain Swapchain texture to render to
     * @param clear_background If true, clear to black. If false, preserve existing content.
     */
    void execute(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain, bool clear_background = false);

    // Primitive drawing. Every `color`/`tint` is a core::Color — the packed
    // ARGB (0xAARRGGBB) word of bracken/core/color.hpp, implicitly
    // constructible from one so hex literals still read as colors.
    void draw_filled_rect(float x, float y, float w, float h, core::Color color);
    void draw_rect_outline(float x, float y, float w, float h, core::Color color, float line_width = 2.0f);
    void draw_circle(float x, float y, float radius, core::Color color, int segments = 24);
    void draw_circle_outline(float x, float y, float radius, core::Color color, float line_width = 2.0f,
                             int segments = 24);
    void draw_line(float x1, float y1, float x2, float y2, core::Color color, float line_width = 2.0f);

    /**
     * Draw a textured quad (image widget). The texture is not copied or
     * retained: it must stay valid until execute() runs. `tint` (packed ARGB,
     * same as every other color parameter) multiplies the texture sample —
     * 0xFFFFFFFF draws the texture unmodified. u0/v0..u1/v1 select the
     * sampled sub-rectangle (defaults cover the whole texture). A null
     * texture falls back to a solid `tint` quad.
     */
    void draw_textured_quad(SDL_GPUTexture* texture, float x, float y, float w, float h,
                            core::Color tint = core::Color::white(), float u0 = 0.0f, float v0 = 0.0f,
                            float u1 = 1.0f, float v1 = 1.0f);

    // Text drawing. `color` is decoded identically to the shape primitives.
    // The referenced characters are NOT copied: `text` must stay valid until
    // execute() runs (FramePacket arena storage satisfies this — the packet
    // is retired only after the frame is rendered).
    void draw_text(std::string_view text, float x, float y, core::Color color, float scale = 1.0f);

    // Composite UI elements. `label` must stay valid until execute() (see draw_text).
    void draw_button(float x, float y, float w, float h, const std::string& label, core::Color color, bool selected);

    /**
     * Register `texture` for use from the sim stage and return the handle that
     * names it (contracts/ui_texture.hpp). Tier: `main-thread`. The texture is
     * NOT owned or retained — the host keeps it alive for as long as any frame
     * may reference the id. Returns an invalid id for a null texture.
     */
    [[nodiscard]] UiTextureId register_texture(SDL_GPUTexture* texture);

    /** Release the slot. The id is not recycled, so a stale id resolves to
     *  null (diagnosable) rather than to somebody else's texture. */
    void unregister_texture(UiTextureId id);

    /** The texture behind `id`, or null when the id is invalid, unregistered
     *  or already released. */
    [[nodiscard]] SDL_GPUTexture* texture_for(UiTextureId id) const;

    /**
     * Clip every SUBSEQUENT recorded primitive and text draw to this
     * rectangle, in screen pixels, until clear_clip_rect(). Recording phase
     * only. A clip change opens a new draw range exactly the way a texture
     * change does, so it costs one extra draw call per change.
     */
    void set_clip_rect(float x, float y, float w, float h);
    void clear_clip_rect();

    // Text measurement against the real font metrics (for centering,
    // alignment and wrapping). Both return 0 before init() / when no font
    // could be loaded.
    int text_width(std::string_view text, float scale = 1.0f) const;
    int text_height(float scale = 1.0f) const;

    /** The frozen glyph-metrics table the text renderer exported at init —
     *  what Application copies out for the sim stage. Invalid before init()
     *  or when no font could be loaded. */
    [[nodiscard]] const bracken::FontMetrics& metrics() const;

    // Access text renderer for advanced usage
    TextRenderer* text_renderer() { return text_renderer_.get(); }
    const TextRenderer* text_renderer() const { return text_renderer_.get(); }

    int width() const { return width_; }
    int height() const { return height_; }

    // Must match Vertex2D layout: position (2), texcoord (2), color (4)
    struct UIVertex {
        float x, y;       // Position
        float u, v;       // Texcoord (unused for solid colors, set to 0)
        float r, g, b, a; // Color
    };

    // Recording introspection (stats/tests): vertices, texture-switch draw
    // ranges, and queued text draws recorded since begin().
    size_t batch_vertex_count() const { return vertex_batch_.size(); }
    size_t batch_draw_count() const { return draw_ranges_.size(); }
    size_t queued_text_count() const { return queued_text_draws_.size(); }
    const std::vector<UIVertex>& batch_vertices() const { return vertex_batch_; }

    /**
     * One contiguous span of recorded work of a single KIND, in submission
     * order. This is the z-order fix, exposed for tests.
     *
     * The UI pass used to run two hard phases — every primitive, then every
     * queued text — so a panel recorded AFTER a label could not cover it. No
     * amount of care at the call site fixed that; the ordering was structural.
     * Recording now appends to this list instead, and execute() walks it,
     * alternating the UI and Text pipeline binds. A HUD that draws its panel
     * and then its labels produces exactly two ops and therefore exactly the
     * command stream it always did — the extra binds are paid only by content
     * that actually interleaves.
     */
    struct Op {
        enum class Kind : uint8_t { Geometry, Text };
        Kind kind = Kind::Geometry;
        /// Index into draw_ranges_ (Geometry) or queued_text_draws_ (Text).
        size_t first = 0;
        size_t count = 0;
    };
    const std::vector<Op>& ops() const { return ops_; }

private:
    void draw_quad(float x, float y, float w, float h, const glm::vec4& color);

    int width_ = 0;
    int height_ = 0;

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipeline_registry_ = nullptr;

    std::unique_ptr<TextRenderer> text_renderer_;

    glm::mat4 projection_{};

    // SDL3 GPU resources
    std::unique_ptr<gpu::GPUBuffer> vertex_buffer_;
    SDL_GPUTexture* dummy_texture_ = nullptr; // 1x1 white bound for solid-color draw ranges
    SDL_GPUSampler* sampler_ = nullptr;       // linear/clamp; shared by solid + textured draws

    // Vertex batching. The CPU batch is unbounded; the GPU buffer starts at
    // INITIAL_VERTEX_CAPACITY vertices and is grown inside execute() when a
    // frame records more (growth is logged; on allocation failure the batch
    // is truncated to what fits rather than drawing stale buffer contents).
    static constexpr size_t INITIAL_VERTEX_CAPACITY = 16384;
    std::vector<UIVertex> vertex_batch_;

    // A contiguous run of vertices drawn with one texture binding and one
    // scissor, in submission order (painter's algorithm). texture == nullptr
    // means solid color (dummy white texture, has_texture = 0).
    struct DrawRange {
        size_t first_vertex = 0;
        uint32_t vertex_count = 0;
        SDL_GPUTexture* texture = nullptr;
        SDL_Rect clip{};
        bool clip_enabled = false;
    };
    std::vector<DrawRange> draw_ranges_;

    /// Range accepting vertices for `texture` under the current clip; extends
    /// the last range when texture AND clip both match, otherwise opens a new
    /// one at the batch end. Also opens a new Geometry op when the previous op
    /// was text.
    DrawRange& range_for(SDL_GPUTexture* texture);

    /// Drop recorded vertices/ranges beyond `max_vertices` (rounded down to
    /// whole triangles). Fallback when growing the GPU buffer fails.
    void truncate_batch(size_t max_vertices);

    /// Ensure the GPU vertex buffer can hold the recorded batch, growing it if
    /// needed; when growth fails the batch is truncated to what fits (logged).
    void reserve_gpu_capacity();

    // Queued text draws (executed during execute() phase). Views into
    // caller-owned strings that must stay valid until execute() — no copies.
    struct QueuedTextDraw {
        std::string_view text;
        float x{}, y{};
        core::Color color{};
        float scale{};
        SDL_Rect clip{};
        bool clip_enabled = false;
    };
    std::vector<QueuedTextDraw> queued_text_draws_;

    // Submission-ordered spans of geometry/text — the z-order fix. See Op.
    std::vector<Op> ops_;

    // Per-queued-text-draw run span in the TextRenderer's queue, filled in
    // execute()'s phase 1. A text draw contributes 0 runs when every glyph was
    // invisible, which is why this is a span and not an index.
    struct TextRunSpan {
        size_t first = 0;
        size_t count = 0;
    };
    std::vector<TextRunSpan> text_run_spans_;

    // Recording-phase clip state (see set_clip_rect). Not a stack: the caller
    // owns nesting and hands in the already-intersected rectangle, which is
    // the same contract scene::UiClipCommand carries.
    SDL_Rect clip_rect_{};
    bool clip_enabled_ = false;

    // Registered UI textures; index = UiTextureId::value - 1. Cleared slots
    // hold null and are never recycled — a stale id must resolve to nothing,
    // not to whatever was registered next.
    std::vector<SDL_GPUTexture*> textures_;

    /// Reset the recording buffers. One place, called from begin() and from
    /// every execute() exit — a missed clear leaks a frame's UI into the next.
    void clear_recording();
};

} // namespace bracken::render
