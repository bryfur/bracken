#include "bracken/render/text_renderer.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float4.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_gpu.h"
#include "SDL3/SDL_pixels.h"
#include "SDL3/SDL_stdinc.h"
#include "SDL3/SDL_surface.h"
#include "SDL3_ttf/SDL_ttf.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace bracken::render {

namespace gpu = bracken::gpu;

TextRenderer::TextRenderer() = default;

TextRenderer::~TextRenderer() {
    shutdown();
}

bool TextRenderer::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry) {
    if (initialized_) {
        return true;
    }

    device_ = &device;
    pipeline_registry_ = &pipeline_registry;

    if (!TTF_Init()) {
        BRACKEN_LOG_ERROR("text", "Failed to initialize SDL_ttf: {}", SDL_GetError());
        return false;
    }

    // Try to load a default font
    const char* font_paths[] = {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                                "/System/Library/Fonts/Helvetica.ttc",
                                R"(C:\Windows\Fonts\arial.ttf)"};

    for (const char* path : font_paths) {
        font_ = TTF_OpenFont(path, font_size_);
        if (font_) {
            BRACKEN_LOG_INFO("text", "Loaded font: {}", path);
            break;
        }
    }

    if (!font_) {
        BRACKEN_LOG_WARN("text", "Could not load any font, text rendering disabled");
        // Continue without font - not fatal
    }

    // Create dynamic vertex buffer for text quads (batched)
    // Each vertex: x, y, u, v (4 floats), 6 vertices per glyph quad
    vertex_buffer_ = gpu::GPUBuffer::create_dynamic(
        device, gpu::GPUBuffer::Type::Vertex, MAX_TEXT_QUADS * VERTICES_PER_QUAD * FLOATS_PER_VERTEX * sizeof(float));

    if (!vertex_buffer_) {
        BRACKEN_LOG_ERROR("text", "Failed to create text vertex buffer");
        return false;
    }

    // Create sampler for the glyph atlas
    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;

    sampler_ = SDL_CreateGPUSampler(device.handle(), &sampler_info);
    if (!sampler_) {
        BRACKEN_LOG_ERROR("text", "Failed to create text sampler: {}", SDL_GetError());
        return false;
    }

    // Create the glyph atlas texture and its persistent upload transfer buffer
    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_2D;
    tex_info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    tex_info.width = ATLAS_SIZE;
    tex_info.height = ATLAS_SIZE;
    tex_info.layer_count_or_depth = 1;
    tex_info.num_levels = 1;
    tex_info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;

    atlas_texture_ = SDL_CreateGPUTexture(device.handle(), &tex_info);
    if (!atlas_texture_) {
        BRACKEN_LOG_ERROR("text", "Failed to create glyph atlas texture: {}", SDL_GetError());
        return false;
    }

    SDL_GPUTransferBufferCreateInfo transfer_info = {};
    transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    transfer_info.size = static_cast<Uint32>(ATLAS_SIZE) * ATLAS_SIZE * 4;

    atlas_transfer_ = SDL_CreateGPUTransferBuffer(device.handle(), &transfer_info);
    if (!atlas_transfer_) {
        BRACKEN_LOG_ERROR("text", "Failed to create glyph atlas transfer buffer: {}", SDL_GetError());
        return false;
    }

    atlas_pixels_.assign(static_cast<size_t>(ATLAS_SIZE) * ATLAS_SIZE * 4, 0);
    // Mark the whole atlas dirty so the first flush uploads defined (zero)
    // contents everywhere, including padding between glyph cells.
    mark_dirty_rows(0, ATLAS_SIZE);

    initialized_ = true;
    return true;
}

void TextRenderer::shutdown() {
    if (device_) {
        if (atlas_texture_) {
            SDL_ReleaseGPUTexture(device_->handle(), atlas_texture_);
            atlas_texture_ = nullptr;
        }
        if (atlas_transfer_) {
            SDL_ReleaseGPUTransferBuffer(device_->handle(), atlas_transfer_);
            atlas_transfer_ = nullptr;
        }
        if (sampler_) {
            SDL_ReleaseGPUSampler(device_->handle(), sampler_);
            sampler_ = nullptr;
        }
    }

    glyphs_.clear();
    atlas_pixels_.clear();
    packer_.reset();
    queued_runs_.clear();
    batch_vertices_.clear();

    vertex_buffer_.reset();

    if (font_) {
        TTF_CloseFont(font_);
        font_ = nullptr;
    }
    if (initialized_) {
        TTF_Quit();
        initialized_ = false;
    }

    device_ = nullptr;
    pipeline_registry_ = nullptr;
}

void TextRenderer::set_projection(const glm::mat4& projection) {
    projection_ = projection;
}

void TextRenderer::release_pending_resources() {
    if (!device_) {
        return;
    }
    current_frame_++;
    batch_full_warned_ = false;
}

void TextRenderer::mark_dirty_rows(int y0, int y1) {
    y0 = std::max(y0, 0);
    y1 = std::min(y1, ATLAS_SIZE);
    if (y0 >= y1) {
        return;
    }
    if (dirty_y0_ >= dirty_y1_) {
        dirty_y0_ = y0;
        dirty_y1_ = y1;
    } else {
        dirty_y0_ = std::min(dirty_y0_, y0);
        dirty_y1_ = std::max(dirty_y1_, y1);
    }
}

void TextRenderer::blit_glyph_to_atlas(const SDL_Surface* rgba, int x, int y) {
    const auto* src = static_cast<const uint8_t*>(rgba->pixels);
    uint8_t* dst = atlas_pixels_.data() + (static_cast<size_t>(y) * ATLAS_SIZE + x) * 4;
    const size_t row_bytes = static_cast<size_t>(rgba->w) * 4;
    for (int row = 0; row < rgba->h; ++row) {
        std::memcpy(dst + static_cast<size_t>(row) * ATLAS_SIZE * 4, src + static_cast<size_t>(row) * rgba->pitch,
                    row_bytes);
    }
}

void TextRenderer::evict_all_glyphs() {
    BRACKEN_LOG_WARN("text", "Glyph atlas overflow ({} glyphs) - evicting all; text may glitch for one frame",
                    glyphs_.size());
    glyphs_.clear();
    packer_.reset();
    std::fill(atlas_pixels_.begin(), atlas_pixels_.end(), uint8_t{0});
    mark_dirty_rows(0, ATLAS_SIZE);
}

const TextRenderer::Glyph* TextRenderer::ensure_glyph(uint32_t codepoint) {
    auto it = glyphs_.find(codepoint);
    if (it != glyphs_.end()) {
        return &it->second;
    }
    if (!font_) {
        return nullptr;
    }

    Glyph glyph{};
    int minx = 0;
    int maxx = 0;
    int miny = 0;
    int maxy = 0;
    int advance = 0;
    if (TTF_GetGlyphMetrics(font_, codepoint, &minx, &maxx, &miny, &maxy, &advance)) {
        glyph.advance = advance;
    }

    // TTF_RenderGlyph_Blended returns a full line-height cell whose origin is
    // the pen position at the top of the line box, so a string is laid out by
    // placing cells at the pen and advancing by the glyph advance.
    const SDL_Color white = {255, 255, 255, 255};
    SDL_Surface* surface = TTF_RenderGlyph_Blended(font_, codepoint, white);
    if (surface) {
        SDL_Surface* converted = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_RGBA32);
        SDL_DestroySurface(surface);
        if (converted) {
            // Fully transparent cells (e.g. space) take no atlas space.
            bool any_visible = false;
            const auto* pixels = static_cast<const uint8_t*>(converted->pixels);
            for (int row = 0; row < converted->h && !any_visible; ++row) {
                const uint8_t* row_px = pixels + static_cast<size_t>(row) * converted->pitch;
                for (int col = 0; col < converted->w; ++col) {
                    if (row_px[col * 4 + 3] != 0) {
                        any_visible = true;
                        break;
                    }
                }
            }

            if (any_visible) {
                int x = 0;
                int y = 0;
                bool packed = packer_.pack(converted->w + GLYPH_PADDING, converted->h + GLYPH_PADDING, x, y);
                if (!packed) {
                    // Atlas full: evict everything and retry. Quads emitted
                    // earlier this frame keep stale UVs for one frame.
                    evict_all_glyphs();
                    packed = packer_.pack(converted->w + GLYPH_PADDING, converted->h + GLYPH_PADDING, x, y);
                }
                if (packed) {
                    blit_glyph_to_atlas(converted, x, y);
                    mark_dirty_rows(y, y + converted->h + GLYPH_PADDING);
                    glyph.atlas_x = static_cast<uint16_t>(x);
                    glyph.atlas_y = static_cast<uint16_t>(y);
                    glyph.width = static_cast<uint16_t>(converted->w);
                    glyph.height = static_cast<uint16_t>(converted->h);
                    glyph.has_bitmap = true;
                }
            }
            SDL_DestroySurface(converted);
        }
    }

    auto result = glyphs_.emplace(codepoint, glyph);
    return &result.first->second;
}

void TextRenderer::flush_atlas(SDL_GPUCommandBuffer* cmd) {
    if (dirty_y0_ >= dirty_y1_ || !atlas_texture_ || !atlas_transfer_ || !device_ || !cmd) {
        return;
    }

    const int rows = dirty_y1_ - dirty_y0_;
    const size_t band_bytes = static_cast<size_t>(rows) * ATLAS_SIZE * 4;

    // cycle=true: the previous frame's upload may still be in flight
    void* mapped = SDL_MapGPUTransferBuffer(device_->handle(), atlas_transfer_, true);
    if (!mapped) {
        BRACKEN_LOG_ERROR("text", "Failed to map glyph atlas transfer buffer: {}", SDL_GetError());
        return;
    }
    std::memcpy(mapped, atlas_pixels_.data() + static_cast<size_t>(dirty_y0_) * ATLAS_SIZE * 4, band_bytes);
    SDL_UnmapGPUTransferBuffer(device_->handle(), atlas_transfer_);

    SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(cmd);
    if (!copy_pass) {
        BRACKEN_LOG_ERROR("text", "Failed to begin copy pass for glyph atlas: {}", SDL_GetError());
        return;
    }

    SDL_GPUTextureTransferInfo src = {};
    src.transfer_buffer = atlas_transfer_;
    src.offset = 0;
    src.pixels_per_row = ATLAS_SIZE;
    src.rows_per_layer = static_cast<Uint32>(rows);

    SDL_GPUTextureRegion dst = {};
    dst.texture = atlas_texture_;
    dst.x = 0;
    dst.y = static_cast<Uint32>(dirty_y0_);
    dst.w = ATLAS_SIZE;
    dst.h = static_cast<Uint32>(rows);
    dst.d = 1;

    // cycle=false: keep previously uploaded glyph regions intact
    SDL_UploadToGPUTexture(copy_pass, &src, &dst, false);
    SDL_EndGPUCopyPass(copy_pass);

    dirty_y0_ = 0;
    dirty_y1_ = 0;
}

size_t TextRenderer::build_string_quads(std::string_view text, float x, float y, float scale, std::vector<float>& out,
                                        size_t max_floats) {
    constexpr float INV_ATLAS = 1.0f / static_cast<float>(ATLAS_SIZE);
    size_t quads = 0;
    int pen = 0; // unscaled pixels from the string origin
    uint32_t prev_cp = 0;

    for (size_t i = 0; i < text.size();) {
        const uint32_t cp = decode_utf8(text, i);
        const Glyph* glyph = ensure_glyph(cp);
        if (!glyph) {
            continue;
        }

        if (prev_cp != 0) {
            int kerning = 0;
            if (TTF_GetGlyphKerning(font_, prev_cp, cp, &kerning)) {
                pen += kerning;
            }
        }

        if (glyph->has_bitmap) {
            if (out.size() + VERTICES_PER_QUAD * FLOATS_PER_VERTEX > max_floats) {
                if (!batch_full_warned_) {
                    BRACKEN_LOG_WARN("text", "Text quad budget ({}) exhausted - dropping glyphs this frame",
                                    MAX_TEXT_QUADS);
                    batch_full_warned_ = true;
                }
                break;
            }

            const float x0 = x + static_cast<float>(pen) * scale;
            const float y0 = y;
            const float x1 = x0 + static_cast<float>(glyph->width) * scale;
            const float y1 = y0 + static_cast<float>(glyph->height) * scale;
            const float u0 = static_cast<float>(glyph->atlas_x) * INV_ATLAS;
            const float v0 = static_cast<float>(glyph->atlas_y) * INV_ATLAS;
            const float u1 = static_cast<float>(glyph->atlas_x + glyph->width) * INV_ATLAS;
            const float v1 = static_cast<float>(glyph->atlas_y + glyph->height) * INV_ATLAS;

            const float vertices[] = {x0, y0, u0, v0, x1, y0, u1, v0, x1, y1, u1, v1,
                                      x0, y0, u0, v0, x1, y1, u1, v1, x0, y1, u0, v1};
            out.insert(out.end(), std::begin(vertices), std::end(vertices));
            ++quads;
        }

        pen += glyph->advance;
        prev_cp = cp;
    }
    return quads;
}

void TextRenderer::prepare_text_textures(SDL_GPUCommandBuffer* cmd, const std::vector<std::string>& texts) {
    if (!font_) {
        return;
    }
    for (const auto& text : texts) {
        std::string_view sv = text;
        for (size_t i = 0; i < sv.size();) {
            ensure_glyph(decode_utf8(sv, i));
        }
    }
    flush_atlas(cmd);
}

void TextRenderer::create_pending_textures(SDL_GPUCommandBuffer* cmd) {
    BRACKEN_PROFILE_ZONE_COLOR("TextRenderer::create_pending_textures", 0xC0392B);
    flush_atlas(cmd);
}

void TextRenderer::draw_text(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* render_pass, std::string_view text, float x,
                             float y, uint32_t color, float scale) {
    if (!font_ || text.empty() || !device_ || !pipeline_registry_ || !cmd || !render_pass) {
        return;
    }

    // Build quads; this rasterizes any missing glyphs into the CPU atlas.
    std::vector<float> vertices;
    const size_t quads =
        build_string_quads(text, x, y, scale, vertices, MAX_TEXT_QUADS * VERTICES_PER_QUAD * FLOATS_PER_VERTEX);
    if (quads == 0) {
        return;
    }
    if (dirty_y0_ < dirty_y1_) {
        // New glyphs are not on the GPU yet (we are inside a render pass and
        // cannot upload). They flush at the next copy pass; draw next frame.
        return;
    }

    // Upload vertex data
    vertex_buffer_->update(cmd, vertices.data(), vertices.size() * sizeof(float));

    // Bind text pipeline
    auto* text_pipeline = pipeline_registry_->get_text_pipeline();
    if (text_pipeline) {
        text_pipeline->bind(render_pass);
    }

    // Push uniforms - projection matrix to vertex shader
    SDL_PushGPUVertexUniformData(cmd, 0, &projection_, sizeof(glm::mat4));

    // Push text color to fragment shader (ABGR format - same as rest of renderer)
    glm::vec4 text_color((color & 0xFF) / 255.0f, ((color >> 8) & 0xFF) / 255.0f, ((color >> 16) & 0xFF) / 255.0f,
                         ((color >> 24) & 0xFF) / 255.0f);
    SDL_PushGPUFragmentUniformData(cmd, 0, &text_color, sizeof(glm::vec4));

    // Bind atlas texture and sampler
    SDL_GPUTextureSamplerBinding tex_binding = {};
    tex_binding.texture = atlas_texture_;
    tex_binding.sampler = sampler_;
    SDL_BindGPUFragmentSamplers(render_pass, 0, &tex_binding, 1);

    // Bind vertex buffer
    SDL_GPUBufferBinding vb_binding = {};
    vb_binding.buffer = vertex_buffer_->handle();
    vb_binding.offset = 0;
    SDL_BindGPUVertexBuffers(render_pass, 0, &vb_binding, 1);

    SDL_DrawGPUPrimitives(render_pass, static_cast<Uint32>(quads * VERTICES_PER_QUAD), 1, 0, 0);
}

void TextRenderer::draw_text_centered(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* render_pass, std::string_view text,
                                      float x, float y, uint32_t color, float scale) {
    int width = get_text_width(text, scale);
    draw_text(cmd, render_pass, text, x - width / 2.0f, y, color, scale);
}

int TextRenderer::get_text_width(std::string_view text, float scale) {
    if (!font_ || text.empty()) {
        return 0;
    }

    int w = 0;
    int h = 0;
    TTF_GetStringSize(font_, text.data(), text.size(), &w, &h);
    return static_cast<int>(w * scale);
}

int TextRenderer::get_text_height(float scale) {
    if (!font_) {
        return 0;
    }
    return static_cast<int>(TTF_GetFontHeight(font_) * scale);
}

void TextRenderer::queue_text_draw(std::string_view text, float x, float y, uint32_t color, float scale) {
    if (!font_ || text.empty()) {
        return;
    }

    const size_t first_vertex = batch_vertices_.size() / FLOATS_PER_VERTEX;
    const size_t quads =
        build_string_quads(text, x, y, scale, batch_vertices_, MAX_TEXT_QUADS * VERTICES_PER_QUAD * FLOATS_PER_VERTEX);
    if (quads == 0) {
        return;
    }

    TextRun run{};
    run.first_vertex = first_vertex;
    run.vertex_count = static_cast<uint32_t>(quads * VERTICES_PER_QUAD);
    run.color = color;
    queued_runs_.push_back(run);
}

void TextRenderer::upload_queued_text(SDL_GPUCommandBuffer* cmd) {
    BRACKEN_PROFILE_ZONE_COLOR("TextRenderer::upload_queued_text", 0xC0392B);
    if (!cmd) {
        return;
    }

    // Upload any glyphs rasterized this frame (no-op when the atlas is clean)
    flush_atlas(cmd);

    if (batch_vertices_.empty() || !vertex_buffer_) {
        return;
    }

    // Upload all batched vertex data in one copy pass
    vertex_buffer_->update(cmd, batch_vertices_.data(), batch_vertices_.size() * sizeof(float));
}

void TextRenderer::draw_queued_text(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* render_pass) {
    BRACKEN_PROFILE_ZONE_COLOR("TextRenderer::draw_queued_text", 0xC0392B);
    if (queued_runs_.empty() || !cmd || !render_pass || !pipeline_registry_ || !atlas_texture_) {
        queued_runs_.clear();
        batch_vertices_.clear();
        return;
    }

    auto* text_pipeline = pipeline_registry_->get_text_pipeline();
    if (!text_pipeline) {
        queued_runs_.clear();
        batch_vertices_.clear();
        return;
    }
    text_pipeline->bind(render_pass);

    // Projection, atlas and vertex buffer are invariant across all runs
    SDL_PushGPUVertexUniformData(cmd, 0, &projection_, sizeof(glm::mat4));

    SDL_GPUTextureSamplerBinding tex_binding = {};
    tex_binding.texture = atlas_texture_;
    tex_binding.sampler = sampler_;
    SDL_BindGPUFragmentSamplers(render_pass, 0, &tex_binding, 1);

    SDL_GPUBufferBinding vb_binding = {};
    vb_binding.buffer = vertex_buffer_->handle();
    vb_binding.offset = 0;
    SDL_BindGPUVertexBuffers(render_pass, 0, &vb_binding, 1);

    // Merge contiguous same-color runs into a single draw; push the color
    // uniform only when it changes (common case: same color for many labels).
    const auto push_color = [&](uint32_t color) {
        glm::vec4 text_color((color & 0xFF) / 255.0f, ((color >> 8) & 0xFF) / 255.0f, ((color >> 16) & 0xFF) / 255.0f,
                             ((color >> 24) & 0xFF) / 255.0f);
        SDL_PushGPUFragmentUniformData(cmd, 0, &text_color, sizeof(glm::vec4));
    };

    size_t draw_first = queued_runs_.front().first_vertex;
    uint32_t draw_count = 0;
    uint32_t draw_color = queued_runs_.front().color;
    push_color(draw_color);

    for (const auto& run : queued_runs_) {
        const bool contiguous = run.first_vertex == draw_first + draw_count;
        if (run.color == draw_color && contiguous) {
            draw_count += run.vertex_count;
            continue;
        }
        if (draw_count > 0) {
            SDL_DrawGPUPrimitives(render_pass, draw_count, 1, static_cast<Uint32>(draw_first), 0);
        }
        if (run.color != draw_color) {
            push_color(run.color);
            draw_color = run.color;
        }
        draw_first = run.first_vertex;
        draw_count = run.vertex_count;
    }
    if (draw_count > 0) {
        SDL_DrawGPUPrimitives(render_pass, draw_count, 1, static_cast<Uint32>(draw_first), 0);
    }

    // Clear queued data for next frame
    queued_runs_.clear();
    batch_vertices_.clear();
}

void TextRenderer::draw_text_immediate(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* render_pass, std::string_view text,
                                       float x, float y, uint32_t color, float scale) {
    // Legacy path: appends into the staging batch and draws at that offset.
    // Only renders correctly if the vertex data was already uploaded; prefer
    // queue_text_draw + upload_queued_text + draw_queued_text.
    if (!font_ || text.empty() || !device_ || !pipeline_registry_ || !cmd || !render_pass) {
        return;
    }

    const size_t immediate_offset = batch_vertices_.size();
    const size_t quads =
        build_string_quads(text, x, y, scale, batch_vertices_, MAX_TEXT_QUADS * VERTICES_PER_QUAD * FLOATS_PER_VERTEX);
    if (quads == 0) {
        return;
    }

    // Bind text pipeline
    auto* text_pipeline = pipeline_registry_->get_text_pipeline();
    if (text_pipeline) {
        text_pipeline->bind(render_pass);
    }

    // Push uniforms
    SDL_PushGPUVertexUniformData(cmd, 0, &projection_, sizeof(glm::mat4));
    glm::vec4 text_color((color & 0xFF) / 255.0f, ((color >> 8) & 0xFF) / 255.0f, ((color >> 16) & 0xFF) / 255.0f,
                         ((color >> 24) & 0xFF) / 255.0f);
    SDL_PushGPUFragmentUniformData(cmd, 0, &text_color, sizeof(glm::vec4));

    // Bind atlas texture and sampler
    SDL_GPUTextureSamplerBinding tex_binding = {};
    tex_binding.texture = atlas_texture_;
    tex_binding.sampler = sampler_;
    SDL_BindGPUFragmentSamplers(render_pass, 0, &tex_binding, 1);

    SDL_GPUBufferBinding vb_binding = {};
    vb_binding.buffer = vertex_buffer_->handle();
    vb_binding.offset = static_cast<Uint32>(immediate_offset * sizeof(float));
    SDL_BindGPUVertexBuffers(render_pass, 0, &vb_binding, 1);

    SDL_DrawGPUPrimitives(render_pass, static_cast<Uint32>(quads * VERTICES_PER_QUAD), 1, 0, 0);
}

} // namespace bracken::render
