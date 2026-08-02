#include "gpu_texture.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_types.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_gpu.h"
#include "SDL3/SDL_pixels.h"
#include "SDL3/SDL_stdinc.h"
#include "SDL3/SDL_surface.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <SDL3_image/SDL_image.h>
#include <string>

namespace bracken::gpu {

namespace {
constexpr const char* LOG_TAG = "gpu";
} // namespace

// =============================================================================
// GPUTexture Implementation
// =============================================================================
// Thread-safety: texture creation/destruction is main-thread tier. Every
// factory funnels through GPUDevice::create_texture and every destructor
// through GPUDevice::release_texture, both of which assert the contract;
// the asserts repeated below exist to fail at the caller-visible entry point.

GPUTexture::~GPUTexture() {
    if (device_ && texture_) {
        device_->release_texture(texture_);
    }
}

GPUTexture::GPUTexture(GPUTexture&& other) noexcept
    : device_(other.device_),
      texture_(other.texture_),
      width_(other.width_),
      height_(other.height_),
      format_(other.format_),
      is_render_target_(other.is_render_target_),
      is_depth_(other.is_depth_),
      layers_(other.layers_),
      mip_levels_(other.mip_levels_) {
    other.device_ = nullptr;
    other.texture_ = nullptr;
}

GPUTexture& GPUTexture::operator=(GPUTexture&& other) noexcept {
    if (this != &other) {
        if (device_ && texture_) {
            device_->release_texture(texture_);
        }

        device_ = other.device_;
        texture_ = other.texture_;
        width_ = other.width_;
        height_ = other.height_;
        format_ = other.format_;
        is_render_target_ = other.is_render_target_;
        is_depth_ = other.is_depth_;
        layers_ = other.layers_;
        mip_levels_ = other.mip_levels_;

        other.device_ = nullptr;
        other.texture_ = nullptr;
    }
    return *this;
}

uint32_t GPUTexture::calculate_mip_levels(int width, int height) {
    return static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}

size_t GPUTexture::get_bytes_per_pixel(SDL_GPUTextureFormat format) {
    // Covers every format the engine can actually meet here: the TextureFormat
    // enum (gpu_types.hpp), swapchain candidates, and the HDR scene format.
    // Unknown formats return 0 so callers fail loudly instead of mis-sizing
    // the transfer buffer and silently truncating the upload.
    switch (format) {
        case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
        // Integer single-channel: the GI probe-state atlas
        // (render/gi/probe_volume.cpp) is uploaded from a CPU byte mirror, and
        // an unlisted format here would reject that upload rather than
        // mis-size it.
        case SDL_GPU_TEXTUREFORMAT_R8_UINT:
            return 1;
        case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
            return 2;
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_D32_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
            return 4;
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
            return 8;
        default:
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::get_bytes_per_pixel: unsupported format {}",
                              static_cast<int>(format));
            return 0;
    }
}

bool GPUTexture::reload_from_file(const std::string& path, bool generate_mipmaps) {
    // Hot-reload swap: creates the fresh texture and releases the old one.
    BRACKEN_ASSERT_MAIN_THREAD();
    if (!device_) {
        return false;
    }
    // Preserve the creation format (notably RGBA8_SRGB for albedo art) —
    // reloading with the load_from_file default would silently drop sRGB and
    // wash out color textures.
    TextureFormat creation_format = from_sdl_format(format_);
    if (to_sdl_format(creation_format) != format_) {
        BRACKEN_LOG_ERROR(
            LOG_TAG, "GPUTexture::reload_from_file: cannot reload '{}': texture format {} has no TextureFormat mapping",
            path.c_str(), static_cast<int>(format_));
        return false;
    }
    auto fresh = load_from_file(*device_, path, generate_mipmaps, creation_format);
    if (!fresh) {
        return false;
    }

    if (texture_) {
        device_->release_texture(texture_);
    }
    texture_ = fresh->texture_;
    width_ = fresh->width_;
    height_ = fresh->height_;
    format_ = fresh->format_;
    mip_levels_ = fresh->mip_levels_;
    layers_ = fresh->layers_;
    // Prevent fresh's destructor from releasing the texture we just stole.
    fresh->texture_ = nullptr;
    fresh->device_ = nullptr;
    return true;
}

std::unique_ptr<GPUTexture> GPUTexture::load_from_file(GPUDevice& device, const std::string& path,
                                                       bool generate_mipmaps, TextureFormat format) {
    // Load image via SDL_image
    SDL_Surface* surface = IMG_Load(path.c_str());
    if (!surface) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::load_from_file: Failed to load '{}': {}", path.c_str(), SDL_GetError());
        return nullptr;
    }

    // Convert to RGBA8 format if needed
    SDL_Surface* converted = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_RGBA32);
    SDL_DestroySurface(surface);

    if (!converted) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::load_from_file: Failed to convert surface: {}", SDL_GetError());
        return nullptr;
    }

    // Create texture from pixel data
    auto texture = create_2d(device, converted->w, converted->h, format, converted->pixels, generate_mipmaps);

    SDL_DestroySurface(converted);
    return texture;
}

std::unique_ptr<GPUTexture> GPUTexture::create_2d(GPUDevice& device, int width, int height, TextureFormat format,
                                                  const void* pixels, bool generate_mipmaps) {
    BRACKEN_ASSERT_MAIN_THREAD();
    auto texture = std::unique_ptr<GPUTexture>(new GPUTexture());
    texture->device_ = &device;
    texture->width_ = width;
    texture->height_ = height;
    texture->format_ = to_sdl_format(format);
    texture->mip_levels_ = generate_mipmaps ? calculate_mip_levels(width, height) : 1;

    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_2D;
    tex_info.format = texture->format_;
    tex_info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    // Mipmap generation requires COLOR_TARGET usage
    if (generate_mipmaps) {
        tex_info.usage |= SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    }
    tex_info.width = static_cast<Uint32>(width);
    tex_info.height = static_cast<Uint32>(height);
    tex_info.layer_count_or_depth = 1;
    tex_info.num_levels = texture->mip_levels_;

    texture->texture_ = device.create_texture(tex_info);
    if (!texture->texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d: Failed to create texture: {}", SDL_GetError());
        return nullptr;
    }

    // Upload pixel data if provided
    if (pixels) {
        // Create transfer buffer
        size_t bytes_per_pixel = get_bytes_per_pixel(texture->format_);
        if (bytes_per_pixel == 0) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d: Cannot upload pixels for unsupported format");
            return nullptr;
        }
        size_t data_size = static_cast<size_t>(width) * static_cast<size_t>(height) * bytes_per_pixel;
        SDL_GPUTransferBufferCreateInfo transfer_info = {};
        transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        transfer_info.size = static_cast<Uint32>(data_size);

        SDL_GPUTransferBuffer* transfer = device.create_transfer_buffer(transfer_info);
        if (!transfer) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d: Failed to create transfer buffer: {}", SDL_GetError());
            return nullptr;
        }

        // Map and copy
        void* mapped = device.map_transfer_buffer(transfer, false);
        if (!mapped) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d: Failed to map transfer buffer: {}", SDL_GetError());
            device.release_transfer_buffer(transfer);
            return nullptr;
        }
        std::memcpy(mapped, pixels, data_size);
        device.unmap_transfer_buffer(transfer);

        // Upload via copy pass
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device.handle());
        if (!cmd) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d: Failed to acquire command buffer: {}", SDL_GetError());
            device.release_transfer_buffer(transfer);
            return nullptr;
        }

        SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(cmd);
        if (!copy_pass) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d: Failed to begin copy pass: {}", SDL_GetError());
            SDL_CancelGPUCommandBuffer(cmd);
            device.release_transfer_buffer(transfer);
            return nullptr;
        }

        SDL_GPUTextureTransferInfo src = {};
        src.transfer_buffer = transfer;
        src.offset = 0;

        SDL_GPUTextureRegion dst = {};
        dst.texture = texture->texture_;
        dst.w = static_cast<Uint32>(width);
        dst.h = static_cast<Uint32>(height);
        dst.d = 1;

        SDL_UploadToGPUTexture(copy_pass, &src, &dst, false);
        SDL_EndGPUCopyPass(copy_pass);

        // Generate mipmaps if requested (must be called outside any pass)
        if (generate_mipmaps && texture->mip_levels_ > 1) {
            SDL_GenerateMipmapsForGPUTexture(cmd, texture->texture_);
        }

        SDL_SubmitGPUCommandBuffer(cmd);
        device.release_transfer_buffer(transfer);
    }

    return texture;
}

std::unique_ptr<GPUTexture> GPUTexture::create_2d(GPUDevice& device, int width, int height, SDL_GPUTextureFormat format,
                                                  SDL_GPUTextureUsageFlags usage) {
    BRACKEN_ASSERT_MAIN_THREAD();
    auto texture = std::unique_ptr<GPUTexture>(new GPUTexture());
    texture->device_ = &device;
    texture->width_ = width;
    texture->height_ = height;
    texture->format_ = format;
    texture->mip_levels_ = 1;

    // Check if this is a depth format
    texture->is_depth_ =
        (format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT || format == SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT ||
         format == SDL_GPU_TEXTUREFORMAT_D16_UNORM);

    // Check if this is a render target (color or depth-stencil)
    texture->is_render_target_ =
        (usage & (SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) != 0;

    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_2D;
    tex_info.format = format;
    tex_info.usage = usage;
    tex_info.width = static_cast<Uint32>(width);
    tex_info.height = static_cast<Uint32>(height);
    tex_info.layer_count_or_depth = 1;
    tex_info.num_levels = 1;

    texture->texture_ = device.create_texture(tex_info);
    if (!texture->texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d: Failed to create texture: {}", SDL_GetError());
        return nullptr;
    }

    return texture;
}

std::unique_ptr<GPUTexture> GPUTexture::create_render_target(GPUDevice& device, int width, int height,
                                                             TextureFormat format) {
    auto texture = std::unique_ptr<GPUTexture>(new GPUTexture());
    texture->device_ = &device;
    texture->width_ = width;
    texture->height_ = height;
    texture->format_ = to_sdl_format(format);
    texture->is_render_target_ = true;
    texture->mip_levels_ = 1;

    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_2D;
    tex_info.format = texture->format_;
    tex_info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
    tex_info.width = static_cast<Uint32>(width);
    tex_info.height = static_cast<Uint32>(height);
    tex_info.layer_count_or_depth = 1;
    tex_info.num_levels = 1;

    texture->texture_ = device.create_texture(tex_info);
    if (!texture->texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_render_target: Failed to create texture: {}", SDL_GetError());
        return nullptr;
    }

    return texture;
}

std::unique_ptr<GPUTexture> GPUTexture::create_depth(GPUDevice& device, int width, int height) {
    auto texture = std::unique_ptr<GPUTexture>(new GPUTexture());
    texture->device_ = &device;
    texture->width_ = width;
    texture->height_ = height;
    texture->format_ = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    texture->is_depth_ = true;
    texture->mip_levels_ = 1;

    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_2D;
    tex_info.format = texture->format_;
    // Depth is sampled by GTAO/TAA/fog and the RT compute passes. It cannot be
    // a compute storage texture: SDL explicitly restricts depth usage to
    // DEPTH_STENCIL_TARGET | SAMPLER. RT therefore binds it through
    // SDL_BindGPUComputeSamplers with a point-clamp sampler, which is portable
    // across Metal/Vulkan/D3D12 and matches this usage exactly.
    tex_info.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
    tex_info.width = static_cast<Uint32>(width);
    tex_info.height = static_cast<Uint32>(height);
    tex_info.layer_count_or_depth = 1;
    tex_info.num_levels = 1;

    texture->texture_ = device.create_texture(tex_info);
    if (!texture->texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_depth: Failed to create texture: {}", SDL_GetError());
        return nullptr;
    }

    return texture;
}

std::unique_ptr<GPUTexture> GPUTexture::create_depth_array(GPUDevice& device, int width, int height, int layers) {
    auto texture = std::unique_ptr<GPUTexture>(new GPUTexture());
    texture->device_ = &device;
    texture->width_ = width;
    texture->height_ = height;
    texture->layers_ = layers;
    texture->format_ = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    texture->is_depth_ = true;
    texture->is_render_target_ = true;
    texture->mip_levels_ = 1;

    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_2D_ARRAY;
    tex_info.format = texture->format_;
    tex_info.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
    tex_info.width = static_cast<Uint32>(width);
    tex_info.height = static_cast<Uint32>(height);
    tex_info.layer_count_or_depth = static_cast<Uint32>(layers);
    tex_info.num_levels = 1;

    texture->texture_ = device.create_texture(tex_info);
    if (!texture->texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_depth_array: Failed to create texture: {}", SDL_GetError());
        return nullptr;
    }

    return texture;
}

std::unique_ptr<GPUTexture> GPUTexture::create_2d_array(GPUDevice& device, int width, int height, int layers,
                                                        TextureFormat format, const void** layer_data,
                                                        bool generate_mipmaps) {
    auto texture = std::unique_ptr<GPUTexture>(new GPUTexture());
    texture->device_ = &device;
    texture->width_ = width;
    texture->height_ = height;
    texture->layers_ = layers;
    texture->format_ = to_sdl_format(format);
    texture->is_depth_ = false;
    texture->is_render_target_ = false;
    texture->mip_levels_ = generate_mipmaps ? calculate_mip_levels(width, height) : 1;

    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_2D_ARRAY;
    tex_info.format = texture->format_;
    tex_info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    // Mipmap generation requires COLOR_TARGET usage
    if (generate_mipmaps) {
        tex_info.usage |= SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    }
    tex_info.width = static_cast<Uint32>(width);
    tex_info.height = static_cast<Uint32>(height);
    tex_info.layer_count_or_depth = static_cast<Uint32>(layers);
    tex_info.num_levels = texture->mip_levels_;

    texture->texture_ = device.create_texture(tex_info);
    if (!texture->texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d_array: Failed to create texture: {}", SDL_GetError());
        return nullptr;
    }

    // Upload all layers through one transfer buffer / command buffer, then a
    // single mipmap-generation call covering every layer. Any upload failure
    // releases the texture and returns nullptr (header contract; a
    // valid-looking texture with no pixel data would render garbage with no
    // way for the caller to detect it).
    if (layer_data) {
        size_t bytes_per_pixel = get_bytes_per_pixel(texture->format_);
        if (bytes_per_pixel == 0) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d_array: Cannot upload layers for unsupported format");
            return nullptr;
        }
        size_t layer_size = static_cast<size_t>(width) * height * bytes_per_pixel;

        SDL_GPUTransferBufferCreateInfo transfer_info = {};
        transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        transfer_info.size = static_cast<Uint32>(layer_size * static_cast<size_t>(layers));

        SDL_GPUTransferBuffer* transfer = device.create_transfer_buffer(transfer_info);
        if (!transfer) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d_array: Failed to create transfer buffer: {}",
                              SDL_GetError());
            return nullptr;
        }

        void* mapped = device.map_transfer_buffer(transfer, false);
        if (!mapped) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d_array: Failed to map transfer buffer: {}",
                              SDL_GetError());
            device.release_transfer_buffer(transfer);
            return nullptr;
        }
        for (int layer = 0; layer < layers; ++layer) {
            if (layer_data[layer]) {
                std::memcpy(static_cast<char*>(mapped) + static_cast<size_t>(layer) * layer_size, layer_data[layer],
                            layer_size);
            }
        }
        device.unmap_transfer_buffer(transfer);

        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device.handle());
        if (!cmd) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d_array: Failed to acquire command buffer: {}",
                              SDL_GetError());
            device.release_transfer_buffer(transfer);
            return nullptr;
        }

        SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(cmd);
        if (!copy_pass) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_2d_array: Failed to begin copy pass: {}", SDL_GetError());
            SDL_CancelGPUCommandBuffer(cmd);
            device.release_transfer_buffer(transfer);
            return nullptr;
        }

        for (int layer = 0; layer < layers; ++layer) {
            if (!layer_data[layer]) {
                continue;
            }
            SDL_GPUTextureTransferInfo src = {};
            src.transfer_buffer = transfer;
            src.offset = static_cast<Uint32>(static_cast<size_t>(layer) * layer_size);

            SDL_GPUTextureRegion dst = {};
            dst.texture = texture->texture_;
            dst.w = static_cast<Uint32>(width);
            dst.h = static_cast<Uint32>(height);
            dst.d = 1;
            dst.mip_level = 0;
            dst.layer = static_cast<Uint32>(layer);

            SDL_UploadToGPUTexture(copy_pass, &src, &dst, false);
        }
        SDL_EndGPUCopyPass(copy_pass);

        // One call generates mips for every array layer (must be outside any pass).
        if (generate_mipmaps && texture->mip_levels_ > 1) {
            SDL_GenerateMipmapsForGPUTexture(cmd, texture->texture_);
        }

        SDL_SubmitGPUCommandBuffer(cmd);
        device.release_transfer_buffer(transfer);
    }

    return texture;
}

std::unique_ptr<GPUTexture> GPUTexture::create_cube(GPUDevice& device, int size, SDL_GPUTextureFormat format,
                                                    SDL_GPUTextureUsageFlags usage, bool mipmaps) {
    auto texture = std::unique_ptr<GPUTexture>(new GPUTexture());
    texture->device_ = &device;
    texture->width_ = size;
    texture->height_ = size;
    texture->layers_ = 6;
    texture->format_ = format;
    texture->is_render_target_ = (usage & (SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) != 0;
    texture->mip_levels_ = mipmaps ? calculate_mip_levels(size, size) : 1;

    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_CUBE;
    tex_info.format = format;
    tex_info.usage = usage;
    tex_info.width = static_cast<Uint32>(size);
    tex_info.height = static_cast<Uint32>(size);
    tex_info.layer_count_or_depth = 6;
    tex_info.num_levels = texture->mip_levels_;

    texture->texture_ = device.create_texture(tex_info);
    if (!texture->texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_cube: Failed to create texture: {}", SDL_GetError());
        return nullptr;
    }

    return texture;
}

std::unique_ptr<GPUTexture> GPUTexture::create_depth_stencil(GPUDevice& device, int width, int height) {
    auto texture = std::unique_ptr<GPUTexture>(new GPUTexture());
    texture->device_ = &device;
    texture->width_ = width;
    texture->height_ = height;
    texture->format_ = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT;
    texture->is_depth_ = true;
    texture->mip_levels_ = 1;

    SDL_GPUTextureCreateInfo tex_info = {};
    tex_info.type = SDL_GPU_TEXTURETYPE_2D;
    tex_info.format = texture->format_;
    tex_info.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
    tex_info.width = static_cast<Uint32>(width);
    tex_info.height = static_cast<Uint32>(height);
    tex_info.layer_count_or_depth = 1;
    tex_info.num_levels = 1;

    texture->texture_ = device.create_texture(tex_info);
    if (!texture->texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::create_depth_stencil: Failed to create texture: {}", SDL_GetError());
        return nullptr;
    }

    return texture;
}

void GPUTexture::upload(SDL_GPUCommandBuffer* cmd, const void* pixels, int width, int height) {
    if (!cmd || !pixels || !device_ || !texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::upload: Invalid parameters");
        return;
    }

    if (width != width_ || height != height_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::upload: Dimension mismatch");
        return;
    }

    size_t bytes_per_pixel = get_bytes_per_pixel(format_);
    if (bytes_per_pixel == 0) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::upload: Unsupported texture format, upload rejected");
        return;
    }
    size_t data_size = static_cast<size_t>(width) * static_cast<size_t>(height) * bytes_per_pixel;

    SDL_GPUTransferBufferCreateInfo transfer_info = {};
    transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    transfer_info.size = static_cast<Uint32>(data_size);

    SDL_GPUTransferBuffer* transfer = device_->create_transfer_buffer(transfer_info);
    if (!transfer) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::upload: Failed to create transfer buffer: {}", SDL_GetError());
        return;
    }

    void* mapped = device_->map_transfer_buffer(transfer, false);
    if (!mapped) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::upload: Failed to map transfer buffer: {}", SDL_GetError());
        device_->release_transfer_buffer(transfer);
        return;
    }
    std::memcpy(mapped, pixels, data_size);
    device_->unmap_transfer_buffer(transfer);

    SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(cmd);
    if (!copy_pass) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUTexture::upload: Failed to begin copy pass: {}", SDL_GetError());
        device_->release_transfer_buffer(transfer);
        return;
    }

    SDL_GPUTextureTransferInfo src = {};
    src.transfer_buffer = transfer;
    src.offset = 0;

    SDL_GPUTextureRegion dst = {};
    dst.texture = texture_;
    dst.w = static_cast<Uint32>(width);
    dst.h = static_cast<Uint32>(height);
    dst.d = 1;

    SDL_UploadToGPUTexture(copy_pass, &src, &dst, false);
    SDL_EndGPUCopyPass(copy_pass);

    // Releasing before the caller submits `cmd` is safe by SDL3 GPU contract:
    // resources are reference-tracked as they are recorded into command
    // buffers, and SDL_ReleaseGPUTransferBuffer only marks the buffer for
    // deletion — actual destruction is deferred until every command buffer
    // referencing it has finished executing on the GPU.
    // Note this method allocates a fresh transfer buffer per call; callers
    // uploading every frame should keep a persistent staging buffer instead
    // (as the text renderer does).
    device_->release_transfer_buffer(transfer);
}

// =============================================================================
// SamplerConfig Implementation
// =============================================================================

SamplerConfig SamplerConfig::linear_repeat() {
    return SamplerConfig{
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
        .address_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
    };
}

SamplerConfig SamplerConfig::linear_clamp() {
    return SamplerConfig{
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
        .address_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
    };
}

SamplerConfig SamplerConfig::nearest_repeat() {
    return SamplerConfig{
        .min_filter = SDL_GPU_FILTER_NEAREST,
        .mag_filter = SDL_GPU_FILTER_NEAREST,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
    };
}

SamplerConfig SamplerConfig::nearest_clamp() {
    return SamplerConfig{
        .min_filter = SDL_GPU_FILTER_NEAREST,
        .mag_filter = SDL_GPU_FILTER_NEAREST,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
    };
}

SamplerConfig SamplerConfig::anisotropic(float max_aniso) {
    return SamplerConfig{
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
        .address_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .max_anisotropy = max_aniso,
        .enable_anisotropy = true,
    };
}

// =============================================================================
// GPUSampler Implementation
// =============================================================================

GPUSampler::~GPUSampler() {
    if (device_ && sampler_) {
        device_->release_sampler(sampler_);
    }
}

GPUSampler::GPUSampler(GPUSampler&& other) noexcept : device_(other.device_), sampler_(other.sampler_) {
    other.device_ = nullptr;
    other.sampler_ = nullptr;
}

GPUSampler& GPUSampler::operator=(GPUSampler&& other) noexcept {
    if (this != &other) {
        if (device_ && sampler_) {
            device_->release_sampler(sampler_);
        }

        device_ = other.device_;
        sampler_ = other.sampler_;

        other.device_ = nullptr;
        other.sampler_ = nullptr;
    }
    return *this;
}

std::unique_ptr<GPUSampler> GPUSampler::create(GPUDevice& device, const SamplerConfig& config) {
    auto sampler = std::unique_ptr<GPUSampler>(new GPUSampler());
    sampler->device_ = &device;

    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = config.min_filter;
    sampler_info.mag_filter = config.mag_filter;
    sampler_info.mipmap_mode = config.mipmap_mode;
    sampler_info.address_mode_u = config.address_u;
    sampler_info.address_mode_v = config.address_v;
    sampler_info.address_mode_w = config.address_w;
    sampler_info.mip_lod_bias = config.mip_lod_bias;
    sampler_info.min_lod = config.min_lod;
    sampler_info.max_lod = config.max_lod;
    sampler_info.enable_anisotropy = config.enable_anisotropy;
    sampler_info.max_anisotropy = config.max_anisotropy;
    sampler_info.enable_compare = false;

    sampler->sampler_ = device.create_sampler(sampler_info);
    if (!sampler->sampler_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUSampler::create: Failed to create sampler: {}", SDL_GetError());
        return nullptr;
    }

    return sampler;
}

} // namespace bracken::gpu
