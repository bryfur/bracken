// SceneRenderer: swapchain capture.
//
// One function, and it is here for one reason: it is the only place in the
// renderer that downloads from the GPU, blocks on a fence and links an image
// encoder (SDL3_image). That is a different kind of work from anything else
// in the frame path, and none of the other translation units should be
// paying for its includes.
//
// The frame-lifecycle BRANCH (end_frame_and_wait vs end_frame) deliberately
// stays in the orchestrator, in render_frame: which of the two ends the frame
// is a lifecycle decision, not a capture detail. Only the capture body lives
// here.

#include "scene_renderer.hpp"
#include "scene_renderer_internal.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/render/render_context.hpp"

#include <SDL3/SDL_gpu.h>
#include <SDL3_image/SDL_image.h>
#include <string>
#include <utility>

namespace bracken::scene {

namespace gpu = bracken::gpu;

void SceneRenderer::capture_screenshot_and_end_frame() {
    const std::string path = std::move(pending_screenshot_path_);
    pending_screenshot_path_.clear();

    const uint32_t w = static_cast<uint32_t>(context_->width());
    const uint32_t h = static_cast<uint32_t>(context_->height());
    gpu::GPUDevice& device = context_->device();

    SDL_GPUTransferBufferCreateInfo tb_info{};
    tb_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    tb_info.size = w * h * 4;
    SDL_GPUTransferBuffer* transfer = device.create_transfer_buffer(tb_info);
    if (!transfer) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Screenshot: failed to create download buffer: {}", SDL_GetError());
        end_frame();
        return;
    }

    // Record the download on this frame's command buffer, then submit with a
    // fence and block — the transfer buffer is only coherent after the GPU is
    // done. Copy passes can't nest in render passes; all passes ended above.
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(context_->current_command_buffer());
    SDL_GPUTextureRegion src{};
    src.texture = current_swapchain_;
    src.w = w;
    src.h = h;
    src.d = 1;
    SDL_GPUTextureTransferInfo dst{};
    dst.transfer_buffer = transfer;
    SDL_DownloadFromGPUTexture(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);

    const SDL_GPUTextureFormat gpu_format = context_->swapchain_format();
    current_swapchain_ = nullptr;
    if (!context_->end_frame_and_wait()) {
        device.release_transfer_buffer(transfer);
        return;
    }

    SDL_PixelFormat pixel_format;
    switch (gpu_format) {
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
        pixel_format = SDL_PIXELFORMAT_BGRA32;
        break;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
        pixel_format = SDL_PIXELFORMAT_RGBA32;
        break;
    default:
        BRACKEN_LOG_WARN(LOG_TAG, "Screenshot: unexpected swapchain format {}, assuming RGBA8",
                         static_cast<int>(gpu_format));
        pixel_format = SDL_PIXELFORMAT_RGBA32;
        break;
    }

    bool saved = false;
    if (void* pixels = device.map_transfer_buffer(transfer, false)) {
        SDL_Surface* surface =
            SDL_CreateSurfaceFrom(static_cast<int>(w), static_cast<int>(h), pixel_format, pixels, static_cast<int>(w * 4));
        if (surface) {
            // RGB24 drops the alpha channel — swapchain alpha is undefined and
            // would render the PNG transparent in viewers.
            if (SDL_Surface* rgb = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_RGB24)) {
                saved = IMG_SavePNG(rgb, path.c_str());
                SDL_DestroySurface(rgb);
            }
            SDL_DestroySurface(surface);
        }
        device.unmap_transfer_buffer(transfer);
    }
    device.release_transfer_buffer(transfer);

    if (saved) {
        BRACKEN_LOG_INFO(LOG_TAG, "SceneRenderer: screenshot saved to '{}' ({}x{})", path.c_str(), w, h);
    } else {
        BRACKEN_LOG_ERROR(LOG_TAG, "Screenshot: failed to save '{}': {}", path, SDL_GetError());
    }
}

} // namespace bracken::scene
