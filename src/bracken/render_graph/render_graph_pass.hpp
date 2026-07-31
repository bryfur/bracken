#pragma once

#include "bracken/render_graph/render_graph_resource.hpp"
#include <functional>
#include <SDL3/SDL_gpu.h>
#include <string>
#include <vector>

namespace bracken::render_graph {

class RenderPassContext;

struct ColorAttachment {
    ResourceHandle handle;
    SDL_GPUColorTargetInfo info{};
};

struct DepthAttachment {
    ResourceHandle handle;
    SDL_GPUDepthStencilTargetInfo info{};
};

using PassExecuteFn = std::function<void(RenderPassContext&)>;

// A pass is a callback plus declared reads/writes. Whether the graph wraps the
// callback in an SDL render pass is driven entirely by managed_render_pass +
// the declared attachments; there is no pass-type enum (a former PassType field
// was inert metadata and was removed — compute/blit passes just use the raw
// command buffer from RenderPassContext).
struct PassNode {
    std::string name;

    std::vector<ResourceHandle> reads;
    std::vector<ResourceHandle> writes;

    // Render-target outputs (for graphics passes the graph manages).
    std::vector<ColorAttachment> color_attachments;
    DepthAttachment depth_attachment{};
    bool has_depth = false;

    // If true, the graph will SDL_BeginGPURenderPass / SDL_EndGPURenderPass
    // around the execute callback using color_attachments + depth_attachment.
    bool managed_render_pass = false;

    PassExecuteFn execute;
};

} // namespace bracken::render_graph
