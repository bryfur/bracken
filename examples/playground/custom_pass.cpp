#include "custom_pass.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"

#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_stdinc.h>
#include <string>
#include <utility>

namespace playground {

namespace {
constexpr const char* LOG_TAG = "playground";

/// Where `bracken_compile_shaders(playground DIRECTORY .../shaders)` stages the
/// game's own SPIR-V, relative to the executable. It is NOT `shaders/`:
/// `bracken_copy_shaders` deletes and re-copies that directory on every build,
/// so anything a host puts there vanishes intermittently.
constexpr const char* k_host_shader_dir = "shaders_playground/";
} // namespace

bool CustomTintPass::enabled_by_environment() {
    const char* env = SDL_getenv("PLAYGROUND_CUSTOM_PASS");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool CustomTintPass::install(bracken::scene::SceneRenderer& renderer) {
    namespace gpu = bracken::gpu;
    namespace scene = bracken::scene;

    gpu::PipelineRegistry* registry = renderer.pipeline_registry();
    if (registry == nullptr || !registry->is_initialized()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "CustomTintPass: install() needs a live renderer — call it after init_renderer()");
        return false;
    }

    // Step 1: tell the loader where this game's SPIR-V lives. Roots resolve in
    // registration order with the ENGINE root always first, so this can never
    // shadow an engine shader — and, read the other way, the pipeline below can
    // name an engine shader and get it.
    const char* base_path = SDL_GetBasePath();
    const std::string host_root = std::string(base_path != nullptr ? base_path : "") + k_host_shader_dir;
    // false here just means "already a root" (install() called twice); it is
    // not a failure, which is why the result is deliberately discarded.
    (void)registry->add_shader_search_path(host_root);

    // Step 2: describe the pipeline. Everything not set below keeps the
    // catalogue's own defaults, which is the point of the defaults being
    // byte-identical — a host that says nothing about depth inherits the
    // engine's reversed-Z convention instead of discovering it.
    gpu::HostGraphicsPipelineDesc desc;
    desc.name = "PlaygroundTint";
    desc.vertex_shader = "fullscreen.vert.spv";        // engine root (root 0)
    desc.fragment_shader = "playground_tint.frag.spv"; // this game's root
    // A fullscreen triangle is generated from SV_VertexID, so there is no
    // vertex buffer and nothing to cull.
    desc.vertex_layout = gpu::VertexLayout::None;
    desc.cull_mode = SDL_GPU_CULLMODE_NONE;
    // No depth attachment at all: `depth_target = false` forces test and write
    // off whatever else the desc says, exactly as it does for a catalogue row.
    desc.depth_target = false;
    desc.color = gpu::TargetFormat::SceneColor; // RGBA16F linear HDR
    // AdditivePreserveAlpha, not Alpha: the scene target's DESTINATION ALPHA is
    // metadata later passes read, and a fullscreen resolve must not grow it.
    // The shader therefore writes premultiplied RGB and its alpha output is
    // ignored by the blend state.
    desc.blend_mode = gpu::BlendMode::AdditivePreserveAlpha;

    pipeline_ = registry->register_graphics_pipeline(std::move(desc));
    if (!pipeline_.is_valid()) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "CustomTintPass: pipeline registration failed. Is '{}' staged next to the executable? "
                          "(bracken_compile_shaders writes it there at build time)",
                          host_root);
        return false;
    }

    // Step 3: the hook. Order 0 with nothing else registered is trivial, but it
    // is the field that makes two hooks in one slot a total order rather than a
    // race between two registrations.
    hook_ = renderer.render_hooks().add(scene::RenderHookSlot::AfterOpaque, "PlaygroundTint", /*order=*/0,
                                        [this](const scene::RenderHookContext& ctx) { draw(ctx); });
    if (!hook_.is_valid()) {
        registry->unregister_pipeline(pipeline_);
        pipeline_ = {};
        BRACKEN_LOG_ERROR(LOG_TAG, "CustomTintPass: render hook was refused");
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "CustomTintPass: installed (AfterOpaque), host shaders at {}", host_root);
    return true;
}

void CustomTintPass::uninstall(bracken::scene::SceneRenderer& renderer) {
    if (hook_.is_valid()) {
        (void)renderer.render_hooks().remove(hook_);
        hook_ = {};
    }
    if (pipeline_.is_valid()) {
        if (bracken::gpu::PipelineRegistry* registry = renderer.pipeline_registry(); registry != nullptr) {
            registry->unregister_pipeline(pipeline_);
        }
        pipeline_ = {};
    }
}

void CustomTintPass::draw(const bracken::scene::RenderHookContext& ctx) {
    if (ctx.cmd == nullptr || ctx.pipelines == nullptr || ctx.scene_color == nullptr) {
        return;
    }

    // RE-FETCH BY HANDLE, EVERY FRAME. Caching the GPUPipeline* at install time
    // is the one mistake this API cannot protect you from: a shader hot reload
    // (BRACKEN_HOT_RELOAD=1) destroys every pipeline and rebuilds it, and the
    // cached pointer becomes a use-after-free the first time a .spv is saved.
    // The lookup is a generation check plus one map probe.
    bracken::gpu::GPUPipeline* pipeline = ctx.pipelines->get_pipeline(pipeline_);
    if (pipeline == nullptr) {
        return; // mid-hot-reload, or the pipeline failed to rebuild
    }

    SDL_GPUColorTargetInfo color{};
    color.texture = ctx.scene_color;
    // LOAD, never CLEAR: the frame's opaque and transparent geometry is already
    // in this texture and this pass adds to it.
    color.load_op = SDL_GPU_LOADOP_LOAD;
    color.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(ctx.cmd, &color, 1, nullptr);
    if (pass == nullptr) {
        BRACKEN_LOG_ERROR(LOG_TAG, "CustomTintPass: SDL_BeginGPURenderPass failed: {}", SDL_GetError());
        return;
    }
    pipeline->bind(pass);
    // Fragment uniform slot 0 == `register(b0, space3)` in the HLSL. SDL3 GPU's
    // spaces are fixed: space1 is vertex uniforms, space3 is fragment uniforms.
    const glm::vec4 tint = tint_;
    SDL_PushGPUFragmentUniformData(ctx.cmd, 0, &tint, static_cast<Uint32>(sizeof(tint)));
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    // NOT OPTIONAL. SDL3 GPU offers no way to ask whether a pass is open, so a
    // hook that returns with one still open corrupts every later pass in the
    // frame and fails inside the driver, far from here.
    SDL_EndGPURenderPass(pass);

    ++executions_;
}

} // namespace playground
