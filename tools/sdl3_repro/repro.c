// Minimal SDL3 GPU repro: does a compute dispatch that writes a 3D storage
// image work, and if not, what binding does it actually want?
//
// Four cases, each fully independent (own device, own resources), run in one
// process so a fault in an early one does not hide a later one — the process
// reports which case it is entering BEFORE running it and flushes, so the last
// line printed names the case that crashed.
//
//   2d          control: 2D texture, uimage2D shader, one dispatch
//   3d_volume   3D texture bound as a whole volume, uimage3D shader
//   3d_layer0   3D texture, uimage2D shader, layer = 0
//   3d_slices   3D texture, uimage2D shader, one pass per Z slice (layer = z)
//
// Build: cc repro.c -o repro $(pkg-config --cflags --libs sdl3)

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define RES 32

static void *read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_size = (size_t)n;
    return buf;
}

static SDL_GPUComputePipeline *make_pipeline(SDL_GPUDevice *dev, const char *spv) {
    size_t size = 0;
    void *code = read_file(spv, &size);
    if (!code) return NULL;
    SDL_GPUComputePipelineCreateInfo info = {0};
    info.code = code;
    info.code_size = size;
    info.entrypoint = "main";
    info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    info.num_readwrite_storage_textures = 1;
    info.threadcount_x = 4;
    info.threadcount_y = 4;
    info.threadcount_z = 4;
    SDL_GPUComputePipeline *p = SDL_CreateGPUComputePipeline(dev, &info);
    free(code);
    if (!p) fprintf(stderr, "  pipeline failed: %s\n", SDL_GetError());
    return p;
}

static SDL_GPUTextureUsageFlags g_usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
static const char *g_entry = "main";

static SDL_GPUTexture *make_texture(SDL_GPUDevice *dev, bool three_d) {
    SDL_GPUTextureCreateInfo t = {0};
    t.type = three_d ? SDL_GPU_TEXTURETYPE_3D : SDL_GPU_TEXTURETYPE_2D;
    t.format = SDL_GPU_TEXTUREFORMAT_R32_UINT;
    t.usage = g_usage;
    t.width = RES;
    t.height = RES;
    t.layer_count_or_depth = three_d ? RES : 1;
    t.num_levels = 1;
    SDL_GPUTexture *tex = SDL_CreateGPUTexture(dev, &t);
    if (!tex) fprintf(stderr, "  texture failed: %s\n", SDL_GetError());
    return tex;
}

// Returns true if the case completed without the process dying.
static bool run_case(const char *name, const char *spv, bool three_d, int slices,
                     Uint32 threads_z) {
    printf("== case %-10s ", name);
    fflush(stdout);

    SDL_GPUDevice *dev = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, NULL);
    if (!dev) { printf("device failed: %s\n", SDL_GetError()); return false; }

    printf("[usage=0x%02x supports=%d] ", g_usage,
           SDL_GPUTextureSupportsFormat(dev, SDL_GPU_TEXTUREFORMAT_R32_UINT,
                                        three_d ? SDL_GPU_TEXTURETYPE_3D : SDL_GPU_TEXTURETYPE_2D,
                                        g_usage));
    fflush(stdout);

    size_t size = 0;
    void *code = read_file(spv, &size);
    if (!code) { SDL_DestroyGPUDevice(dev); return false; }
    SDL_GPUComputePipelineCreateInfo info = {0};
    info.code = code;
    info.code_size = size;
    info.entrypoint = g_entry;
    info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    info.num_readwrite_storage_textures = 1;
    info.threadcount_x = threads_z == 4 ? 4 : 8;
    info.threadcount_y = threads_z == 4 ? 4 : 8;
    info.threadcount_z = threads_z;
    SDL_GPUComputePipeline *pipe = SDL_CreateGPUComputePipeline(dev, &info);
    free(code);
    if (!pipe) { printf("pipeline failed: %s\n", SDL_GetError()); SDL_DestroyGPUDevice(dev); return false; }

    SDL_GPUTexture *tex = make_texture(dev, three_d);
    if (!tex) { SDL_ReleaseGPUComputePipeline(dev, pipe); SDL_DestroyGPUDevice(dev); return false; }

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(dev);
    for (int s = 0; s < slices; ++s) {
        SDL_GPUStorageTextureReadWriteBinding b = {0};
        b.texture = tex;
        b.mip_level = 0;
        b.layer = (Uint32)s;
        b.cycle = false;
        SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &b, 1, NULL, 0);
        if (!pass) { printf("pass failed: %s\n", SDL_GetError()); break; }
        SDL_BindGPUComputePipeline(pass, pipe);
        SDL_DispatchGPUCompute(pass, RES / (threads_z == 4 ? 4 : 8),
                               RES / (threads_z == 4 ? 4 : 8),
                               threads_z == 4 ? RES / 4 : 1);
        SDL_EndGPUComputePass(pass);
    }
    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (fence) {
        SDL_WaitForGPUFences(dev, true, &fence, 1);
        SDL_ReleaseGPUFence(dev, fence);
    }

    printf("OK\n");
    fflush(stdout);
    SDL_ReleaseGPUTexture(dev, tex);
    SDL_ReleaseGPUComputePipeline(dev, pipe);
    SDL_DestroyGPUDevice(dev);
    return true;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    SDL_SetHint(SDL_HINT_GPU_DRIVER, "vulkan");
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    const char *only = getenv("ONLY");
    if (getenv("USAGE")) {
        g_usage = (SDL_GPUTextureUsageFlags)strtoul(getenv("USAGE"), NULL, 0);
    }

    if (!only || !SDL_strcmp(only, "2d"))
        run_case("2d", "write2d.spv", false, 1, 1);
    if (!only || !SDL_strcmp(only, "3d_volume"))
        run_case("3d_volume", "write3d.spv", true, 1, 4);
    if (!only || !SDL_strcmp(only, "3d_layer0"))
        run_case("3d_layer0", "write3d_as2d.spv", true, 1, 1);
    if (!only || !SDL_strcmp(only, "3d_slices"))
        run_case("3d_slices", "write3d_as2d.spv", true, RES, 1);
    // THE DECISIVE ONE: the engine's OWN compiled SPIR-V (HLSL via
    // SDL_shadercross) in this minimal harness. If the hand-written GLSL passes
    // and this faults, the difference is the shader toolchain, not SDL3.
    if (only && !SDL_strcmp(only, "engine")) {
        g_entry = "CSMain"; // the engine's HLSL entry point, not GLSL's "main"
        run_case("engine", "engine_clear.spv", true, 1, 4);
    }

    SDL_Quit();
    printf("done\n");
    return 0;
}
