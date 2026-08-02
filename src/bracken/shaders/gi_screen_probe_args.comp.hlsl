// gi_screen_probe_args.comp.hlsl - turn the march pass's miss counter into an
// SDL_GPUIndirectDispatchCommand. One thread, three uints, and the whole reason
// R4a put `ComputePass::dispatch_indirect` in the gpu/ layer.
//
// Phase R4b of docs/rt_lighting_plan.md §5.5.
//
// THE POINT IS NOT THE SAVING, IT IS THE SHAPE. The BVH pass's workgroup count
// is a number only the GPU knows: it is how many of this frame's 182400 probe
// rays the screen march could not resolve, which depends on the depth buffer.
// The CPU's options without indirection are (a) dispatch for the worst case and
// have 90% of the threads exit on their first instruction, or (b) read the
// counter back, which costs a fence and a frame of latency in the middle of the
// frame. Neither is a trade worth making when SDL3 exposes
// SDL_DispatchGPUComputeIndirect and always has.
//
// It is also, deliberately, THE FIRST RUNTIME USE of that wrapper. R4a landed
// it untested-at-runtime and said so; this pass is what exercises it.

#include "gi_screen_probe.hlsli"

// ---------------------------------------------------------------------------
// Uniforms (set 2, b0) — mirrors gpu::GiScreenProbeArgsUniforms.
// ---------------------------------------------------------------------------
[[vk::binding(0, 2)]]
cbuffer GiScreenProbeArgsUniforms : register(b0, space2) {
    uint missCapacity;
    uint threadsPerGroup;
    uint _argsPad0;
    uint _argsPad1;
};

// --- read-only storage buffers (1) ---
// The miss buffer, read for its header word alone. It is the SAME buffer the
// BVH pass reads its records from, which is what keeps that pass at eight
// read-only storage buffers instead of nine (SDL3 GPU's ceiling).
[[vk::binding(0, 0)]] ByteAddressBuffer missBuf : register(t0, space0);

// --- read-write (set 1) ---
// Exactly SDL_GPUIndirectDispatchCommand: three Uint32 group counts at offset
// 0. No count, no stride — one dispatch (gpu/gpu_compute.hpp records the
// requirement read off SDL_gpu.h).
[[vk::binding(0, 1)]] RWByteAddressBuffer dispatchArgs : register(u0, space1);

[numthreads(1, 1, 1)]
void CSMain() {
    uint count = min(missBuf.Load(0), missCapacity);
    uint groups = (count + threadsPerGroup - 1u) / threadsPerGroup;
    dispatchArgs.Store3(0, uint3(groups, 1u, 1u));
}
