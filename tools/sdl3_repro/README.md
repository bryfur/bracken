# SDL3 GPU 3D-storage-image repro

Standalone, ~150 lines of C against **system** SDL3 — no engine, no CMake target,
not built by anything. It exists because the raster GI path (voxel cone tracing,
`docs/rt_lighting_plan.md` §7) faulted inside the NVIDIA driver with no
diagnostic, and reasoning about it produced four wrong answers in a row.

## What it settled

The fault was **`SDL_GenerateMipmapsForGPUTexture` on a 3D texture**. Everything
else about 3D storage images works: creating one mipmapped, binding it
read-write at mip 0, writing it from compute, and every usage-flag combination
tried. The engine now builds its mip chain with a compute shader
(`voxel_mip.comp.hlsl`) and never calls SDL's generator on a volume.

Two things about the method, both worth reusing:

* **The fault is ASYNCHRONOUS.** It surfaces at whichever dispatch runs next, so
  bisecting by removing a suspect just moves the reported line onto the next
  pass. Bisect by removing everything AFTER a suspect instead.
* **Check the entry point.** An early run of this program reported the engine's
  own SPIR-V failing to create a pipeline, which looked like a toolchain problem
  and was not — the harness hardcoded `"main"` while the engine's HLSL entry is
  `CSMain`.

## Build and run

```sh
cd tools/sdl3_repro
for f in write3d write2d write3d_as2d; do glslc -fshader-stage=compute $f.comp -o $f.spv; done
cc repro.c -o repro $(pkg-config --cflags --libs sdl3)
./repro                      # all cases
ONLY=3d_volume ./repro       # one case
USAGE=0x60 ONLY=3d_volume ./repro   # a specific usage-flag combination
```

Each case builds its own device and resources so an early fault cannot hide a
later one, and prints the case name **before** running it — the last line printed
names the case that crashed.
