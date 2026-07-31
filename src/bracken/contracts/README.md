# `bracken/contracts/` — the leaf layer

Pure-data types that two or more layers of the engine have to agree on.

**The rule, enforced by `src/bracken/core/module_layering.test.cpp`:**

> A header under `contracts/` may include `bracken/core/*`, other
> `bracken/contracts/*` headers, glm, and the C++ standard library. Nothing
> else. In particular it may not include `gpu/`, `render/`, `scene/`,
> `animation/`, `systems/`, `effects/`, `dev/` or a root header.

That is the whole definition of the layer: it sits directly above `core/` and
below everything else, so any module may include from it and it may include from
nothing above it. It is what makes the engine's header graph a DAG.

## Why these files, and why their namespaces did not change

Each of these types was filed under whichever module happened to author it, not
under the layer that needs it — `EnvironmentState` under `render/` even though
`FramePacket` carries one, `CameraState` and `Frustum` under `scene/` even though
every sub-renderer takes them, `GraphicsSettings` and `RenderStats` at the root
even though `scene/` reads and writes them. Each of those misfilings *reversed*
a dependency edge, and together they were the reason the production module graph
had a six-module strongly-connected component.

A header's directory and its namespace are independent, so the namespaces are
deliberately unchanged: `EnvironmentState` is still `bracken::render::`,
`CameraState` and `Frustum` are still `bracken::scene::`. Moving a file here is
an include-line rewrite, never a rename — no host code changes except the path.

## What is here

| Header | Namespace | Contract |
|---|---|---|
| `model_handle.hpp` | `bracken` | `ModelHandle` / `INVALID_MODEL_HANDLE` — how a frame command names a mesh |
| `environment.hpp` | `bracken::render` | `EnvironmentState` — sun/sky/fog, linear-space, carried by the packet |
| `camera_state.hpp` | `bracken::scene` | `CameraState` — the view/projection the frame is drawn from |
| `frustum.hpp` / `.cpp` | `bracken::scene` | `Plane` / `Frustum` — culling primitive shared by every pass |
| `velocity_math.hpp` | `bracken::scene` | reprojection helpers for motion vectors |
| `light.hpp` | `bracken::render::lighting` | `PointLight` / `SpotLight` / `LightHeader` — the host-facing light contract. The cluster and shadow-atlas *budgets* stay renderer-internal in `render/lighting/light.hpp` |
| `graphics_settings.hpp` / `.cpp` | `bracken` | `GraphicsSettings` — host-set quality knobs |
| `render_stats.hpp` | `bracken` | `RenderStats` / `FrameDiagnostics` — what the renderer reports back |

## What does *not* belong here

Anything with behaviour that owns a resource, anything that needs a GPU or SDL
type, and anything a single module is the only consumer of. A type that only
`render/` uses belongs in `render/`; putting it here to "share" it is how the
layer stops being a leaf.
