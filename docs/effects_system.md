# Particle Effects System

Data-driven mesh-particle effects. The engine owns the definition structs, the
runtime simulation, and the rendering; **parsing effect files (JSON or
otherwise) is the host's job** — the engine never does file I/O for effects.

| Layer | Files | Namespace |
|---|---|---|
| Definitions (pure data) | `src/engine/effect_definition.hpp` | `mmo::engine` |
| Runtime simulation | `src/engine/systems/effect_system.hpp/.cpp` | `mmo::engine::systems` |
| Rendering | `src/engine/render/effect_renderer.hpp/.cpp` | `mmo::engine::render` |

When using `Application`/`SceneRenderer`, the host only touches the first
layer: build `EffectDefinition`s, then queue spawns on the `RenderScene`. The
`SceneRenderer` owns the `EffectSystem` and `EffectRenderer` internally —
it updates, terrain-clamps, and draws particles as part of `render_frame`.

## Spawning effects (host side)

```cpp
#include "engine/effect_definition.hpp"

// Definitions must outlive every effect spawned from them — the runtime
// stores raw pointers into them. Keep them in a long-lived registry.
mmo::engine::EffectDefinition fireball = make_fireball_definition();

// Per frame, in on_render():
scene.add_particle_effect_spawn(&fireball, position, direction, range);
```

`SceneRenderer::render_frame` drains the spawn queue each frame (this is why
`RenderScene` is passed by non-const reference), pre-uploads any models the
effect references, and simulates + renders from then on. Particle models are
looked up by name in the `ModelManager` — load or `register_model` them
before the first spawn.

Driving `EffectSystem` directly (without `SceneRenderer`) is also supported:
`spawn_effect(definition, position, direction, range)`, then
`update(dt)` or `update(dt, terrain_height_fn)` per frame, then render from
`get_effects()`. The update overload taking a terrain sampler is templated —
any callable `float(float x, float z)` works, with no `std::function`
overhead in the per-particle loop.

## Definition structures

All in `effect_definition.hpp` (verbatim fields, defaults in parentheses):

- **`EffectDefinition`** — `name`, `emitters` (vector), `duration` (1.0),
  `loop` (false), `default_range` (100.0, used when spawn passes `range < 0`).
- **`EmitterDefinition`** — `name`; `particle_type` (`"mesh"`; `"sprite"` is
  declared but **not implemented** — only mesh particles render); `model`
  (ModelManager name); `spawn_mode` (`BURST` = all at once after `delay`,
  `CONTINUOUS` = `spawn_rate`/sec); `spawn_count` (1); `spawn_rate` (10.0);
  `particle_lifetime` (1.0); `velocity`; `rotation`; `appearance`;
  `delay` (0.0); `duration` (−1.0 = run for `particle_lifetime`).
- **`VelocityDefinition`** — `type`, `speed` (100), `direction`,
  `spread_angle` (degrees), `gravity` (vec3), `drag` (0–1), plus
  orbital params (`orbit_radius`, `orbit_speed` rot/sec, `orbit_height_base`,
  `height_variation`) and arc params (`arc_radius`, `arc_height_base`,
  `arc_height_amplitude`, `arc_tilt_amplitude`).
- **`RotationDefinition`** — `initial_rotation` (degrees),
  `rotation_rate` (degrees/sec), `face_velocity` (orient along movement).
- **`AppearanceDefinition`** — `scale_over_lifetime` / `opacity_over_lifetime`
  (curves), `color_tint`, `color_end` + `use_color_gradient`.

### Velocity types (`VelocityType`)

| Type | Behavior |
|---|---|
| `DIRECTIONAL` | Move along the spawn direction, ± `spread_angle` random cone |
| `RADIAL` | Shoot outward in random directions (bursts, explosions) |
| `ORBITAL` | Orbit the spawn point at `orbit_radius`, optional vertical oscillation |
| `CUSTOM` | Fixed world-space `direction` regardless of spawn direction |
| `ARC` | Swing along an arc around the spawn point (melee slashes) |

### Curves (`CurveType`)

`Curve { type, start_value, end_value, fade_start }` evaluated over normalized
particle age (0→1): `CONSTANT`, `LINEAR`, `EASE_IN`, `EASE_OUT`,
`EASE_IN_OUT`, and `FADE_OUT_LATE` (hold `start_value` until `fade_start`,
default 0.8, then blend to `end_value`).

## Runtime model

`systems/effect_system.hpp`: `EffectInstance` → `EmitterInstance` →
`Particle`. Instances hold `const` pointers back into their definitions
(hence the lifetime rule above). Per `update`:

1. Emitters age; `BURST` fires once after `delay`, `CONTINUOUS` spawns on a
   `1/spawn_rate` cadence while active.
2. Each particle integrates velocity/gravity/drag, applies its velocity-type
   behavior, advances rotation, and evaluates appearance curves.
3. With a terrain sampler, particles are clamped to terrain height (no
   bounce — they slide on the ground).
4. Dead particles are erased; completed effects are compacted away.

Introspection: `effect_count()`, `particle_count()`, `clear()`.

## Rendering

`EffectRenderer` draws particles GPU-instanced through the `Particle`
pipeline — `particle.vert` + `particle.frag`, unlit additive (ONE/ONE) into
the HDR scene target, so bright particles bloom naturally. Per-particle
`{transform, tint+opacity}` records are packed into a per-frame storage
buffer (uploaded in `SceneRenderer`'s consolidated `FrameUploads` copy pass),
then each (emitter, mesh) batch is a single instanced draw with a small
instance-offset uniform — camera uniforms are pushed once per pass.

## Effect files

Serialization format and loading live with the host. The conventional shape is
one JSON file per `EffectDefinition` mirroring the struct fields — but the
engine only ever sees the populated structs, so any source works (JSON files,
code-built definitions in tests, editor output).
