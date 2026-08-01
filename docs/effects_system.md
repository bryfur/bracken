# Particle Effects System

Data-driven mesh-particle effects. The engine owns the definition structs, the
runtime simulation, and the rendering; **parsing effect files (JSON or
otherwise) is the host's job** — the engine never does file I/O for effects.

| Layer | Files | Namespace |
|---|---|---|
| Definitions (pure data) | `src/bracken/effects/effect_definition.hpp` | `bracken` |
| Runtime simulation | `src/bracken/effects/effect_system.hpp/.cpp` | `bracken::systems` |
| Rendering | `src/bracken/effects/effect_renderer.hpp/.cpp` | `bracken::render` |

When using `Application`, the host only touches the first layer: build
`EffectDefinition`s, then emit spawns through the frame sink
(`scene::FrameSink::spawn_effect`). **`Application` owns the `EffectSystem`**
— it is sim-stage state, spawned/updated/snapshotted inside the sim job — and
`SceneRenderer` owns only the `EffectRenderer`, which draws the packet's
snapshot. The host never calls into either.

## Spawning effects (host side)

```cpp
#include "bracken/effects/effect_definition.hpp"

// Definitions must outlive every effect spawned from them — the runtime
// stores raw pointers into them. Keep them in a long-lived registry.
bracken::EffectDefinition fireball = make_fireball_definition();

// When the event happens, while describing the frame in on_simulate():
sim.frame().spawn_effect(&fireball, position, direction);
```

The spawn is a *command in the packet*, not an immediate call. The engine's
**sim-stage tail** — after `on_simulate` returns, before `seal()` — drains the
frame's spawns into the `EffectSystem` in deterministic (lane, emission)
order, runs `EffectSystem::update` on the job pool with the terrain sampler,
and writes the resulting effect-instance snapshot into the packet
(`Application::run_sim_frame`, `application.cpp`). Each packet's spawns are
consumed exactly once — there is no persistent spawn queue, so emit once per
event.

The renderer then draws **that snapshot** and never the live `EffectSystem`,
which the next sim frame is already mutating. `SceneRenderer::render_frame`
pre-uploads any models the snapshot's emitters reference. Particle models are
looked up by name in the `ModelManager` — load or `register_model` them
before the first spawn so the first frame can render. A model registered
late is picked up automatically: unresolved lookups are retried every frame
(they are never cached; the miss is warned once per emitter definition).

Driving `EffectSystem` directly (without `Application`) is also supported:
`spawn_effect(definition, position, direction)`, then
`update(dt)` or `update(dt, terrain_height_fn)` per frame, then render from
`get_effects()`. The update overload taking a terrain sampler is templated —
any callable `float(float x, float z)` works, with no `std::function`
overhead in the per-particle loop. Note that `update()` parallelizes
particle integration/compaction internally on the job pool (spawning stays
serial — results are bit-identical at any worker count), so the terrain
sampler is invoked from job workers and must be safe to call concurrently
(a pure function of `(x, z)`; the engine's heightmap sampler qualifies).
Explicit-pool forms `update(JobSystem&, dt[, fn])` exist for caller-owned
pools; the plain forms service `JobSystem::instance()`.

## Definition structures

All in `effects/effect_definition.hpp` (verbatim fields, defaults in parentheses):

- **`EffectDefinition`** — `name`, `emitters` (vector), `duration` (−1.0 =
  emitter-driven; when ≥ 0, emitters stop spawning once the effect has been
  alive this long and the effect completes as soon as its remaining
  particles die).
- **`EmitterDefinition`** — `name`; `model` (ModelManager name — every
  particle renders as this mesh); `spawn_mode` (`BURST` = all at once after
  `delay`, `CONTINUOUS` = `spawn_rate`/sec); `spawn_count` (1); `spawn_rate`
  (10.0; **must be > 0 and finite** — invalid rates are warned once at spawn
  and the emitter stays inert); `spawn_volume` (vec3 box half-extents around
  the emitter origin — each particle spawns at origin + a uniform random
  offset inside the box; `{0,0,0}` = point emitter, the default. Area
  effects — rain, snow, fog banks — set this instead of spawning many point
  effects); `particle_lifetime` (1.0); `velocity`;
  `rotation`; `appearance`; `delay` (0.0); `duration` (−1.0 = run for
  `particle_lifetime`).
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

`effects/effect_system.hpp`: `EffectInstance` → `EmitterInstance` →
`Particle`. Instances hold `const` pointers back into their definitions
(hence the lifetime rule above). Per `update`:

1. Emitters age; `BURST` fires once after `delay`, `CONTINUOUS` spawns on a
   `1/spawn_rate` cadence while active (a non-positive or non-finite
   `spawn_rate` never spawns — validated with a warning at spawn, and the
   update loop guards against it independently). An effect-level
   `EffectDefinition::duration` ≥ 0 gates all spawning off once the effect
   has outlived it.
2. Each particle integrates velocity/gravity/drag, applies its velocity-type
   behavior, advances rotation, and evaluates appearance curves.
3. With a terrain sampler, particles are clamped to terrain height (no
   bounce — they slide on the ground).
4. Dead particles are erased; completed effects are compacted away (all
   emitters done and particles dead, or effect-level duration elapsed and
   particles dead).

Introspection: `effect_count()`, `particle_count()`, `clear()`.

## Rendering

`EffectRenderer` draws particles GPU-instanced through the `Particle`
pipeline — `particle.vert` + `particle.frag`, unlit additive into the HDR
scene target, so bright particles bloom naturally. The blend is configured
`(SRC_ALPHA, ONE)`; the fragment shader writes alpha 1.0, so the effective
color blend is `(ONE, ONE)` (changing the shader's output alpha would change
the blend — opacity is instead folded into the emitted RGB). Per-particle
`{transform, tint+opacity}` records are packed into a per-frame storage
buffer (uploaded in `SceneRenderer`'s consolidated `FrameUploads` copy pass),
then each (emitter, mesh) batch is a single instanced draw with a small
instance-offset uniform — camera uniforms are pushed once per pass.

## Effect files

Serialization format and loading live with the host. The conventional shape is
one JSON file per `EffectDefinition` mirroring the struct fields — but the
engine only ever sees the populated structs, so any source works (JSON files,
code-built definitions in tests, editor output).
