#pragma once

// bracken.hpp — the supported surface.
//
// This is an umbrella over the headers a game is expected to include: the
// exact set the reference consumer (`examples/playground`) uses, nothing more.
// Including it gives you `Application`, the frame-description API, the
// sub-renderers a host configures, animation, physics, procedural generation
// and the dev overlay.
//
// **What "supported" means here, and what it does not.** It means: these are
// the headers to reach for, they are the ones the example uses, and they are
// the ones `public_headers.test.cpp` keeps free of third-party leakage
// (no Jolt, no ImGui, no tinygltf, no stb_ in this include tree — see
// `<module>/internal/` below). It is **not** a stability promise. The engine
// is pre-1.0 and this release changed six of these signatures; publishing an
// ABI or API contract on top of that would be signalling, not ergonomics.
// Expect breaks, expect them to be compiler-caught, and read the phase notes
// in `docs/refactor_plan.md` when they happen.
//
// **Public vs internal.** A header under `src/bracken/<module>/internal/` is
// not part of this surface and may include third-party headers freely
// (`physics/internal/jolt_job_system.hpp` includes Jolt). Everything else
// under `src/bracken/` is reachable by a host, but only what is listed below
// is *meant* to be: the rest is engine plumbing that happens to be
// header-visible. If you find yourself including something that is not here,
// that is a signal the engine is missing an API — say so.
//
// **Cost.** This pulls in the renderer and physics headers whether or not a
// TU needs them. It exists for discoverability and for the getting-started
// path; a translation unit in a build you care about should include the two
// or three headers it actually uses. There is deliberately no precompiled
// header behind this (see `docs/refactor_plan.md` §6).
//
// Include root is `src/`, so this is `#include "bracken/bracken.hpp"`.

// ---------------------------------------------------------------------------
// Host lifecycle — subclass one type, override three methods
// ---------------------------------------------------------------------------
#include "bracken/application.hpp"      // Application + SimFrame / MainFrame; owns SDL and the pipelined loop

// ---------------------------------------------------------------------------
// Describing a frame — what the host writes each simulation step
// ---------------------------------------------------------------------------
#include "bracken/scene/frame_sink.hpp" // FrameSink: the host-writable half of FrameBuilder (draws, lights, UI, effects)
#include "bracken/contracts/environment.hpp" // EnvironmentState: sun, sky gradient, hemisphere ambient, fog — all linear-space
#include "bracken/contracts/light.hpp"  // PointLightDesc / SpotLightDesc: host light descriptors, distinct from the GPU PODs
#include "bracken/contracts/font_metrics.hpp" // FontMetrics: the frozen glyph table sim-stage text layout measures against
#include "bracken/contracts/ui_texture.hpp"   // UiTextureId: the packet-safe name for a UI texture (register it main-thread)

// ---------------------------------------------------------------------------
// The UI layer — immediate-mode declarations, solved and emitted from on_simulate
// ---------------------------------------------------------------------------
#include "bracken/ui/ui_types.hpp"   // Rect/Edges/Axis/Align/Size/BoxStyle/Style/WidgetId + id()
#include "bracken/ui/layout.hpp"     // LayoutTree: the pure row/column/grid solver
#include "bracken/ui/nine_slice.hpp" // build_nine_slice: the nine patches that tile a box
#include "bracken/ui/ui_input.hpp"   // InputFrame + InputCapture: what the UI reads, and what it took
#include "bracken/ui/ui_context.hpp" // ui::Context: containers, widgets, focus, scrolling, capture

// ---------------------------------------------------------------------------
// Assets — models, terrain heightfields
// ---------------------------------------------------------------------------
#include "bracken/model_loader.hpp"     // Model, Mesh, Material, ModelManager, ModelHandle, ModelLoadReport
#include "bracken/heightmap.hpp"        // Heightmap, HeightRange, WorldBounds: the quantized terrain heightfield
#include "bracken/heightmap_grid.hpp"   // HeightmapGrid: N x M heightmap PAGES against one shared quantization range
#include "bracken/scene/level.hpp"      // Level: placed instances, markers, terrain, water — save, load and resolve a scene

// ---------------------------------------------------------------------------
// The renderer — the frame orchestrator plus the sub-renderers a host configures
// ---------------------------------------------------------------------------
#include "bracken/scene/scene_renderer.hpp"    // SceneRenderer: drives all sub-renderers from one sealed FramePacket
#include "bracken/render/terrain_renderer.hpp" // TerrainRenderer: heightfield terrain, splat layers, tile culling
#include "bracken/render/grass_renderer.hpp"   // GrassRenderer: blade grass and instanced ground cover
#include "bracken/render/water_renderer.hpp"   // WaterRenderer: water surface + the compute wave simulation
#include "bracken/render/ibl.hpp"              // IBL: the sky prefilter chain feeding image-based lighting
#include "bracken/render/auto_exposure.hpp"    // AutoExposure: compute histogram exposure, and its manual override

// ---------------------------------------------------------------------------
// Extending the renderer from a game — your own pipelines, your own passes
// ---------------------------------------------------------------------------
#include "bracken/gpu/host_pipeline.hpp"  // HostGraphicsPipelineDesc/HostComputePipelineDesc + their handles
#include "bracken/scene/render_hook.hpp"  // RenderHookSlot/RenderHookContext/RenderHookRegistry: run your own pass

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
#include "bracken/systems/camera_controller.hpp" // CameraController + CameraModeConfig: orbit/follow, shake, focus, fov

// ---------------------------------------------------------------------------
// Animation — per-instance player OR layer rig + data-driven state machine + foot IK
//
// AnimationPlayer and AnimationRig are both supported and neither deprecates
// the other: the player is the one-clip fast path with half the memory, the rig
// is what you reach for when a character needs a second simultaneous animation
// source. See animation_rig.hpp's header comment before "unifying" them.
// ---------------------------------------------------------------------------
#include "bracken/animation/animation_types.hpp"         // Skeleton, AnimationClip, MAX_BONES
#include "bracken/animation/animation_player.hpp"        // AnimationPlayer: sampling and crossfade
#include "bracken/animation/pose.hpp"                    // BoneTransform/PoseView/BoneMask + the blend operators
#include "bracken/animation/clip_sampler.hpp"            // ClipBindings (shared per model) + ClipSampler
#include "bracken/animation/blend_space.hpp"             // BlendSpace1D/2D: a position in, weighted clips out
#include "bracken/animation/root_motion.hpp"             // extract locomotion from the clip, strip it from the pose
#include "bracken/animation/animation_rig.hpp"           // AnimationRig: N masked, weighted, Override/Additive layers
#include "bracken/animation/animation_state_machine.hpp" // AnimationStateMachine: parameters in, transitions out
#include "bracken/animation/ik_solver.hpp"               // two-bone IK, applied by the host after the animation update

// ---------------------------------------------------------------------------
// Particle effects
// ---------------------------------------------------------------------------
#include "bracken/animation/attachment.hpp"     // put a sword in a hand: bone-space attachments
#include "bracken/navigation/nav_grid.hpp"      // walkability + A* pathfinding
#include "bracken/physics/character_controller.hpp" // capsule controller: slopes, steps, jump
#include "bracken/action_map.hpp"                // named game actions bound to keys/mouse/pad, with rebinding
#include "bracken/async_model_loader.hpp"      // decode models on the pool, upload at the fence
#include "bracken/core/asset/asset_pack.hpp"   // ship assets as one file instead of thousands
#include "bracken/core/random.hpp"             // deterministic RNG: seed it, do not derive from game state
#include "bracken/audio/audio_engine.hpp"      // playback device, mixer, 3D voices — Application::audio()
#include "bracken/audio/music/song.hpp"         // Song/Track/Pattern — the data play_music() takes
#include "bracken/audio/synth/instrument.hpp"   // patches: the synthesizer's authoring surface
#include "bracken/audio/synth/render.hpp"       // bake a patch into a clip: SFX without audio assets
#include "bracken/audio/wav_writer.hpp"         // dump a rendered mix for inspection
#include "bracken/save/save_document.hpp"      // saved games: typed, versioned, atomic, checksummed
#include "bracken/effects/effect_definition.hpp" // EffectDefinition and its curves/emitters — spawned through the FrameSink

// ---------------------------------------------------------------------------
// Physics — rigid bodies over Jolt, behind a glm-only pimpl
// ---------------------------------------------------------------------------
#include "bracken/physics/physics_world.hpp" // PhysicsWorld: fixed-step bodies and ray casts; no Jolt type is visible here

// ---------------------------------------------------------------------------
// Procedural generation
// ---------------------------------------------------------------------------
#include "bracken/procedural/tree_generator.hpp" // TreeGenerator + TreeParams
#include "bracken/procedural/rock_generator.hpp" // RockGenerator + RockParams

// ---------------------------------------------------------------------------
// Engine services a host talks to directly
// ---------------------------------------------------------------------------
#include "bracken/core/logger.hpp"          // core::Logger, LogSink, BRACKEN_LOG_* — the engine's one log channel
#include "bracken/core/jobs/job_system.hpp" // core::jobs::JobSystem: the always-on pool the host may also submit to
#include "bracken/dev/dev_ui.hpp"           // dev::DevUI: the Dear ImGui overlay (a no-op stub when BRACKEN_ENABLE_DEV_UI=OFF)

// Not included on purpose:
//   bracken/testing/frame_test.hpp — a GoogleTest fixture for hosts. It belongs
//   to your test TUs, not to your game, so it stays an explicit include.
