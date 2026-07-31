#pragma once

#include <cstdint>

namespace bracken::scene {

class FramePacket;

/**
 * A 64-bit fingerprint of everything a sealed FramePacket describes.
 *
 * WHY THIS EXISTS. The playground smoke run is the engine's integration test,
 * but its render statistics and its screenshot are both downstream of the GPU,
 * so neither is a trustworthy regression gate on its own: driver-level
 * floating-point differences move pixels, and the auto-exposure histogram
 * readback moves `auto_ev` between otherwise identical runs. The digest is
 * taken on the CPU side of that boundary — it covers exactly what the host
 * described and what the engine's sim-stage tail appended, and nothing that a
 * GPU or driver can perturb.
 *
 * WHAT IT COVERS. Every sealed command, in packet order: model and skinned
 * draws (including packed bone-palette contents), billboards, debug lines,
 * effect spawns, point and spot lights, UI commands (text contents included),
 * the camera and environment state, the skybox/ground/grass toggles, the
 * effect-instance snapshot, and the scene/sky clocks. A swapped flag, a
 * transposed light field, a dropped draw, a shifted camera or a changed bone
 * matrix all move the value.
 *
 * WHAT IT DELIBERATELY EXCLUDES.
 *  - Raw pointers. `EffectSpawnCommand::definition` and
 *    `EffectEmitterSnapshot::definition` are host-owned addresses that ASLR
 *    changes on every launch; their `name` strings are hashed instead, which
 *    is what actually identifies the asset.
 *  - `dt`, `time` and `frame_index`. Wall-clock-derived under the default
 *    timing mode, so including them would make the digest unstable for reasons
 *    that have nothing to do with the frame's content. Pin the timestep with
 *    Application::set_fixed_timestep when you want run-to-run reproducibility.
 *
 * DETERMINISM CONTRACT. Floats are hashed by bit pattern rather than value, so
 * the digest distinguishes +0.0 from -0.0 and is stable across NaN payloads.
 * The traversal order is packet order, which the seal already fixes to a
 * deterministic (lane-index, allocation-order) sequence — so the digest is
 * bit-identical at any worker count, the same discipline the Prepare-stage
 * fan-out follows.
 *
 * Tier: worker-safe on a SEALED packet (pure read).
 */
[[nodiscard]] uint64_t packet_digest(const FramePacket& packet);

} // namespace bracken::scene
