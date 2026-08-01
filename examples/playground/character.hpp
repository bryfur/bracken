#pragma once

#include "bracken/model_loader.hpp"
#include <memory>

namespace playground {

// These models are built in code (the medieval village itself is loaded from
// generated .glb files — see tools/build_medieval_assets.py).
// The character is a blocky biped whose joints use the engine's default
// FootIK bone names, with four authored clips: Idle, Walk, Run, Attack.
std::unique_ptr<bracken::Model> build_character_model();

// Emissive gold sphere (the collectible).
std::unique_ptr<bracken::Model> build_orb_model();

// Polished chrome sphere (metallic 1.0, near-zero roughness) placed by the
// spawn clearing — the specular-IBL proof piece: it mirrors the prefiltered
// sky, sun disc included.
std::unique_ptr<bracken::Model> build_chrome_sphere_model();

// Small emissive octahedron used as the particle mesh for effects.
std::unique_ptr<bracken::Model> build_shard_model();

/** A small carried lantern — the bone-attachment demo's prop. Sized and lit
 *  like something a person holds, not like the pickup orbs. */
std::unique_ptr<bracken::Model> build_lantern_model();

// Soft grey blob used as the chimney-smoke particle mesh.
std::unique_ptr<bracken::Model> build_smoke_puff_model();

// Thin crossed vertical quads — the falling rain-streak particle mesh.
std::unique_ptr<bracken::Model> build_rain_streak_model();

// Tiny droplet sphere — the rain-splash burst particle mesh.
std::unique_ptr<bracken::Model> build_splash_drop_model();

} // namespace playground
