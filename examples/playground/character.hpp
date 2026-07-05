#pragma once

#include "bracken/model_loader.hpp"
#include <memory>

namespace playground {

// All example assets are built in code — the repo ships no binary assets.
// The character is a blocky biped whose joints use the engine's default
// FootIK bone names, with four authored clips: Idle, Walk, Run, Attack.
std::unique_ptr<bracken::Model> build_character_model();

// Emissive gold sphere (the collectible).
std::unique_ptr<bracken::Model> build_orb_model();

// Small emissive octahedron used as the particle mesh for effects.
std::unique_ptr<bracken::Model> build_shard_model();

} // namespace playground
