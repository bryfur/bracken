#pragma once

#include "bracken/animation/animation_types.hpp"
#include <glm/glm.hpp>
#include <span>

namespace bracken::animation {

// Two-bone IK solver. Modifies bone_matrices and world_transforms in place.
// Spans must point to contiguous storage of at least MAX_BONES glm::mat4.
void solve_two_bone_ik(std::span<glm::mat4> bone_matrices, std::span<glm::mat4> world_transforms,
                       const Skeleton& skeleton, int upper_idx, int lower_idx, int end_idx, const glm::vec3& target,
                       const glm::vec3& pole_hint);

// High-level foot IK: adjusts pelvis and solves both legs. Terrain offsets are
// world-space heights; `scale` (the instance's uniform world scale) converts
// them into model space — the instance transform itself never participates.
void apply_foot_ik(std::span<glm::mat4> bone_matrices, std::span<glm::mat4> world_transforms, const Skeleton& skeleton,
                   const FootIKData& ik, float scale, float left_terrain_offset, float right_terrain_offset);

// Apply procedural lean to all spine descendants. Precondition: joints are
// ordered parents-before-children (descendants are marked in a single
// index-order pass) — guaranteed for skeletons from ModelLoader::load_glb.
void apply_body_lean(std::span<glm::mat4> bone_matrices, std::span<glm::mat4> world_transforms,
                     const Skeleton& skeleton, int spine_index, float forward_lean, float lateral_lean);

} // namespace bracken::animation
