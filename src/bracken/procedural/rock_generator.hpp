#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

namespace bracken {
struct Model;
}

namespace bracken::procedural {

/**
 * Parameters for procedural rock generation.
 * Based on the SDF + marching cubes approach from
 * przemyslawzaworski/Unity-Procedural-Rock-Generation (MIT).
 */
struct RockParams {
    // Shape seed. Hashed internally, so the full uint32 range yields distinct
    // rocks (seeds are reduced to 2^20 effective shape variants).
    uint32_t seed = 880;
    int resolution = 40;                          // Grid resolution (40 = fast, 80 = detailed)
    float scale = 1.0f;                           // Output mesh scale
    int steps = 20;                               // SDF iteration count (more = more complex shape)
    float smoothness = 0.05f;                     // Boolean blend smoothness
    float displacement_scale = 0.15f;             // Surface noise amplitude
    float displacement_spread = 10.0f;            // Surface noise frequency
    glm::vec4 color = {0.5f, 0.48f, 0.45f, 1.0f}; // Base rock color (vertex-colored; no texture support)
};

/**
 * Procedural rock mesh generator using SDF + marching cubes.
 */
class RockGenerator {
public:
    [[nodiscard]] static std::unique_ptr<Model> generate(const RockParams& params);

    // Presets
    [[nodiscard]] static std::unique_ptr<Model> generate_boulder(uint32_t seed = 0);
    [[nodiscard]] static std::unique_ptr<Model> generate_slate(uint32_t seed = 0);
    [[nodiscard]] static std::unique_ptr<Model> generate_spire(uint32_t seed = 0);
    [[nodiscard]] static std::unique_ptr<Model> generate_cluster(uint32_t seed = 0);
    [[nodiscard]] static std::unique_ptr<Model> generate_mossy(uint32_t seed = 0);
};

} // namespace bracken::procedural
