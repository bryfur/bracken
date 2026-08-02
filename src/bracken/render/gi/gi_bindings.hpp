#pragma once

/**
 * What a consumer of the probe field needs handed to it: the three atlas
 * bindings `shaders/gi_sample.hlsli` names, and the per-cascade window corners
 * its addressing needs.
 *
 * A struct rather than four more parameters on `TerrainRenderer::render` and
 * `GrassRenderer::render`, whose signatures already carry eleven; and a struct
 * rather than a `ProbeVolume*`, because `render/gi/probe_volume.hpp` drags the
 * whole `gpu/` layer in and the sub-renderers only ever need these bytes.
 *
 * Tier: main-thread (it holds live GPU handles for exactly one frame).
 */

#include "bracken/render/gi/gi_math.hpp"

#include <SDL3/SDL_gpu.h>
#include <array>
#include <glm/glm.hpp>

namespace bracken::render::gi {

struct GiBindings {
    /**
     * Irradiance, visibility, probe metadata — IN THE SLOT ORDER every scene
     * fragment shader declares them, because they are bound as one contiguous
     * run. The first two carry the linear clamp sampler (their border rings
     * exist so a bilinear tap at a payload edge wraps the octahedron); the
     * third carries the point clamp sampler and is read with Load().
     */
    std::array<SDL_GPUTextureSamplerBinding, 3> textures{};

    /// `render::gi::cascade_base()` per cascade, xyz used, w unused. Sent to
    /// the shaders rather than re-derived there from the camera position, so a
    /// shader and `ProbeVolume` can never disagree about which probe answers
    /// for which point in the world.
    std::array<glm::ivec4, CASCADE_COUNT> cascade_base{};

    /// 1 only when the field exists, is placed, and `gi_mode != GiMode::Off`.
    /// Every shader gates every tap on this; 0 is the pre-R1c ambient exactly.
    float enabled = 0.0f;
};

} // namespace bracken::render::gi
