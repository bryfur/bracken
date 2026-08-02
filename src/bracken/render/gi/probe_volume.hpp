#pragma once

#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/render/gi/gi_math.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

namespace bracken::render::gi {

namespace gpu = bracken::gpu;

/** Per-cascade accounting. Everything here is a fact about the STRUCTURE —
 *  there is no radiance to report until R1b. */
struct GiCascadeStats {
    float spacing = 0.0f;              ///< world units between adjacent probes
    glm::ivec3 base{0};                ///< absolute coord of the (0,0,0) corner probe
    std::uint32_t probes = 0;          ///< always PROBES_PER_CASCADE once allocated
    std::uint32_t probes_invalidated = 0; ///< entered the window THIS frame
    std::size_t atlas_bytes = 0;       ///< irradiance + visibility + state
};

/**
 * What the probe field costs and what it did this frame.
 *
 * `probes_invalidated` is the number that matters while nothing consumes the
 * field: it is the only observable proof the scroll is working, and it is what
 * R1b's ray budget will be spent on. A stationary camera must report 0 after
 * the first frame; a moving one must report a plane's worth, not a field's.
 */
struct GiStats {
    bool active = false;
    std::uint32_t cascades = 0;
    std::uint32_t probes = 0;
    std::uint32_t probes_invalidated = 0;
    /// Running total since init(). The per-frame figure is zero on most frames
    /// (a cascade only scrolls when the camera crosses a whole probe spacing),
    /// so a run's evidence that the scroll happened at all has to be cumulative
    /// — sampling `probes_invalidated` at an arbitrary frame proves nothing.
    std::uint64_t probes_invalidated_total = 0;
    std::size_t atlas_bytes = 0;
    std::array<GiCascadeStats, CASCADE_COUNT> per_cascade{};
};

/**
 * The scrolled probe clipmap: three camera-centred cascades of octahedral
 * irradiance + visibility probes, and the CPU addressing that says which texel
 * belongs to which point in the world.
 *
 * AFTER R1a THIS LIGHTS NOTHING. It allocates its atlases, snaps each cascade
 * to whole probe-spacing steps as the camera moves, and marks the probes that
 * entered the window as uninitialized. No ray is cast, no probe is shaded, and
 * no shader samples any of these textures — that is R1b (update/trace) and R1c
 * (consumption in the forward pass and the fog march). Splitting it this way
 * keeps the structure reviewable on its own: the addressing is the part that is
 * easy to get subtly wrong and impossible to see wrong once radiance is on top
 * of it.
 *
 * PER CASCADE, three 2D textures over the tiling defined in `gi_math.hpp`:
 *   - irradiance, RGBA16F, 8x8 texels/probe (6x6 payload + 1 border ring)
 *     -> 2304 x 192, 3.375 MB
 *   - visibility, RG16F, 16x16 texels/probe (14x14 payload + border), storing
 *     mean and mean-squared ray distance for the Chebyshev leak test
 *     -> 4608 x 384, 6.75 MB
 *   - probe state, R8_UINT, one texel/probe -> 288 x 24, 6.75 KB
 * i.e. ~10.13 MB per cascade, ~30.4 MB for the field. That is the whole memory
 * story; `docs/rt_lighting_plan.md` §6 budgets ~35 MB for it.
 *
 * Tier: main-thread (owns GPU resources, records copy passes).
 */
class ProbeVolume {
public:
    ProbeVolume() = default;
    ~ProbeVolume();
    ProbeVolume(const ProbeVolume&) = delete;
    ProbeVolume& operator=(const ProbeVolume&) = delete;
    ProbeVolume(ProbeVolume&&) = delete;
    ProbeVolume& operator=(ProbeVolume&&) = delete;

    /// Allocate every cascade's atlases. Fails (and allocates nothing) if any
    /// texture cannot be created — a half-built field would be worse than none.
    [[nodiscard]] bool init(gpu::GPUDevice& device);
    void shutdown();
    [[nodiscard]] bool is_ready() const { return ready_; }

    /**
     * Follow the camera: re-snap every cascade and mark entering probes.
     *
     * `cmd` may be null (a host or test driving the CPU side alone), in which
     * case the state textures are left stale and re-uploaded on the next call
     * that has a command buffer. NO RENDER PASS may be active — the state
     * upload opens its own copy pass.
     */
    void update(SDL_GPUCommandBuffer* cmd, const glm::vec3& camera_position);

    [[nodiscard]] const GiStats& stats() const noexcept { return stats_; }

    /// Atlases, for R1b's compute passes. Null before init() or out of range.
    [[nodiscard]] gpu::GPUTexture* irradiance(int cascade) const;
    [[nodiscard]] gpu::GPUTexture* visibility(int cascade) const;
    [[nodiscard]] gpu::GPUTexture* probe_state(int cascade) const;

    /// CPU-side scroll state, for tests and the dev overlay.
    [[nodiscard]] const ProbeCascade& cascade(int index) const;

private:
    struct Cascade {
        ProbeCascade cpu;
        std::unique_ptr<gpu::GPUTexture> irradiance;
        std::unique_ptr<gpu::GPUTexture> visibility;
        std::unique_ptr<gpu::GPUTexture> state;
    };

    gpu::GPUDevice* device_ = nullptr;
    std::array<Cascade, CASCADE_COUNT> cascades_;
    bool ready_ = false;
    GiStats stats_{};
};

} // namespace bracken::render::gi
