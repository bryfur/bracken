#pragma once
/**
 * Engine heightmap data structure for terrain rendering.
 * Contains only the data and methods needed by the renderer.
 * Game code converts from its own heightmap format to this — through
 * Heightmap::from_floats, which owns the scan and the 16-bit packing.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace bracken {

/** Where a heightmap sits in the world: a square starting at (origin_x,
 *  origin_z) and `size` units on a side. */
struct WorldBounds {
    float origin_x = 0.0f;
    float origin_z = 0.0f;
    float size = 0.0f;
};

/** The height range a heightmap's 16-bit samples are normalized against. */
struct HeightRange {
    float min = 0.0f;
    float max = 0.0f;
};

struct Heightmap {
    uint32_t resolution = 0;

    float world_origin_x = 0.0f;
    float world_origin_z = 0.0f;
    float world_size = 0.0f;

    // Min/max height range used for 16-bit normalization
    float min_height = -500.0f;
    float max_height = 500.0f;

    // Height data as 16-bit normalized values
    // Real height = (uint16_value / 65535.0) * (max_height - min_height) + min_height
    std::vector<uint16_t> height_data;

    /**
     * Build a heightmap from world-unit heights in row-major
     * `z * resolution + x` order — the engine does the min/max scan and the
     * 16-bit packing every host used to hand-roll.
     *
     * The scanned range is the tightest the data allows, which is also the
     * most precise. It does mean the encoding of EVERY sample depends on the
     * extremes, so editing one height re-quantizes the whole map; a host that
     * needs a stable encoding (a reproducible integration test, a streamed
     * world whose tiles must agree) passes its own range to the overload
     * below instead.
     *
     * Returns an invalid (empty) heightmap — see valid() — and logs an error
     * when `resolution < 2` or `heights.size() != resolution * resolution`.
     */
    [[nodiscard]] static Heightmap from_floats(std::span<const float> heights, uint32_t resolution,
                                               WorldBounds bounds);

    /** As above, against a caller-supplied range. Heights outside it are
     *  clamped rather than wrapping through the 16-bit cast. An empty or
     *  inverted range packs everything to 0. */
    [[nodiscard]] static Heightmap from_floats(std::span<const float> heights, uint32_t resolution, WorldBounds bounds,
                                               HeightRange range);

    /** Usable for sampling: at least 2x2 (world queries interpolate over
     *  `resolution - 1` cells, which UNDERFLOWS at resolution 0) and fully
     *  populated. Application::set_heightmap rejects anything else. */
    [[nodiscard]] bool valid() const {
        return resolution >= 2 && height_data.size() == static_cast<std::size_t>(resolution) * resolution;
    }

    float get_height_local(uint32_t local_x, uint32_t local_z) const {
        if (local_x >= resolution || local_z >= resolution) {
            return 0.0f;
        }
        uint16_t raw = height_data[local_z * resolution + local_x];
        float normalized = raw / 65535.0f;
        return normalized * (max_height - min_height) + min_height;
    }

    float get_height_world(float world_x, float world_z) const {
        float u = (world_x - world_origin_x) / world_size;
        float v = (world_z - world_origin_z) / world_size;
        u = std::fmax(0.0f, std::fmin(1.0f, u));
        v = std::fmax(0.0f, std::fmin(1.0f, v));

        float tx = u * (resolution - 1);
        float tz = v * (resolution - 1);
        uint32_t x0 = static_cast<uint32_t>(tx);
        uint32_t z0 = static_cast<uint32_t>(tz);
        uint32_t x1 = std::min(x0 + 1, resolution - 1);
        uint32_t z1 = std::min(z0 + 1, resolution - 1);
        float fx = tx - x0;
        float fz = tz - z0;

        float h00 = get_height_local(x0, z0);
        float h10 = get_height_local(x1, z0);
        float h01 = get_height_local(x0, z1);
        float h11 = get_height_local(x1, z1);
        float h0 = h00 * (1.0f - fx) + h10 * fx;
        float h1 = h01 * (1.0f - fx) + h11 * fx;
        return h0 * (1.0f - fz) + h1 * fz;
    }

    /** Unit surface normal by central differences over one grid cell;
     *  (0, 1, 0) where the surface is degenerate. */
    glm::vec3 get_normal_world(float world_x, float world_z) const {
        float eps = world_size / (resolution - 1);
        float hL = get_height_world(world_x - eps, world_z);
        float hR = get_height_world(world_x + eps, world_z);
        float hD = get_height_world(world_x, world_z - eps);
        float hU = get_height_world(world_x, world_z + eps);

        const glm::vec3 n{hL - hR, 2.0f * eps, hD - hU};
        const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (len > 0.0001f) {
            return n / len;
        }
        return {0.0f, 1.0f, 0.0f};
    }
};

} // namespace bracken
