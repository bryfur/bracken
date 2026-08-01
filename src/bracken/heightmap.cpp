#include "bracken/heightmap.hpp"

#include "bracken/core/logger.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace bracken {

namespace {

/** The shared body: validate, then quantize against `range`.
 *
 *  The packing is deliberately the exact arithmetic hosts hand-rolled —
 *  `(h - min) / (max - min)` (division, not a reciprocal multiply, which is
 *  not bit-identical) and a TRUNCATING cast — so moving a host onto
 *  from_floats reproduces its old terrain sample for sample. The clamp is the
 *  one behavioral addition: an out-of-range height used to convert to a
 *  garbage uint16 (or trap), and now saturates. */
Heightmap pack(std::span<const float> heights, uint32_t resolution, WorldBounds bounds, HeightRange range) {
    Heightmap map;
    const std::size_t expected = static_cast<std::size_t>(resolution) * resolution;
    if (resolution < 2) {
        BRACKEN_LOG_ERROR("heightmap", "from_floats: resolution {} is below the 2x2 minimum", resolution);
        return map;
    }
    if (heights.size() != expected) {
        BRACKEN_LOG_ERROR("heightmap", "from_floats: got {} heights for a {}x{} map (expected {})", heights.size(),
                          resolution, resolution, expected);
        return map;
    }

    map.resolution = resolution;
    map.world_origin_x = bounds.origin_x;
    map.world_origin_z = bounds.origin_z;
    map.world_size = bounds.size;
    map.min_height = range.min;
    map.max_height = range.max;
    map.height_data.resize(expected);

    const float span = range.max - range.min;
    for (std::size_t i = 0; i < expected; ++i) {
        float normalized = span > 0.0f ? (heights[i] - range.min) / span : 0.0f;
        normalized = std::clamp(normalized, 0.0f, 1.0f);
        map.height_data[i] = static_cast<uint16_t>(normalized * 65535.0f);
    }
    return map;
}

} // namespace

Heightmap Heightmap::from_floats(std::span<const float> heights, uint32_t resolution, WorldBounds bounds) {
    HeightRange range{0.0f, 0.0f};
    if (!heights.empty()) {
        range.min = std::numeric_limits<float>::max();
        range.max = std::numeric_limits<float>::lowest();
        for (float h : heights) {
            range.min = std::min(range.min, h);
            range.max = std::max(range.max, h);
        }
    }
    return pack(heights, resolution, bounds, range);
}

Heightmap Heightmap::from_floats(std::span<const float> heights, uint32_t resolution, WorldBounds bounds,
                                 HeightRange range) {
    return pack(heights, resolution, bounds, range);
}

} // namespace bracken
