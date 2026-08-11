#include "bracken/heightmap_grid.hpp"

#include "bracken/core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>

namespace bracken {

namespace {
constexpr const char* LOG_TAG = "heightmap";

/// Row-major index of page (px,pz). Callers have already range-checked.
std::size_t page_index(const HeightmapGrid& grid, uint32_t px, uint32_t pz) {
    return static_cast<std::size_t>(pz) * grid.pages_x + px;
}

/// The page index a world coordinate falls in, CLAMPED to the grid. This is
/// the sampling path's addressing: outside the grid the edge page answers,
/// and it then clamps the sample to its own square — which is exactly how a
/// single Heightmap behaves outside its bounds.
uint32_t clamped_page_axis(float coord, float origin, float page_size, uint32_t page_count) {
    if (page_count == 0 || !(page_size > 0.0f)) {
        return 0;
    }
    const float f = std::floor((coord - origin) / page_size);
    // NaN first, and explicitly: `f <= 0` and `f >= count-1` are BOTH false for
    // a NaN, so without this the function falls through to a float->uint32_t
    // cast of a NaN, which is undefined behaviour and in practice an
    // out-of-range page index. A NaN query has no containing page; page 0 is the
    // same answer the clamp gives for anything else off the low edge.
    if (!(f == f)) {
        return 0;
    }
    if (f <= 0.0f) {
        return 0;
    }
    if (f >= static_cast<float>(page_count - 1)) {
        return page_count - 1;
    }
    return static_cast<uint32_t>(f);
}

} // namespace

HeightmapGrid HeightmapGrid::from_single(const Heightmap& map) {
    HeightmapGrid grid;
    grid.pages_x = 1;
    grid.pages_z = 1;
    grid.page_size = map.world_size;
    grid.origin_x = map.world_origin_x;
    grid.origin_z = map.world_origin_z;
    grid.range = HeightRange{map.min_height, map.max_height};
    grid.pages.push_back(map);
    return grid;
}

HeightmapGrid HeightmapGrid::create(uint32_t pages_x, uint32_t pages_z, float page_size, float origin_x,
                                    float origin_z, HeightRange range) {
    HeightmapGrid grid;
    grid.pages_x = pages_x;
    grid.pages_z = pages_z;
    grid.page_size = page_size;
    grid.origin_x = origin_x;
    grid.origin_z = origin_z;
    grid.range = range;
    grid.pages.resize(static_cast<std::size_t>(pages_x) * pages_z);
    return grid;
}

bool HeightmapGrid::set_page(uint32_t px, uint32_t pz, std::span<const float> heights, uint32_t resolution) {
    if (px >= pages_x || pz >= pages_z || pages.size() != static_cast<std::size_t>(pages_x) * pages_z) {
        BRACKEN_LOG_ERROR(LOG_TAG, "HeightmapGrid::set_page: page ({},{}) is outside a {}x{} grid", px, pz, pages_x,
                          pages_z);
        return false;
    }
    const WorldBounds bounds{origin_x + static_cast<float>(px) * page_size,
                             origin_z + static_cast<float>(pz) * page_size, page_size};
    // The EXPLICIT-range overload, always: the scanning one would quantize
    // this page against its own extremes and step at every seam it shares.
    Heightmap packed = Heightmap::from_floats(heights, resolution, bounds, range);
    if (!packed.valid()) {
        // from_floats already logged which of the two preconditions failed.
        return false;
    }
    pages[page_index(*this, px, pz)] = std::move(packed);
    return true;
}

void HeightmapGrid::clear_page(uint32_t px, uint32_t pz) {
    if (px >= pages_x || pz >= pages_z || pages.size() != static_cast<std::size_t>(pages_x) * pages_z) {
        return;
    }
    pages[page_index(*this, px, pz)] = Heightmap{};
}

float HeightmapGrid::geometry_tolerance(float reference) {
    return std::max(0.01f, 1e-5f * std::abs(reference));
}

core::Result<void, std::string> HeightmapGrid::validate() const {
    if (pages_x == 0 || pages_z == 0) {
        return core::make_err(std::format("HeightmapGrid: degenerate page count {}x{}", pages_x, pages_z));
    }
    if (!(page_size > 0.0f)) {
        return core::make_err(std::format("HeightmapGrid: page_size {} must be positive", page_size));
    }
    const std::size_t expected = static_cast<std::size_t>(pages_x) * pages_z;
    if (pages.size() != expected) {
        return core::make_err(
            std::format("HeightmapGrid: {} pages for a {}x{} grid (expected {})", pages.size(), pages_x, pages_z,
                        expected));
    }

    uint32_t shared_resolution = 0;
    for (uint32_t pz = 0; pz < pages_z; ++pz) {
        for (uint32_t px = 0; px < pages_x; ++px) {
            const Heightmap& map = pages[page_index(*this, px, pz)];
            if (!map.valid()) {
                continue; // not resident — a legal state, not an error
            }
            // THE SEAM TRAP, and the reason this function exists: two pages
            // packed against different ranges disagree about the heights on
            // the line they share. No tolerance — a range is copied, never
            // recomputed, so any difference at all is an authoring mistake.
            if (map.min_height != range.min || map.max_height != range.max) {
                return core::make_err(std::format(
                    "HeightmapGrid: page ({},{}) is packed against range [{}, {}] but the grid's is [{}, {}] — "
                    "independently quantized pages step at their shared edge; use set_page or the explicit-range "
                    "Heightmap::from_floats overload",
                    px, pz, map.min_height, map.max_height, range.min, range.max));
            }
            if (shared_resolution == 0) {
                shared_resolution = map.resolution;
            } else if (map.resolution != shared_resolution) {
                return core::make_err(std::format("HeightmapGrid: page ({},{}) has resolution {} but a sibling has {}",
                                                  px, pz, map.resolution, shared_resolution));
            }
            if (std::abs(map.world_size - page_size) > geometry_tolerance(page_size)) {
                return core::make_err(std::format("HeightmapGrid: page ({},{}) spans {} world units, grid pages are {}",
                                                  px, pz, map.world_size, page_size));
            }
            const float expect_x = origin_x + static_cast<float>(px) * page_size;
            const float expect_z = origin_z + static_cast<float>(pz) * page_size;
            if (std::abs(map.world_origin_x - expect_x) > geometry_tolerance(expect_x) ||
                std::abs(map.world_origin_z - expect_z) > geometry_tolerance(expect_z)) {
                return core::make_err(std::format(
                    "HeightmapGrid: page ({},{}) sits at ({}, {}) but the grid addresses it at ({}, {})", px, pz,
                    map.world_origin_x, map.world_origin_z, expect_x, expect_z));
            }
        }
    }
    return core::Ok();
}

std::size_t HeightmapGrid::resident_pages() const {
    std::size_t count = 0;
    for (const Heightmap& map : pages) {
        if (map.valid()) {
            ++count;
        }
    }
    return count;
}

uint32_t HeightmapGrid::page_resolution() const {
    for (const Heightmap& map : pages) {
        if (map.valid()) {
            return map.resolution;
        }
    }
    return 0;
}

glm::ivec2 HeightmapGrid::page_at(float world_x, float world_z) const {
    if (pages_x == 0 || pages_z == 0 || !(page_size > 0.0f)) {
        return {-1, -1};
    }
    // A NaN coordinate is outside every page, and it has to be said outright:
    // every comparison below is false for a NaN, so it would otherwise fall
    // through to the clamped answer and report page (0,0) as "containing" it.
    if (!(world_x == world_x) || !(world_z == world_z)) {
        return {-1, -1};
    }
    // Inclusive on the far edge: the grid's outer boundary line belongs to
    // the last page, the same way a page's own far edge belongs to it.
    if (world_x < origin_x || world_x > origin_x + world_size_x() || world_z < origin_z ||
        world_z > origin_z + world_size_z()) {
        return {-1, -1};
    }
    return {static_cast<int>(clamped_page_axis(world_x, origin_x, page_size, pages_x)),
            static_cast<int>(clamped_page_axis(world_z, origin_z, page_size, pages_z))};
}

const Heightmap* HeightmapGrid::page(int px, int pz) const {
    if (px < 0 || pz < 0 || static_cast<uint32_t>(px) >= pages_x || static_cast<uint32_t>(pz) >= pages_z ||
        pages.size() != static_cast<std::size_t>(pages_x) * pages_z) {
        return nullptr;
    }
    const Heightmap& map = pages[page_index(*this, static_cast<uint32_t>(px), static_cast<uint32_t>(pz))];
    return map.valid() ? &map : nullptr;
}

glm::vec2 HeightmapGrid::page_min_xz(int px, int pz) const {
    if (px < 0 || pz < 0 || static_cast<uint32_t>(px) >= pages_x || static_cast<uint32_t>(pz) >= pages_z) {
        return {0.0f, 0.0f};
    }
    return {origin_x + static_cast<float>(px) * page_size, origin_z + static_cast<float>(pz) * page_size};
}

glm::vec2 HeightmapGrid::page_max_xz(int px, int pz) const {
    if (px < 0 || pz < 0 || static_cast<uint32_t>(px) >= pages_x || static_cast<uint32_t>(pz) >= pages_z) {
        return {0.0f, 0.0f};
    }
    const glm::vec2 lo = page_min_xz(px, pz);
    return {lo.x + page_size, lo.y + page_size};
}

float HeightmapGrid::get_height_world(float world_x, float world_z, float fallback) const {
    if (pages.size() != static_cast<std::size_t>(pages_x) * pages_z || pages.empty()) {
        return fallback;
    }
    const uint32_t px = clamped_page_axis(world_x, origin_x, page_size, pages_x);
    const uint32_t pz = clamped_page_axis(world_z, origin_z, page_size, pages_z);
    const Heightmap& map = pages[page_index(*this, px, pz)];
    if (!map.valid()) {
        return fallback;
    }
    // No arithmetic in front of the page's own sampler: for a 1x1 grid this
    // IS Heightmap::get_height_world, bit for bit, and outside the grid the
    // page's internal [0,1] clamp produces the grid edge height.
    return map.get_height_world(world_x, world_z);
}

glm::vec3 HeightmapGrid::get_normal_world(float world_x, float world_z) const {
    const uint32_t resolution = page_resolution();
    if (resolution < 2 || !(page_size > 0.0f)) {
        return {0.0f, 1.0f, 0.0f};
    }
    // The same stencil Heightmap::get_normal_world uses, but every tap goes
    // through the GRID sampler, so a stencil straddling a seam reads the
    // neighbour instead of clamping against an interior page edge.
    const float eps = page_size / static_cast<float>(resolution - 1);
    const float hL = get_height_world(world_x - eps, world_z);
    const float hR = get_height_world(world_x + eps, world_z);
    const float hD = get_height_world(world_x, world_z - eps);
    const float hU = get_height_world(world_x, world_z + eps);

    const glm::vec3 n{hL - hR, 2.0f * eps, hD - hU};
    const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 0.0001f) {
        return n / len;
    }
    return {0.0f, 1.0f, 0.0f};
}

bool HeightmapGrid::resident_at(float world_x, float world_z) const {
    if (pages.size() != static_cast<std::size_t>(pages_x) * pages_z || pages.empty()) {
        return false;
    }
    const uint32_t px = clamped_page_axis(world_x, origin_x, page_size, pages_x);
    const uint32_t pz = clamped_page_axis(world_z, origin_z, page_size, pages_z);
    return pages[page_index(*this, px, pz)].valid();
}

} // namespace bracken
