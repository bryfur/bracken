#pragma once

/**
 * The probe update BUDGET and the cascade ROTA — how many rays phase R1b is
 * allowed to cast per frame, and which cascade gets them.
 *
 * Split out of `gi_math.hpp` because this is the one part of the probe field
 * that reads a setting: `gi_math.hpp` is deliberately glm + `<std>` only, so it
 * can be tested with nothing linked, and pulling `GraphicsSettings` in there to
 * serve two tables would cost that. Everything here is `constexpr` and free of
 * GPU types, so it tests just as headlessly.
 *
 * See `docs/rt_lighting_plan.md` §5.2 for why the budget is per FRAME rather
 * than per pixel: that is the single property that makes a probe field
 * affordable on hardware that cannot trace one ray per pixel.
 */

#include "bracken/contracts/graphics_settings.hpp"
#include "bracken/render/gi/gi_math.hpp"

#include <cstdint>

namespace bracken::render::gi {

/// What one cascade gets when its turn comes round.
struct GiBudget {
    /// Rays cast from each updated probe. More rays is a better estimate per
    /// update, not a faster response — the response rate is the hysteresis.
    std::uint32_t rays_per_probe = 0;
    /// Probes updated per cascade per turn. The reciprocal of the fraction of
    /// the cascade covered, so a cascade is fully refreshed every
    /// PROBES_PER_CASCADE / this turns.
    std::uint32_t probes_per_turn = 0;
};

/**
 * Rays/probe {32, 64, 96} and update fraction {1/8, 1/4, 1/4}, per the plan's
 * tier examples.
 *
 * Low is the iGPU tier and trades BOTH knobs down; Medium and High share the
 * 1/4 fraction and differ only in rays, because past 1/4 the field's response
 * lag stops being the limiting artefact and per-update variance starts being
 * one. (At 1/4 and the cascade-0-every-frame rota, cascade 0 refreshes fully
 * every 4 frames — well inside the hysteresis's own time constant.)
 */
[[nodiscard]] constexpr GiBudget gi_budget(GiQuality quality) {
    switch (quality) {
    case GiQuality::Low: return {32u, static_cast<std::uint32_t>(PROBES_PER_CASCADE) / 8u};
    case GiQuality::Medium: return {64u, static_cast<std::uint32_t>(PROBES_PER_CASCADE) / 4u};
    case GiQuality::High: return {96u, static_cast<std::uint32_t>(PROBES_PER_CASCADE) / 4u};
    }
    return {64u, static_cast<std::uint32_t>(PROBES_PER_CASCADE) / 4u};
}

/**
 * THE ROTA. Cascade 0 updates every frame; cascade c > 0 updates every 2^c
 * frames, phased so that no two coarse cascades ever land on the same frame.
 *
 *   frame:  0     1     2     3     4     5     6     7
 *   c0:     x     x     x     x     x     x     x     x
 *   c1:           x           x           x           x     (frame % 2 == 1)
 *   c2:                 x                       x           (frame % 4 == 2)
 *
 * Two properties, both deliberate. Fine cascades update more often because they
 * cover the geometry the camera is actually standing in, where a stale bounce
 * is visible; cascade 2 is 600-unit spacing at the edge of the draw distance,
 * where a four-frame lag is not a thing anyone can see. And because `c1`'s
 * frames are odd while `c2`'s are even, AT MOST TWO cascades trace in any one
 * frame — the peak cost is 2x a cascade's budget, not 3x, with no scheduler.
 */
[[nodiscard]] constexpr bool gi_cascade_updates_on_frame(int cascade, std::uint64_t frame) {
    if (cascade <= 0) {
        return true;
    }
    if (cascade >= CASCADE_COUNT) {
        return false;
    }
    const std::uint64_t period = 1ull << static_cast<unsigned>(cascade);
    return (frame % period) == (period >> 1);
}

/// Cascades that trace on the worst frame of the rota (see the table above).
inline constexpr int GI_MAX_CASCADES_PER_FRAME = 2;

/// Peak rays in any one frame at this quality — the number the cost of the
/// feature should be quoted against, not the per-cascade budget.
[[nodiscard]] constexpr std::uint32_t gi_peak_rays_per_frame(GiQuality quality) {
    const GiBudget b = gi_budget(quality);
    return b.rays_per_probe * b.probes_per_turn * static_cast<std::uint32_t>(GI_MAX_CASCADES_PER_FRAME);
}

/// Ray-result slots the trace buffer must hold: one cascade's worst-case turn.
/// Quality-independent so that changing quality never reallocates.
[[nodiscard]] constexpr std::uint32_t gi_max_rays_per_dispatch() {
    return gi_budget(GiQuality::High).rays_per_probe * gi_budget(GiQuality::High).probes_per_turn;
}

/// Update-list records the buffer must hold: the largest per-turn probe count.
[[nodiscard]] constexpr std::uint32_t gi_max_probes_per_dispatch() {
    return gi_budget(GiQuality::High).probes_per_turn;
}

} // namespace bracken::render::gi
