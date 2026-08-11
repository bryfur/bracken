#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <span>

namespace bracken {

/**
 * One frame's downloaded coarse depth, as a VALUE.
 *
 * Everything the Prepare stage needs to occlusion-cull a draw and nothing it
 * must not have: no texture, no device, no renderer pointer. That is not
 * fastidiousness — `ScenePrep` structurally cannot reach a main-thread API
 * (scene_prep.hpp's hard rule), so the download, the fence and the mapping all
 * live on the renderer side and the answer crosses the boundary as POD plus a
 * span over a buffer the renderer owns for the frame.
 *
 * THE DEPTH IS A FRAME OLD, AND THAT IS STRUCTURAL. It is mip `level` of the
 * FARTHEST-surface pyramid built over the previous frame's depth prepass:
 * Prepare runs before this frame's prepass, and SDL3 has no non-blocking map,
 * so there is no arrangement in which a CPU cull reads current depth. Every
 * conservatism knob below exists because of that one fact.
 *
 * ADDRESSING. `mip_w`/`mip_h` are the dimensions of the level actually
 * downloaded, but a footprint is computed from `screen_w`/`screen_h` and
 * `level` — the pyramid's last texel on an odd axis is OVERSIZED (see
 * render/hzb.hpp), so `uv * mip_w` under-shoots the rightmost texel a box
 * touches and would silently make the test non-conservative.
 */
struct OcclusionField {
    /// Row-major `mip_w * mip_h` device-depth values, reversed-Z (larger =
    /// closer), each the FARTHEST surface anywhere in its footprint.
    std::span<const float> depth;
    int mip_w = 0;
    int mip_h = 0;
    /// Mip 0's dimensions, i.e. the screen the depth was rasterized at.
    int screen_w = 0;
    int screen_h = 0;
    /// Which level `depth` holds; footprints are addressed at `screen / 2^level`.
    int level = 0;

    /// The UNJITTERED view-projection the depth was rasterized with.
    glm::mat4 view_proj{1.0f};

    /// World distance the camera moved since then. Every tested AABB is
    /// inflated by it, so a box that only *became* visible because the camera
    /// slid sideways is no longer provably behind the stale depth.
    float camera_travel = 0.0f;
    /// Past this much travel the stale depth describes a different scene and
    /// the whole test is skipped. The inflation above is a bound on a small
    /// translation, not a licence for a teleport.
    float max_camera_travel = 250.0f;
    /// Dot of this frame's camera forward with the depth frame's. 1 = no
    /// rotation.
    float camera_rotation_cos = 1.0f;
    /// Below this, the frame rotated far enough that geometry outside the old
    /// frustum is now on screen, and the old depth says nothing about it.
    /// cos(2.5 deg) — roughly one frame of a fast mouse flick at 60 Hz.
    float min_camera_rotation_cos = 0.99905f;

    /// Never test anything whose conservative screen rect covers more than
    /// this fraction of the frame. THE OCCLUDER FEEDBACK LOOP: an object culled
    /// this frame writes no prepass depth, so it cannot occlude next frame; for
    /// a pebble that is free, for a cathedral it means everything behind it
    /// pops back in and the scene oscillates at 30 Hz. This threshold is the
    /// only guard, and it is a GUESS — see docs/gap_streaming.md, it has not
    /// been measured against a knee.
    float max_tested_screen_fraction = 0.15f;

    /// Extra device-depth margin a box must be behind by. 0 is already
    /// conservative; a positive value buys slack against depth precision at
    /// range. Reversed-Z, so this is subtracted from the occluder side.
    float depth_slack = 0.0f;

    /** Is there an answer worth asking for this frame? False whenever the
     *  camera moved or rotated enough that a frame-old depth buffer stops
     *  describing this view. Depends only on scene state and settings — never
     *  on GPU timing — so it is reproducible run to run. */
    [[nodiscard]] bool usable() const noexcept {
        return !depth.empty() && mip_w > 0 && mip_h > 0 && screen_w > 0 && screen_h > 0 && level >= 0 &&
               static_cast<int>(depth.size()) >= mip_w * mip_h && camera_travel <= max_camera_travel &&
               camera_rotation_cos >= min_camera_rotation_cos;
    }
};

/**
 * A world AABB's conservative screen-space footprint under a view-projection.
 *
 * `valid` is false when ANY corner is at or behind the near plane (clip
 * w <= epsilon). Such a box is never culled: the perspective divide is
 * meaningless there, and a straddling box's projection is not the hull of its
 * projected corners.
 */
struct OcclusionBounds {
    /// [0,1] UV rect, y down (texel row 0 is the top of the frame, matching a
    /// straight copy of the depth target). NOT clamped to the screen — the
    /// clamp happens in the footprint walk, and `screen_fraction` deliberately
    /// measures the unclamped rect.
    glm::vec2 min_uv{0.0f};
    glm::vec2 max_uv{0.0f};
    /// MAX device depth over the eight corners. Under reversed-Z that is the
    /// CLOSEST point of the box, and because device depth is monotone in view
    /// depth and view depth attains its extremes at a convex hull's vertices,
    /// no interior point of the box is closer. That is what makes testing one
    /// scalar against the field sound rather than approximate.
    float nearest_depth = 0.0f;
    bool valid = false;

    /** Fraction of the frame the rect covers, capped at 1. */
    [[nodiscard]] float screen_fraction() const noexcept;
};

/** Project the eight corners and take the axis-aligned hull. Pure. */
[[nodiscard]] OcclusionBounds project_aabb(const glm::mat4& view_proj, const glm::vec3& aabb_min,
                                           const glm::vec3& aabb_max);

/**
 * Is the box provably behind the field EVERYWHERE it could appear?
 *
 * True only when every texel the (one-texel-dilated, screen-clamped) footprint
 * touches holds a surface CLOSER than the box's nearest point. False on any
 * doubt at all: no field, an unusable field, invalid bounds, a rect over
 * `max_tested_screen_fraction`, or a footprint that lands entirely off screen.
 *
 * ONE-SIDEDNESS, EXACTLY AS FAR AS IT GOES. For a static scene under a static
 * camera and a correct farthest-surface pyramid this cannot return true for a
 * visible box, and contracts/occlusion.test.cpp brute-forces that against a
 * reference. Under camera motion it is a HEURISTIC: the depth is a frame old,
 * so a box the old view hid may be visible now. `camera_travel` inflation,
 * `max_camera_travel` and `min_camera_rotation_cos` bound that; they do not
 * eliminate it. This is why the feature ships off by default.
 *
 * Pure — the CPU source of truth a GPU cull shader would mirror.
 */
[[nodiscard]] bool occluded_by_field(const OcclusionField& field, const OcclusionBounds& bounds);

/**
 * The whole test as the Prepare stage calls it: inflate by camera travel,
 * project, test. One call so the inflation cannot be forgotten at a call site.
 */
[[nodiscard]] bool occlusion_cull_aabb(const OcclusionField& field, const glm::vec3& aabb_min,
                                       const glm::vec3& aabb_max);

} // namespace bracken
