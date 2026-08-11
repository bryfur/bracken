#include "bracken/contracts/occlusion.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace bracken {

namespace {

/// Clip w at or below this is "on or behind the near plane": the perspective
/// divide there is meaningless and a box straddling it has no axis-aligned
/// screen hull derivable from its corners.
constexpr float k_min_clip_w = 1e-4f;

} // namespace

float OcclusionBounds::screen_fraction() const noexcept {
    if (!valid) {
        return 1.0f; // "as large as possible", i.e. refuse to test
    }
    const float w = std::max(0.0f, max_uv.x - min_uv.x);
    const float h = std::max(0.0f, max_uv.y - min_uv.y);
    return std::min(1.0f, w * h);
}

OcclusionBounds project_aabb(const glm::mat4& view_proj, const glm::vec3& aabb_min, const glm::vec3& aabb_max) {
    OcclusionBounds out;
    glm::vec2 lo(std::numeric_limits<float>::max());
    glm::vec2 hi(std::numeric_limits<float>::lowest());
    float nearest = std::numeric_limits<float>::lowest();

    for (int corner = 0; corner < 8; ++corner) {
        const glm::vec3 p((corner & 1) ? aabb_max.x : aabb_min.x, (corner & 2) ? aabb_max.y : aabb_min.y,
                          (corner & 4) ? aabb_max.z : aabb_min.z);
        const glm::vec4 clip = view_proj * glm::vec4(p, 1.0f);
        if (!(clip.w > k_min_clip_w)) {
            // Straddles or is behind the near plane (NaN lands here too, which
            // is the answer we want). Never culled.
            return out; // valid == false
        }
        const float inv_w = 1.0f / clip.w;
        const glm::vec3 ndc(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
        // NDC -> UV with y DOWN: ndc.y == -1 is the top of the frame, which is
        // texel row 0 of a straight copy of the depth target.
        const glm::vec2 uv(ndc.x * 0.5f + 0.5f, ndc.y * 0.5f + 0.5f);
        lo = glm::min(lo, uv);
        hi = glm::max(hi, uv);
        nearest = std::max(nearest, ndc.z);
    }

    out.min_uv = lo;
    out.max_uv = hi;
    out.nearest_depth = nearest;
    out.valid = true;
    return out;
}

bool occluded_by_field(const OcclusionField& field, const OcclusionBounds& bounds) {
    if (!field.usable() || !bounds.valid) {
        return false;
    }
    if (bounds.screen_fraction() > field.max_tested_screen_fraction) {
        return false;
    }

    // Footprint in MIP-0 columns/rows, then divided down by the level. Doing it
    // this way rather than as `uv * mip_w` matters because the pyramid's last
    // texel on an odd axis is oversized (render/hzb.hpp), so mip dimensions are
    // NOT `screen >> level` and scaling by them addresses one texel too far
    // left near the right-hand edge.
    //
    // Precisely one texel, and no more: `screen < (mip + 1) * 2^level`, so the
    // two addressings can differ by at most 1 — which the dilation below
    // already covers. This is belt AND braces on purpose: it makes the walk
    // correct on its own terms rather than correct only because of the
    // dilation, so tightening the dilation later cannot silently open a hole.
    const int scale = 1 << field.level;
    const auto to_texel = [scale](float uv, int screen_extent, int mip_extent) {
        const float column = uv * static_cast<float>(screen_extent);
        const int texel = static_cast<int>(std::floor(column)) / scale;
        return std::clamp(texel, 0, mip_extent - 1);
    };

    // Reject a rect that lies entirely outside the frame: the depth buffer says
    // nothing about it, and "no data" must never read as "occluded".
    if (bounds.max_uv.x < 0.0f || bounds.min_uv.x > 1.0f || bounds.max_uv.y < 0.0f || bounds.min_uv.y > 1.0f) {
        return false;
    }

    const glm::vec2 lo = glm::clamp(bounds.min_uv, glm::vec2(0.0f), glm::vec2(1.0f));
    const glm::vec2 hi = glm::clamp(bounds.max_uv, glm::vec2(0.0f), glm::vec2(1.0f));

    // One texel of dilation on every side. It costs at most a 2-texel-wider
    // walk and it makes the footprint a strict SUPERSET of the true one under
    // any half-texel or rounding convention — and a superset can only ever make
    // the test refuse to cull, never make it cull more.
    const int x0 = std::max(0, to_texel(lo.x, field.screen_w, field.mip_w) - 1);
    const int x1 = std::min(field.mip_w - 1, to_texel(hi.x, field.screen_w, field.mip_w) + 1);
    const int y0 = std::max(0, to_texel(lo.y, field.screen_h, field.mip_h) - 1);
    const int y1 = std::min(field.mip_h - 1, to_texel(hi.y, field.screen_h, field.mip_h) + 1);
    if (x1 < x0 || y1 < y0) {
        return false;
    }

    // Occluded iff EVERY texel's farthest surface is still closer than the
    // box's nearest point. Reversed-Z: larger is closer, so "closer than the
    // box" is `texel > nearest_depth`. A background texel holds the clear value
    // 0.0 (the far plane) and therefore never occludes anything, which is what
    // makes the sky-behind-a-prop case come out right for free.
    const float threshold = bounds.nearest_depth + field.depth_slack;
    for (int y = y0; y <= y1; ++y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(field.mip_w);
        for (int x = x0; x <= x1; ++x) {
            const float texel = field.depth[row + static_cast<std::size_t>(x)];
            if (!(texel > threshold)) {
                return false;
            }
        }
    }
    return true;
}

bool occlusion_cull_aabb(const OcclusionField& field, const glm::vec3& aabb_min, const glm::vec3& aabb_max) {
    if (!field.usable()) {
        return false;
    }
    // Inflate by how far the camera travelled since the depth was rasterized.
    // Widening the box can only widen its footprint and can only move
    // `nearest_depth` closer, both of which make culling harder — the inflation
    // is monotone in the safe direction by construction, which is the property
    // the test asserts rather than a hope.
    const glm::vec3 pad(field.camera_travel);
    const OcclusionBounds bounds = project_aabb(field.view_proj, aabb_min - pad, aabb_max + pad);
    return occluded_by_field(field, bounds);
}

} // namespace bracken
