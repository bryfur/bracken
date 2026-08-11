#include "bracken/animation/blend_space.hpp"

#include "bracken/core/assert.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace bracken::animation {
namespace {

/// Signed angle from a to b about the origin, in radians. Zero when either
/// vector has no direction — a sample at the origin is "not moving", and
/// pretending it points somewhere is what makes a centre sample fight with its
/// neighbours.
[[nodiscard]] float signed_angle(glm::vec2 a, glm::vec2 b) noexcept {
    constexpr float k_min_len_sq = 1e-12f;
    const float la = a.x * a.x + a.y * a.y;
    const float lb = b.x * b.x + b.y * b.y;
    if (la < k_min_len_sq || lb < k_min_len_sq) {
        return 0.0f;
    }
    return std::atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y);
}

} // namespace

int find_clip_index(const std::vector<AnimationClip>& clips, std::string_view name) noexcept {
    for (std::size_t i = 0; i < clips.size(); ++i) {
        if (clips[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// 1D
// ---------------------------------------------------------------------------

bool BlendSpace1D::add_sample(int clip_index, float position, float speed_scale) noexcept {
    if (clip_index < 0 || count_ >= k_max_blend_samples) {
        return false;
    }
    samples_[static_cast<std::size_t>(count_)] = Sample{clip_index, position, speed_scale};
    ++count_;
    finalized_ = false;
    return true;
}

void BlendSpace1D::finalize() noexcept {
    std::stable_sort(samples_.begin(), samples_.begin() + count_,
                     [](const Sample& a, const Sample& b) { return a.position < b.position; });
    finalized_ = true;
}

std::uint32_t BlendSpace1D::resolve(float x, std::span<WeightedClip> out) const noexcept {
    if (count_ <= 0 || out.empty()) {
        return 0;
    }
    BRACKEN_ASSERT(finalized_ || count_ == 1);

    if (count_ == 1) {
        out[0] = WeightedClip{samples_[0].clip_index, 1.0f, samples_[0].speed_scale, 0.0f};
        return 1;
    }

    const std::size_t last = static_cast<std::size_t>(count_) - 1;

    // Clamp, never extrapolate: outside the authored range the nearest clip
    // plays at full weight.
    if (x <= samples_[0].position) {
        out[0] = WeightedClip{samples_[0].clip_index, 1.0f, samples_[0].speed_scale, 0.0f};
        return 1;
    }
    if (x >= samples_[last].position) {
        out[0] = WeightedClip{samples_[last].clip_index, 1.0f, samples_[last].speed_scale, 0.0f};
        return 1;
    }

    std::size_t i = 0;
    while (i + 1 < last && samples_[i + 1].position <= x) {
        ++i;
    }

    const float p0 = samples_[i].position;
    const float p1 = samples_[i + 1].position;
    const float span = p1 - p0;
    // Coincident samples: no interval to interpolate across, so the lower one
    // wins outright rather than dividing by zero.
    const float t = (span > 0.0f) ? (x - p0) / span : 0.0f;

    if (t <= k_blend_weight_epsilon) {
        out[0] = WeightedClip{samples_[i].clip_index, 1.0f, samples_[i].speed_scale, 0.0f};
        return 1;
    }
    if (t >= 1.0f - k_blend_weight_epsilon) {
        out[0] = WeightedClip{samples_[i + 1].clip_index, 1.0f, samples_[i + 1].speed_scale, 0.0f};
        return 1;
    }
    if (out.size() < 2) {
        out[0] = WeightedClip{(t < 0.5f ? samples_[i].clip_index : samples_[i + 1].clip_index), 1.0f,
                              (t < 0.5f ? samples_[i].speed_scale : samples_[i + 1].speed_scale), 0.0f};
        return 1;
    }

    out[0] = WeightedClip{samples_[i].clip_index, 1.0f - t, samples_[i].speed_scale, 0.0f};
    out[1] = WeightedClip{samples_[i + 1].clip_index, t, samples_[i + 1].speed_scale, 0.0f};
    return 2;
}

// ---------------------------------------------------------------------------
// 2D
// ---------------------------------------------------------------------------

bool BlendSpace2D::add_sample(int clip_index, glm::vec2 position, float speed_scale) noexcept {
    if (clip_index < 0 || count_ >= k_max_blend_samples) {
        return false;
    }
    samples_[static_cast<std::size_t>(count_)] = Sample{clip_index, position, speed_scale};
    ++count_;
    return true;
}

std::uint32_t BlendSpace2D::resolve(glm::vec2 p, std::span<WeightedClip> out) const noexcept {
    if (count_ <= 0 || out.empty()) {
        return 0;
    }
    if (count_ == 1) {
        out[0] = WeightedClip{samples_[0].clip_index, 1.0f, samples_[0].speed_scale, 0.0f};
        return 1;
    }

    const float r = std::sqrt(p.x * p.x + p.y * p.y);
    std::array<float, k_max_blend_samples> weights{};

    for (int i = 0; i < count_; ++i) {
        const glm::vec2 pi = samples_[static_cast<std::size_t>(i)].position;
        const float ri = std::sqrt(pi.x * pi.x + pi.y * pi.y);
        float w = 1.0f;

        for (int j = 0; j < count_; ++j) {
            if (j == i) {
                continue;
            }
            const glm::vec2 pj = samples_[static_cast<std::size_t>(j)].position;
            const float rj = std::sqrt(pj.x * pj.x + pj.y * pj.y);

            const float avg = 0.5f * (ri + rj);
            const float inv_avg = (avg > 1e-8f) ? (1.0f / avg) : 0.0f;

            const glm::vec2 v_ij((rj - ri) * inv_avg, signed_angle(pi, pj) * k_direction_scale);
            const glm::vec2 v_ip((r - ri) * inv_avg, signed_angle(pi, p) * k_direction_scale);

            const float denom = v_ij.x * v_ij.x + v_ij.y * v_ij.y;
            // Coincident samples cannot constrain each other; h = 1 leaves them
            // both fully weighted, which is what makes duplicates split evenly.
            const float h = (denom > 1e-12f)
                                ? std::clamp(1.0f - (v_ip.x * v_ij.x + v_ip.y * v_ij.y) / denom, 0.0f, 1.0f)
                                : 1.0f;
            w = std::min(w, h);
            if (w <= 0.0f) {
                break;
            }
        }

        weights[static_cast<std::size_t>(i)] = (w > k_blend_weight_epsilon) ? w : 0.0f;
    }

    float sum = 0.0f;
    for (int i = 0; i < count_; ++i) {
        sum += weights[static_cast<std::size_t>(i)];
    }

    if (sum <= 0.0f) {
        // Everything culled — the query is outside the cover. Fall back to the
        // nearest sample rather than returning nothing or dividing by zero.
        int nearest = 0;
        float best = std::numeric_limits<float>::max();
        for (int i = 0; i < count_; ++i) {
            const glm::vec2 d = samples_[static_cast<std::size_t>(i)].position - p;
            const float dist = d.x * d.x + d.y * d.y;
            if (dist < best) {
                best = dist;
                nearest = i;
            }
        }
        out[0] = WeightedClip{samples_[static_cast<std::size_t>(nearest)].clip_index, 1.0f,
                              samples_[static_cast<std::size_t>(nearest)].speed_scale, 0.0f};
        return 1;
    }

    const float inv_sum = 1.0f / sum;
    std::uint32_t written = 0;
    for (int i = 0; i < count_ && written < out.size(); ++i) {
        if (weights[static_cast<std::size_t>(i)] <= 0.0f) {
            continue;
        }
        out[written] = WeightedClip{samples_[static_cast<std::size_t>(i)].clip_index,
                                    weights[static_cast<std::size_t>(i)] * inv_sum,
                                    samples_[static_cast<std::size_t>(i)].speed_scale, 0.0f};
        ++written;
    }
    return written;
}

float effective_duration(std::span<const WeightedClip> active) noexcept {
    float sum_w = 0.0f;
    float sum_d = 0.0f;
    for (const WeightedClip& c : active) {
        if (c.weight <= 0.0f || c.duration <= 0.0f) {
            continue;
        }
        const float scale = (std::abs(c.speed_scale) > 1e-6f) ? c.speed_scale : 1.0f;
        sum_d += c.weight * (c.duration / scale);
        sum_w += c.weight;
    }
    return (sum_w > 0.0f && sum_d > 0.0f) ? (sum_d / sum_w) : 0.0f;
}

} // namespace bracken::animation
