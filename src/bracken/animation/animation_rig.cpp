#include "bracken/animation/animation_rig.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace bracken::animation {
namespace {

constexpr const char* LOG_TAG = "animation";

/// The empty clip table `evaluate()` falls back to when the caller has not
/// bound bindings yet. Returning an empty span from `for_clip` makes every
/// sample read the rest pose, which is the defined "no animation" answer.
const ClipBindings& null_bindings() {
    static const ClipBindings k_empty;
    return k_empty;
}

} // namespace

void AnimationRig::reset() {
    for (auto& m : bone_matrices_) {
        m = glm::mat4(1.0f);
    }
    for (auto& m : world_transforms_) {
        m = glm::mat4(1.0f);
    }
    composed_.fill(BoneTransform{});
    used_joint_count_ = 0;
    root_delta_ = RootMotionDelta{};
    for (int i = 0; i < layer_count_; ++i) {
        layers_[static_cast<std::size_t>(i)].clock = Clock{};
        layers_[static_cast<std::size_t>(i)].phase = 0.0f;
        for (auto& s : layers_[static_cast<std::size_t>(i)].samplers) {
            s.reset();
        }
    }
}

void AnimationRig::bind(const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
                        const ClipBindings& bindings) {
    bindings_ = &bindings;
    joint_count_ = std::min(skeleton.joints.size(), static_cast<std::size_t>(MAX_BONES));
    if (bindings.clip_count() != clips.size()) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "AnimationRig::bind: bindings were built for {} clips but {} were passed; "
                          "clips beyond the bindings will sample as the rest pose",
                          bindings.clip_count(), clips.size());
    }
    if (skeleton.joints.size() > static_cast<std::size_t>(MAX_BONES)) {
        BRACKEN_LOG_WARN(LOG_TAG,
                         "AnimationRig::bind: skeleton has {} joints, clamping to MAX_BONES ({}); "
                         "joints beyond the clamp are not animated",
                         skeleton.joints.size(), MAX_BONES);
    }
    for (int i = 0; i < layer_count_; ++i) {
        size_layer_storage(layers_[static_cast<std::size_t>(i)]);
    }
}

void AnimationRig::size_layer_storage(Layer& layer) {
    // How many samplers this layer can actually use. A clip layer needs exactly
    // two (current + crossfade partner); a blend space needs one per sample.
    std::size_t needed = 2;
    if (layer.source == LayerSource::Blend1D) {
        needed = static_cast<std::size_t>(std::max(1, layer.space_1d.sample_count()));
    } else if (layer.source == LayerSource::Blend2D) {
        needed = static_cast<std::size_t>(std::max(1, layer.space_2d.sample_count()));
    }
    for (std::size_t i = 0; i < layer.samplers.size(); ++i) {
        layer.samplers[i].bind(i < needed ? joint_count_ : 0u, i < needed && layer.desc.cache_cursors);
    }
}

int AnimationRig::add_layer(LayerDesc desc) {
    if (layer_count_ >= k_max_animation_layers) {
        BRACKEN_LOG_WARN(LOG_TAG, "AnimationRig::add_layer: '{}' refused, the stack already holds {} layers",
                         desc.name, k_max_animation_layers);
        return -1;
    }
    const int index = layer_count_;
    Layer& l = layers_[static_cast<std::size_t>(index)];
    l = Layer{};
    l.desc = std::move(desc);
    ++layer_count_;
    if (bindings_ != nullptr) {
        size_layer_storage(l);
    }
    return index;
}

LayerDesc& AnimationRig::layer(int index) noexcept {
    if (index < 0 || index >= layer_count_) {
        discard_ = LayerDesc{};
        return discard_;
    }
    return layers_[static_cast<std::size_t>(index)].desc;
}

const LayerDesc& AnimationRig::layer(int index) const noexcept {
    if (index < 0 || index >= layer_count_) {
        return discard_;
    }
    return layers_[static_cast<std::size_t>(index)].desc;
}

void AnimationRig::set_layer_weight(int index, float w) noexcept {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    layers_[static_cast<std::size_t>(index)].desc.weight = w;
}

void AnimationRig::play_clip(int index, int clip_index, float crossfade) {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    Layer& l = layers_[static_cast<std::size_t>(index)];
    const bool was_blend = l.source != LayerSource::Clip;
    l.source = LayerSource::Clip;
    if (was_blend && bindings_ != nullptr) {
        size_layer_storage(l);
    }

    // AnimationPlayer::crossfade_to, statement for statement.
    if (clip_index == l.clock.current_clip && !was_blend) {
        return;
    }
    l.clock.prev_clip = l.clock.current_clip;
    l.clock.prev_time = l.clock.time;
    l.clock.blend_factor = 0.0f;
    l.clock.blend_duration = crossfade;
    l.clock.current_clip = clip_index;
    l.clock.time = 0.0f;
    l.clock.prev_sample_time = 0.0f;
    l.clock.wrapped = false;
}

void AnimationRig::set_blend_space_1d(int index, const BlendSpace1D& space) {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    Layer& l = layers_[static_cast<std::size_t>(index)];
    l.source = LayerSource::Blend1D;
    l.space_1d = space;
    if (!l.space_1d.finalized()) {
        l.space_1d.finalize();
    }
    if (bindings_ != nullptr) {
        size_layer_storage(l);
    }
}

void AnimationRig::set_blend_space_2d(int index, const BlendSpace2D& space) {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    Layer& l = layers_[static_cast<std::size_t>(index)];
    l.source = LayerSource::Blend2D;
    l.space_2d = space;
    if (bindings_ != nullptr) {
        size_layer_storage(l);
    }
}

void AnimationRig::set_blend_position(int index, float x) noexcept {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    layers_[static_cast<std::size_t>(index)].blend_position.x = x;
}

void AnimationRig::set_blend_position(int index, glm::vec2 p) noexcept {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    layers_[static_cast<std::size_t>(index)].blend_position = p;
}

glm::vec2 AnimationRig::blend_position(int index) const noexcept {
    if (index < 0 || index >= layer_count_) {
        return glm::vec2(0.0f);
    }
    return layers_[static_cast<std::size_t>(index)].blend_position;
}

void AnimationRig::set_layer_time(int index, float t) noexcept {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    layers_[static_cast<std::size_t>(index)].clock.time = t;
    layers_[static_cast<std::size_t>(index)].clock.prev_sample_time = t;
}

float AnimationRig::layer_time(int index) const noexcept {
    if (index < 0 || index >= layer_count_) {
        return 0.0f;
    }
    return layers_[static_cast<std::size_t>(index)].clock.time;
}

void AnimationRig::set_layer_speed(int index, float s) noexcept {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    layers_[static_cast<std::size_t>(index)].clock.speed = s;
}

void AnimationRig::set_layer_loop(int index, bool loop) noexcept {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    layers_[static_cast<std::size_t>(index)].clock.loop = loop;
}

void AnimationRig::set_layer_playing(int index, bool v) noexcept {
    if (index < 0 || index >= layer_count_) {
        return;
    }
    layers_[static_cast<std::size_t>(index)].clock.playing = v;
}

bool AnimationRig::layer_playing(int index) const noexcept {
    if (index < 0 || index >= layer_count_) {
        return false;
    }
    return layers_[static_cast<std::size_t>(index)].clock.playing;
}

float AnimationRig::layer_phase(int index) const noexcept {
    if (index < 0 || index >= layer_count_) {
        return 0.0f;
    }
    return layers_[static_cast<std::size_t>(index)].phase;
}

int AnimationRig::layer_clip(int index) const noexcept {
    if (index < 0 || index >= layer_count_) {
        return -1;
    }
    return layers_[static_cast<std::size_t>(index)].clock.current_clip;
}

void AnimationRig::write_palette(std::span<glm::mat4> dst) const {
    const std::size_t used = std::min(dst.size(), used_joint_count_);
    std::copy_n(bone_matrices_.data(), used, dst.data());
    std::fill(dst.begin() + static_cast<std::ptrdiff_t>(used), dst.end(), glm::mat4(1.0f));
}

// ---------------------------------------------------------------------------
// The frame
// ---------------------------------------------------------------------------

void AnimationRig::advance_clock(Layer& l, const std::vector<AnimationClip>& clips, float dt) noexcept {
    Clock& c = l.clock;
    c.wrapped = false;
    c.prev_sample_time = c.time;

    if (clips.empty() || !c.playing) {
        // Frozen: neither the clock nor the crossfade advances. AnimationPlayer
        // returns early here for the same reason, and because the pose is a pure
        // function of the frozen clock, re-sampling it produces the same
        // matrices it produced last frame.
        return;
    }

    if (l.source != LayerSource::Clip) {
        // Blend-space layers keep one clock; the per-sample times are derived
        // from it (or from the synchronised phase) in evaluate_layer.
        c.time += dt * c.speed;
        return;
    }

    if (c.current_clip < 0 || c.current_clip >= static_cast<int>(clips.size())) {
        c.current_clip = 0;
    }
    const auto& clip = clips[static_cast<std::size_t>(c.current_clip)];

    c.time += dt * c.speed;
    if (clip.duration <= 0.0f) {
        // Zero-duration clip (single keyframe pose); fmod would be UB.
        c.time = 0.0f;
        c.playing = false;
    } else if (c.time >= clip.duration) {
        if (c.loop) {
            c.time = std::fmod(c.time, clip.duration);
            c.wrapped = true;
        } else {
            c.time = clip.duration;
            c.playing = false;
        }
    }

    if (c.blend_factor < 1.0f) {
        if (c.blend_duration > 0.0f) {
            c.blend_factor += dt / c.blend_duration;
        } else {
            c.blend_factor = 1.0f;
        }
        if (c.blend_factor >= 1.0f) {
            c.blend_factor = 1.0f;
            c.prev_clip = -1;
        }

        if (c.prev_clip >= 0 && c.prev_clip < static_cast<int>(clips.size())) {
            const auto& prev = clips[static_cast<std::size_t>(c.prev_clip)];
            c.prev_time += dt * c.speed;
            if (prev.duration > 0.0f && c.prev_time > prev.duration) {
                c.prev_time = std::fmod(c.prev_time, prev.duration);
            }
        }
    }
}

void AnimationRig::evaluate_layer(Layer& l, const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
                                  PoseSpan out, PoseScratch& scratch) {
    const ClipBindings& bindings = (bindings_ != nullptr) ? *bindings_ : null_bindings();
    const std::size_t count = std::min(out.size(), joint_count_);
    const PoseSpan dst = out.subspan(0, count);

    if (l.source == LayerSource::Clip) {
        l.samplers[0].sample(skeleton, clips, bindings, l.clock.current_clip, l.clock.time, dst);

        const bool blending = (l.clock.blend_factor < 1.0f && l.clock.prev_clip >= 0 &&
                               l.clock.prev_clip < static_cast<int>(clips.size()));
        if (blending) {
            const PoseSpan prev = scratch.sample_span().subspan(0, count);
            l.samplers[1].sample(skeleton, clips, bindings, l.clock.prev_clip, l.clock.prev_time, prev);
            // mix / slerp / mix, in that order, on (prev -> current) — the same
            // three calls in the same order as AnimationPlayer's crossfade. Not
            // routed through blend_override on purpose: that operator's
            // end-point early-outs would diverge at f == 0 exactly.
            const float t = l.clock.blend_factor;
            for (std::size_t i = 0; i < count; ++i) {
                dst[i].translation = glm::mix(prev[i].translation, dst[i].translation, t);
                dst[i].rotation = glm::slerp(prev[i].rotation, dst[i].rotation, t);
                dst[i].scale = glm::mix(prev[i].scale, dst[i].scale, t);
            }
        }
        return;
    }

    // ---- blend space ------------------------------------------------------
    std::array<WeightedClip, k_max_blend_samples> active{};
    const std::uint32_t n = (l.source == LayerSource::Blend1D)
                                ? l.space_1d.resolve(l.blend_position.x, active)
                                : l.space_2d.resolve(l.blend_position, active);
    if (n == 0) {
        rest_pose(skeleton, dst);
        return;
    }
    for (std::uint32_t k = 0; k < n; ++k) {
        const int ci = active[k].clip_index;
        active[k].duration =
            (ci >= 0 && ci < static_cast<int>(clips.size())) ? clips[static_cast<std::size_t>(ci)].duration : 0.0f;
    }

    // Phase-synchronised clock: one normalized phase for the whole layer, so
    // every sample sits at the same point in its own stride. Without this a
    // walk/run blend skates.
    const float phase_prev = l.phase;
    if (l.desc.sync_duration) {
        const float eff = effective_duration(std::span<const WeightedClip>(active.data(), n));
        if (eff > 0.0f) {
            const float advanced = l.phase + (l.clock.time - l.clock.prev_sample_time) / eff;
            l.phase = advanced - std::floor(advanced);
            l.clock.wrapped = (advanced >= 1.0f) || (advanced < 0.0f);
        }
    }

    auto sample_time = [&](std::uint32_t k, float ph, float raw) {
        if (l.desc.sync_duration) {
            return ph * active[k].duration;
        }
        return (active[k].duration > 0.0f) ? std::fmod(std::fabs(raw), active[k].duration) : 0.0f;
    };

    // Normalized incremental blend: start from the first contribution, then fold
    // each subsequent one in at w_k / (accumulated + w_k). Exactly the first
    // sample when n == 1, and independent of the weights' absolute scale.
    float accumulated = 0.0f;
    for (std::uint32_t k = 0; k < n; ++k) {
        const std::size_t slot = std::min<std::size_t>(k, l.samplers.size() - 1);
        const float t_now = sample_time(k, l.phase, l.clock.time);
        if (k == 0) {
            l.samplers[slot].sample(skeleton, clips, bindings, active[k].clip_index, t_now, dst);
            accumulated = active[k].weight;
            continue;
        }
        const PoseSpan tmp = scratch.sample_span().subspan(0, count);
        l.samplers[slot].sample(skeleton, clips, bindings, active[k].clip_index, t_now, tmp);
        const float denom = accumulated + active[k].weight;
        BlendParams p;
        p.weight = (denom > 0.0f) ? (active[k].weight / denom) : 0.0f;
        blend_override(dst, tmp, p);
        accumulated = denom;
    }

    // Root motion from a blend space: translations sum weighted, rotation comes
    // from the DOMINANT sample. That rotation is DISCONTINUOUS when the dominant
    // sample changes — a documented limitation, not an oversight; a slerp chain
    // over an unordered active set has no canonical order.
    if (root_motion_.joint >= 0 && &l == &layers_[0]) {
        RootMotionDelta acc;
        std::uint32_t dominant = 0;
        for (std::uint32_t k = 1; k < n; ++k) {
            if (active[k].weight > active[dominant].weight) {
                dominant = k;
            }
        }
        for (std::uint32_t k = 0; k < n; ++k) {
            const float t_now = sample_time(k, l.phase, l.clock.time);
            const float t_prev = sample_time(k, phase_prev, l.clock.prev_sample_time);
            const RootMotionDelta d = extract_root_motion(clips, bindings, active[k].clip_index, t_prev, t_now,
                                                          l.clock.wrapped, root_motion_);
            acc.translation += d.translation * active[k].weight;
            if (k == dominant) {
                acc.rotation = d.rotation;
            }
        }
        root_delta_ = acc;
    }
}

void AnimationRig::evaluate(const Skeleton& skeleton, const std::vector<AnimationClip>& clips, float dt,
                            PoseScratch& scratch) {
    BRACKEN_PROFILE_ZONE("AnimationRig::evaluate");
    root_delta_ = RootMotionDelta{};

    const std::size_t count = std::min(skeleton.joints.size(), static_cast<std::size_t>(MAX_BONES));
    if (count == 0) {
        return;
    }
    // bind() sizes the cursor storage; a rig evaluated against a skeleton it was
    // never bound to would silently sample fewer joints than it writes. That is
    // a HOST mistake and it stays checked in release.
    BRACKEN_CONTRACT(joint_count_ == count,
                     "AnimationRig::evaluate: bound for {} joints, evaluated against {} — call bind() first",
                     joint_count_, count);

    const PoseSpan composed(composed_.data(), count);

    if (layer_count_ == 0) {
        rest_pose(skeleton, composed);
    }

    for (int i = 0; i < layer_count_; ++i) {
        Layer& l = layers_[static_cast<std::size_t>(i)];
        // Clocks advance even for a muted layer: a layer faded back in must not
        // resume from where it was faded out.
        advance_clock(l, clips, dt);

        if (i == 0) {
            // THE BASE. Its mode/weight/mask are ignored by design — it writes
            // the composed pose directly, which is also what removes the
            // `lerp(x, y, 1.0f)` that would spoil bit-identity with the player.
            evaluate_layer(l, skeleton, clips, composed, scratch);
            continue;
        }

        if (l.desc.weight <= 0.0f || l.desc.mask.empty()) {
            continue; // nothing this layer could change
        }

        const PoseSpan layer_pose = scratch.layer_span().subspan(0, count);
        evaluate_layer(l, skeleton, clips, layer_pose, scratch);

        BlendParams p;
        p.weight = l.desc.weight;
        p.mask = &l.desc.mask;
        p.joint_weights = l.desc.joint_weights;

        if (l.desc.mode == BlendMode::Override) {
            blend_override(composed, layer_pose, p);
        } else {
            const PoseSpan reference = scratch.reference_span().subspan(0, count);
            if (l.desc.additive_reference_clip >= 0 && bindings_ != nullptr) {
                sample_clip_pose(skeleton, clips, *bindings_, l.desc.additive_reference_clip,
                                 l.desc.additive_reference_time, reference);
            } else {
                rest_pose(skeleton, reference);
            }
            blend_additive(composed, layer_pose, reference, p);
        }
    }

    if (root_motion_.joint >= 0) {
        if (layer_count_ > 0 && layers_[0].source == LayerSource::Clip) {
            const Clock& c = layers_[0].clock;
            root_delta_ = extract_root_motion(clips, (bindings_ != nullptr) ? *bindings_ : null_bindings(),
                                              c.current_clip, c.prev_sample_time, c.time, c.wrapped, root_motion_);
        }
        if (root_motion_.strip_from_pose) {
            const PoseSpan reference = scratch.reference_span().subspan(0, count);
            rest_pose(skeleton, reference);
            strip_root_motion(composed, reference, root_motion_);
        }
    }

    const std::size_t used = pose_to_matrices(skeleton, ConstPoseSpan(composed_.data(), count),
                                              std::span<glm::mat4>(world_transforms_.data(), MAX_BONES),
                                              std::span<glm::mat4>(bone_matrices_.data(), MAX_BONES));

    // Shrink-only tail identity fill, exactly as AnimationPlayer does it: a
    // per-update rewrite of both MAX_BONES arrays is ~8 KB of dead stores per
    // instance per frame.
    for (std::size_t i = used; i < used_joint_count_; ++i) {
        bone_matrices_[i] = glm::mat4(1.0f);
        world_transforms_[i] = glm::mat4(1.0f);
    }
    used_joint_count_ = used;
}

} // namespace bracken::animation
