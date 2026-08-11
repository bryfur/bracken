#include "animation_state_machine.hpp"

#include "bracken/core/profiler.hpp"

#include <algorithm>
#include <cmath>

namespace bracken::animation {

bool TransitionCondition::evaluate(const std::vector<ParamValue>& params) const {
    if (param_index < 0 || param_index >= static_cast<int>(params.size())) {
        return false;
    }

    const auto& val = params[param_index];

    switch (op) {
        case Op::IS_TRUE:
            return val.type == ParamType::Bool && val.b;
        case Op::IS_FALSE:
            return val.type == ParamType::Bool && !val.b;
        case Op::GT:
            return val.type == ParamType::Float && val.f > threshold;
        case Op::LT:
            return val.type == ParamType::Float && val.f < threshold;
        case Op::EQ:
            return val.type == ParamType::Float && std::abs(val.f - threshold) < 0.001f;
        case Op::NE:
            return val.type == ParamType::Float && std::abs(val.f - threshold) >= 0.001f;
    }
    return false;
}

void AnimationStateMachine::add_state(AnimState state) {
    std::string name = state.name;
    std::sort(state.transitions.begin(), state.transitions.end(),
              [](const StateTransition& a, const StateTransition& b) { return a.priority > b.priority; });

    auto it = state_name_to_index_.find(name);
    if (it != state_name_to_index_.end()) {
        states_[it->second] = std::move(state);
    } else {
        int index = static_cast<int>(states_.size());
        state_name_to_index_[name] = index;
        states_.push_back(std::move(state));
    }
    // Whatever a rig currently has installed was installed from the OLD table.
    // Replacing a state in place keeps its index, so without this the
    // "already applied" fast path in update(AnimationRig&) would skip
    // re-installing it and the layer would keep playing the previous clip or
    // blend space forever.
    invalidate_applied();
}

void AnimationStateMachine::set_default_state(const std::string& name) {
    default_state_ = name;
}

int AnimationStateMachine::ensure_param(const std::string& name) {
    auto it = param_name_to_index_.find(name);
    if (it != param_name_to_index_.end()) {
        return it->second;
    }
    int index = static_cast<int>(params_.size());
    param_name_to_index_[name] = index;
    params_.emplace_back(0.0f);
    return index;
}

int AnimationStateMachine::find_param(const std::string& name) const {
    auto it = param_name_to_index_.find(name);
    if (it != param_name_to_index_.end()) {
        return it->second;
    }
    return -1;
}

bool AnimationStateMachine::bind_clips(const std::vector<AnimationClip>& clips) {
    bool all_found = true;
    for (auto& state : states_) {
        state.clip_index = -1;
        for (size_t i = 0; i < clips.size(); i++) {
            if (clips[i].name == state.clip_name) {
                state.clip_index = static_cast<int>(i);
                break;
            }
        }
        if (state.clip_index < 0) {
            all_found = false;
        }
    }

    for (auto& state : states_) {
        for (auto& transition : state.transitions) {
            auto sit = state_name_to_index_.find(transition.target_state);
            transition.target_index = (sit != state_name_to_index_.end()) ? sit->second : -1;

            for (auto& cond : transition.conditions) {
                cond.param_index = ensure_param(cond.param_name);
            }
            for (auto& cond : transition.any_of) {
                cond.param_index = ensure_param(cond.param_name);
            }
        }
    }

    bound_ = true;
    // Clip INDICES just moved. A rig driven by this machine is holding indices
    // resolved against the previous clip table, so the next update() must
    // re-install the current state even though its index has not changed —
    // this is what a model hot-reload looks like from here.
    invalidate_applied();

    if (!default_state_.empty()) {
        auto it = state_name_to_index_.find(default_state_);
        if (it != state_name_to_index_.end()) {
            current_state_ = it->second;
        }
    }

    return all_found;
}

int AnimationStateMachine::param_index(const std::string& name) {
    return ensure_param(name);
}

int AnimationStateMachine::find_param_index(const std::string& name) const {
    return find_param(name);
}

void AnimationStateMachine::set_float(const std::string& name, float v) {
    set_float(ensure_param(name), v);
}

void AnimationStateMachine::set_bool(const std::string& name, bool v) {
    set_bool(ensure_param(name), v);
}

float AnimationStateMachine::get_float(const std::string& name) const {
    return get_float(find_param(name));
}

bool AnimationStateMachine::get_bool(const std::string& name) const {
    return get_bool(find_param(name));
}

void AnimationStateMachine::set_float(int param_index, float v) {
    if (param_index < 0 || param_index >= static_cast<int>(params_.size())) {
        return;
    }
    params_[param_index] = ParamValue{v};
}

void AnimationStateMachine::set_bool(int param_index, bool v) {
    if (param_index < 0 || param_index >= static_cast<int>(params_.size())) {
        return;
    }
    params_[param_index] = ParamValue{v};
}

float AnimationStateMachine::get_float(int param_index) const {
    if (param_index >= 0 && param_index < static_cast<int>(params_.size()) &&
        params_[param_index].type == ParamType::Float) {
        return params_[param_index].f;
    }
    return 0.0f;
}

bool AnimationStateMachine::get_bool(int param_index) const {
    if (param_index >= 0 && param_index < static_cast<int>(params_.size()) &&
        params_[param_index].type == ParamType::Bool) {
        return params_[param_index].b;
    }
    return false;
}

const std::string& AnimationStateMachine::current_state() const {
    static const std::string empty;
    if (current_state_ >= 0 && current_state_ < static_cast<int>(states_.size())) {
        return states_[current_state_].name;
    }
    return empty;
}

void AnimationStateMachine::enter_state(int state_index, AnimationPlayer& player, float crossfade) {
    if (state_index < 0 || state_index >= static_cast<int>(states_.size())) {
        return;
    }

    const auto& state = states_[state_index];
    if (state.clip_index < 0) {
        return;
    }

    current_state_ = state_index;
    player.crossfade_to(state.clip_index, crossfade);
    player.set_playing(true);
    player.set_loop(state.loop);
    player.set_speed(state.speed);
}

int AnimationStateMachine::next_transition(const AnimState& state, bool clip_ended,
                                           float& out_crossfade) const {
    for (const auto& transition : state.transitions) {
        bool passes = clip_ended;
        if (!passes) {
            passes = true;
            for (const auto& cond : transition.conditions) {
                if (!cond.evaluate(params_)) {
                    passes = false;
                    break;
                }
            }
            if (passes && !transition.any_of.empty()) {
                bool any = false;
                for (const auto& cond : transition.any_of) {
                    if (cond.evaluate(params_)) {
                        any = true;
                        break;
                    }
                }
                passes = any;
            }
        }

        if (passes) {
            out_crossfade = transition.crossfade_duration;
            return transition.target_index;
        }
    }
    return -1;
}

void AnimationStateMachine::update(AnimationPlayer& player) {
    // update() is also the transition-evaluation entry point; one zone covers both.
    BRACKEN_PROFILE_ZONE_COLOR("AnimationStateMachine::update", 0x8E44AD);
    if (!bound_ || current_state_ < 0) {
        return;
    }
    if (current_state_ >= static_cast<int>(states_.size())) {
        return;
    }

    const auto& state = states_[current_state_];

    bool clip_ended = !state.loop && !player.is_playing();

    float crossfade = 0.0f;
    const int target = next_transition(state, clip_ended, crossfade);
    if (target >= 0) {
        enter_state(target, player, crossfade);
    }
}

void AnimationStateMachine::enter_state(int state_index, AnimationRig& rig, int layer, float crossfade) {
    if (state_index < 0 || state_index >= static_cast<int>(states_.size())) {
        return;
    }

    const auto& state = states_[state_index];
    const int target_layer = (layer >= 0) ? layer : state.layer;

    // A Clip state with no bound clip is refused, exactly as the player overload
    // refuses it — entering a state that would sample nothing leaves the
    // character frozen with no diagnostic. A blend-space state has no
    // clip_index, so the check does not apply to it.
    if (state.source == LayerSource::Clip && state.clip_index < 0) {
        return;
    }

    current_state_ = state_index;
    switch (state.source) {
        case LayerSource::Clip:
            rig.play_clip(target_layer, state.clip_index, crossfade);
            break;
        case LayerSource::Blend1D:
            // Installing a blend space sizes the layer's samplers, so this
            // ALLOCATES. That is why it happens on state entry and never
            // per-frame, and why applied_state_ exists.
            rig.set_blend_space_1d(target_layer, state.blend_space_1d);
            break;
        case LayerSource::Blend2D:
            rig.set_blend_space_2d(target_layer, state.blend_space_2d);
            break;
    }
    rig.set_layer_playing(target_layer, true);
    rig.set_layer_loop(target_layer, state.loop);
    rig.set_layer_speed(target_layer, state.speed);

    applied_state_ = state_index;
    applied_layer_ = target_layer;
}

void AnimationStateMachine::update(AnimationRig& rig, int layer) {
    BRACKEN_PROFILE_ZONE_COLOR("AnimationStateMachine::update(rig)", 0x8E44AD);
    if (!bound_ || current_state_ < 0) {
        return;
    }
    if (current_state_ >= static_cast<int>(states_.size())) {
        return;
    }

    const int target_layer = (layer >= 0) ? layer : states_[current_state_].layer;

    // The DEFAULT state is never "entered" — bind_clips just sets it — so
    // install it the first time we see a rig. The player path gets away without
    // this because an AnimationPlayer already defaults to clip 0; a layer whose
    // blend space was never installed would sample nothing at all.
    if (applied_state_ != current_state_ || applied_layer_ != target_layer) {
        enter_state(current_state_, rig, layer, 0.0f);
    }

    const auto& state = states_[current_state_];

    // Blend position is pushed EVERY update, not only on entry: the whole point
    // of a blend space is that it tracks a continuously-varying parameter.
    if (state.source != LayerSource::Clip) {
        glm::vec2 p = rig.blend_position(target_layer);
        if (state.blend_param_x >= 0) {
            p.x = get_float(state.blend_param_x);
        }
        if (state.blend_param_y >= 0) {
            p.y = get_float(state.blend_param_y);
        }
        rig.set_blend_position(target_layer, p);
    }

    const bool clip_ended = !state.loop && !rig.layer_playing(target_layer);

    float crossfade = 0.0f;
    const int target = next_transition(state, clip_ended, crossfade);
    if (target >= 0) {
        enter_state(target, rig, layer, crossfade);
    }
}

} // namespace bracken::animation
