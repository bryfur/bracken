#pragma once

#include "bracken/animation/animation_player.hpp"
#include "bracken/animation/animation_rig.hpp"
#include "bracken/animation/blend_space.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace bracken::animation {

struct TransitionCondition {
    enum class Op : uint8_t { GT, LT, EQ, NE, IS_TRUE, IS_FALSE };

    std::string param_name;
    int param_index = -1;
    Op op = Op::GT;
    float threshold = 0.0f;

    bool evaluate(const std::vector<ParamValue>& params) const;
};

struct StateTransition {
    std::string target_state;
    int target_index = -1;
    // Transition fires when every `conditions` (AND) AND at least one of
    // `any_of` (OR, when non-empty) evaluates true.
    std::vector<TransitionCondition> conditions;
    std::vector<TransitionCondition> any_of;
    float crossfade_duration = 0.2f;
    int priority = 0;
};

struct AnimState {
    std::string name;
    std::string clip_name;
    int clip_index = -1;
    bool loop = true;
    float speed = 1.0f;
    std::vector<StateTransition> transitions;

    // ---- rig-driving fields, ignored entirely by update(AnimationPlayer&) ----
    //
    // A state can drive ONE layer of an AnimationRig, with either a single clip
    // (the default, identical to the player behaviour) or a blend space whose
    // position is fed from named parameters. The player overload predates all of
    // this and is untouched by it: a host that never builds a rig sees exactly
    // the state machine it saw before.

    /// Which rig layer this state drives. `update(AnimationRig&, int)` uses its
    /// `layer` argument when that argument is >= 0 and this field otherwise, so
    /// a machine whose states all live on one layer needs no per-state setup.
    int layer = 0;

    /// What the state installs on its layer when entered.
    LayerSource source = LayerSource::Clip;

    /// Used when `source` is `Blend1D` / `Blend2D`. Samples carry clip INDICES,
    /// not names — resolve them with `find_clip_index` at setup, because
    /// `bind_clips` only resolves `clip_name`.
    BlendSpace1D blend_space_1d{};
    BlendSpace2D blend_space_2d{};

    /// Parameter indices driving the blend position each update; -1 = leave that
    /// axis alone. Resolve them once with `param_index`.
    int blend_param_x = -1;
    int blend_param_y = -1;
};

class AnimationStateMachine {
public:
    AnimationStateMachine() = default;

    void add_state(AnimState state);
    void set_default_state(const std::string& name);

    [[nodiscard]] bool bind_clips(const std::vector<AnimationClip>& clips);

    // Resolve a parameter name to a stable index, creating the parameter if
    // needed. Resolve once at setup and use the index overloads below on the
    // per-frame path — the string overloads hash the name on every call.
    int param_index(const std::string& name);
    // Lookup without creating; returns -1 when the parameter doesn't exist.
    int find_param_index(const std::string& name) const;

    // String-keyed convenience API (thin wrappers over the index overloads).
    void set_float(const std::string& name, float v);
    void set_bool(const std::string& name, bool v);
    float get_float(const std::string& name) const;
    bool get_bool(const std::string& name) const;

    // Index-keyed API for per-entity-per-frame updates (no hashing).
    // Out-of-range indices are ignored on set and return defaults on get.
    void set_float(int param_index, float v);
    void set_bool(int param_index, bool v);
    float get_float(int param_index) const;
    bool get_bool(int param_index) const;

    void update(AnimationPlayer& player);

    /**
     * Drive ONE layer of an `AnimationRig` instead of a player.
     *
     * `layer` >= 0 overrides every state's own `AnimState::layer`; `layer` < 0
     * (the default) uses each state's field, which is what lets one machine own
     * a base layer and another own an upper-body layer without either knowing
     * about the other.
     *
     * Every existing behaviour is preserved exactly — transition priority,
     * AND/OR conditions, the `!loop && !playing` clip-ended trigger — because
     * both overloads run the SAME evaluator (`next_transition`). A state whose
     * `source` is a blend space additionally pushes its blend parameters into
     * the layer every update, which is what makes "states drive a blend space"
     * true rather than aspirational.
     */
    void update(AnimationRig& rig, int layer = -1);

    const std::string& current_state() const;
    bool is_bound() const { return bound_; }

private:
    std::vector<AnimState> states_;
    std::unordered_map<std::string, int> state_name_to_index_;

    std::vector<ParamValue> params_;
    std::unordered_map<std::string, int> param_name_to_index_;

    int current_state_ = -1;
    std::string default_state_;
    bool bound_ = false;

    // What has actually been installed on a rig, so update(AnimationRig&) can
    // apply the DEFAULT state once without re-installing (and re-allocating) a
    // blend space every frame. The player path has no equivalent and needs
    // none: a player defaults to clip 0 and a state's clip is applied on the
    // first transition.
    int applied_state_ = -1;
    int applied_layer_ = -1;

    int ensure_param(const std::string& name);
    int find_param(const std::string& name) const;

    void enter_state(int state_index, AnimationPlayer& player, float crossfade);
    void enter_state(int state_index, AnimationRig& rig, int layer, float crossfade);

    // The one transition evaluator both update() overloads share. Returns the
    // index of the state to enter, or -1 to stay put. Factored so the player and
    // rig paths cannot drift — a second copy of this loop is how "the rig
    // ignores transition priority" bugs get written.
    [[nodiscard]] int next_transition(const AnimState& state, bool clip_ended,
                                      float& out_crossfade) const;
};

} // namespace bracken::animation
