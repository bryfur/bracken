#pragma once

#include "bracken/animation/animation_player.hpp"
#include "bracken/animation/animation_state_machine.hpp"
#include "bracken/animation/animation_types.hpp"
#include "bracken/application.hpp"
#include "bracken/effect_definition.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/scene/render_scene.hpp"
#include "bracken/scene/ui_scene.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace playground {

/**
 * Orb-collector demo: walk a procedurally-built character around procedural
 * terrain and pick up glowing orbs. Doubles as the engine's integration test —
 * it deliberately touches every major subsystem (terrain, grass, skybox,
 * instanced props, skinned animation + state machine + foot IK, dynamic
 * point/spot lights, particle effects, billboards, debug lines, UI, graphics
 * settings). Run with --smoke N to self-drive for N frames and exit.
 */
class PlaygroundGame : public bracken::Application {
public:
    explicit PlaygroundGame(int smoke_frames = -1) : smoke_frames_(smoke_frames) {}

    // Public wrappers so main() can drive the protected lifecycle hooks.
    bool start() { return on_init(); }
    void stop() { on_shutdown(); }

    /** Where the smoke run saves its screenshot (main() verifies it after). */
    static constexpr const char* SMOKE_SCREENSHOT_PATH = "playground_smoke.png";

protected:
    bool on_init() override;
    void on_shutdown() override;
    void on_update(float dt) override;
    void on_render() override;

private:
    struct Orb {
        glm::vec3 pos;
        float bob_phase;
    };
    struct Prop {
        bracken::ModelHandle model;
        glm::mat4 transform;
    };

    void build_heightmap();
    void register_models();
    void scatter_props();
    void setup_state_machine();
    void setup_effects();

    void update_smoke_driver(float dt);
    void handle_input(float dt);
    void update_player(float dt);
    void update_camera(float dt);
    void update_orbs(float dt);
    void build_ui();

    float rand01();
    glm::vec3 random_ground_pos(float margin);

    // World
    bracken::Heightmap heightmap_;
    std::vector<Prop> props_;
    std::vector<Orb> orbs_;
    uint32_t rng_ = 0x2f6e2b1u;

    // Player
    bracken::ModelHandle player_model_ = bracken::INVALID_MODEL_HANDLE;
    glm::vec3 player_pos_{0.0f};
    glm::vec3 player_vel_{0.0f};
    float player_speed_ = 0.0f;
    bool running_input_ = false;
    glm::vec3 move_dir_{0.0f}; // set by handle_input / smoke driver
    bracken::animation::AnimationPlayer anim_player_;
    bracken::animation::AnimationStateMachine anim_sm_;
    int param_speed_ = -1;     // anim_sm_ parameter indices, resolved once in
    int param_attacking_ = -1; // setup_state_machine (no per-frame name hashing)
    bracken::animation::RotationSmoother facing_;
    bracken::animation::FootIKSmoother foot_smoother_;
    glm::mat4 player_transform_{1.0f};
    float attack_timer_ = 0.0f;

    // Game state
    bracken::ModelHandle orb_model_ = bracken::INVALID_MODEL_HANDLE;
    int orbs_collected_ = 0;
    bool flashlight_ = false;
    bool debug_draw_ = false;
    std::vector<glm::vec3> pending_pickup_bursts_;
    bool pending_attack_arc_ = false;

    // Effect definitions (must outlive all spawned instances)
    bracken::EffectDefinition pickup_burst_;
    bracken::EffectDefinition attack_arc_;

    // Frame scenes (members so vector capacity persists across frames)
    bracken::scene::RenderScene scene_;
    bracken::scene::UIScene ui_;

    float time_ = 0.0f;
    float last_dt_ = 0.0f;
    int frame_ = 0;
    int smoke_frames_ = -1; // >0: self-drive and quit after this many frames
    int screenshot_counter_ = 0;
};

} // namespace playground
