#pragma once

#include "bracken/animation/animation_player.hpp"
#include "bracken/animation/animation_state_machine.hpp"
#include "bracken/animation/animation_types.hpp"
#include "bracken/application.hpp"
#include "bracken/contracts/environment.hpp"
#include "bracken/effects/effect_definition.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/physics/physics_world.hpp"
#include "bracken/scene/frame_sink.hpp"
#include "day_weather.hpp"
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace playground {

/**
 * Medieval-village demo: walk a villager around a hamlet assembled from the
 * modular building kit (tools/build_medieval_assets.py) and pick up glowing
 * orbs. Doubles as the engine's integration test — it deliberately touches
 * every major subsystem (terrain + splatmap, grass with exclusion zones,
 * skybox, glb model loading with full PBR materials, instanced props, skinned
 * animation + state machine + foot IK, dynamic point/spot lights, particle
 * effects incl. continuous chimney smoke, billboards, debug lines, UI,
 * graphics settings, the water renderer + wave sim via the west river). Run
 * with --smoke N to self-drive for N frames and exit.
 *
 * Ported to the pipelined frame API (docs/multithreading_plan.md Phase 4):
 * the whole game frame — input, character/animation/IK, camera, orbs,
 * effects, frame description — lives in on_simulate (a pool job, overlapped
 * with the previous frame's rendering). on_main_frame is the frame fence:
 * render-side touches (water ripples, smoke-run settings flips + stats,
 * screenshots) and the DevUI panel-data exchange happen there, with no sim
 * in flight.
 *
 * The village assets are optional: without assets/generated (see the tool's
 * header comment) the demo falls back to the procedural-only scene so the
 * smoke test still passes on an asset-less checkout.
 */
class PlaygroundGame : public bracken::Application {
public:
    explicit PlaygroundGame(int smoke_frames = -1) : smoke_frames_(smoke_frames) {}

    /** Where the smoke run saves its screenshot (main() verifies it after). */
    static constexpr const char* SMOKE_SCREENSHOT_PATH = "playground_smoke.png";

protected:
    bool on_init() override;
    void on_shutdown() override;
    void on_simulate(bracken::SimFrame& sim) override;
    void on_main_frame(bracken::MainFrame& mf) override;

private:
    struct Orb {
        glm::vec3 pos;
        float bob_phase;
    };
    struct Prop {
        bracken::ModelHandle model;
        glm::mat4 transform;
    };
    struct PendingRipple {
        glm::vec3 pos;
        float radius;
        float strength;
    };
    /**
     * A static prop recorded at placement time for setup_physics(): the world
     * AABB is derived from the model bounds x the placement transform once
     * physics is up (models are all loaded by then). trunk=true shrinks the
     * box to the tree's trunk footprint so canopies don't block movement;
     * trunk_half > 0 is a measured per-model footprint half-extent
     * (pre-scale model units, box centered on the placement origin), 0 falls
     * back to the AABB-fraction rule (procedural generator trees).
     */
    struct PendingCollider {
        bracken::ModelHandle model;
        glm::mat4 transform;
        bool trunk;
        float trunk_half = 0.0f;
    };

    /**
     * Fence-owned copy of everything the DevUI "Playground" panel shows or
     * edits. The overlay draws DURING rendering, while the NEXT frame's sim
     * job runs on the pool (see dev_ui.hpp "PANEL DATA SAFETY"), so the panel
     * body reads and writes ONLY this struct; sync_dev_panel() — called from
     * on_main_frame, the one main-thread point with no sim in flight —
     * refreshes the display fields from live game state and applies the
     * widget edits back to it.
     */
    struct DevPanelView {
        // Read-only display, refreshed at the fence.
        int orbs_collected = 0;
        glm::vec3 player_pos{0.0f};
        std::string anim_state;
        bool village_loaded = false;
        // Chrome-ball physics state (copied from the sim-written mirrors —
        // the fence never calls into PhysicsWorld itself; physics is sim-stage).
        glm::vec3 ball_pos{0.0f};
        float ball_speed = 0.0f;
        bool ball_active = false;
        // Interactive widgets (panel-owned; applied to game state at the fence).
        bool flashlight = false;
        bool debug_draw = false;
        int spawn_burst_clicks = 0;
        // Time of day + weather (mirrors DayWeather's control fields; readouts
        // below are refreshed from the live sim state every fence).
        float tod_hour = 0.0f;
        float tod_speed = 0.0f;
        bool tod_animate = true;
        int weather_mode = 0;      // WeatherMode as int (Auto/Clear/Overcast/Rain)
        float rain_slider = 0.85f; // manual target for the Rain mode
        float overcast_now = 0.0f; // readout
        float rain_now = 0.0f;     // readout
        std::string weather_label; // readout
    };

    void build_heightmap();
    void register_models();
    void setup_environment();
    void setup_village(); // village model + splatmap + grass exclusions
    void scatter_wilds(); // applies the hand-authored layout (scene_layout.cpp)
                          // + the grass-tuft ground cover; procedural fallback
    void build_orb_spots();
    void setup_state_machine();
    void setup_effects();
    void setup_physics(); // world + terrain heightfield + static prop boxes
                          // + chrome ball + character pusher capsule; runs
                          // after the heightmap/village/props exist
    void setup_smoke_run(); // one-time smoke config (env-var hooks, forced
                            // settings, camera framing) — main thread, at init

    // Sim-side (on_simulate; pool job — no GPU/SDL/main-thread-tier calls):
    void update_smoke_driver(float dt); // movement synthesis: orbit/chase/attack
    void handle_input(const bracken::SimFrame& sim);
    void update_player(float dt);
    void update_camera(bracken::SimFrame& sim); // non-const: records camera-forward feedback
    void update_orbs(float dt);
    void update_chimneys(float dt);
    void update_day_weather(float dt); // sun clock + weather -> env staging,
                                       // rain layer/splash/ripple emission
    void update_physics(float dt); // kinematic pusher, kick, step, readback
    void build_frame(bracken::SimFrame& sim); // describe the frame via sim.frame()
    void build_ui(bracken::scene::FrameSink& fb);

    // Fence-side (on_main_frame; main thread, no sim in flight):
    void sync_dev_panel(); // apply panel edits + refresh the panel snapshot
    void update_smoke_fence(bracken::MainFrame& mf); // settings flips, probe,
                                                     // screenshot, exit + stats

    float rand01();
    glm::vec3 random_ground_pos(float margin);
    /** Village metres (layout space) -> world units, terrain height applied. */
    glm::vec3 village_pos(float vx, float vz, float vy = 0.0f);

    // World
    bracken::Heightmap heightmap_;
    std::vector<Prop> props_;
    std::vector<Orb> orbs_;
    uint32_t rng_ = 0x2f6e2b1u;

    // Physics (bracken::physics). Sim-stage owned after init: everything
    // below is touched only from on_init/on_shutdown (no sim in flight) and
    // on_simulate — NEVER from on_main_frame. The fence reads only the plain
    // ball_* mirrors the sim frame wrote (single writer per phase, ordering
    // rides the sim kick/join edges — same contract as env_dirty_).
    bracken::physics::PhysicsWorld physics_;
    bool physics_ready_ = false;
    std::vector<bracken::physics::BodyId> static_bodies_; // heightfield + statics
    // The heightfield alone — excluded from the character's wall query
    // (terrain height is heightmap-snapped, not collided).
    bracken::physics::BodyId terrain_body_ = bracken::physics::INVALID_BODY_ID;
    bracken::physics::BodyId ball_body_ = bracken::physics::INVALID_BODY_ID;
    bracken::physics::BodyId character_body_ = bracken::physics::INVALID_BODY_ID;
    float ball_radius_ = 45.0f; // refined from the chrome model bounds at init
    std::vector<PendingCollider> pending_colliders_; // filled by scatter_wilds,
                                                     // drained by setup_physics
    // Kick state: one kick per attack swing — armed on the swing's rising
    // edge, disarmed when it lands a kick or the swing ends.
    bool attack_was_active_ = false;
    bool kick_armed_ = false;
    // Sim-written mirrors for the dev panel snapshot (read at the fence).
    glm::vec3 ball_pos_{0.0f};
    float ball_speed_ = 0.0f;
    bool ball_active_ = false;

    // Village
    bool has_village_ = false;
    bracken::ModelHandle village_model_ = bracken::INVALID_MODEL_HANDLE;
    glm::mat4 village_transform_{1.0f};
    std::vector<glm::vec3> orb_spots_;    // world space
    std::vector<glm::vec3> chimney_tops_; // world space
    float smoke_spawn_timer_ = 0.0f;
    // Set by update_chimneys, consumed by build_frame later in the same
    // on_simulate call (spawns are queued into that frame's packet).
    bool pending_chimney_smoke_ = false;
    // The scene's golden-hour sun — the smoke driver's day-cycle swing must
    // restore THIS look, not the engine default.
    glm::vec3 scene_sun_dir_{0.0f};
    glm::vec3 scene_sun_color_{1.0f};

    // Environment — the host-owned sun/sky/fog state. The renderer's live
    // EnvironmentState is render-side in the pipelined loop, so the game
    // never touches it after init; it edits this copy (sim- or fence-side —
    // the phases never overlap) and STAGES it into the frame
    // (fb.set_environment) when it changed. Staging only on change keeps the
    // DevUI Environment tab's live edits effective between host changes.
    bracken::render::EnvironmentState env_;
    // Set wherever env_ is edited (init/fence), cleared by build_frame when
    // it stages env_. Single writer per pipeline phase; ordering rides the
    // sim kick/join edges — no atomics needed.
    bool env_dirty_ = false;

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

    // River (wave sim). The WaterRenderer is render-side: sim code queues
    // ripple impulses here and on_main_frame forwards them at the fence
    // (single writer per phase, ordered by the sim kick/join edges).
    bool river_active_ = false; // cached at init; immutable afterwards
    std::vector<PendingRipple> pending_ripples_;
    float wade_ripple_timer_ = 0.0f;    // rate-limits wading ripple impulses
    float ambient_ripple_timer_ = 0.0f; // occasional ambient rings on the river

    // Game state
    bracken::ModelHandle orb_model_ = bracken::INVALID_MODEL_HANDLE;
    bracken::ModelHandle chrome_model_ = bracken::INVALID_MODEL_HANDLE; // specular-IBL proof prop
    glm::mat4 chrome_transform_{1.0f};
    int orbs_collected_ = 0;
    bool flashlight_ = false;
    bool debug_draw_ = false;
    std::vector<glm::vec3> pending_pickup_bursts_;
    bool pending_attack_arc_ = false;

    // Time of day + weather (sim-owned; see day_weather.hpp). When disabled
    // (PLAYGROUND_SUN_ELEVATION pins the sun, or plain --smoke runs keep the
    // legacy fixed environment deterministic) it never writes env_.
    DayWeather day_weather_;
    bool day_weather_enabled_ = true;
    // Rain emission state (sim-owned). The rain layer is a short continuous
    // volume effect re-spawned above the player on a cadence (continuous
    // emitters cannot move); splashes are burst effects at ray-cast impact
    // points, drained by build_frame like the pickup bursts.
    float rain_layer_timer_ = 0.0f;
    float splash_accum_ = 0.0f;
    bool pending_rain_spawn_ = false;
    std::vector<glm::vec3> pending_splashes_;

    // Dev overlay panel data (see DevPanelView). The *_applied_ baselines let
    // the fence edge-detect widget changes so sim-side writes (the F key,
    // the advancing clock, Auto-mode weather) aren't overwritten by a stale
    // widget value.
    DevPanelView panel_;
    bool panel_applied_flashlight_ = false;
    bool panel_applied_debug_draw_ = false;
    float panel_applied_tod_hour_ = 0.0f;
    float panel_applied_tod_speed_ = 0.0f;
    bool panel_applied_tod_animate_ = true;
    int panel_applied_weather_mode_ = 0;
    float panel_applied_rain_slider_ = 0.85f;

    // Effect definitions (must outlive all spawned instances)
    bracken::EffectDefinition pickup_burst_;
    bracken::EffectDefinition attack_arc_;
    bracken::EffectDefinition chimney_smoke_;
    bracken::EffectDefinition rain_layer_;
    bracken::EffectDefinition rain_splash_;

    float time_ = 0.0f; // mirror of sim.time() (accumulated dt)
    int frame_ = 0;     // 1-based sim frame counter (sim.frame_index() + 1)
    int smoke_frames_ = -1;       // >0: self-drive and quit after this many frames
    uint64_t smoke_start_ns_ = 0; // first-fence timestamp for the run-average fps
    bool smoke_local_shadows_ = true; // PLAYGROUND_LOCAL_SHADOWS=0 forces off (A/B hook)
    bool smoke_sun_shadows_off_ = false; // PLAYGROUND_SUN_SHADOWS=0 forces the CSM off
    bool smoke_exposure_probe_ = false; // PLAYGROUND_EXPOSURE_PROBE=1 prints per-frame adapted EV
    bool smoke_sun_forced_ = false;     // PLAYGROUND_SUN_ELEVATION=<deg> pins the sun for the run
    bool smoke_aa_forced_ = false;      // PLAYGROUND_AA=0/1/2 pins the AA mode for the run
    bool smoke_ao_forced_ = false;      // PLAYGROUND_AO=0/1/2 pins the AO mode for the run
    // PLAYGROUND_CAMERA="vx,vz,yaw,pitch,dist_m" (village metres) pins the
    // camera for the whole run — the art close-up hook.
    bool smoke_cam_forced_ = false;
    bool smoke_cam_settled_ = false; // springs pre-settled on the first pinned frame
    glm::vec3 smoke_cam_target_{0.0f};
    float smoke_cam_yaw_ = 0.0f;
    float smoke_cam_pitch_ = 10.0f;
    float smoke_cam_dist_ = 500.0f;
    glm::vec3 smoke_forced_sun_dir_{0.0f}; // the pinned sun_dir (direction light travels)
    int screenshot_counter_ = 0;
};

} // namespace playground
