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
#include "sfx.hpp"

#include "bracken/action_map.hpp"
#include "bracken/animation/attachment.hpp"
#include "bracken/contracts/shadow_capsule.hpp"
#include "bracken/core/random.hpp"
#include "bracken/navigation/nav_grid.hpp"
#include "bracken/physics/character_controller.hpp"
#include <glm/glm.hpp>
#include <map>
#include <string>
#include <vector>

namespace playground {

/** Bundle every .glb under assets/models into one pack file and exit. The
 *  shippable form of the asset directory: one artifact, one file handle, and a
 *  sorted TOC so lookups are a binary search rather than a scan. */
[[nodiscard]] bool build_asset_pack(const std::string& out_path);


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
 * The village assets live in assets/ and are tracked in git. They are still
 * optional at build time: on a sparse checkout without them the demo falls
 * back to the procedural-only scene so the smoke test still passes.
 */
class PlaygroundGame : public bracken::Application {
public:
    /** @param smoke_frames  >0: self-drive for N frames and exit.
     *  @param engine_camera false (--engine-camera=off) makes the demo drive
     *         the camera rig by hand instead of letting the engine do it —
     *         the coverage that keeps Application's escape hatch honest.
     *  @param rt_bench (--rt-bench N) turns the same self-driven run into a
     *         MEASUREMENT JIG: pinned camera, no screenshot, none of the
     *         scripted mid-run settings flips, and a machine-readable per-pass
     *         timing table at exit instead of the [smoke] lines. */
    explicit PlaygroundGame(int smoke_frames = -1, bool engine_camera = true, bool rt_bench = false)
        : engine_camera_(engine_camera), smoke_frames_(smoke_frames), rt_bench_(rt_bench) {}

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
    /// Pin rt_shadows/rt_reflections from PLAYGROUND_RT. An UNSET var means
    /// both OFF: the flags are persisted in settings.cfg, so "leave them
    /// alone" silently inherited ray tracing from whatever ran last.
    void apply_rt_mode(bracken::GraphicsSettings& gs) const;
    /// Pin gi_mode from PLAYGROUND_GI, on exactly the same terms as
    /// apply_rt_mode: an UNSET var means OFF, because gi_mode is persisted too
    /// and "leave it alone" would inherit a 30 MB probe field from whatever ran
    /// last.
    void apply_gi_mode(bracken::GraphicsSettings& gs) const;

    // Sim-side (on_simulate; pool job — no GPU/SDL/main-thread-tier calls):
    void update_smoke_driver(float dt); // movement synthesis: orbit/chase/attack
    void handle_input(const bracken::SimFrame& sim);
    void quicksave();  // writes playground_quicksave.sav (atomic)
    void quickload();  // restores it if present
    void update_player(float dt);
    /// Resolve the capsule bones once, at model load. See capsule_bones_.
    void resolve_capsule_bones(const bracken::Model& model);
    /// Rebuild `shadow_capsules_` from the character's LIVE joint world
    /// transforms. Call after every procedural layer has been applied and
    /// before the frame is described.
    void rebuild_shadow_capsules(const bracken::Model& model);
    void update_camera(bracken::SimFrame& sim); // non-const: records camera-forward feedback
    void update_orbs(float dt);
    void update_wanderer(float dt); // nav-grid pathing demo
    void update_chimneys(float dt);
    void update_day_weather(float dt); // sun clock + weather -> env staging,
                                       // rain layer/splash/ripple emission
    void update_physics(float dt); // kinematic pusher, kick, step, readback
    void build_frame(bracken::SimFrame& sim); // describe the frame via sim.frame()
    void build_ui(const bracken::SimFrame& sim, bracken::scene::FrameSink& fb);

    // Fence-side (on_main_frame; main thread, no sim in flight):
    void sync_dev_panel(); // apply panel edits + refresh the panel snapshot
    void update_smoke_fence(bracken::MainFrame& mf); // settings flips, probe,
                                                     // screenshot, exit + stats
    /// --rt-bench: sample this frame's pass times, and at the end of the run
    /// print the table + exit. Replaces update_smoke_fence entirely — the
    /// scripted flips it runs would change what is being measured mid-run.
    void update_rt_bench_fence();
    void print_rt_bench_report();

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

    SfxBank sfx_{};                     // synthesized at boot; no audio assets
    // Footfalls are driven by distance travelled, not by a timer: that way the
    // step rate follows the animation instead of drifting against it when the
    // player accelerates.
    // Named actions rather than scancodes at the call site. Bindings load
    // from playground_bindings.cfg when present, so they are rebindable
    // without a rebuild.
    bracken::ActionMap actions_;
    bracken::ActionId act_forward_{}, act_back_{}, act_left_{}, act_right_{};
    bracken::ActionId act_run_{}, act_attack_{}, act_flashlight_{};
    bracken::ActionId act_quicksave_{}, act_quickload_{};
    bracken::ActionId act_jump_{}, act_crouch_{};

    // --- Tier 3 demos ---------------------------------------------------
    /// Replaces the old `player_pos_.y = terrain_height(...)` snap: real
    /// slopes, step-up, air time and jump.
    bracken::physics::CharacterController character_;
    /// A lantern carried in the character's hand, placed by the bone rather
    /// than by hand-written matrix math.
    bracken::animation::Attachment lantern_attach_{};
    bracken::ModelHandle lantern_model_ = bracken::INVALID_MODEL_HANDLE;

    // --- Ray-traced shadow proxies (docs/rt_lighting_plan.md §5.8) --------
    /// The character's capsule occluders, rebuilt from the LIVE skeleton every
    /// frame after the animation + IK + lean layers have run, and emitted
    /// through the frame sink. A member rather than a local so the per-frame
    /// derivation allocates nothing once warm.
    ///
    /// Why the host owns this and not the engine: the engine has no idea which
    /// bones of an arbitrary skeleton are a torso and which are a sleeve, and
    /// guessing from names is exactly the kind of content assumption the
    /// engine/game split exists to keep out. Deriving the proxy is a game's
    /// job; testing it against rays is the engine's.
    std::vector<bracken::render::rt::ShadowCapsule> shadow_capsules_;
    /// Joint indices resolved ONCE at load (find_joint is a linear name scan).
    /// `valid` is false when the character model's bone names did not match,
    /// in which case the derivation falls back to a single whole-body capsule.
    struct CapsuleBones {
        bracken::animation::JointIndex hips{}, spine{}, head{};
        bracken::animation::JointIndex left_upper_leg{}, left_lower_leg{}, left_foot{};
        bracken::animation::JointIndex right_upper_leg{}, right_lower_leg{}, right_foot{};
        bracken::animation::JointIndex left_arm{}, right_arm{};
        bool valid = false;
    };
    CapsuleBones capsule_bones_{};
    /// Logged once, not per frame: a name mismatch is a standing content
    /// condition, so the warning is a fact about the asset and not an event.
    bool capsule_fallback_logged_ = false;
    /// Walkability + A* over the terrain, for the wandering villager.
    bracken::nav::NavGrid nav_;
    /// A villager that paths around the village on the nav grid.
    struct Wanderer {
        glm::vec3 position{0.0f};
        float facing = 0.0f;
        std::vector<glm::vec3> path;
        std::size_t waypoint = 0;
        float repath_timer = 0.0f;
    } wanderer_;
    bool nav_debug_draw_ = false;
    bool jump_requested_ = false; // edge, consumed by the controller
    bool crouch_held_ = false;
    int smoke_grounded_frames_ = 0;
    int smoke_air_frames_ = 0;
    float smoke_max_height_ = 0.0f;
    glm::vec3 wanderer_start_{0.0f};

    // Footfalls are driven by distance travelled, not by a timer, so the step
    // rate follows the animation instead of drifting against it as the player
    // accelerates. Only the synth build has footsteps to schedule.
    float step_distance_ = 0.0f;
    bool was_in_water_ = false;         // rising edge = entry splash
    // Footstep pitch scatter. Seeded to a constant, so steps vary from each
    // other but a run is reproducible — determinism belongs in the seed, not
    // in deriving the value from game state that may not be varying.
    bracken::core::Rng step_rng_{0x5EED'F007ULL};
    // Generated LODs are OFF here by default, and that is a measured decision,
    // not an oversight. A/B on this scene (PLAYGROUND_LOD=1 turns them on):
    //   off: 538 draw calls, 6,243,298 triangles, 86-87 fps
    //   on : 604 draw calls, 5,646,904 triangles, 86-88 fps
    // -9.6% triangles for +12% draw calls and no frame-time change, because
    // this scene's flora are already 40-1700 triangles each and the frame is
    // fill/pass-bound (render 7.9ms of which present_wait 3.4ms), not
    // vertex-bound. The engine feature is real and tested; this particular
    // scene is simply not one that benefits.
    bool lod_disabled_ = true;
    // PLAYGROUND_ASYNC=0 restores serial model loading (the A/B baseline).
    bool async_load_disabled_ = false;
    std::string rt_mode_; // PLAYGROUND_RT: shadows | reflections | both
    /// PLAYGROUND_RT_LOCAL: ratio | off. Pinned on the same rule as
    /// PLAYGROUND_RT — unset means OFF, never "keep what was persisted".
    std::string rt_local_mode_;
    std::string gi_mode_; // PLAYGROUND_GI: probes | probes_screen | off
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

    // Let the engine drive the camera rig (the default; see
    // Application::set_camera_driven_by_engine). Immutable after construction.
    bool engine_camera_ = true;

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

    // ---- --rt-bench: the traversal A/B jig ------------------------------
    // The plan of record (docs/rt_lighting_plan.md §5.0) makes this the gate
    // for every traversal performance claim, so it is deliberately dumb: fixed
    // timestep, pinned camera, no scripted flips, no screenshot, and the raw
    // per-frame samples kept so the report can quote a p95 rather than only a
    // mean (a mean hides exactly the rebuild spikes this measures).
    bool rt_bench_ = false;
    static constexpr int RT_BENCH_WARMUP_FRAMES = 30; // pipeline/PSO/history warm-up, discarded
    /// Per-pass CPU record times, keyed by the pass name. The key is a
    /// std::string copy, not the PassTiming string_view: the view points at a
    /// literal in engine code, which is stable, but a map keyed on an owned
    /// string is the honest thing to hand to a report that outlives the frame.
    std::map<std::string, std::vector<float>> rt_bench_pass_ms_;
    std::vector<float> rt_bench_sim_ms_;
    std::vector<float> rt_bench_render_ms_;
    std::vector<float> rt_bench_fence_ms_;
    std::vector<float> rt_bench_present_wait_ms_;
    /// R5b denoiser diagnostics, sampled from RtStats::denoise whenever a
    /// lagged readback has landed. These are the ONLY evidence a static
    /// screenshot cannot supply: a denoiser lagging behind a moving sun and one
    /// tracking it perfectly produce identical frames whenever nothing has
    /// changed for thirty frames, so the accumulation window and the gradient
    /// fire rate are what have to be reported instead.
    std::vector<float> rt_bench_shadow_accum_;
    std::vector<float> rt_bench_shadow_fire_;
    std::vector<float> rt_bench_local_accum_;
    std::vector<float> rt_bench_local_fire_;
    /**
     * PLAYGROUND_SUN_SWEEP=<deg/s> — a continuous sun-azimuth sweep for the
     * duration of an --rt-bench run.
     *
     * Without it there is nothing to measure. Every smoke and bench run in this
     * repo pins a static environment on purpose (that is what makes the luma
     * figures comparable), and a static sun is exactly the case R5b's
     * lighting-change invalidation is inert in. The sweep is derived from the
     * FRAME INDEX at the pinned 1/60 timestep, never from wall clock, so the
     * run remains bit-reproducible.
     *
     * Azimuth rather than elevation: it moves every shadow on the ground plane
     * without changing the sun's intensity or the sky's colour, so the
     * accumulation counts below answer for the ANGULAR half of the gradient
     * alone rather than for a confounded mixture of the two.
     */
    float sun_sweep_deg_per_s_ = 0.0f;
    int rt_bench_sampled_frames_ = 0;
    int screenshot_counter_ = 0;
};

} // namespace playground
