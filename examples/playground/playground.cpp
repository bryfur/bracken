#include "playground.hpp"
#include "music.hpp"

#include "bracken/audio/audio_engine.hpp"
#include "bracken/async_model_loader.hpp"
#include "bracken/core/asset/asset_pack.hpp"
#include "bracken/save/save_document.hpp"

#include "character.hpp"
#include "scene_layout.hpp"
#include "bracken/animation/ik_solver.hpp"
#include "bracken/contracts/light.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/dev/dev_ui.hpp"
#include "bracken/procedural/rock_generator.hpp"
#include "bracken/procedural/tree_generator.hpp"
#include "bracken/render/auto_exposure.hpp"
#include "bracken/render/grass_renderer.hpp"
#include "bracken/render/ibl.hpp"
#include "bracken/render/terrain_renderer.hpp"
#include "bracken/render/water_renderer.hpp"
#include "bracken/scene/scene_renderer.hpp"
#include "bracken/systems/camera_controller.hpp"
#include <SDL3/SDL_timer.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <optional>
#include <string_view>
#include <SDL3/SDL.h>
#ifdef BRACKEN_DEV_UI
#include <imgui.h> // panel bodies only — DevUI itself needs no #ifdef
#endif

// Generated alongside the .glb files; the build stays valid without it
// (procedural-only fallback scene).
#if __has_include("village_layout.h")
#include "village_layout.h"
#define PLAYGROUND_HAS_VILLAGE 1
#else
#define PLAYGROUND_HAS_VILLAGE 0
#endif

namespace playground {

using namespace bracken;
namespace anim = bracken::animation;
namespace lighting = bracken::render::lighting;

namespace {
constexpr float WORLD = 8000.0f;
constexpr float WORLD_MARGIN = 250.0f;
// The village sits at the world centre on a flattened plateau.
constexpr float VILLAGE_CX = WORLD * 0.5f;
constexpr float VILLAGE_CZ = WORLD * 0.5f;
constexpr float PLATEAU_HEIGHT = 25.0f;
constexpr float PLATEAU_RADIUS = 2400.0f; // fully flat inside this
constexpr float PLATEAU_BLEND = 3300.0f;  // rolling hills resume beyond this
constexpr int ORB_COUNT = 15;
constexpr float ORB_PICKUP_RADIUS = 60.0f;
constexpr float ORB_HOVER = 26.0f;
// The river along the west edge of the map: a meandering north-south channel
// carved into the heightmap. All values in world units; the centerline is
// river_center_x(wz). Water sits at RIVER_WATER_LEVEL; the carve guarantees
// banks crest above it so the flat water plane never floods the meadow.
constexpr float RIVER_WATER_LEVEL = -25.0f;
constexpr float RIVER_DEPTH = 110.0f;        // bed depth below water level (centre)
constexpr float RIVER_HALF_WIDTH = 320.0f;   // full-depth channel half-width
constexpr float RIVER_BANK_HALF_WIDTH = 520.0f; // bank crest distance
constexpr float RIVER_INFLUENCE = 900.0f;    // carve blends into raw terrain here
constexpr float RIVER_BANK_HEIGHT = 30.0f;   // bank crest above water level margin
constexpr float RIVER_FLOW_SPEED = 55.0f;    // units/s, flowing north (-z)

float river_center_x(float wz) {
    return 700.0f + 220.0f * std::sin(wz * 0.00035f + 1.0f) + 90.0f * std::sin(wz * 0.0011f + 3.0f);
}
constexpr float WALK_SPEED = 140.0f;
constexpr float RUN_SPEED = 300.0f;
constexpr float ATTACK_DURATION = 0.45f;
constexpr float CHIMNEY_SMOKE_PERIOD = 3.0f; // re-spawn cadence per chimney

// --- Physics tuning (world units are centimetre-ish: gravity -980/s^2) ---
constexpr float PHYS_GRAVITY_Y = -980.0f;
constexpr float PHYS_FIXED_DT = 1.0f / 60.0f;
// The chrome ball, now a toy: heavy enough to feel solid when pushed, bouncy
// enough that a kick visibly arcs and rebounds off walls.
constexpr float BALL_MASS = 8.0f;
constexpr float BALL_FRICTION = 0.4f;
constexpr float BALL_RESTITUTION = 0.75f;
constexpr float BALL_SPAWN_RAISE = 150.0f; // drop-and-settle proof on frame 1
// Character pusher capsule, sized to the blocky biped (model bounds are
// ~62 wide x 182 tall, feet at y=0): radius 32, cylinder half-height 59
// -> total capsule height 2*(59+32) = 182.
constexpr float CHAR_CAPSULE_RADIUS = 32.0f;
constexpr float CHAR_CAPSULE_HALF_CYL = 59.0f;
constexpr float CHAR_CAPSULE_CENTER_Y = CHAR_CAPSULE_HALF_CYL + CHAR_CAPSULE_RADIUS; // feet on ground
// Kick: attack swing within range and roughly facing the ball. Range is
// player-centre to ball-SURFACE (the ball is 45 units of radius by itself).
constexpr float KICK_RANGE = 150.0f;
constexpr float KICK_FACING_COS = 0.35f; // ~70 deg half-cone
// Impulse = mass * delta-v (Jolt semantics): 3600/8 = 450 u/s out and up —
// a ~1 m-high, ~4 m arc under -980 gravity. (The suggested 2200/900 pair
// gives the 8 kg ball only 275/112 u/s — a 6 cm "arc" — so both components
// are tuned up to actually loft it.)
constexpr float KICK_IMPULSE_FLAT = 3600.0f;
constexpr float KICK_IMPULSE_UP = 3600.0f;
// Village statics: the merged village.glb has no per-building meshes to
// measure, but the layout's footprint rects are exact; eave height runs
// 8.3-13 m in the kit, so 9 m boxes stop anything a kick can loft.
constexpr float BUILDING_COLLIDER_HEIGHT = 900.0f;
constexpr float BUILDING_COLLIDER_SINK = 50.0f; // bury the base to avoid a lip
// Tree trunks: height capped at 40% of the AABB (the canopy must not block
// a lofted ball). Nature-pack trees use MEASURED footprint half-extents
// (below, pre-scale model units); the generic 25%-of-AABB fraction is only
// the fallback for procedural generator trees, whose AABB is mostly trunk.
constexpr float TRUNK_XZ_FRACTION = 0.25f;
constexpr float TRUNK_HEIGHT_FRACTION = 0.40f;
// Measured trunk footprints for the nature pack: XZ extent of each .glb's
// vertices below waist height (canopy AABBs are 4-6 m wide, visible trunks
// 0.5-2 m — an AABB fraction gives fat invisible boxes that blocked the
// mushrooms/ferns hand-placed at tree bases in scene_layout.cpp). Includes
// the root flare. Order matters: "giantpine" also contains "pine".
constexpr float TRUNK_HALF_GIANTPINE = 100.0f;
constexpr float TRUNK_HALF_COMMONTREE = 75.0f;
constexpr float TRUNK_HALF_BIRCH = 38.0f;
constexpr float TRUNK_HALF_PINE = 30.0f;

/** Measured trunk half-extent for nature-pack trees; 0 = AABB-fraction rule. */
float nature_trunk_half(std::string_view name) {
    if (name.find("giantpine") != std::string_view::npos) {
        return TRUNK_HALF_GIANTPINE;
    }
    if (name.find("commontree") != std::string_view::npos) {
        return TRUNK_HALF_COMMONTREE;
    }
    if (name.find("birch") != std::string_view::npos) {
        return TRUNK_HALF_BIRCH;
    }
    if (name.find("pine") != std::string_view::npos) {
        return TRUNK_HALF_PINE;
    }
    return 0.0f;
}

// --- Weather / rain tuning (see update_day_weather) ---
constexpr float RAIN_LAYER_PERIOD = 0.5f;    // streak-volume re-spawn cadence
constexpr float RAIN_LAYER_LIFETIME = 0.62f; // streak fall time
constexpr float RAIN_LAYER_HEIGHT = 520.0f;  // volume center above the player
constexpr float RAIN_FALL_SPEED = 1350.0f;
constexpr float SPLASH_RATE = 90.0f;      // impact points/sec at full intensity
constexpr int SPLASH_MAX_PER_FRAME = 6;   // burst-effect budget
constexpr float SPLASH_RADIUS = 750.0f;   // impact disc around the player
constexpr float SPLASH_RAY_TOP = 2500.0f; // impact rays start above every rooftop
constexpr int RAIN_RIPPLE_BUDGET = 4;     // of the water sim's 8 impulses/frame

/** Static-collider classification by ModelManager name (see scatter_wilds). */
enum class ColliderKind { None, Box, Trunk };
ColliderKind classify_collider(std::string_view name) {
    // Solid props: crates, wagons, fences, brick piles.
    if (name.starts_with("prop_")) {
        return ColliderKind::Box;
    }
    // Real rocks only — pebbles are ankle-high dressing, no collider.
    if (name.find("rock_medium") != std::string_view::npos ||
        name.find("rock_big") != std::string_view::npos) {
        return ColliderKind::Box;
    }
    // Trees (nature pack + procedural fallback): trunk-footprint box.
    if (name.find("tree") != std::string_view::npos || name.find("pine") != std::string_view::npos ||
        name.find("birch") != std::string_view::npos) {
        return ColliderKind::Trunk;
    }
    // Flowers, grass tufts, clover, bushes, ferns, plants, mushrooms, pebbles.
    return ColliderKind::None;
}

glm::vec3 forward_from_yaw(float yaw) {
    // The character model faces -Z; rotate_y(yaw) maps that to this direction.
    return {-std::sin(yaw), 0.0f, -std::cos(yaw)};
}

/** Asset path anchored to the executable dir (same policy as the shaders). */
std::string asset_path(const std::string& rel) {
    const char* base = SDL_GetBasePath();
    return base ? std::string(base) + rel : rel;
}
} // namespace

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

bool PlaygroundGame::on_init() {
    // A/B hook: PLAYGROUND_LOD=1 builds LOD chains for the flora, 0 (the
    // default, see playground.hpp) skips them, so the two runs differ in
    // exactly one thing.
    // Ray tracing needs CPU-side geometry to build its BVH from, and models
    // release it at upload — so the decision has to be made BEFORE anything
    // loads, not when the structure is built.
    if (const char* rt = SDL_getenv("PLAYGROUND_RT"); rt != nullptr) {
        rt_mode_ = rt;
    }
    // PLAYGROUND_RT_LOCAL=ratio|full|off — traced shadows for the village's point
    // and spot lights. Read here for the same reason as the two above: it
    // decides whether a BVH is needed, and models release their CPU geometry
    // at upload.
    if (const char* rl = SDL_getenv("PLAYGROUND_RT_LOCAL"); rl != nullptr) {
        rt_local_mode_ = rl;
    }
    // PLAYGROUND_GI=probes|probes_screen|voxel|off, read here for the same reason
    // PLAYGROUND_RT is: the setting has to be pinned before init_renderer()
    // could load a persisted one.
    if (const char* gi = SDL_getenv("PLAYGROUND_GI"); gi != nullptr) {
        gi_mode_ = gi;
    }

    if (const char* env = SDL_getenv("PLAYGROUND_LOD"); env != nullptr) {
        lod_disabled_ = env[0] == '0';
    }
    // PLAYGROUND_ASYNC=0 forces the old one-at-a-time load, so the async win is
    // a measurement rather than a claim.
    if (const char* env = SDL_getenv("PLAYGROUND_ASYNC"); env != nullptr && env[0] == '0') {
        async_load_disabled_ = true;
    }

    // A smoke run is a regression gate, so it must not read or write the
    // developer's persisted GraphicsSettings. This has to happen BEFORE
    // init_renderer(), which is where the engine auto-loads that file.
    //
    // It was not hermetic, and that cost two full investigations: any earlier
    // run that turned ray tracing on (PLAYGROUND_RT, or a tick in the dev
    // overlay) wrote rt_shadows=1 at exit, so every later plain `--smoke`
    // silently traced; and a full 240-frame run persists its end-of-run
    // sweep state (shadow_mode, ao_mode), which moved the reported mean_luma
    // by more than a point with no code change at all. The file is also a
    // single process-global path, so two concurrent runs raced on it.
    if (smoke_frames_ > 0) {
        set_settings_persistence(false);
    }

    if (!init_renderer(1600, 900, "bracken playground", WORLD, WORLD)) {
        return false;
    }

    // Before ANY model loads: ray tracing builds its BVH from CPU-side
    // geometry, and upload releases it. The decision cannot wait until the
    // structure is built, because by then the data is gone -- which is exactly
    // why this is NOT gated on PLAYGROUND_RT. The dev overlay can turn ray
    // tracing on at any moment (Graphics panel), and a toggle that silently
    // does nothing because the geometry was freed an hour ago is worse than
    // the memory this costs. A shipping game that knows it will never trace
    // should leave this off and pay nothing.
    models().set_retain_cpu_geometry(true);

    // Dear ImGui developer overlay (F10). In stub builds
    // (BRACKEN_ENABLE_DEV_UI=OFF) init_dev_ui() returns false and every DevUI
    // call is a harmless no-op, so nothing here needs guarding except the
    // panel body, which calls ImGui:: directly.
    const bool dev_ui_ok = init_dev_ui();
    BRACKEN_LOG_INFO("playground", "Dev UI {}",
                     dev_ui_ok ? "initialized (F10 toggles it)" : "unavailable (stub build or init failed)");
    // Handled inside DevUI so F10 works even while an ImGui widget has
    // keyboard focus (InputHandler never sees captured key-downs).
    dev_ui().set_toggle_key(SDL_SCANCODE_F10);
#ifdef BRACKEN_DEV_UI
    dev_ui().add_panel("Playground", [this] {
        // Drawn DURING rendering, while the next frame's sim job mutates the
        // live game state on the pool — so this body reads and writes ONLY
        // panel_, the fence-owned copy sync_dev_panel() exchanges with the
        // sim state in on_main_frame (dev_ui.hpp "PANEL DATA SAFETY").
        ImGui::Text("Orbs collected: %d", panel_.orbs_collected);
        ImGui::Text("Player pos: %.1f, %.1f, %.1f", panel_.player_pos.x, panel_.player_pos.y, panel_.player_pos.z);
        ImGui::Text("Anim state: %s", panel_.anim_state.c_str());
        ImGui::Text("Village: %s",
                    panel_.village_loaded ? "loaded" : "unavailable (run tools/build_medieval_assets.py)");
        ImGui::Text("Ball pos: %.1f, %.1f, %.1f (%s)", panel_.ball_pos.x, panel_.ball_pos.y, panel_.ball_pos.z,
                    panel_.ball_active ? "active" : "asleep");
        ImGui::Text("Ball speed: %.1f", panel_.ball_speed);
        if (ImGui::Button("Spawn pickup burst")) {
            ++panel_.spawn_burst_clicks; // applied at the next fence
        }
        ImGui::Checkbox("Flashlight", &panel_.flashlight);
        ImGui::Checkbox("Debug draw", &panel_.debug_draw);

        ImGui::SeparatorText("Time of day");
        ImGui::SliderFloat("Hour", &panel_.tod_hour, 0.0f, 23.99f, "%.2f");
        ImGui::SliderFloat("Day speed", &panel_.tod_speed, 0.0f, 900.0f, "%.0f s/s");
        ImGui::Checkbox("Sun animates", &panel_.tod_animate);
        if (ImGui::Button("Dawn")) panel_.tod_hour = 5.8f;
        ImGui::SameLine();
        if (ImGui::Button("Noon")) panel_.tod_hour = 12.0f;
        ImGui::SameLine();
        if (ImGui::Button("Dusk")) panel_.tod_hour = 18.2f;
        ImGui::SameLine();
        if (ImGui::Button("Midnight")) panel_.tod_hour = 0.0f;

        ImGui::SeparatorText("Weather");
        ImGui::Combo("Mode", &panel_.weather_mode, "Auto\0Clear\0Overcast\0Rain\0");
        ImGui::SliderFloat("Rain intensity", &panel_.rain_slider, 0.05f, 1.0f, "%.2f");
        ImGui::Text("Now: %s (overcast %.2f, rain %.2f)", panel_.weather_label.c_str(),
                    panel_.overcast_now, panel_.rain_now);
    });
#endif

    build_heightmap();
    set_heightmap(heightmap_);

    // The river (carved into the heightmap above): one water region along
    // the west edge, flowing north. The rect generously covers the channel;
    // terrain above the water level hides the plane via the depth test. The
    // interactive sim window covers the stretch nearest the village.
    if (auto* water = scene_renderer().water()) {
        render::WaterRegion river;
        river.min_x = 150.0f;
        river.max_x = 1400.0f;
        river.min_z = 0.0f;
        river.max_z = WORLD;
        river.height = RIVER_WATER_LEVEL;
        river.flow = {0.0f, -1.0f}; // downstream = north
        river.flow_speed = RIVER_FLOW_SPEED;
        river.sim_center = {river_center_x(VILLAGE_CZ), VILLAGE_CZ};
        river.sim_extent = 2048.0f;
        water->set_region(river);
        // Cached for sim code: the WaterRenderer itself is render-side, but
        // whether the river exists is immutable after this point.
        river_active_ = water->active();
    }

    register_models();
    setup_environment();
    setup_village();
    setup_state_machine();
    setup_effects();
    scatter_wilds();
    build_orb_spots();

    for (int i = 0; i < ORB_COUNT; ++i) {
        // Orb 0 takes spot 0 (beside the spawn — see build_orb_spots).
        glm::vec3 spot = i == 0 ? orb_spots_[0]
                                : orb_spots_[static_cast<size_t>(rand01() * 0.999f * orb_spots_.size())];
        orbs_.push_back({spot + glm::vec3(0, ORB_HOVER, 0), rand01() * 6.28f});
    }

    // Spawn on the plaza's brick centrepiece: the middle of the square is the
    // one spot where a ~520-unit orbit camera clears every building.
    player_pos_ = village_pos(0.3f, 0.6f);

    // Chrome sphere on the plaza — the specular-IBL proof piece: a polished
    // metal ball that mirrors the prefiltered dusk sky and the village.
    chrome_transform_ = glm::translate(glm::mat4(1.0f), village_pos(9.0f, -6.5f, 1.4f));

    // Physics last: needs the heightmap, the village layout and every placed
    // prop (scatter_wilds recorded their collider descs), the chrome spawn
    // spot and the player spawn.
    setup_physics();

    // --- Tier 3: character controller ---------------------------------
    {
        bracken::physics::CharacterConfig cc;
        cc.radius = 26.0f;
        cc.stand_height = 150.0f;
        cc.crouch_height = 85.0f;
        cc.step_height = 45.0f;      // village doorsills and the tavern porch
        cc.max_slope_degrees = 50.0f;
        cc.gravity = -2400.0f;       // snappier than earth: game feel, not physics
        cc.jump_speed = 850.0f;
        cc.ground_snap = 50.0f;
        character_.configure(cc);
        character_.set_position(player_pos_);
    }

    // --- Tier 3: bone attachment --------------------------------------
    // A lantern in the character's hand, placed BY THE BONE. The engine finds
    // the joint; the host supplies only a grip offset.
    if (Model* pm = models().get_model(player_model_); pm != nullptr) {
        // The playground's rig (character.cpp) names its arm bones
        // "RightArm"/"LeftArm"; the alternatives cover common exported rigs so
        // the demo keeps working if the character is swapped for real art.
        for (const char* candidate : {"RightArm", "mixamorig:RightHand", "hand.R", "Hand_R"}) {
            lantern_attach_ = bracken::animation::make_attachment(
                pm->skeleton, candidate,
                // The joint is the SHOULDER; the arm box runs from y=153 down to
                // y=103 (character.cpp), so the hand is ~52 units along the
                // bone. Offsetting by a token amount buried the lantern inside
                // the upper arm.
                glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, -56.0f, 0.0f)));
            if (lantern_attach_.valid()) {
                BRACKEN_LOG_INFO("playground", "lantern attached to bone '{}'", candidate);
                break;
            }
        }
        // Ray-traced shadow proxies (docs/rt_lighting_plan.md §5.8). Resolved
        // here, next to the lantern, because it is the same operation for the
        // same reason: find_joint is a linear name scan, so it happens once at
        // load and the INDEX is what the per-frame code keeps.
        resolve_capsule_bones(*pm);
    }

    // Reuse an existing small mesh as the carried lantern rather than adding
    // an asset the repo does not have.
    lantern_model_ = models().get_handle("lantern");

    // --- Ray tracing (opt-in) -----------------------------------------
    // PLAYGROUND_RT=shadows | reflections | both, or the dev overlay's
    // Graphics panel at runtime — both go through the same GraphicsSettings
    // flags, and the engine builds/refreshes the acceleration structure from
    // the frame packet itself. Off unless asked for: this is software ray
    // tracing over a BVH, and it is not free.
    // PINS both flags rather than only setting them when the var is present:
    // an unset PLAYGROUND_RT means "no ray tracing", and used to mean "keep
    // whatever the last run persisted". That is how a plain `--smoke` ended up
    // building a 27 MB acceleration structure and tracing, with no env var set
    // and both GraphicsSettings defaults false. The playground is a demo and
    // an integration harness, so an explicit pin is the right trade: an
    // interactive session that wants ray tracing ticks the Graphics panel.
    {
        auto edit = edit_graphics_settings();
        apply_rt_mode(*edit);
        apply_gi_mode(*edit);
        // No request_graphics_settings() here: the edit handle commits on
        // scope exit if anything changed (the old code requested the same
        // value twice into the same slot).
    }

    // --- Tier 3: nav grid ---------------------------------------------
    {
        bracken::nav::NavGridConfig ng;
        ng.cell_size = 120.0f;
        ng.origin = glm::vec3(0.0f);
        ng.width = static_cast<int>(WORLD / ng.cell_size);
        ng.depth = ng.width;
        ng.max_slope_degrees = 45.0f; // slightly under the controller's 50
        ng.max_step = 45.0f;
        nav_.build(
            ng, [this](float x, float z) { return get_terrain_height(x, z); },
            [this](const glm::vec3& p) {
                // Anything the character cannot stand in is off the mesh. One
                // capsule overlap test per cell, reusing the same physics the
                // controller collides against — so the path and the walking
                // agree by construction.
                //
                // The terrain heightfield is EXCLUDED: it is the thing being
                // walked on, and a capsule resting on the ground always
                // overlaps it. Including it marked every cell blocked and the
                // grid came out 0% walkable.
                std::array<bracken::physics::PhysicsWorld::ShapeContact, 4> hits{};
                const glm::vec3 center = p + glm::vec3(0.0f, 80.0f, 0.0f);
                return physics_.collide_capsule(center, 45.0f, 26.0f, hits, /*include_moving=*/false,
                                                terrain_body_) > 0;
            });
        wanderer_.position = player_pos_ + glm::vec3(300.0f, 0.0f, 300.0f);
        wanderer_.position.y = get_terrain_height(wanderer_.position.x, wanderer_.position.z);
        wanderer_start_ = wanderer_.position;
    }

    set_camera_driven_by_engine(engine_camera_);
    auto& cam = camera();
    cam.set_terrain_height_func([this](float x, float z) { return get_terrain_height(x, z); });
    cam.set_screen_size(screen_width(), screen_height());
    auto cfg = cam.get_config();
    cfg.distance = 420.0f;
    cfg.height_offset = 130.0f;
    cam.set_config(cfg);
    cam.set_target(player_pos_ + glm::vec3(0, 110, 0));

    // The village theme, played by the engine's synthesizer. There is no
    // audio file: the whole score is generated from oscillators at runtime.
    // Quieter than the default so it sits under the scene rather than on it.
    // Actions, bound to defaults then overridden by a bindings file if the
    // player has one. The engine never learns what "attack" means — it only
    // reports that the thing this host named "attack" went down.
    act_forward_ = actions_.add("move_forward");
    act_back_ = actions_.add("move_backward");
    act_left_ = actions_.add("move_left");
    act_right_ = actions_.add("move_right");
    act_run_ = actions_.add("run");
    act_attack_ = actions_.add("attack");
    act_flashlight_ = actions_.add("flashlight");

    actions_.bind(act_forward_, bracken::Binding::key(SDL_SCANCODE_W));
    actions_.bind(act_back_, bracken::Binding::key(SDL_SCANCODE_S));
    actions_.bind(act_left_, bracken::Binding::key(SDL_SCANCODE_A));
    actions_.bind(act_right_, bracken::Binding::key(SDL_SCANCODE_D));
    actions_.bind(act_run_, bracken::Binding::key(SDL_SCANCODE_LSHIFT));
    actions_.bind(act_run_, bracken::Binding::key(SDL_SCANCODE_RSHIFT));
    actions_.bind(act_attack_, bracken::Binding::mouse(bracken::MouseButton::Left));
    actions_.bind(act_attack_, bracken::Binding::controller(bracken::ControllerButton::Primary));
    actions_.bind(act_flashlight_, bracken::Binding::key(SDL_SCANCODE_F));
    act_quicksave_ = actions_.add("quicksave");
    act_quickload_ = actions_.add("quickload");
    actions_.bind(act_quicksave_, bracken::Binding::key(SDL_SCANCODE_F9));
    actions_.bind(act_quickload_, bracken::Binding::key(SDL_SCANCODE_F11));
    act_jump_ = actions_.add("jump");
    act_crouch_ = actions_.add("crouch");
    actions_.bind(act_jump_, bracken::Binding::key(SDL_SCANCODE_SPACE));
    actions_.bind(act_jump_, bracken::Binding::controller(bracken::ControllerButton::Secondary));
    actions_.bind(act_crouch_, bracken::Binding::key(SDL_SCANCODE_LCTRL));
    actions_.bind(actions_.add("nav_debug"), bracken::Binding::key(SDL_SCANCODE_N));
    (void)actions_.load("playground_bindings.cfg"); // absent = defaults stand

    audio().set_music_gain(0.55f);
    sfx_.build(audio());
    audio().play_music(build_village_theme());

    if (smoke_frames_ > 0) {
        set_collect_render_stats(true);
        setup_smoke_run();
    }
    return true;
}

void PlaygroundGame::on_shutdown() {
    // Main thread, no sim in flight — the only other PhysicsWorld toucher
    // (on_simulate) is done for good.
    if (physics_ready_) {
        if (character_body_ != physics::INVALID_BODY_ID) {
            physics_.remove_body(character_body_);
        }
        if (ball_body_ != physics::INVALID_BODY_ID) {
            physics_.remove_body(ball_body_);
        }
        for (physics::BodyId id : static_bodies_) {
            physics_.remove_body(id);
        }
        static_bodies_.clear();
        physics_.shutdown();
        physics_ready_ = false;
    }
    shutdown_renderer();
}

void PlaygroundGame::build_heightmap() {
    constexpr uint32_t RES = 257;
    // Author in world units; Heightmap::from_floats owns the packing. The
    // range is given explicitly rather than scanned because this demo is also
    // the regression gate: a scanned range would re-quantize the whole map
    // whenever a single sample moved, so the terrain would shift under an
    // unrelated edit.
    constexpr bracken::HeightRange RANGE{-160.0f, 150.0f}; // riverbed carves to ~-135
    std::vector<float> heights(static_cast<size_t>(RES) * RES);
    for (uint32_t z = 0; z < RES; ++z) {
        for (uint32_t x = 0; x < RES; ++x) {
            float wx = WORLD * static_cast<float>(x) / (RES - 1);
            float wz = WORLD * static_cast<float>(z) / (RES - 1);
            float h = 40.0f * std::sin(wx * 0.0011f + 0.5f) * std::cos(wz * 0.0008f) +
                      15.0f * std::sin(wx * 0.0040f + 2.0f) * std::sin(wz * 0.0035f) +
                      5.0f * std::sin(wx * 0.0130f) * std::cos(wz * 0.0110f + 1.0f);
            // Flatten a plateau for the village, blending back into the hills.
            float d = glm::distance(glm::vec2(wx, wz), glm::vec2(VILLAGE_CX, VILLAGE_CZ));
            float w = glm::smoothstep(PLATEAU_RADIUS, PLATEAU_BLEND, d);
            h = glm::mix(PLATEAU_HEIGHT, h, w);
#if PLAYGROUND_HAS_VILLAGE
            // Terraced patches under every building footprint: the outlying
            // farmsteads sit past the plateau rim, and the merged village
            // model is baked flat at the plateau height.
            for (const auto& b : village::BUILDINGS) {
                const float s = village::VILLAGE_SCALE;
                const float x0 = VILLAGE_CX + b.min_x * s, x1 = VILLAGE_CX + b.max_x * s;
                const float z0 = VILLAGE_CZ + b.min_z * s, z1 = VILLAGE_CZ + b.max_z * s;
                const float dx = std::max({x0 - wx, 0.0f, wx - x1});
                const float dz = std::max({z0 - wz, 0.0f, wz - z1});
                const float dist = std::sqrt(dx * dx + dz * dz);
                constexpr float TERRACE_BLEND = 500.0f; // 5 m skirt around the footprint
                h = glm::mix(PLATEAU_HEIGHT, h, glm::smoothstep(0.0f, TERRACE_BLEND, dist));
            }
#endif
            // The river carve runs last so it wins over the plateau and the
            // building terraces: a parabolic channel below RIVER_WATER_LEVEL,
            // banks pinned above it (so the flat water plane stays contained),
            // then a blend back into whatever the terrain was doing.
            {
                const float river_d = std::abs(wx - river_center_x(wz));
                // Dry floor for the strip flanking the channel: the raw hills
                // dip below the water level in places, and the flat water
                // plane would pool in any such dip inside the water region.
                const float dry = RIVER_WATER_LEVEL + 12.0f;
                if (river_d < RIVER_INFLUENCE) {
                    const float bed = RIVER_WATER_LEVEL - RIVER_DEPTH;
                    const float bank = RIVER_WATER_LEVEL + RIVER_BANK_HEIGHT;
                    float channel;
                    if (river_d < RIVER_HALF_WIDTH) {
                        // Parabolic bed: deepest at the centerline.
                        const float t = river_d / RIVER_HALF_WIDTH;
                        channel = glm::mix(bed, RIVER_WATER_LEVEL - 10.0f, t * t);
                    } else {
                        // Waterline up to the bank crest.
                        const float t = glm::smoothstep(RIVER_HALF_WIDTH, RIVER_BANK_HALF_WIDTH, river_d);
                        channel = glm::mix(RIVER_WATER_LEVEL - 10.0f, std::max(h, bank), t);
                    }
                    h = glm::mix(channel, std::max(h, dry),
                                 glm::smoothstep(RIVER_BANK_HALF_WIDTH, RIVER_INFLUENCE, river_d));
                } else if (river_d < RIVER_INFLUENCE + 700.0f) {
                    h = std::max(h, dry);
                }
            }
            heights[z * RES + x] = h;
        }
    }
    heightmap_ = bracken::Heightmap::from_floats(heights, RES, {0.0f, 0.0f, WORLD}, RANGE);
}

glm::vec3 PlaygroundGame::village_pos(float vx, float vz, float vy) {
#if PLAYGROUND_HAS_VILLAGE
    const float s = village::VILLAGE_SCALE;
#else
    const float s = 100.0f;
#endif
    const float x = VILLAGE_CX + vx * s;
    const float z = VILLAGE_CZ + vz * s;
    return {x, heightmap_.get_height_world(x, z) + vy * s, z};
}

void PlaygroundGame::register_models() {
    auto& mm = models();
    // register_model finishes the model (bounds where unset, bounding sphere,
    // `loaded`) and reports why it refused one; a demo that ignored that would
    // fail later as an invisible model.
    auto register_or_log = [&mm](const char* name, std::unique_ptr<Model> model) {
        auto handle = mm.register_model(name, std::move(model));
        if (!handle) {
            BRACKEN_LOG_ERROR("playground", "register_model('{}'): {}", name, to_string(handle.error()));
            return INVALID_MODEL_HANDLE;
        }
        return handle.value();
    };
    player_model_ = register_or_log("player", build_character_model());
    // The bone-attachment demo's carried prop.
    register_or_log("lantern", build_lantern_model());
    orb_model_ = register_or_log("orb", build_orb_model());
    chrome_model_ = register_or_log("chrome_sphere", build_chrome_sphere_model());
    register_or_log("orb_shard", build_shard_model()); // referenced by name from effects
    register_or_log("smoke_puff", build_smoke_puff_model());
    register_or_log("rain_streak", build_rain_streak_model());
    register_or_log("splash_drop", build_splash_drop_model());

    // The medieval kit + stylized nature pack (both converted by
    // tools/build_medieval_assets.py). Absence is not an error: the
    // playground degrades to the procedural-only scene.
    village_model_ = mm.load_model("village", asset_path("assets/models/village.glb"));
    has_village_ = village_model_ != INVALID_MODEL_HANDLE;

    // Decoded on the worker pool rather than one at a time on this thread.
    std::vector<bracken::AsyncModelLoader::Request> nature_requests;
    if (has_village_) {
        for (const char* name :
             {"prop_crate", "prop_wagon", "prop_fence", "prop_fence_ext", "prop_bricks",
              "nature_commontree_1", "nature_commontree_2", "nature_commontree_3", "nature_birch_1",
              "nature_birch_2", "nature_pine_1", "nature_pine_2", "nature_pine_3", "nature_giantpine_1",
              "nature_giantpine_2", "nature_bush_common", "nature_bush_common_flowers", "nature_bush_large",
              "nature_fern_2", "nature_rock_medium_1", "nature_rock_medium_2", "nature_rock_medium_3",
              "nature_rock_big_1", "nature_grass_common_short", "nature_grass_common_tall",
              "nature_grass_wispy_short", "nature_grass_wispy_tall", "nature_grass_wide_short",
              "nature_clover_1", "nature_flower_1_group", "nature_flower_2_group", "nature_flower_3_group",
              "nature_rock_medium_4", "nature_rock_big_2", "nature_grass_wide_tall", "nature_grass_wheat",
              "nature_clover_2", "nature_plant_1", "nature_plant_2", "nature_plant_3", "nature_bush_long_1",
              "nature_flower_4_group", "nature_flower_6", "nature_flower_7_group", "nature_mushroom_common",
              "nature_mushroom_redcap", "nature_pebble_round_1", "nature_pebble_round_2", "nature_pebble_round_3",
              "nature_pebble_square_1", "nature_pebble_square_2"}) {
            // Queue rather than load: decoding is worker-safe, so the whole
            // pack is decoded in parallel and only the GPU upload comes back
            // to this thread. (The LOD path stays synchronous — build_lods
            // has to run between decode and upload, and PLAYGROUND_LOD=1 is
            // a measurement hook, not the default.)
            const std::string path = asset_path(std::string("assets/models/") + name + ".glb");
            if (lod_disabled_ && !async_load_disabled_) {
                nature_requests.push_back({name, path});
            } else if (lod_disabled_) {
                (void)mm.load_model(name, path); // PLAYGROUND_ASYNC=0: the old serial path
            } else {
                constexpr float k_lod_ratios[] = {0.35f, 0.10f};
                constexpr float k_lod_thresholds[] = {0.040f, 0.014f};
                (void)mm.load_model_with_lods(name, path, k_lod_ratios, k_lod_thresholds);
            }
        }

        if (!nature_requests.empty()) {
            // PLAYGROUND_PACK=<file> loads through an asset pack instead of
            // loose files — build one with `playground --build-pack out.pack`.
            std::optional<bracken::core::asset::AssetPack> pack;
            if (const char* pack_path = SDL_getenv("PLAYGROUND_PACK")) {
                auto opened = bracken::core::asset::AssetPack::open(pack_path);
                if (opened) {
                    pack = std::move(opened).value();
                    // Sources become logical pack names rather than paths.
                    for (auto& r : nature_requests) {
                        r.source = "models/" + r.name + ".glb";
                    }
                } else {
                    BRACKEN_LOG_WARN("playground", "PLAYGROUND_PACK='{}': {}; using loose files",
                                     pack_path, bracken::core::asset::to_string(opened.error()));
                }
            }

            bracken::AsyncModelLoader loader(mm, pack ? &*pack : nullptr);
            loader.request_all(nature_requests);
            // Nothing to overlap with during boot, so drain — which helps the
            // pool while it waits rather than idling.
            const auto completed = loader.drain();
            BRACKEN_LOG_INFO("playground", "Decoded {} nature models on the pool ({} failed)",
                             completed.size(), loader.failed_count());
        }
    } else {
        BRACKEN_LOG_WARN("playground",
                         "village assets not found ({}) — run tools/build_medieval_assets.py; "
                         "falling back to the procedural-only scene",
                         asset_path("assets/models/village.glb"));
        // Procedural stand-ins for the fallback world only.
        register_or_log("rock_boulder", procedural::RockGenerator::generate_boulder(11));
        register_or_log("rock_slate", procedural::RockGenerator::generate_slate(29));
        register_or_log("rock_cluster", procedural::RockGenerator::generate_cluster(43));
        register_or_log("tree_oak", procedural::TreeGenerator::generate_oak(7));
        register_or_log("tree_pine", procedural::TreeGenerator::generate_pine(21));
        register_or_log("tree_birch", procedural::TreeGenerator::generate_birch(33));
    }
}

void PlaygroundGame::setup_environment() {
    // Golden hour: low warm sun out of the south-west so the plaza-facing
    // facades catch the light; sky/fog/clouds tuned to match. All linear.
    // Written into the HOST-owned env_ (staged into frame 0's packet by
    // build_frame — the renderer adopts it before its first pass, so the IBL
    // init build already sees this look and the smoke run's refresh count
    // stays at init + the two sun swings).
    auto& env = env_;
    // ~55 deg elevation: late afternoon. The plaza is a courtyard ringed by
    // 6-10 m buildings, so anything much lower buries it in cast shadow;
    // at 55 deg shadows run ~0.7x building height — long enough to read as
    // evening, short enough that the square stays sunlit.
    scene_sun_dir_ = glm::normalize(glm::vec3(0.30f, -1.15f, -0.75f));
    scene_sun_color_ = {1.0f, 0.87f, 0.68f};
    env.sun_dir = scene_sun_dir_;
    env.sun_color = scene_sun_color_;
    env.sky_zenith = {0.19f, 0.30f, 0.55f};
    env.sky_horizon = {0.74f, 0.58f, 0.44f};
    env.ground_color = {0.14f, 0.11f, 0.08f};
    env.ambient_sky = {0.42f, 0.42f, 0.50f};
    env.ambient_ground = {0.30f, 0.25f, 0.19f};
    env.fog_color_horizon = {0.70f, 0.56f, 0.44f};
    env.fog_color_sun = {0.98f, 0.72f, 0.48f};
    env.fog_start = 2800.0f;
    env.fog_end = 13000.0f;
    env.fog_height_falloff = 0.004f; // thins quickly with altitude — keeps rooftops crisp
    env.cloud_coverage = 0.42f;
    env.cloud_density = 0.9f;
    env.fog_volumetric_sun_color = {1.0f, 0.62f, 0.35f};
    // A tenth of the default extinction: at dusk the sun-phase inscatter is
    // weak, so extinction dominates — the default 0.0015/unit reads as a
    // black wash over everything past ~1000 units (optical depth 1 at only
    // ~650). 0.00015 keeps a soft haze without eating the ground.
    env.fog_volumetric_density = 0.00015f;
    env.fog_volumetric_scatter = 0.22f;
    env.fog_volumetric_end = 2600.0f;
    // A whisper of warm grade; the CDL stays near-neutral on purpose.
    env.grade_temperature = 0.08f;
    env.grade_slope = {1.02f, 1.0f, 0.97f};
    env_dirty_ = true; // stage into the first frame's packet
}

void PlaygroundGame::setup_village() {
#if PLAYGROUND_HAS_VILLAGE
    if (!has_village_) {
        return;
    }
    namespace vl = village;
    const float S = vl::VILLAGE_SCALE;
    const float ground = heightmap_.get_height_world(VILLAGE_CX, VILLAGE_CZ);
    village_transform_ = glm::translate(glm::mat4(1.0f), {VILLAGE_CX, ground, VILLAGE_CZ});

    auto world_rect = [&](const vl::Rect& r, float pad) {
        return render::GrassExclusionRect{VILLAGE_CX + (r.min_x - pad) * S, VILLAGE_CZ + (r.min_z - pad) * S,
                                          VILLAGE_CX + (r.max_x + pad) * S, VILLAGE_CZ + (r.max_z + pad) * S};
    };

    // No grass through the cobbles or inside buildings.
    std::vector<render::GrassExclusionRect> excl;
    excl.push_back(world_rect(vl::PLAZA, 0.3f));
    for (const auto& s : vl::STREETS) {
        excl.push_back(world_rect(s, 0.3f));
    }
    for (const auto& b : vl::BUILDINGS) {
        excl.push_back(world_rect(b, 0.5f));
    }
    if (auto* grass = scene_renderer().grass()) {
        grass->set_exclusion_rects(std::move(excl));
    }

    // Splatmap: packed earth under the village fading out to meadow, dirt
    // lanes continuing where the cobbled streets end. R=grass G=dirt B=rock A=sand.
    constexpr uint32_t SPLAT_RES = 512; // 15.6 units/texel — keeps lane edges smooth
    std::vector<uint8_t> splat(SPLAT_RES * SPLAT_RES * 4);
    auto village_metres = [&](float wx, float wz) {
        return glm::vec2((wx - VILLAGE_CX) / S, (wz - VILLAGE_CZ) / S);
    };
    for (uint32_t ty = 0; ty < SPLAT_RES; ++ty) {
        for (uint32_t tx = 0; tx < SPLAT_RES; ++tx) {
            const float wx = WORLD * (static_cast<float>(tx) + 0.5f) / SPLAT_RES;
            const float wz = WORLD * (static_cast<float>(ty) + 0.5f) / SPLAT_RES;
            const glm::vec2 v = village_metres(wx, wz);

            // Village core: dirt weight from a soft disc around the plaza.
            float dirt = 1.0f - glm::smoothstep(14.0f, 26.0f, glm::length(v));
            // Lanes: extend each street outward past the village edge.
            auto lane = [&](glm::vec2 a, glm::vec2 b, float half_w) {
                glm::vec2 ab = b - a;
                float t = glm::clamp(glm::dot(v - a, ab) / glm::dot(ab, ab), 0.0f, 1.0f);
                float d = glm::length(v - (a + ab * t));
                return (1.0f - glm::smoothstep(half_w, half_w + 2.5f, d)) * (1.0f - 0.65f * t);
            };
            dirt = std::max({dirt,
                             lane({2.0f, -9.0f}, {2.0f, -34.0f}, 2.2f),   // north
                             lane({12.0f, 0.0f}, {36.0f, 0.0f}, 2.2f),    // east
                             lane({-4.0f, 9.0f}, {-4.0f, 34.0f}, 2.2f)}); // south
            // Rocky wash where the hills get steep (plateau rim and beyond).
            const float dist_c = glm::distance(glm::vec2(wx, wz), glm::vec2(VILLAGE_CX, VILLAGE_CZ));
            float rock = 0.35f * glm::smoothstep(PLATEAU_RADIUS, PLATEAU_BLEND + 600.0f, dist_c);
            // Sand banks along the river: full strength over the bed and
            // shoreline, fading out over the bank crest.
            const float river_d = std::abs(wx - river_center_x(wz));
            const float sand = 1.0f - glm::smoothstep(RIVER_HALF_WIDTH + 100.0f,
                                                      RIVER_BANK_HALF_WIDTH + 300.0f, river_d);

            const float g = glm::clamp(dirt * (1.0f - sand), 0.0f, 1.0f);
            rock = glm::clamp(rock * (1.0f - sand), 0.0f, 1.0f);
            const float r = glm::clamp(1.0f - g - rock - sand, 0.0f, 1.0f);
            uint8_t* px = &splat[(ty * SPLAT_RES + tx) * 4];
            px[0] = static_cast<uint8_t>(r * 255.0f);
            px[1] = static_cast<uint8_t>(g * 255.0f);
            px[2] = static_cast<uint8_t>(rock * 255.0f);
            px[3] = static_cast<uint8_t>(sand * 255.0f);
        }
    }
    scene_renderer().terrain().update_splatmap(splat.data(), SPLAT_RES);

    // Chimney mouths (world space) for the smoke emitters.
    for (const auto& c : vl::CHIMNEYS) {
        glm::vec3 p = village_pos(c.x, c.z);
        p.y += c.y * S;
        chimney_tops_.push_back(p);
    }

    // All prop/flora dressing (crates, wagons, fences, trees, ...) is the
    // hand-authored layout applied in scatter_wilds().
#endif
}

void PlaygroundGame::scatter_wilds() {
    auto& mm = models();

    if (!has_village_) {
        // Fallback dressing: procedural trees + rocks scattered over the hills.
        const ModelHandle trees[] = {mm.get_handle("tree_oak"), mm.get_handle("tree_pine"),
                                     mm.get_handle("tree_birch")};
        const ModelHandle rocks[] = {mm.get_handle("rock_boulder"), mm.get_handle("rock_slate"),
                                     mm.get_handle("rock_cluster")};
        auto place = [this](ModelHandle handle, float scale, float sink, bool trunk) {
            glm::vec3 pos = random_ground_pos(400.0f);
            const float d = glm::distance(glm::vec2(pos.x, pos.z), glm::vec2(VILLAGE_CX, VILLAGE_CZ));
            if (d < PLATEAU_RADIUS + 300.0f) {
                return; // keep the central clearing open
            }
            if (pos.y < RIVER_WATER_LEVEL + 15.0f) {
                return; // nothing grows in the river channel
            }
            glm::mat4 t = glm::translate(glm::mat4(1.0f), pos - glm::vec3(0, sink, 0));
            t = glm::rotate(t, rand01() * 6.28f, glm::vec3(0, 1, 0));
            t = glm::scale(t, glm::vec3(scale));
            props_.push_back({handle, t});
            pending_colliders_.push_back({handle, t, trunk});
        };
        for (int i = 0; i < 70; ++i) {
            place(trees[i % 3], 8.0f + rand01() * 5.0f, 3.0f, /*trunk=*/true);
        }
        for (int i = 0; i < 25; ++i) {
            float s = 10.0f + rand01() * 16.0f;
            place(rocks[i % 3], s, s * 0.4f, /*trunk=*/false);
        }
        return;
    }

#if PLAYGROUND_HAS_VILLAGE
    // Everything with a silhouette — props, trees, bushes, rocks, flower
    // clumps, mushrooms, path pebbles — is hand-placed area by area in
    // scene_layout.cpp. This function just resolves the layout to world
    // transforms and lays the ground-cover tuft carpet underneath it.
    for (const auto& p : layout::build_layout()) {
        const ModelHandle h = mm.get_handle(p.model);
        if (h == INVALID_MODEL_HANDLE) {
            continue;
        }
        glm::mat4 t = glm::translate(glm::mat4(1.0f), village_pos(p.x, p.z, p.y) - glm::vec3(0, p.sink, 0));
        t = glm::rotate(t, glm::radians(p.rot_deg), glm::vec3(0, 1, 0));
        t = glm::scale(t, glm::vec3(p.scale));
        props_.push_back({h, t});
        // Record a physics collider for everything with a real silhouette
        // (setup_physics builds the static boxes once the world is up).
        const ColliderKind kind = classify_collider(p.model);
        if (kind != ColliderKind::None) {
            const bool trunk = kind == ColliderKind::Trunk;
            pending_colliders_.push_back({h, t, trunk, trunk ? nature_trunk_half(p.model) : 0.0f});
        }
    }

    auto in_clearing = [](float vx, float vz) {
        for (const auto& c : layout::CLEARINGS) {
            if (vx > c.x0 && vx < c.x1 && vz > c.z0 && vz < c.z1) {
                return true;
            }
        }
        return false;
    };

    // Ground cover: instanced nature-pack grass tufts (with clover accents)
    // on a jittered grid — the one fill layer that stays procedural. A lawn
    // is texture, not composition; every distinct object is in the layout.
    // No wheat here: the golden tufts are reserved for the farm field and
    // cottage garden, where they read as crops instead of stray orange clumps.
    const ModelHandle tufts[] = {
        mm.get_handle("nature_grass_common_short"), mm.get_handle("nature_grass_common_tall"),
        mm.get_handle("nature_grass_wispy_short"), mm.get_handle("nature_grass_wispy_tall"),
        mm.get_handle("nature_grass_wide_short"), mm.get_handle("nature_grass_wide_tall")};
    const ModelHandle clover[] = {mm.get_handle("nature_clover_1"), mm.get_handle("nature_clover_2")};
    auto blocked_flora = [&](float vx, float vz) {
        if (in_clearing(vx, vz)) {
            return true;
        }
        // Nothing grows on the riverbed or the wet shoreline.
        const float wx = VILLAGE_CX + vx * village::VILLAGE_SCALE;
        const float wz = VILLAGE_CZ + vz * village::VILLAGE_SCALE;
        if (heightmap_.get_height_world(wx, wz) < RIVER_WATER_LEVEL + 15.0f) {
            return true;
        }
        auto in_rect = [&](float x0, float z0, float x1, float z1, float pad) {
            return vx > x0 - pad && vx < x1 + pad && vz > z0 - pad && vz < z1 + pad;
        };
        if (in_rect(village::PLAZA.min_x, village::PLAZA.min_z, village::PLAZA.max_x, village::PLAZA.max_z, 0.5f)) {
            return true;
        }
        for (const auto& st : village::STREETS) {
            if (in_rect(st.min_x, st.min_z, st.max_x, st.max_z, 0.5f)) {
                return true;
            }
        }
        for (const auto& b : village::BUILDINGS) {
            if (in_rect(b.min_x, b.min_z, b.max_x, b.max_z, 0.6f)) {
                return true;
            }
        }
        return false;
    };
    constexpr float FLORA_R0 = 13.5f, FLORA_R1 = 42.5f, FLORA_STEP = 1.2f;
    for (float gx = -FLORA_R1; gx < FLORA_R1; gx += FLORA_STEP) {
        for (float gz = -FLORA_R1; gz < FLORA_R1; gz += FLORA_STEP) {
            const float vx = gx + (rand01() - 0.5f) * FLORA_STEP;
            const float vz = gz + (rand01() - 0.5f) * FLORA_STEP;
            const float rr = std::sqrt(vx * vx + vz * vz);
            if (rr < FLORA_R0 || rr > FLORA_R1 || blocked_flora(vx, vz)) {
                continue;
            }
            glm::mat4 t = glm::translate(glm::mat4(1.0f), village_pos(vx, vz) - glm::vec3(0, 3.0f, 0));
            t = glm::rotate(t, rand01() * 6.28f, glm::vec3(0, 1, 0));
            const float roll = rand01();
            if (roll < 0.12f) { // clover patches thread through the tufts
                t = glm::scale(t, glm::vec3(0.6f + rand01() * 0.3f));
                props_.push_back({clover[static_cast<int>(rand01() * 1.999f)], t});
            } else {
                t = glm::scale(t, glm::vec3(0.45f + rand01() * 0.3f));
                props_.push_back({tufts[static_cast<int>(rand01() * 5.999f)], t});
            }
        }
    }
#endif
}

void PlaygroundGame::build_orb_spots() {
#if PLAYGROUND_HAS_VILLAGE
    if (has_village_) {
        namespace vl = village;
        // Spot 0 sits just off the spawn point: the smoke driver's player
        // reaches it within a few frames, so every smoke run proves the
        // pickup path (burst + shake + respawn + counter) end to end.
        orb_spots_.push_back(village_pos(1.0f, 1.3f));
        // In front of every door...
        for (const auto& d : vl::DOORS) {
            orb_spots_.push_back(village_pos(d.x, d.z));
        }
        // ...around the plaza...
        for (auto [x, z] : {std::pair{-9.0f, -6.0f}, {9.0f, 6.0f}, {-9.0f, 6.5f}, {0.0f, -6.5f},
                            {6.0f, 0.5f}, {-6.5f, 0.0f}}) {
            orb_spots_.push_back(village_pos(x, z));
        }
        // ...down the lanes and just past the village edge.
        for (auto [x, z] : {std::pair{2.0f, -12.0f}, {2.0f, -17.0f}, {16.0f, 0.5f}, {20.5f, 7.2f},
                            {-4.0f, 12.0f}, {-4.0f, 16.5f}, {-13.0f, -8.0f}, {8.0f, 14.0f},
                            {2.0f, -24.0f}, {26.0f, 0.0f}, {-4.0f, 24.0f}, {-20.0f, 10.0f}}) {
            orb_spots_.push_back(village_pos(x, z));
        }
        return;
    }
#endif
    for (int i = 0; i < 24; ++i) {
        orb_spots_.push_back(random_ground_pos(600.0f));
    }
}

void PlaygroundGame::setup_state_machine() {
    using TC = anim::TransitionCondition;
    auto cond = [](const char* name, TC::Op op, float threshold = 0.0f) {
        TC c;
        c.param_name = name;
        c.op = op;
        c.threshold = threshold;
        return c;
    };
    auto to = [](const char* target, TC c, float xfade, int priority = 0) {
        anim::StateTransition t;
        t.target_state = target;
        t.conditions.push_back(std::move(c));
        t.crossfade_duration = xfade;
        t.priority = priority;
        return t;
    };
    const auto attack_now = [&] { return to("Attack", cond("attacking", TC::Op::IS_TRUE), 0.08f, 10); };

    anim::AnimState idle;
    idle.name = "Idle";
    idle.clip_name = "Idle";
    idle.transitions.push_back(attack_now());
    idle.transitions.push_back(to("Walk", cond("speed", TC::Op::GT, 20.0f), 0.15f));

    anim::AnimState walk;
    walk.name = "Walk";
    walk.clip_name = "Walk";
    walk.transitions.push_back(attack_now());
    walk.transitions.push_back(to("Run", cond("speed", TC::Op::GT, 240.0f), 0.15f));
    walk.transitions.push_back(to("Idle", cond("speed", TC::Op::LT, 20.0f), 0.2f));

    anim::AnimState run;
    run.name = "Run";
    run.clip_name = "Run";
    run.transitions.push_back(attack_now());
    run.transitions.push_back(to("Walk", cond("speed", TC::Op::LT, 220.0f), 0.2f));

    anim::AnimState attack;
    attack.name = "Attack";
    attack.clip_name = "Attack";
    attack.loop = false;
    attack.transitions.push_back(to("Idle", cond("attacking", TC::Op::IS_FALSE), 0.2f));

    anim_sm_.add_state(std::move(idle));
    anim_sm_.add_state(std::move(walk));
    anim_sm_.add_state(std::move(run));
    anim_sm_.add_state(std::move(attack));
    anim_sm_.set_default_state("Idle");
    if (!anim_sm_.bind_clips(models().get_model(player_model_)->animations)) {
        std::cout << "[playground] warning: state machine has states with unbound clips\n";
    }

    // Resolve per-frame parameter names to indices once; update_player uses
    // the index overloads so no name is hashed per frame.
    param_speed_ = anim_sm_.param_index("speed");
    param_attacking_ = anim_sm_.param_index("attacking");
}

void PlaygroundGame::setup_effects() {
    {
        pickup_burst_.name = "pickup_burst";
        pickup_burst_.duration = 1.0f;
        EmitterDefinition e;
        e.name = "shards";
        e.model = "orb_shard";
        e.spawn_mode = SpawnMode::BURST;
        e.spawn_count = 26;
        e.particle_lifetime = 0.8f;
        e.velocity.type = VelocityType::RADIAL;
        e.velocity.speed = 260.0f;
        e.velocity.gravity = {0.0f, -500.0f, 0.0f};
        e.velocity.drag = 0.4f;
        e.rotation.rotation_rate = {180.0f, 240.0f, 90.0f};
        e.appearance.scale_over_lifetime = {CurveType::LINEAR, 1.0f, 0.25f, 0.8f};
        e.appearance.opacity_over_lifetime = {CurveType::FADE_OUT_LATE, 1.0f, 0.0f, 0.55f};
        e.appearance.color_tint = {1.0f, 0.85f, 0.4f, 1.0f};
        pickup_burst_.emitters.push_back(std::move(e));
    }
    {
        attack_arc_.name = "attack_arc";
        attack_arc_.duration = 0.6f;
        EmitterDefinition e;
        e.name = "slash";
        e.model = "orb_shard";
        e.spawn_mode = SpawnMode::BURST;
        e.spawn_count = 12;
        e.particle_lifetime = 0.35f;
        e.velocity.type = VelocityType::ARC;
        e.velocity.speed = 140.0f;
        e.velocity.arc_radius = 70.0f;
        e.velocity.arc_height_base = 95.0f;
        e.velocity.arc_height_amplitude = 30.0f;
        e.rotation.face_velocity = true;
        e.appearance.opacity_over_lifetime = {CurveType::LINEAR, 1.0f, 0.0f, 0.8f};
        e.appearance.color_tint = {0.85f, 0.92f, 1.0f, 1.0f};
        attack_arc_.emitters.push_back(std::move(e));
    }
    {
        // Lazy wood-smoke column; re-spawned per chimney every
        // CHIMNEY_SMOKE_PERIOD seconds so the plume never runs dry.
        chimney_smoke_.name = "chimney_smoke";
        chimney_smoke_.duration = CHIMNEY_SMOKE_PERIOD + 0.5f;
        EmitterDefinition e;
        e.name = "plume";
        e.model = "smoke_puff";
        e.spawn_mode = SpawnMode::CONTINUOUS;
        e.spawn_rate = 7.0f;
        e.duration = CHIMNEY_SMOKE_PERIOD + 0.5f;
        e.particle_lifetime = 4.5f;
        e.velocity.type = VelocityType::DIRECTIONAL;
        e.velocity.speed = 55.0f;             // up the spawn direction (+Y)
        e.velocity.gravity = {14.0f, 6.0f, -7.0f}; // buoyancy + a breath of wind
        e.velocity.drag = 0.12f;
        e.rotation.rotation_rate = {0.0f, 25.0f, 0.0f};
        e.appearance.scale_over_lifetime = {CurveType::EASE_OUT, 0.8f, 3.4f, 1.0f};
        e.appearance.opacity_over_lifetime = {CurveType::FADE_OUT_LATE, 0.5f, 0.0f, 0.35f};
        e.appearance.color_tint = {0.72f, 0.70f, 0.68f, 0.55f};
        chimney_smoke_.emitters.push_back(std::move(e));
    }
    {
        // Rain streak layer: a box volume of fast-falling streaks re-spawned
        // above the player every RAIN_LAYER_PERIOD (continuous emitters can't
        // move — same cadence trick as the chimneys). spawn_rate is retuned
        // per sim frame from the live rain intensity (update_day_weather).
        rain_layer_.name = "rain_layer";
        rain_layer_.duration = RAIN_LAYER_PERIOD + RAIN_LAYER_LIFETIME + 0.1f;
        EmitterDefinition e;
        e.name = "streaks";
        e.model = "rain_streak";
        e.spawn_mode = SpawnMode::CONTINUOUS;
        e.spawn_rate = 400.0f; // placeholder; overwritten from rain intensity
        e.duration = RAIN_LAYER_PERIOD; // spawn window only; drops outlive it
        e.particle_lifetime = RAIN_LAYER_LIFETIME;
        e.spawn_volume = {850.0f, 90.0f, 850.0f};
        e.velocity.type = VelocityType::CUSTOM;
        e.velocity.direction = glm::normalize(glm::vec3(0.10f, -1.0f, 0.045f)); // slight wind lean
        e.velocity.speed = RAIN_FALL_SPEED;
        e.appearance.opacity_over_lifetime = {CurveType::FADE_OUT_LATE, 0.55f, 0.0f, 0.75f};
        e.appearance.color_tint = {0.55f, 0.65f, 0.90f, 0.5f};
        rain_layer_.emitters.push_back(std::move(e));
    }
    {
        // Rain splash: a flat crown ring plus a small vertical spurt at the
        // impact point update_day_weather ray-cast onto terrain/props/roofs.
        rain_splash_.name = "rain_splash";
        rain_splash_.duration = 0.45f;
        EmitterDefinition ring;
        ring.name = "crown";
        ring.model = "splash_drop";
        ring.spawn_mode = SpawnMode::BURST;
        ring.spawn_count = 5;
        ring.particle_lifetime = 0.30f;
        ring.velocity.type = VelocityType::RADIAL;
        ring.velocity.speed = 60.0f;
        ring.appearance.scale_over_lifetime = {CurveType::EASE_OUT, 1.0f, 0.4f, 1.0f};
        ring.appearance.opacity_over_lifetime = {CurveType::LINEAR, 0.85f, 0.0f, 1.0f};
        ring.appearance.color_tint = {0.50f, 0.60f, 0.80f, 0.6f};
        rain_splash_.emitters.push_back(std::move(ring));
        EmitterDefinition spurt;
        spurt.name = "spurt";
        spurt.model = "splash_drop";
        spurt.spawn_mode = SpawnMode::BURST;
        spurt.spawn_count = 3;
        spurt.particle_lifetime = 0.28f;
        spurt.velocity.type = VelocityType::DIRECTIONAL; // spawn direction = +Y
        spurt.velocity.speed = 130.0f;
        spurt.velocity.gravity = {0.0f, -900.0f, 0.0f};
        spurt.appearance.scale_over_lifetime = {CurveType::EASE_OUT, 0.8f, 0.3f, 1.0f};
        spurt.appearance.opacity_over_lifetime = {CurveType::LINEAR, 0.8f, 0.0f, 1.0f};
        spurt.appearance.color_tint = {0.55f, 0.65f, 0.85f, 0.6f};
        rain_splash_.emitters.push_back(std::move(spurt));
    }
}

// Bring up bracken::physics and populate it: the terrain heightfield (river
// carve included), a static box per placed prop with a real silhouette
// (buildings from the layout rects, crates/wagons/fences/bricks/rocks from
// their model AABBs, trees as trunk-footprint boxes), the chrome ball as a
// dynamic sphere, and a kinematic capsule that shadows the character so
// walking into the ball pushes it. Main thread, end of on_init — after this,
// PhysicsWorld is touched exclusively from on_simulate until on_shutdown.
void PlaygroundGame::setup_physics() {
    physics::PhysicsConfig cfg;
    cfg.gravity = {0.0f, PHYS_GRAVITY_Y, 0.0f};
    cfg.max_bodies = 4096; // heightfield + ~a few hundred statics + ball + capsule
    cfg.max_body_pairs = 8192;
    cfg.max_contact_constraints = 4096;
    cfg.fixed_dt = PHYS_FIXED_DT;
    cfg.max_steps_per_update = 4;
    physics_ready_ = physics_.init(cfg, core::jobs::JobSystem::instance());
    if (!physics_ready_) {
        BRACKEN_LOG_WARN("playground", "physics init failed — chrome ball stays decorative");
        pending_colliders_.clear();
        return;
    }

    // Terrain: the render heightmap, decoded to world-height floats in the
    // same row-major (z * resolution + x) layout. Heights are absolute world
    // Y, so the field origin sits at y=0.
    {
        const uint32_t res = heightmap_.resolution;
        std::vector<float> samples(static_cast<size_t>(res) * res);
        for (uint32_t z = 0; z < res; ++z) {
            for (uint32_t x = 0; x < res; ++x) {
                samples[static_cast<size_t>(z) * res + x] = heightmap_.get_height_local(x, z);
            }
        }
        terrain_body_ = physics_.add_heightfield(
            {heightmap_.world_origin_x, 0.0f, heightmap_.world_origin_z}, heightmap_.world_size, samples, res);
        static_bodies_.push_back(terrain_body_);
    }

    auto add_static_box = [this](const glm::vec3& center, const glm::vec3& half_extents) {
        physics::BodyDesc desc;
        desc.type = physics::BodyType::Static;
        desc.position = center;
        const physics::BodyId id = physics_.add_box(desc, half_extents);
        if (id != physics::INVALID_BODY_ID) {
            static_bodies_.push_back(id);
        }
    };

#if PLAYGROUND_HAS_VILLAGE
    // Buildings: village.glb is one merged mesh, but the layout rects are the
    // exact footprints (the same ones the push-out collision + grass
    // exclusions use). One box per building, 9 m tall, base sunk a little.
    if (has_village_) {
        const float S = village::VILLAGE_SCALE;
        for (const auto& b : village::BUILDINGS) {
            const float cx = VILLAGE_CX + (b.min_x + b.max_x) * 0.5f * S;
            const float cz = VILLAGE_CZ + (b.min_z + b.max_z) * 0.5f * S;
            const float ground = heightmap_.get_height_world(cx, cz);
            add_static_box({cx, ground - BUILDING_COLLIDER_SINK + BUILDING_COLLIDER_HEIGHT * 0.5f, cz},
                           {(b.max_x - b.min_x) * 0.5f * S, BUILDING_COLLIDER_HEIGHT * 0.5f,
                            (b.max_z - b.min_z) * 0.5f * S});
        }
    }
#endif

    // Placed props: world-space AABB = model-local AABB carried through the
    // placement transform (|M| * half-extents — exact for these TRS
    // transforms, conservative in general). Trees keep only a trunk box.
    for (const auto& pc : pending_colliders_) {
        const Model* model = models().get_model(pc.model);
        if (model == nullptr || !model->loaded) {
            continue;
        }
        const glm::vec3 lcenter = model->bounds.center();
        const glm::vec3 lhalf = model->bounds.size() * 0.5f;
        const glm::vec3 wcenter = glm::vec3(pc.transform * glm::vec4(lcenter, 1.0f));
        glm::mat3 absm{pc.transform};
        for (int c = 0; c < 3; ++c) {
            absm[c] = glm::abs(absm[c]);
        }
        glm::vec3 whalf = absm * lhalf;
        glm::vec3 center = wcenter;
        if (pc.trunk) {
            // Trunk footprint; height capped at 40% of the AABB height,
            // measured up from the AABB base so the canopy never blocks
            // movement (or a lofted ball). Nature-pack trees use the measured
            // per-model footprint centered on the placement origin — the
            // trunk grows from the model origin, while the AABB center is
            // dragged sideways by the canopy lean. Procedural trees keep the
            // AABB-fraction rule (their AABB is mostly trunk).
            const float base_y = wcenter.y - whalf.y;
            const float trunk_half_h = whalf.y * TRUNK_HEIGHT_FRACTION;
            if (pc.trunk_half > 0.0f) {
                // Uniform TRS placement: scale = any basis column's length.
                const float scale = glm::length(glm::vec3(pc.transform[0]));
                whalf.x = whalf.z = pc.trunk_half * scale;
                center.x = pc.transform[3].x;
                center.z = pc.transform[3].z;
            } else {
                whalf.x *= TRUNK_XZ_FRACTION;
                whalf.z *= TRUNK_XZ_FRACTION;
            }
            whalf.y = trunk_half_h;
            center.y = base_y + trunk_half_h;
        }
        if (whalf.x < 1.0f || whalf.y < 1.0f || whalf.z < 1.0f) {
            continue; // degenerate/flat dressing — not worth a body
        }
        add_static_box(center, whalf);
    }
    pending_colliders_.clear();
    pending_colliders_.shrink_to_fit();

    // The chrome ball: a dynamic sphere at the plaza spot, raised so the
    // first frames show it drop, bounce and settle (visible physics proof —
    // the smoke run exercises this automatically). The mesh is a UV sphere
    // centred on the model origin (bounds ±radius), so body transform ==
    // render transform with no offset.
    {
        const Model* chrome = models().get_model(chrome_model_);
        if (chrome != nullptr && chrome->loaded) {
            ball_radius_ = chrome->width() * 0.5f; // 45 for the stock model
        }
        physics::BodyDesc desc;
        desc.type = physics::BodyType::Dynamic;
        desc.position = glm::vec3(chrome_transform_[3]) + glm::vec3(0.0f, BALL_SPAWN_RAISE, 0.0f);
        desc.mass = BALL_MASS;
        desc.friction = BALL_FRICTION;
        desc.restitution = BALL_RESTITUTION;
        desc.start_active = true;
        ball_body_ = physics_.add_sphere(desc, ball_radius_);
        // Keep render + panel mirrors coherent until the first sim frame.
        chrome_transform_ = glm::translate(glm::mat4(1.0f), desc.position);
        ball_pos_ = desc.position;
        ball_active_ = true;
    }

    // The character pusher: a kinematic capsule move_kinematic()-ed onto the
    // heightmap-driven character every sim frame, so Jolt's
    // kinematic-vs-dynamic response lets the villager dribble the ball by
    // walking into it. Locomotion itself stays heightmap-driven, but
    // update_player depenetrates against the static bodies via
    // collide_capsule — walls, trunks and props block the character too.
    {
        physics::BodyDesc desc;
        desc.type = physics::BodyType::Kinematic;
        desc.position = player_pos_ + glm::vec3(0.0f, CHAR_CAPSULE_CENTER_Y, 0.0f);
        character_body_ = physics_.add_capsule(desc, CHAR_CAPSULE_HALF_CYL, CHAR_CAPSULE_RADIUS);
    }

    BRACKEN_LOG_INFO("playground", "physics up: {} static bodies, ball radius {}", static_bodies_.size(),
                     ball_radius_);
}

// ---------------------------------------------------------------------------
// Simulate (pool job — docs/multithreading_plan.md §3.1)
// ---------------------------------------------------------------------------

void PlaygroundGame::on_simulate(SimFrame& sim) {
    const float dt = sim.dt();
    time_ = static_cast<float>(sim.time());
    frame_ = static_cast<int>(sim.frame_index()) + 1; // 1-based, like the old per-update counter

    if (smoke_frames_ > 0) {
        update_smoke_driver(dt);
    } else {
        handle_input(sim);
    }
    update_player(dt);
    update_camera(sim);
    update_orbs(dt);
    update_wanderer(dt);
    update_chimneys(dt);
    update_day_weather(dt);
    // Physics is sim-stage: step it after the character/orb update so the
    // pusher capsule chases this frame's player position, and before
    // build_frame so the ball renders where it landed this frame.
    update_physics(dt);

    // Put the listener on the camera. play_sound/set_listener are worker-safe,
    // so this is a plain call from the sim job — no locking, no deferral.
    audio().set_listener(camera().get_position(), camera().get_forward(),
                         glm::vec3(0.0f, 1.0f, 0.0f));

    build_frame(sim);
}

void PlaygroundGame::handle_input(const SimFrame& sim) {
    // The UI gets first refusal, and it gets it for the WHOLE function — the
    // raw snapshot is deliberately never bound to a name here. Masking only
    // the ActionMap would leave every `was_key_just_pressed` below reading
    // straight through the gate: typing "F1" into the demo menu's text field
    // would toggle AO, and Escape would quit the game instead of leaving the
    // field. `ui_capture_` is what the UI took while describing the PREVIOUS
    // frame (build_ui runs at the end of the frame, after this), so the gate
    // is one frame late — see the note at build_ui, and ui::Context's class
    // comment for why one frame is inherent to the design rather than a
    // shortcut taken here.
    const InputSnapshot in = ui_capture_.mask(sim.input());

    // Evaluate every action once, up front: edges are computed against the
    // previous update(), so it must happen exactly once per sim frame and
    // before any query.
    actions_.update(in);

    if (in.was_key_just_pressed(SDL_SCANCODE_ESCAPE)) {
        quit();
    }
    if (actions_.pressed(act_quicksave_)) {
        quicksave();
    }
    if (actions_.pressed(act_quickload_)) {
        quickload();
    }

    if (actions_.pressed(act_flashlight_)) {
        flashlight_ = !flashlight_;
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F3)) {
        debug_draw_ = !debug_draw_;
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F12)) {
        request_screenshot(std::format("screenshot_{:03d}.png", screenshot_counter_++));
    }

    // Graphics-settings toggles. edit_graphics_settings() hands back a scoped
    // copy and requests it on scope exit — untouched frames cost nothing, and
    // there is no "forgot to hand it back" to forget. Applying the settings is
    // GPU/SDL work, so the request is deferred to the next frame fence.
    {
        auto gs = edit_graphics_settings();
        if (in.was_key_just_pressed(SDL_SCANCODE_F1)) {
            // Off -> GTAO Half -> GTAO Full
            gs->ao_mode = static_cast<AOMode>((static_cast<int>(gs->ao_mode) + 1) % 3);
        }
        if (in.was_key_just_pressed(SDL_SCANCODE_F2)) {
            gs->bloom_enabled = !gs->bloom_enabled;
        }
        if (in.was_key_just_pressed(SDL_SCANCODE_F4)) {
            gs->volumetric_fog = !gs->volumetric_fog;
            gs->god_rays = gs->volumetric_fog;
        }
        if (in.was_key_just_pressed(SDL_SCANCODE_F5)) {
            gs->shadow_mode = static_cast<ShadowMode>((static_cast<int>(gs->shadow_mode) + 1) % 3);
        }
        if (in.was_key_just_pressed(SDL_SCANCODE_F6)) {
            gs->aa_mode = static_cast<AAMode>((static_cast<int>(gs->aa_mode) + 1) % 3); // Off -> FXAA -> TAA
        }
        if (in.was_key_just_pressed(SDL_SCANCODE_F7)) {
            gs->ibl_enabled = !gs->ibl_enabled; // sky IBL ambient vs hemisphere fallback
        }
        if (in.was_key_just_pressed(SDL_SCANCODE_F8)) {
            gs->local_shadows = !gs->local_shadows; // point/spot light shadows
        }
    }

    // Camera-relative movement.
    glm::vec3 fwd = camera().get_forward();
    fwd.y = 0.0f;
    fwd = glm::length(fwd) > 0.001f ? glm::normalize(fwd) : glm::vec3(0, 0, -1);
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
    // One normalized 2D read instead of four ifs — and the diagonal is not
    // 1.41x faster, which the hand-rolled version only avoided because it
    // normalized afterwards.
    const glm::vec2 move = actions_.vector(act_left_, act_right_, act_back_, act_forward_);
    const glm::vec3 dir = fwd * move.y + right * move.x;
    move_dir_ = glm::length(dir) > 0.001f ? glm::normalize(dir) : glm::vec3(0.0f);

    running_input_ = actions_.held(act_run_);
    // Latched, not sampled: the controller consumes the edge, so a press
    // between sim frames cannot be missed.
    jump_requested_ = jump_requested_ || actions_.pressed(act_jump_);
    crouch_held_ = actions_.held(act_crouch_);
    if (actions_.pressed(actions_.find("nav_debug"))) {
        nav_debug_draw_ = !nav_debug_draw_; // N: show the villager's route
    }

    if (actions_.pressed(act_attack_) && attack_timer_ <= 0.0f) {
        attack_timer_ = ATTACK_DURATION;
        pending_attack_arc_ = true;
        camera().add_directional_shake(forward_from_yaw(facing_.current), 6.0f, 0.15f);
    }
}

void PlaygroundGame::update_smoke_driver(float dt) {
    // Jump on a fixed cadence so the smoke run actually exercises the
    // character controller's air path (and the counters below prove it did).
    if (frame_ % 47 == 0) {
        jump_requested_ = true;
    }
    if (character_.state().grounded) {
        ++smoke_grounded_frames_;
    } else {
        ++smoke_air_frames_;
    }
    smoke_max_height_ = std::max(smoke_max_height_, player_pos_.y);

    // Self-drive, sim side: orbit the camera, run toward the nearest orb
    // (real pickups), attack periodically. The mid-run settings flips that
    // exercise the post-processing matrix are fence work — see
    // update_smoke_fence; one-time run config is setup_smoke_run. Sun-pinned
    // art runs keep the camera steady, facing the sun, framing the sky.
    if (!smoke_sun_forced_ && !smoke_cam_forced_) {
        camera().rotate_yaw(35.0f * dt);
        // Re-assert every frame: the controller eases pitch back toward its
        // free-play default, so the one-shot set at init decays away.
        camera().set_pitch(14.0f);
    }

    // Chase the nearest orb ON THE PLAZA: the orbit camera has no building
    // collision, so wandering down the lanes would swing it through walls.
    auto on_plaza = [](const glm::vec3& p) {
#if PLAYGROUND_HAS_VILLAGE
        const float S = village::VILLAGE_SCALE;
        return p.x > VILLAGE_CX + (village::PLAZA.min_x - 1.0f) * S &&
               p.x < VILLAGE_CX + (village::PLAZA.max_x + 1.0f) * S &&
               p.z > VILLAGE_CZ + (village::PLAZA.min_z - 1.0f) * S &&
               p.z < VILLAGE_CZ + (village::PLAZA.max_z + 1.0f) * S;
#else
        (void)p;
        return true;
#endif
    };
    const Orb* nearest = nullptr;
    float best = 1e30f;
    for (const auto& orb : orbs_) {
        if (!on_plaza(orb.pos)) {
            continue;
        }
        float d = glm::distance(glm::vec2(orb.pos.x, orb.pos.z), glm::vec2(player_pos_.x, player_pos_.z));
        if (d < best) {
            best = d;
            nearest = &orb;
        }
    }
    // No orb on the plaza right now: idle back toward the plaza centre.
    const glm::vec3 target = nearest ? nearest->pos : glm::vec3(VILLAGE_CX, 0.0f, VILLAGE_CZ);
    glm::vec3 d = target - player_pos_;
    d.y = 0.0f;
    move_dir_ = glm::length(d) > 60.0f ? glm::normalize(d) : glm::vec3(0.0f);
    running_input_ = true;

    if (frame_ % 120 == 60 && attack_timer_ <= 0.0f) {
        attack_timer_ = ATTACK_DURATION;
        pending_attack_arc_ = true;
    }
}

void PlaygroundGame::apply_rt_mode(bracken::GraphicsSettings& gs) const {
    gs.rt_shadows = rt_mode_ == "shadows" || rt_mode_ == "both";
    gs.rt_reflections = rt_mode_ == "reflections" || rt_mode_ == "both";
    // Separate variable rather than another PLAYGROUND_RT word: local-light
    // shadows are orthogonal to the sun and reflections, and the A/B that
    // matters is "same RT effects, local shadows on vs off".
    if (rt_local_mode_ == "full") {
        gs.rt_local_shadows = bracken::LocalRtShadowMode::Full;
    } else if (rt_local_mode_ == "ratio") {
        gs.rt_local_shadows = bracken::LocalRtShadowMode::Ratio;
    } else {
        gs.rt_local_shadows = bracken::LocalRtShadowMode::Off;
    }
}

void PlaygroundGame::apply_gi_mode(bracken::GraphicsSettings& gs) const {
    // `probes_screen` is accepted now so the pin vocabulary does not change
    // when phase R4 makes it mean something; today it behaves as `probes`.
    if (gi_mode_ == "probes") {
        gs.gi_mode = bracken::GiMode::Probes;
    } else if (gi_mode_ == "probes_screen") {
        gs.gi_mode = bracken::GiMode::ProbesScreen;
    } else if (gi_mode_ == "voxel") {
        // The RASTER path: the same probe field fed by a voxel clipmap instead
        // of by rays, so it runs with no acceleration structure at all.
        gs.gi_mode = bracken::GiMode::VoxelProbes;
    } else {
        gs.gi_mode = bracken::GiMode::Off;
    }
}

// One-time smoke-run configuration — the retired driver's frame-1 block, now
// run from the end of on_init (main thread, renderer up, no sim in flight, so
// the render-side touches are legal; the env-var parsing happens once instead
// of from a pool job).
void PlaygroundGame::setup_smoke_run() {
    // Pin the timestep: the smoke run is a fixed FRAME count, so with
    // wall-clock timing the amount of simulated time it covers depends on how
    // fast the machine happens to be. Measured on this tree, three consecutive
    // runs at 62/59/54 fps produced 1065/1091/1109 entities and 468/510/556
    // draw calls — the slower the run, the further the world advanced. At a
    // pinned 1/60 s, frame N is always simulated time N/60 and the run is
    // reproducible, which is what makes it usable as a regression gate.
    set_fixed_timestep(1.0f / 60.0f);

    // One scoped edit for the whole one-time config: every field below lands
    // in the same request, applied at the first frame fence — before frame 0
    // renders.
    auto gs = edit_graphics_settings();

    // START FROM THE ENGINE DEFAULTS. set_settings_persistence(false) in
    // on_init already stops the file being read, so this is belt and braces —
    // but it is also the thing that scales. There used to be an ad-hoc list of
    // fields reset here ("a previous run may have persisted its end-of-run
    // state"), and it had silently fallen behind by ten fields: rt_shadows,
    // rt_reflections, shadow_mode, ao_mode, ao_strength, ao_radius,
    // draw_distance, shadow_distance, local_shadow_spot_budget, vsync_mode.
    // Two of those (rt_*) made plain `--smoke` trace without being asked; two
    // more (shadow_mode=Off, draw_distance) moved mean_luma by 4.4 points with
    // no code change. Enumerating fields does not scale; resetting does.
    // Every override below — and every env hook — runs after this and wins.
    *gs = bracken::GraphicsSettings{};
    // Ray tracing: pinned from PLAYGROUND_RT, off when it is unset.
    apply_rt_mode(*gs);
    // Probe-field GI: pinned from PLAYGROUND_GI, off when it is unset. Same
    // rule and the same reason — gi_mode persists, so an unpinned smoke run
    // would inherit whatever the last interactive session left behind.
    apply_gi_mode(*gs);
    // TAA so the smoke always exercises the temporal path (and the final
    // screenshot captures it). This is also the struct default; kept explicit
    // because PLAYGROUND_AA below overrides it and the intent should read.
    gs->aa_mode = AAMode::TAA;
    // PLAYGROUND_AA=0/1/2 pins Off/FXAA/TAA for the whole run (A/B hook;
    // the mid-run AA flips at 110/160 are skipped while pinned).
    const char* aa_env = SDL_getenv("PLAYGROUND_AA");
    if (aa_env && aa_env[0] >= '0' && aa_env[0] <= '2') {
        gs->aa_mode = static_cast<AAMode>(aa_env[0] - '0');
        smoke_aa_forced_ = true;
    }
    // PLAYGROUND_RT_RANGE=0/1/2/3 pins rt_max_instance_range
    // (Near/Medium/Far/Unlimited) for the whole run. Unset leaves the engine
    // default (Unlimited), so a plain run traces everything it draws exactly as
    // it did before the knob existed. This is the A/B hook the --rt-bench
    // harness needs to measure what the knob actually buys.
    const char* rt_range_env = SDL_getenv("PLAYGROUND_RT_RANGE");
    if (rt_range_env && rt_range_env[0] >= '0' && rt_range_env[0] <= '3') {
        gs->rt_max_instance_range = static_cast<RtRange>(rt_range_env[0] - '0');
    }
    // PLAYGROUND_AO=0/1/2 pins the AO mode (Off/Half/Full) for the whole
    // run (A/B hook; the frame-140 GTAO_Full flip is skipped while pinned).
    const char* ao_env = SDL_getenv("PLAYGROUND_AO");
    if (ao_env && ao_env[0] >= '0' && ao_env[0] <= '2') {
        gs->ao_mode = static_cast<AOMode>(ao_env[0] - '0');
        smoke_ao_forced_ = true;
    }
    // --rt-bench: pin the camera before anything else can move it. A benchmark
    // whose camera orbits measures a different scene on every frame, and the
    // orbit is what the default smoke run does. This vantage is a raised view
    // across the plaza from the village centre — enough of the settlement in
    // frame that traversal has real work to do, and identical on every run
    // because the forced-camera path pre-settles the rig's springs.
    // PLAYGROUND_CAMERA below still wins, so an A/B can pick its own vantage.
    if (rt_bench_) {
        // PLAYGROUND_RT_BENCH_ORBIT=1 gives the bench the SMOKE run's orbiting
        // camera instead, which is the only condition in this repo that
        // reproduces the thing R7 was asked to fix: a moving camera
        // disoccludes, and a disocclusion is where a 1-spp reservoir estimate
        // is at its worst. A pinned camera lets the temporal accumulation reach
        // its ceiling everywhere and therefore UNDERSTATES both the problem and
        // the fix. It is deliberately not the default — a per-pass timing table
        // wants a scene that does not change under it.
        smoke_cam_forced_ = SDL_getenv("PLAYGROUND_RT_BENCH_ORBIT") == nullptr;
        smoke_cam_target_ = village_pos(0.0f, 0.0f, 2.0f);
        smoke_cam_yaw_ = 310.0f;
        smoke_cam_pitch_ = 20.0f;
        smoke_cam_dist_ = 1400.0f; // world units (14 m at VILLAGE_SCALE)

        // PLAYGROUND_SUN_SWEEP=<deg/s>: the only way to measure R5b's
        // lighting-change invalidation, since every other run in this repo pins
        // the environment and a static sun is precisely the case the mechanism
        // is inert in. Bench-only: a --smoke run's screenshot baseline is a
        // function of its sun, and a sweeping one would have no baseline.
        if (const char* sweep = SDL_getenv("PLAYGROUND_SUN_SWEEP"); sweep != nullptr && sweep[0]) {
            sun_sweep_deg_per_s_ = std::max(0.0f, static_cast<float>(SDL_atof(sweep)));
        }
        // The R7 temporal-noise probe, on for every bench run because that is
        // the only place the number is wanted and the only place a spare
        // dispatch is affordable. A --smoke run measures ONE frame and could
        // not report a frame-to-frame delta if it wanted to.
        rt_di_noise_probe_ = true;
    }
    // PLAYGROUND_RT_SPATIAL=<k> / PLAYGROUND_RT_DI_ITERS=<n>: the A/B levers for
    // the R7 package. Unset leaves the shipped defaults (3 neighbours, 2
    // a-trous iterations); `PLAYGROUND_RT_SPATIAL=0 PLAYGROUND_RT_DI_ITERS=1` is
    // the pre-R7 sampling and filtering, which is what the noise metric's
    // "before" column is measured with. Not bench-only: the same A/B has to be
    // available to a --smoke run for the luma baselines to mean anything.
    if (const char* env = SDL_getenv("PLAYGROUND_RT_SPATIAL"); env != nullptr && env[0]) {
        rt_spatial_neighbors_ = static_cast<uint32_t>(std::max(0, SDL_atoi(env)));
    }
    if (const char* env = SDL_getenv("PLAYGROUND_RT_DI_ITERS"); env != nullptr && env[0]) {
        rt_di_spatial_iterations_ = static_cast<uint32_t>(std::clamp(SDL_atoi(env), 1, 2));
    }
    // PLAYGROUND_RT_EMISSIVE=<M_e> / PLAYGROUND_RT_EMISSIVE_LOCAL=<0..1>: the
    // two A/B levers R8's cost table is measured with. `RT_EMISSIVE=0` is the
    // records-OFF column (no emissive candidate stream at all, every RIS weight
    // back to its R5c form); `RT_EMISSIVE_LOCAL=0` is R6's purely global alias
    // draw with the froxel lists built but unused. Same status as
    // PLAYGROUND_RT_SPATIAL: measurement instruments, not quality levels.
    if (const char* env = SDL_getenv("PLAYGROUND_RT_EMISSIVE"); env != nullptr && env[0]) {
        rt_emissive_candidates_ = static_cast<uint32_t>(std::clamp(SDL_atoi(env), 0, 16));
    }
    if (const char* env = SDL_getenv("PLAYGROUND_RT_EMISSIVE_LOCAL"); env != nullptr && env[0]) {
        rt_emissive_froxel_share_ = std::clamp(static_cast<float>(SDL_atof(env)), 0.0f, 1.0f);
    }
    {
        bracken::scene::SceneRenderer::RtLocalTuning tuning;
        tuning.spatial_neighbors = rt_spatial_neighbors_;
        tuning.di_spatial_iterations = rt_di_spatial_iterations_;
        tuning.di_noise_probe = rt_di_noise_probe_;
        tuning.emissive_candidates = rt_emissive_candidates_;
        tuning.emissive_froxel_share = rt_emissive_froxel_share_;
        scene_renderer().set_rt_local_tuning(tuning);
    }
    // PLAYGROUND_CAMERA="vx,vz,yaw,pitch,dist_m" pins the camera at a
    // village-space position for the whole run (art close-up hook).
    if (const char* cam_env = SDL_getenv("PLAYGROUND_CAMERA")) {
        float vx = 0.0f;
        float vz = 0.0f;
        float dist = 5.0f;
        if (std::sscanf(cam_env, "%f,%f,%f,%f,%f", &vx, &vz, &smoke_cam_yaw_, &smoke_cam_pitch_, &dist) >= 2) {
            smoke_cam_forced_ = true;
            smoke_cam_target_ = village_pos(vx, vz, 1.2f);
            smoke_cam_dist_ = dist * 100.0f;
        }
    }
    // (sun_intensity / volumetric_fog / god_rays used to be reset here against
    // a run's persisted end state — the whole-struct reset above covers them,
    // and every other field the list had missed.)
    // PLAYGROUND_SUN_SHADOWS=0 forces the CSM off for the whole run (A/B
    // debug hook, like PLAYGROUND_LOCAL_SHADOWS). Remembered so the frame-200
    // PCF flip stays skipped without re-reading the env var at the fence.
    const char* ss_env = SDL_getenv("PLAYGROUND_SUN_SHADOWS");
    if (ss_env && ss_env[0] == '0') {
        smoke_sun_shadows_off_ = true;
        gs->shadow_mode = ShadowMode::Off;
    }
    const char* ls_env = SDL_getenv("PLAYGROUND_LOCAL_SHADOWS");
    smoke_local_shadows_ = !(ls_env && ls_env[0] == '0');
    gs->local_shadows = smoke_local_shadows_;
    // PLAYGROUND_AUTO_EXPOSURE=0 forces histogram auto exposure off for
    // the whole run (the A/B test hook, like PLAYGROUND_LOCAL_SHADOWS).
    const char* ae_env = SDL_getenv("PLAYGROUND_AUTO_EXPOSURE");
    gs->auto_exposure = !(ae_env && ae_env[0] == '0');
    // Flashlight on for the whole run: the shadow-casting spot follows
    // the player + camera, so its tile re-renders nearly every frame and
    // the character's local shadow is visible in the final screenshot.
    flashlight_ = true;
    // Frame the village for the end-of-run screenshot: pull back, tilt
    // down, and start the orbit so the final frame looks AWAY from the
    // sun — front-lit facades instead of a backlit silhouette (smoke
    // mode only; the orbit adds ~15 deg by the screenshot frame).
    auto cam_cfg = camera().get_config();
    cam_cfg.distance = 520.0f;
    cam_cfg.height_offset = 150.0f;
    camera().set_config(cam_cfg);
    camera().set_pitch(14.0f); // positive pitch looks DOWN (the sun-pin path uses -12 to frame the sky)
    camera().set_yaw(310.0f);
    // Smoke runs finish in well under a second — poll the adapted-EV
    // readback every frame so the end-of-run auto_ev= value is fresh
    // (PLAYGROUND_EXPOSURE_PROBE=1 additionally prints a per-frame
    // adaptation timeline).
    if (auto* auto_exposure = scene_renderer().auto_exposure()) {
        auto_exposure->set_readback_interval(0.0f);
    }
    const char* probe_env = SDL_getenv("PLAYGROUND_EXPOSURE_PROBE");
    smoke_exposure_probe_ = (probe_env && probe_env[0] == '1');
    // PLAYGROUND_GRADE=1 applies a deliberately non-neutral look (warm
    // white balance + punchy CDL) for color-grading A/B evidence; the
    // default run keeps the scene's own (near-neutral) dusk grade.
    // All env edits below go to the HOST-owned env_ (staged next frame).
    const char* grade_env = SDL_getenv("PLAYGROUND_GRADE");
    if (grade_env && grade_env[0] == '1') {
        env_.grade_temperature = 0.4f;
        env_.grade_slope = {1.05f, 1.0f, 0.9f};
        env_.grade_offset = {0.02f, 0.0f, -0.02f};
        env_.grade_power = {1.1f, 1.1f, 1.1f};
        env_dirty_ = true;
    }
    // PLAYGROUND_SUN_ELEVATION=<degrees> pins the sun at that elevation
    // (default azimuth kept) for the whole run — the sky/cloud/fog art
    // hook. The playground plays day/night host: sun intensity and tint
    // follow the elevation the way a game's time-of-day system would.
    // While pinned, the frame-180/195 sun swing and the frame-205 sun dim
    // are skipped so the final screenshot shows the requested sky.
    const char* elev_env = SDL_getenv("PLAYGROUND_SUN_ELEVATION");
    if (elev_env && elev_env[0]) {
        smoke_sun_forced_ = true;
        const float el = glm::radians(static_cast<float>(SDL_atof(elev_env)));
        // Default azimuth: the direction TO the sun of the scene env.
        const glm::vec3 def_to_sun = glm::normalize(-scene_sun_dir_);
        const float az = std::atan2(def_to_sun.z, def_to_sun.x);
        const glm::vec3 to_sun{std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)};
        smoke_forced_sun_dir_ = -glm::normalize(to_sun);
        env_.sun_dir = smoke_forced_sun_dir_;
        // Host-style time-of-day response: dim + warm the direct sun as it
        // drops (the engine's sky handles its own low-sun response).
        const float low_sun = 1.0f - std::clamp(std::sin(el) * 3.0f, 0.0f, 1.0f);
        const float day = std::clamp(std::sin(el) * 2.0f + 0.2f, 0.0f, 1.0f);
        env_.sun_color = glm::mix(glm::vec3(1.0f, 0.95f, 0.9f), glm::vec3(1.0f, 0.55f, 0.3f),
                                  low_sun * 0.8f);
        // Reset the persisted knobs a previous (default) smoke run may
        // have saved, so art shots are reproducible from any state.
        gs->sun_intensity = 1.0f * day;
        gs->volumetric_fog = false;
        gs->god_rays = false;
            // Art-shot framing: look toward the horizon, facing the sun, so
        // the screenshot frames the sky (silver lining / forward scatter
        // are sun-side effects). Camera forward is -(sin yaw, cos yaw).
        camera().set_pitch(-12.0f);
        // +10 deg yaw offset keeps the low sun disc clear of the player.
        const glm::vec2 sun_xz = glm::normalize(glm::vec2(to_sun.x, to_sun.z));
        camera().set_yaw(glm::degrees(std::atan2(-sun_xz.x, -sun_xz.y)) + 10.0f);
        // PLAYGROUND_CLOUDS="coverage,density,scale" overrides the cloud
        // shaping for the run (sky look-dev companion to the sun pin).
        if (const char* clouds_env = SDL_getenv("PLAYGROUND_CLOUDS")) {
            float cov = env_.cloud_coverage, den = env_.cloud_density, scl = env_.cloud_scale;
            if (std::sscanf(clouds_env, "%f,%f,%f", &cov, &den, &scl) >= 1) {
                env_.cloud_coverage = cov;
                env_.cloud_density = den;
                env_.cloud_scale = scl;
            }
        }
        env_dirty_ = true;
    }

    // Smoke runs keep the legacy fixed environment by default so the
    // screenshot and the scripted frame-180/195 sun swings stay
    // deterministic — the day/night + weather system only drives a smoke
    // run when explicitly asked to via the art hooks below (frozen clock,
    // pinned weather, so still deterministic):
    //   PLAYGROUND_TOD_HOUR=<0..24>            pin the sun to that hour
    //   PLAYGROUND_WEATHER=clear|overcast|rain pin the weather (instantly)
    day_weather_enabled_ = false;
    const char* tod_env = SDL_getenv("PLAYGROUND_TOD_HOUR");
    const char* weather_env = SDL_getenv("PLAYGROUND_WEATHER");
    if ((tod_env || weather_env) && !smoke_sun_forced_) {
        day_weather_enabled_ = true;
        day_weather_.animate = false;
        day_weather_.mode = WeatherMode::Clear;
        if (tod_env && tod_env[0]) {
            day_weather_.hour = std::clamp(static_cast<float>(SDL_atof(tod_env)), 0.0f, 23.99f);
        }
        if (weather_env && weather_env[0]) {
            const std::string_view w{weather_env};
            if (w == "overcast") {
                day_weather_.mode = WeatherMode::Overcast;
            } else if (w == "rain") {
                day_weather_.mode = WeatherMode::Rain;
            }
        }
        day_weather_.snap(); // screenshots want the state, not the transition
    }
}

// Fence-side smoke logic, called from on_main_frame: main thread, no sim in
// flight, BEFORE the joined frame renders — the render-side reads here (the
// exposure probe, the end-of-run stats) see that frame's state.
// frame_ was written by the sim frame just joined
// (== mf.frame_index() + 1), so the frame numbering below is unchanged from
// the old driver.
void PlaygroundGame::update_smoke_fence(bracken::MainFrame& mf) {
    BRACKEN_ASSERT(frame_ == static_cast<int>(mf.frame_index()) + 1,
                   "smoke fence out of step with the joined sim frame");
    if (smoke_start_ns_ == 0) {
        smoke_start_ns_ = SDL_GetTicksNS();
    }

    // Save/load coverage for the smoke run, at the fence because save() does
    // blocking file IO and no sim is in flight here. Saving and immediately
    // reloading must be LOSSLESS: if it is, game state is unchanged and the
    // rest of the run is identical. A quicksave that quietly degrades a float
    // or drops an orb position would show up downstream.
    if (frame_ == 150) {
        quicksave();
        quickload();
    }

    // One scoped edit per fence: whatever the scripted flips below touch is
    // requested once on scope exit and applied at the NEXT fence (one frame
    // later than the old immediate apply — invisible in a 240-frame run, and
    // every flip is well clear of the frame-220 screenshot).
    auto gs = edit_graphics_settings();
    if (smoke_exposure_probe_) {
        if (const auto* auto_exposure = scene_renderer().auto_exposure()) {
            if (const auto& state = auto_exposure->latest_state()) {
                std::cout << std::format("[probe] frame={} ev={:.3f} target={:.3f} avg_log2={:.3f}\n", frame_,
                                         state->current_ev, state->target_ev, state->avg_log2_luma);
            }
        }
    }
    if (frame_ == 80 && !smoke_sun_forced_) {
        // Skipped in sun-pinned art runs: the half-res volumetric wash would
        // sit over the sky being photographed (it defaults off in free play).
        gs->volumetric_fog = true;
        gs->god_rays = true;
    }
    if (smoke_local_shadows_ && frame_ == 100) {
        gs->local_shadows = false; // exercise the local-shadow toggle path...
    }
    if (smoke_local_shadows_ && frame_ == 115) {
        gs->local_shadows = true; // ...and the cached-tile revival after it
    }
    if (frame_ == 110 && !smoke_sun_forced_ && !smoke_aa_forced_) {
        gs->aa_mode = AAMode::FXAA; // exercise the FXAA resolve mid-run...
    }
    if (frame_ == 140 && !smoke_ao_forced_) {
        gs->ao_mode = AOMode::GTAO_Full; // exercise the full-res AO path (default is GTAO_Half)
    }
    if (frame_ == 160 && !smoke_sun_forced_ && !smoke_aa_forced_) {
        gs->aa_mode = AAMode::TAA; // ...then back to TAA (history restarts cleanly)
    }
    if (frame_ == 170 && !smoke_sun_forced_) {
        // Show the dev overlay so the end-of-run screenshot proves it renders.
        // Harmless no-op in stub builds (the hidden UI never initialized).
        // Skipped in sun-pinned art runs so the overlay doesn't cover the sky.
        dev_ui().set_visible(true);
    }
    if (frame_ == 180 && !smoke_sun_forced_) {
        // Swing the sun (a "day cycle" step): the sky-relevant environment
        // fields change, so the IBL prefilter chain must refresh — amortized
        // over ~7 frames, never a hitch. Restored below so the final
        // screenshot keeps the scene's golden-hour lighting. Host env copy:
        // staged by the next sim frame, adopted the frame after (one frame
        // later than the old live edit — invisible in a 240-frame run).
        // Skipped when the day/night art hooks drive the environment
        // (PLAYGROUND_TOD_HOUR / PLAYGROUND_WEATHER) — the swing would
        // overwrite their sun with the fixed golden-hour one.
        if (!day_weather_enabled_) {
            env_.sun_dir = glm::normalize(scene_sun_dir_ + glm::vec3(0.25f, -0.15f, 0.2f));
            env_dirty_ = true;
        }
    }
    if (frame_ == 185 && !smoke_sun_forced_) {
        // ...and back OFF: exercises the volumetric teardown path too, and
        // the end-of-run screenshot stays free of the half-res dusk wash.
        gs->volumetric_fog = false;
        gs->god_rays = false;
    }
    if (frame_ == 195 && !smoke_sun_forced_ && !day_weather_enabled_) {
        env_.sun_dir = scene_sun_dir_;
        env_dirty_ = true;
    }
    if (frame_ == 200 && !smoke_sun_shadows_off_) {
        gs->shadow_mode = ShadowMode::PCF;
    }
    if (frame_ == 205 && !smoke_sun_forced_ && !day_weather_enabled_) {
        // "Cloud passes over": dim the direct sun so the local lights (orb
        // pools + flashlight beam) visibly dominate in the screenshot and
        // their shadows are unmistakable. sun_intensity is a shading
        // multiplier only — the IBL sky snapshot is untouched (no refresh).
        gs->sun_intensity = 2.4f; // 80% of the canonical 3.0
    }

    // Late enough that all the settings flips above are visible in the image.
    // PLAYGROUND_RT_LATE=<mode> flips ray tracing on mid-run, which is what
    // the dev overlay's Graphics checkboxes do. It exists because that path
    // was broken while PLAYGROUND_RT (set before any model loads) worked
    // perfectly: the toggle had no acceleration structure, and no CPU geometry
    // left to build one from. Anything that regresses it must fail here.
    if (std::getenv("PLAYGROUND_RT_LATE") != nullptr && frame_ == 5) {
        auto late_edit = edit_graphics_settings();
        auto& late_gs = *late_edit;
        const std::string late_mode = std::getenv("PLAYGROUND_RT_LATE");
        late_gs.rt_shadows = (late_mode == "shadows" || late_mode == "both");
        late_gs.rt_reflections = (late_mode == "reflections" || late_mode == "both");
        request_graphics_settings(late_gs);
        BRACKEN_LOG_INFO("playground", "LATE RT toggle -> shadows={} reflections={}",
                         late_gs.rt_shadows, late_gs.rt_reflections);
    }
    if (frame_ == std::max(1, smoke_frames_ - 20)) {
        request_screenshot(SMOKE_SCREENSHOT_PATH);
    }

    if (frame_ >= smoke_frames_) {
        // render_stats() is nullopt when collection is off; the smoke driver
        // turns it on at init, so a miss here is a real regression, not a
        // reason to print zeros.
        const std::optional<bracken::RenderStats> stats_opt = render_stats();
        const bracken::RenderStats stats = stats_opt.value_or(bracken::RenderStats{});
        if (!stats_opt) {
            std::cout << "[smoke] WARNING: render stats collection was off\n";
        }
        // Application::fps() is a once-per-second windowed counter and reads 0
        // when the whole run finishes in under a second; compute the average
        // over the run instead so the smoke line is meaningful at any speed.
        const double elapsed_s = static_cast<double>(SDL_GetTicksNS() - smoke_start_ns_) /
                                 static_cast<double>(SDL_NS_PER_SECOND);
        const double avg_fps = elapsed_s > 0.0 ? frame_ / elapsed_s : 0.0;
        // IBL refresh proof: init build (1) + the two sun swings above = 3
        // completed chains, each amortized (IBL::UNITS_PER_FRAME passes/frame).
        const auto* ibl = scene_renderer().ibl();
        const uint64_t ibl_refreshes = ibl ? ibl->refresh_count() : 0;
        // Local-shadow proof: total tiles re-rendered over the run (bounded
        // by 6/frame) + how many lights were shadowed on the last frame.
        const auto* ls = scene_renderer().local_shadows();
        const uint64_t local_tiles = ls ? ls->selector().tiles_rendered_total() : 0;
        const int ls_points = ls ? ls->selector().shadowed_point_lights() : 0;
        const int ls_spots = ls ? ls->selector().shadowed_spot_lights() : 0;
        // Auto-exposure proof: the adapted EV from the compute passes (via
        // the fenced debug readback; a frame or two stale, fine for logging).
        const auto* auto_exposure = scene_renderer().auto_exposure();
        const auto& ev_state = auto_exposure ? auto_exposure->latest_state()
                                             : std::optional<bracken::render::AutoExposure::ExposureState>{};
        std::cout << std::format(
            "[smoke] OK frames={} fps={:.0f} draw_calls={} triangles={} entities={} orbs_collected={} "
            "village={} ibl_refreshes={} local_tiles={} local_lights={}p+{}s auto_ev={} driver={}\n",
            frame_, avg_fps, stats.draw_calls, stats.triangle_count, stats.entities_rendered, orbs_collected_,
            has_village_ ? 1 : 0, ibl_refreshes, local_tiles, ls_points, ls_spots,
            ev_state ? std::format("{:.2f}", ev_state->current_ev) : "n/a", gpu_driver_name());
        {   // The settings the run ACTUALLY rendered with. mean_luma is not a
            // reproducible measurement without this: these knobs used to be
            // inherited from a persisted file, and a drifted shadow_mode alone
            // moved the reported baseline by 3 luma points with no code change.
            // `persist=0` says the run neither read nor wrote settings.cfg.
            const auto& g = graphics_settings();
            std::cout << std::format(
                "[smoke] settings persist={} shadow_mode={} ao_mode={} aa_mode={} draw_distance={} "
                "rt_shadows={} rt_reflections={} rt_local_shadows={} rt_range={} sun={:.2f} "
                "local_shadows={}\n",
                settings_persistence() ? 1 : 0, static_cast<int>(g.shadow_mode), static_cast<int>(g.ao_mode),
                static_cast<int>(g.aa_mode), static_cast<int>(g.draw_distance), g.rt_shadows ? 1 : 0,
                g.rt_reflections ? 1 : 0, static_cast<int>(g.rt_local_shadows),
                static_cast<int>(g.rt_max_instance_range), g.sun_intensity,
                g.local_shadows ? 1 : 0);
        }
        {   // Ray-tracing evidence: whether the passes actually ran, not just
            // whether the frame survived them.
            const auto& rt = scene_renderer().raytracing_stats();
            std::cout << std::format(
                "[smoke] rt shadows_traced={} reflections_traced={} local_shadows_traced={} "
                "local_di_traced={} "
                "shadow_rays={} reflection_rays={} local_shadow_rays={} accel={:.1f}MB "
                // R6: emissive MODEL triangles that cleared the power threshold,
                // the scene-space records they flattened into, and whether those
                // records actually fed the DI candidate draw (Full tier only —
                // see the tier note in docs/rt_lighting_plan.md §5.7).
                "emissive_tris={} emissive_records={} emissive_dropped={} emissive_di={} "
                // R8: what the per-froxel emissive bin produced, and whether the
                // trace drew through it (false = R6's scene-wide alias draw).
                "emissive_local={} emissive_froxels={} emissive_entries={} "
                "emissive_froxel_dropped={}\n",
                rt.shadows_traced, rt.reflections_traced, rt.local_shadows_traced,
                rt.local_di_traced, rt.shadow_rays,
                rt.reflection_rays, rt.local_shadow_rays,
                static_cast<double>(rt.acceleration_bytes) / (1024.0 * 1024.0),
                rt.emissive_triangles, rt.emissive_records, rt.emissive_records_dropped,
                rt.emissive_di_sampled, rt.emissive_froxel_local, rt.emissive_froxels_filled,
                rt.emissive_froxel_entries, rt.emissive_froxel_dropped);
            // The structure, itemized. `instances` is post-range-filter, so it
            // is what the rays actually see; `tlas_builds` should read 1 while
            // the instance SET is stable — a moving instance costs a
            // `tlas_refits` tick instead — and is the number a finite
            // rt_max_instance_range trades traversal against.
            std::cout << std::format(
                "[smoke] rt accel instances={} tlas_builds={} tlas_refits={} blas={:.1f}MB "
                "attrs={:.1f}MB tlas={:.3f}MB\n",
                rt.instances, rt.tlas_builds, rt.tlas_refits,
                static_cast<double>(rt.blas_bytes) / (1024.0 * 1024.0),
                static_cast<double>(rt.attribute_bytes) / (1024.0 * 1024.0),
                static_cast<double>(rt.tlas_bytes) / (1024.0 * 1024.0));
        }
        {   // Probe-field GI. Printed only when it is on, because "gi off" on
            // every default run would be one more line of nothing.
            //
            // Two lines, two claims. The FIELD line is R1a's: the clipmap was
            // allocated and scrolled, so `invalidated` must be non-zero on the
            // frame the volume is placed and must settle to a plane's worth (or
            // zero) as the camera moves. The UPDATE line is R1b's: rays were
            // actually cast, probes were actually blended, and the classifier
            // found the number of buried probes a village full of solid
            // buildings ought to contain. `relocated`/`inside` come off a
            // fenced, polled download and therefore describe a recent frame
            // rather than the last one — see render::gi::GiStats.
            //
            // Neither line says anything about the IMAGE, and that is the
            // point: nothing samples the field until R1c, so mean_luma below
            // must be identical with GI on and off.
            const auto& gi = scene_renderer().gi_stats();
            if (gi.active) {
                // The Hi-Z pyramid (R4a) is built only for GiMode::ProbesScreen
                // and read by nothing yet, so the field is APPENDED when it
                // exists rather than printed as `hzb=off` otherwise: a
                // `PLAYGROUND_GI=probes` run must stay byte-identical, line for
                // line, against the pre-R4a baseline.
                std::string hzb_field;
                if (const auto* hzb = scene_renderer().hzb(); hzb != nullptr && hzb->is_ready()) {
                    hzb_field = std::format(" hzb={}x{} mips={}", hzb->width(), hzb->height(), hzb->mip_count());
                }
                std::cout << std::format(
                    "[smoke] gi mode={} quality={} cascades={} probes={} invalidated={} total={} "
                    "atlas={:.1f}MB spacing={:.0f}..{:.0f}{}\n",
                    static_cast<int>(graphics_settings().gi_mode),
                    static_cast<int>(graphics_settings().gi_quality), gi.cascades, gi.probes,
                    gi.probes_invalidated, gi.probes_invalidated_total,
                    static_cast<double>(gi.atlas_bytes) / (1024.0 * 1024.0),
                    gi.per_cascade.front().spacing, gi.per_cascade.back().spacing, hzb_field);
                std::cout << std::format(
                    "[smoke] gi update rays={} probes={} c0={} c1={} c2={} relocated={} inside={} "
                    "rays_total={} probes_total={}\n",
                    gi.rays_this_frame, gi.probes_updated, gi.per_cascade[0].probes_updated,
                    gi.per_cascade[1].probes_updated, gi.per_cascade[2].probes_updated,
                    gi.probes_relocated, gi.probes_inside_geometry, gi.rays_total,
                    gi.probes_updated_total);
                // R4b's screen-probe gather, APPENDED as its own line rather
                // than folded into either above, for the same reason the hzb
                // field is appended: a `PLAYGROUND_GI=probes` run must stay
                // byte-identical line for line against the pre-R4b baseline.
                //
                // `hits` and `world` are the design's economic claim made
                // measurable: most rays are expected to resolve on screen, and
                // `world` is what it costs when they do not. Both are LAGGED —
                // fenced, polled counters, never waited on.
                if (const auto* sp = scene_renderer().screen_probes();
                    sp != nullptr && sp->stats().active) {
                    const auto& s = sp->stats();
                    std::cout << std::format(
                        "[smoke] gi screen sp={}x{} probes={} hits={} world={} world_hits={} "
                        "fb={} dropped={} {:.2f}MB\n",
                        s.grid.x, s.grid.y, s.probes, s.screen_hits, s.world_rays, s.world_hits,
                        s.fallbacks, s.dropped, static_cast<double>(s.bytes) / (1024.0 * 1024.0));
                }
            }
        }
        // Tier-3 evidence: that the controller left the ground, that the nav
        // grid built, and that the villager actually pathed somewhere.
        std::cout << std::format(
            "[smoke] character grounded={} air={} max_y={:.0f} | nav walkable={} | "
            "wanderer waypoints={} moved={:.0f}\n",
            smoke_grounded_frames_, smoke_air_frames_, smoke_max_height_, nav_.walkable_count(),
            wanderer_.path.size(), glm::distance(wanderer_.position, wanderer_start_));
        // Phase timings: CPU-record only (SDL3 has no GPU timestamp API).
        // present_wait is the CPU/GPU-bound signal and is a subset of render.
        std::cout << std::format("[smoke] timing sim={:.2f}ms render={:.2f}ms fence={:.2f}ms present_wait={:.2f}ms\n",
                                 sim_ms(), render_ms(), fence_ms(), present_wait_ms());
        // What the renderer THREW AWAY. CPU-side and driver-independent: a
        // regression in any drop path (unknown handles, skeleton-less skinned
        // draws, palette budget, unresolved effect models, light caps, a lost
        // camera) moves a number here.
        const auto& diag = frame_diagnostics();
        std::cout << std::format("[smoke] diagnostics unknown_model={} no_skeleton={} palettes_over_budget={} "
                                 "effect_models_unresolved={} lights_dropped={} cluster_lights_dropped={} "
                                 "camera_missing={}\n",
                                 diag.draws_dropped_unknown_model, diag.skinned_dropped_no_skeleton,
                                 diag.palettes_over_budget, diag.effect_models_unresolved, diag.lights_dropped,
                                 diag.cluster_lights_dropped, diag.camera_missing ? 1 : 0);
        quit();
    }
}

namespace {
constexpr const char* k_quicksave_path = "playground_quicksave.sav";
/** Bump when the meaning of a saved field changes. Fields that are merely
 *  ADDED need no bump — a reader takes its default when they are absent. */
constexpr uint32_t k_playground_save_version = 1;
} // namespace

void PlaygroundGame::quicksave() {
    bracken::save::SaveDocument doc;
    doc.set_game_version(k_playground_save_version);

    doc.write("player/position", player_pos_);
    doc.write("player/facing", facing_.current);
    doc.write("progress/orbs_collected", orbs_collected_);
    doc.write("world/hour", day_weather_.hour);
    doc.write("world/weather_mode", static_cast<int32_t>(day_weather_.mode));
    doc.write("world/animate_sun", day_weather_.animate);

    // Orb positions: they respawn to random spots, so a reload that put them
    // back at the defaults would visibly differ from what the player saw.
    doc.write_count("orbs", orbs_.size());
    for (std::size_t i = 0; i < orbs_.size(); ++i) {
        doc.write(bracken::save::index_path("orbs", i, "position"), orbs_[i].pos);
    }

    if (auto r = doc.save(k_quicksave_path); !r) {
        BRACKEN_LOG_WARN("playground", "quicksave failed: {}",
                         bracken::save::to_string(r.error()));
    }
}

void PlaygroundGame::quickload() {
    auto loaded = bracken::save::SaveDocument::load(k_quicksave_path);
    if (!loaded) {
        // A missing quicksave is the normal case on a fresh run, not an error.
        if (loaded.error() != bracken::save::SaveError::FileNotFound) {
            BRACKEN_LOG_WARN("playground", "quickload failed: {}",
                             bracken::save::to_string(loaded.error()));
        }
        return;
    }
    const bracken::save::SaveDocument& doc = loaded.value();

    player_pos_ = doc.read_or("player/position", player_pos_);
    facing_.current = doc.read_or("player/facing", facing_.current);
    orbs_collected_ = doc.read_or<int32_t>("progress/orbs_collected", orbs_collected_);
    day_weather_.hour = doc.read_or("world/hour", day_weather_.hour);
    day_weather_.mode =
        static_cast<WeatherMode>(doc.read_or<int32_t>("world/weather_mode",
                                                      static_cast<int32_t>(day_weather_.mode)));
    day_weather_.animate = doc.read_or("world/animate_sun", day_weather_.animate);

    const std::size_t n = std::min(doc.read_count("orbs"), orbs_.size());
    for (std::size_t i = 0; i < n; ++i) {
        orbs_[i].pos = doc.read_or(bracken::save::index_path("orbs", i, "position"), orbs_[i].pos);
    }

    // Snap the weather to its restored targets rather than sliding there from
    // whatever the live state happened to be.
    day_weather_.snap();
    BRACKEN_LOG_INFO("playground", "Quickload: {} orbs, hour {:.1f}", orbs_collected_,
                     day_weather_.hour);
}

void PlaygroundGame::update_player(float dt) {
    float target_speed = glm::length(move_dir_) > 0.01f ? (running_input_ ? RUN_SPEED : WALK_SPEED) : 0.0f;
    player_speed_ += (target_speed - player_speed_) * std::min(1.0f, dt * 8.0f);

    if (player_speed_ > 1.0f && glm::length(move_dir_) > 0.01f) {
        facing_.smooth_toward(std::atan2(-move_dir_.x, -move_dir_.z), dt, 10.0f);
    } else {
        facing_.decay_turn_rate();
    }

    player_vel_ = move_dir_ * player_speed_;

    // Footfalls by distance travelled, so the rate tracks the animation
    // instead of drifting against it as the player accelerates.
    const bool in_water = river_active_ && player_pos_.y < RIVER_WATER_LEVEL;
    if (player_speed_ > 5.0f) {
        step_distance_ += player_speed_ * dt;
        constexpr float k_stride = 95.0f; // world units between footfalls
        if (step_distance_ >= k_stride) {
            step_distance_ -= k_stride;
            audio().play_sound(in_water ? sfx_.footstep_water : sfx_.footstep_grass,
                               {
                                   // Quieter when walking than running: velocity
                                   // is the cheapest cue that a step has weight.
                                   .gain = 0.35f + 0.45f * std::min(player_speed_ / RUN_SPEED, 1.0f),
                                   // Real per-step scatter from a stream seeded
                                   // to a constant: varied every step, and
                                   // still identical run to run. Deriving it
                                   // from player_pos_.x instead was
                                   // reproducible and WRONG — x is constant
                                   // while running along Z, so every footstep
                                   // came out at the same pitch, which is the
                                   // artifact the scatter exists to prevent.
                                   .pitch = step_rng_.next_range(0.92f, 1.08f),
                                   .space = bracken::audio::SoundSpace::World,
                                   .position = player_pos_,
                                   .min_distance = 250.0f,
                                   .max_distance = 3000.0f,
                               });
        }
    } else {
        // Reset so the first step after standing still lands immediately.
        step_distance_ = 90.0f; // just under a stride: next step fires soon
    }

    if (in_water && !was_in_water_) {
        audio().play_sound(sfx_.splash, {
                                            .gain = 0.9f,
                                            .space = bracken::audio::SoundSpace::World,
                                            .position = player_pos_,
                                            .min_distance = 300.0f,
                                            .max_distance = 5000.0f,
                                        });
    }
    was_in_water_ = in_water;
    // Movement, world collision and ground placement are all the character
    // controller's job now (below). What used to live here — a manual
    // `player_pos_ += vel*dt`, an XZ-only depenetration loop, a village
    // rect push-out fallback and a terrain-height snap — is exactly the
    // hand-rolled controller every game rewrites, and it could not express
    // slopes, steps or air time.

    // Character controller instead of snapping to the terrain height. This is
    // what gives the demo slopes, step-up, air time and a real jump — the old
    // one-liner pinned the character to the ground and could not express any
    // of it.
    bracken::physics::CharacterInput cin;
    cin.desired_velocity = player_vel_;
    cin.jump = jump_requested_;
    cin.crouch = crouch_held_;
    jump_requested_ = false;

    // The TERRAIN body is excluded from the controller's depenetration: a
    // capsule resting on a heightfield always overlaps it slightly, and
    // pushing out of that every frame walks the character up into the sky.
    // Ground is resolved by the downward ray instead, which is the same split
    // the old hand-rolled code used.
    character_.update(physics_, cin, dt, /*self=*/character_body_,
                      /*ground=*/terrain_body_);
    player_pos_ = character_.state().position;
    // World bounds are a game rule, not a collision one.
    player_pos_.x = std::clamp(player_pos_.x, WORLD_MARGIN, WORLD - WORLD_MARGIN);
    player_pos_.z = std::clamp(player_pos_.z, WORLD_MARGIN, WORLD - WORLD_MARGIN);
    character_.set_position_preserving_velocity(player_pos_);
    // Animation reacts to what actually happened, not what was asked for.
    player_speed_ = glm::length(
        glm::vec2(character_.state().velocity.x, character_.state().velocity.z));

    player_transform_ = glm::rotate(glm::translate(glm::mat4(1.0f), player_pos_), facing_.current, glm::vec3(0, 1, 0));

    attack_timer_ = std::max(0.0f, attack_timer_ - dt);
    anim_sm_.set_float(param_speed_, player_speed_);
    anim_sm_.set_bool(param_attacking_, attack_timer_ > 0.0f);
    anim_sm_.update(anim_player_);

    Model* model = models().get_model(player_model_);
    anim_player_.update(model->skeleton, model->animations, dt);

    // Procedural layers on top of the sampled animation, as the engine intends:
    // foot IK pins feet to the terrain, body lean follows speed and turn rate.
    const auto& ik = model->foot_ik;
    if (ik.valid) {
        auto world = anim_player_.world_transforms();
        auto foot_offset = [&](int joint) {
            glm::vec3 foot = glm::vec3(player_transform_ * world[joint][3]);
            return get_terrain_height(foot.x, foot.z) - player_pos_.y;
        };
        foot_smoother_.update(foot_offset(ik.left_foot), foot_offset(ik.right_foot), dt);
        anim::apply_foot_ik(anim_player_.mutable_bone_matrices(), anim_player_.mutable_world_transforms(),
                            model->skeleton, ik, 1.0f, foot_smoother_.smoothed_left_offset,
                            foot_smoother_.smoothed_right_offset);
    }
    // Wading: the river is shallow enough to cross; moving through it pushes
    // ripple impulses into the water sim at the player's feet. The
    // WaterRenderer is render-side (its impulse queue feeds the wave sim
    // inside render_frame), so sim code only QUEUES ripples here —
    // on_main_frame forwards them at the fence, before this frame renders.
    if (river_active_) {
        wade_ripple_timer_ -= dt;
        if (player_pos_.y < RIVER_WATER_LEVEL - 2.0f && player_speed_ > 30.0f &&
            wade_ripple_timer_ <= 0.0f) {
            wade_ripple_timer_ = 0.14f;
            const glm::vec3 at = player_pos_ + forward_from_yaw(facing_.current) * 18.0f;
            pending_ripples_.push_back({at, 20.0f, 0.28f * std::min(player_speed_ / RUN_SPEED, 1.0f)});
        }
        // Ambient life: an occasional soft ring somewhere on the nearby
        // stretch (fish rising, drips from the far-bank pines). Sparse and
        // gentle — the river should read calm between events.
        ambient_ripple_timer_ -= dt;
        if (ambient_ripple_timer_ <= 0.0f) {
            ambient_ripple_timer_ = 1.5f + rand01() * 2.5f;
            const float wz = VILLAGE_CZ + (rand01() - 0.5f) * 1800.0f;
            const float wx = river_center_x(wz) + (rand01() - 0.5f) * 1.4f * RIVER_HALF_WIDTH;
            pending_ripples_.push_back({{wx, RIVER_WATER_LEVEL, wz}, 12.0f + rand01() * 18.0f,
                                        0.05f + rand01() * 0.10f});
        }
    }

    float forward_lean = std::min(player_speed_ * 0.0004f, 0.15f);
    float lateral_lean = std::clamp(-facing_.turn_rate * 0.04f, -0.12f, 0.12f);
    anim::apply_body_lean(anim_player_.mutable_bone_matrices(), anim_player_.mutable_world_transforms(),
                          model->skeleton, ik.spine, forward_lean, lateral_lean);

    // LAST, after every procedural layer. The capsules are supposed to be where
    // the character actually IS, and foot IK and body lean both move joints
    // after the clip is sampled — deriving them before this line would put the
    // shadow back a step behind the pose, which is a subtler version of exactly
    // the bug this feature fixes.
    rebuild_shadow_capsules(*model);
}

// ---------------------------------------------------------------------------
// Ray-traced shadow proxies (docs/rt_lighting_plan.md §5.8)
//
// A skinned mesh traces its BIND POSE, and the traced sun mask REPLACES the
// cascade shadow, so before this the character dragged a rigid statue's
// silhouette around the village. The engine now skips skinned instances in its
// any-hit walks and tests these capsules instead.
//
// THE HOST DERIVES THEM, and that division is deliberate: the engine cannot
// know which bones of an arbitrary rig are a torso and which are a sleeve, and
// name-guessing is the sort of content assumption the engine/game split exists
// to keep out. What the host owes is a dozen segments that follow the live
// skeleton; what it gets back is an animated shadow.
// ---------------------------------------------------------------------------
namespace {

/// Radii in MODEL units, which are world units here — the character transform
/// is a translate+rotate with no scale.
///
/// HAND-TUNED against the box geometry in character.cpp, not computed: the mesh
/// is a stack of boxes and a capsule cannot match a box cross section (the
/// torso is 30 wide and 18 deep), so every one of these is a judgement about
/// which silhouette reads better in shadow. They deliberately sit nearer the
/// box's SMALLER half-extent than its larger: an over-fat proxy is the failure
/// you notice, because it turns a figure into a snowman.
constexpr float CAPSULE_R_PELVIS = 12.0f;   // hips box 14 x 8, spine box 15 x 9
constexpr float CAPSULE_R_CHEST = 11.0f;    // spine box 15 x 9
constexpr float CAPSULE_R_HEAD = 9.0f;      // head box 9 x 9
constexpr float CAPSULE_R_UPPER_LEG = 6.0f; // 5.5 half-extent
constexpr float CAPSULE_R_LOWER_LEG = 5.5f; // 5.0 half-extent
constexpr float CAPSULE_R_FOOT = 5.0f;      // 5.0 half-extent
constexpr float CAPSULE_R_ARM = 5.0f;       // 4.5 half-extent

/// The head and the arms have no CHILD joint to run a segment to, so their far
/// end is an offset in the bone's own frame — which is what keeps them rotating
/// with the bone instead of pointing stubbornly up the world axis.
constexpr glm::vec3 CAPSULE_HEAD_TIP{0.0f, 22.0f, 0.0f};  // head box top, y 182
constexpr glm::vec3 CAPSULE_ARM_TIP{0.0f, -50.0f, 0.0f};  // arm box runs 153 -> 103
constexpr glm::vec3 CAPSULE_FOOT_TIP{0.0f, 0.0f, -9.0f};  // foot box reaches forward

} // namespace

void PlaygroundGame::resolve_capsule_bones(const bracken::Model& model) {
    using bracken::animation::find_joint;
    const auto& sk = model.skeleton;
    // The playground's rig (character.cpp) names these to match the engine's
    // default FootIKBoneNames, which is also what docs/skinned_mesh_format.md
    // documents as the convention. A rig that names them differently resolves
    // to invalid indices and takes the whole-body fallback below rather than
    // silently producing capsules parked at the origin.
    capsule_bones_.hips = find_joint(sk, "Hips");
    capsule_bones_.spine = find_joint(sk, "Spine");
    capsule_bones_.head = find_joint(sk, "Head");
    capsule_bones_.left_upper_leg = find_joint(sk, "LeftUpperLeg");
    capsule_bones_.left_lower_leg = find_joint(sk, "LeftLowerLeg");
    capsule_bones_.left_foot = find_joint(sk, "LeftFoot");
    capsule_bones_.right_upper_leg = find_joint(sk, "RightUpperLeg");
    capsule_bones_.right_lower_leg = find_joint(sk, "RightLowerLeg");
    capsule_bones_.right_foot = find_joint(sk, "RightFoot");
    capsule_bones_.left_arm = find_joint(sk, "LeftArm");
    capsule_bones_.right_arm = find_joint(sk, "RightArm");

    const bracken::animation::JointIndex required[] = {
        capsule_bones_.hips,            capsule_bones_.spine,           capsule_bones_.head,
        capsule_bones_.left_upper_leg,  capsule_bones_.left_lower_leg,  capsule_bones_.left_foot,
        capsule_bones_.right_upper_leg, capsule_bones_.right_lower_leg, capsule_bones_.right_foot,
        capsule_bones_.left_arm,        capsule_bones_.right_arm};
    capsule_bones_.valid = true;
    for (const auto& j : required) {
        capsule_bones_.valid = capsule_bones_.valid && j.valid();
    }
    if (capsule_bones_.valid) {
        BRACKEN_LOG_INFO("playground", "RT shadow capsules: resolved 11 bones on the character rig");
    }
}

void PlaygroundGame::rebuild_shadow_capsules(const bracken::Model& model) {
    using bracken::animation::attachment_position;
    using bracken::render::rt::ShadowCapsule;

    shadow_capsules_.clear();

    if (!capsule_bones_.valid) {
        // FALLBACK, and a visibly worse one rather than nothing at all: one
        // capsule down the model's own bounding box. The character still casts
        // a shadow that moves with it — it just does not articulate. Better
        // than the bind-pose statue, worse than the real thing, and it says so
        // once instead of failing quietly.
        if (!capsule_fallback_logged_) {
            capsule_fallback_logged_ = true;
            BRACKEN_LOG_WARN("playground",
                             "RT shadow capsules: the character rig does not use the expected bone "
                             "names (Hips/Spine/Head/...); falling back to ONE whole-body capsule "
                             "from the model bounds. Its shadow will move but not articulate");
        }
        const glm::vec3 lo = model.bounds.min;
        const glm::vec3 hi = model.bounds.max;
        const glm::vec3 mid{(lo.x + hi.x) * 0.5f, 0.0f, (lo.z + hi.z) * 0.5f};
        // Radius from the SMALLER horizontal half-extent, for the reason the
        // per-bone radii are: an over-fat proxy is the visible failure.
        const float radius = 0.5f * std::min(hi.x - lo.x, hi.z - lo.z);
        const glm::vec3 p0 = glm::vec3(player_transform_ *
                                       glm::vec4(mid + glm::vec3(0.0f, lo.y + radius, 0.0f), 1.0f));
        const glm::vec3 p1 = glm::vec3(player_transform_ *
                                       glm::vec4(mid + glm::vec3(0.0f, hi.y - radius, 0.0f), 1.0f));
        shadow_capsules_.push_back(ShadowCapsule{p0, radius, p1, 0.0f});
        return;
    }

    // Every point below comes from the joint's WORLD transform, not its
    // skinning matrix — attachment_position is the engine API that gets that
    // distinction right (see animation/attachment.hpp; using the skinning
    // matrix here would park the whole figure near the origin at a strange
    // angle, which is the classic symptom).
    const auto at = [&](bracken::animation::JointIndex j, const glm::vec3& local = glm::vec3(0.0f)) {
        return attachment_position(player_transform_, anim_player_, j, local);
    };
    const auto add = [&](const glm::vec3& a, const glm::vec3& b, float r) {
        shadow_capsules_.push_back(ShadowCapsule{a, r, b, 0.0f});
    };

    // 11 capsules: torso in two, head, four leg segments, two feet, two arms.
    // The arms and head run to a BONE-LOCAL offset because they have no child
    // joint; everything else runs joint to joint, so it follows the rig with no
    // authored lengths to keep in sync.
    add(at(capsule_bones_.hips), at(capsule_bones_.spine), CAPSULE_R_PELVIS);
    add(at(capsule_bones_.spine), at(capsule_bones_.head), CAPSULE_R_CHEST);
    add(at(capsule_bones_.head), at(capsule_bones_.head, CAPSULE_HEAD_TIP), CAPSULE_R_HEAD);

    add(at(capsule_bones_.left_upper_leg), at(capsule_bones_.left_lower_leg), CAPSULE_R_UPPER_LEG);
    add(at(capsule_bones_.left_lower_leg), at(capsule_bones_.left_foot), CAPSULE_R_LOWER_LEG);
    add(at(capsule_bones_.left_foot), at(capsule_bones_.left_foot, CAPSULE_FOOT_TIP), CAPSULE_R_FOOT);

    add(at(capsule_bones_.right_upper_leg), at(capsule_bones_.right_lower_leg), CAPSULE_R_UPPER_LEG);
    add(at(capsule_bones_.right_lower_leg), at(capsule_bones_.right_foot), CAPSULE_R_LOWER_LEG);
    add(at(capsule_bones_.right_foot), at(capsule_bones_.right_foot, CAPSULE_FOOT_TIP), CAPSULE_R_FOOT);

    add(at(capsule_bones_.left_arm), at(capsule_bones_.left_arm, CAPSULE_ARM_TIP), CAPSULE_R_ARM);
    add(at(capsule_bones_.right_arm), at(capsule_bones_.right_arm, CAPSULE_ARM_TIP), CAPSULE_R_ARM);
}

// AIM the camera. The engine owns the rig and drives it in its sim-stage
// tail (Application::set_camera_driven_by_engine): no set_screen_size, no
// update(dt), no set_camera into the packet, no set_camera_forward here.
// --engine-camera=off flips that off and does all four by hand, which is
// what keeps the escape hatch honest.
void PlaygroundGame::update_camera(SimFrame& sim) {
    auto& cam = camera(); // sim-stage tier under run()
    if (smoke_frames_ <= 0) {
        const auto& in = sim.input();
        cam.set_yaw(in.camera_yaw);
        cam.set_pitch(in.camera_pitch);
        cam.adjust_zoom(in.camera_zoom_delta);
    }
    if (smoke_cam_forced_) {
        auto cfg = cam.get_config();
        cfg.distance = smoke_cam_dist_;
        // The default zoom clamp (600 units) is gameplay tuning; art shots
        // legitimately ask for anything from macro close-ups to aerial surveys.
        cfg.zoom_min = std::min(cfg.zoom_min, smoke_cam_dist_);
        cfg.zoom_max = std::max(cfg.zoom_max, smoke_cam_dist_);
        cfg.height_offset = 0.0f;
        cam.set_config(cfg);
        cam.set_yaw(smoke_cam_yaw_);
        cam.set_pitch(smoke_cam_pitch_);
        cam.set_target(smoke_cam_target_);
        cam.set_target_velocity({0.0f, 0.0f, 0.0f});
        if (!smoke_cam_settled_) {
            // The rig smooths distance/target through damped springs sized
            // for gameplay; a pinned art camera must land on its framing
            // immediately (and identically on every run length), so settle
            // the springs with a burst of fixed steps once.
            for (int i = 0; i < 120; ++i) {
                cam.update(0.05f);
            }
            smoke_cam_settled_ = true;
        }
    } else {
        cam.set_target(player_pos_ + glm::vec3(0, 110, 0));
        cam.set_target_velocity(player_vel_);
    }
    if (!engine_camera_) {
        cam.set_screen_size(sim.screen_width(), sim.screen_height());
        cam.update(sim.dt());
        sim.set_camera_forward({cam.get_forward().x, cam.get_forward().z});
    }
}

void PlaygroundGame::update_orbs(float dt) {
    (void)dt;
    for (auto& orb : orbs_) {
        float d = glm::distance(glm::vec2(orb.pos.x, orb.pos.z), glm::vec2(player_pos_.x, player_pos_.z));
        if (d < ORB_PICKUP_RADIUS) {
            ++orbs_collected_;
            audio().play_sound(sfx_.orb_pickup, {
                                                    .gain = 0.7f,
                                                    .space = bracken::audio::SoundSpace::World,
                                                    .position = orb.pos,
                                                    .min_distance = 400.0f,
                                                    .max_distance = 4000.0f,
                                                });
            pending_pickup_bursts_.push_back(orb.pos);
            camera().add_shake(systems::ShakeType::Impact, 4.0f, 0.25f);
            // Respawn at another spot (not the one it was just taken from).
            glm::vec3 next = orb.pos;
            while (glm::distance(next, orb.pos) < 1.0f) {
                next = orb_spots_[static_cast<size_t>(rand01() * 0.999f * orb_spots_.size())];
            }
            orb.pos = next + glm::vec3(0, ORB_HOVER, 0);
            orb.bob_phase = rand01() * 6.28f;
        }
    }
}

void PlaygroundGame::update_wanderer(float dt) {
    if (!nav_.built()) {
        return;
    }

    wanderer_.repath_timer -= dt;

    const bool needs_path = wanderer_.path.empty() ||
                            wanderer_.waypoint >= wanderer_.path.size() ||
                            wanderer_.repath_timer <= 0.0f;
    if (needs_path) {
        // Head for the player. Re-pathing on a timer rather than every frame:
        // A* over a 100x100 grid is cheap but not free, and a path that is one
        // second stale is invisible at walking speed.
        wanderer_.repath_timer = 1.5f;
        if (nav_.find_path(wanderer_.position, player_pos_, wanderer_.path)) {
            wanderer_.waypoint = 1; // [0] is where we already are
        } else {
            wanderer_.path.clear();
        }
    }

    if (wanderer_.waypoint >= wanderer_.path.size()) {
        return;
    }

    constexpr float k_speed = 190.0f;
    const glm::vec3 target = wanderer_.path[wanderer_.waypoint];
    glm::vec3 to_target = target - wanderer_.position;
    to_target.y = 0.0f;
    const float distance = glm::length(to_target);

    // Advance the waypoint on arrival. The tolerance is generous because the
    // string-pulled path's waypoints are corners, not precise targets.
    if (distance < 60.0f) {
        ++wanderer_.waypoint;
        return;
    }

    const glm::vec3 step = (to_target / distance) * k_speed * dt;
    wanderer_.position += step;
    wanderer_.position.y = get_terrain_height(wanderer_.position.x, wanderer_.position.z);
    wanderer_.facing = std::atan2(-step.x, -step.z);
}

void PlaygroundGame::update_chimneys(float dt) {
    if (chimney_tops_.empty()) {
        return;
    }
    smoke_spawn_timer_ -= dt;
    if (smoke_spawn_timer_ <= 0.0f) {
        smoke_spawn_timer_ += CHIMNEY_SMOKE_PERIOD;
        pending_chimney_smoke_ = true; // emitted into the packet by build_frame, below
    }
}

// Sim-side day/night + weather: advance the clock and the weather machine,
// stage the resulting environment (on-change only — while the system is
// paused and the weather settled it writes nothing, so the DevUI Environment
// tab stays live-editable), and emit the rain: the player-following streak
// volume, splash bursts at ray-cast impact points on terrain/props/roofs,
// and ripple impulses where drops land on the river.
void PlaygroundGame::update_day_weather(float dt) {
    if (!day_weather_enabled_) {
        return;
    }
    day_weather_.update(dt);
    if (day_weather_.dirty()) {
        day_weather_.apply(env_);
        env_dirty_ = true; // staged into the packet by build_frame
    }

    const float rain = day_weather_.rain();
    if (rain <= 0.02f) {
        rain_layer_timer_ = 0.0f;
        splash_accum_ = 0.0f;
        return;
    }

    // Streak layer. Retuning the shared definition's spawn_rate here is safe:
    // definitions are only read later in this same sim frame (the engine tail
    // updates the EffectSystem after on_simulate returns), and a live rate
    // change deliberately reaches the already-falling layers too.
    rain_layer_.emitters[0].spawn_rate = glm::mix(90.0f, 750.0f, rain);
    rain_layer_timer_ -= dt;
    if (rain_layer_timer_ <= 0.0f) {
        rain_layer_timer_ = RAIN_LAYER_PERIOD;
        pending_rain_spawn_ = true; // consumed by build_frame above the player
    }

    // Impact layer, decoupled from the streak particles (no one can pair a
    // specific streak with a specific splash at full rate): a budgeted rate
    // of impact points around the player. Water gets a wave-sim ripple, all
    // other hits get a splash burst where the ray landed — the same static
    // bodies the chrome ball bounces off, so crates, roofs and trunks splash.
    splash_accum_ += rain * SPLASH_RATE * dt;
    int splashes = static_cast<int>(splash_accum_);
    splash_accum_ -= static_cast<float>(splashes);
    splashes = std::min(splashes, SPLASH_MAX_PER_FRAME);
    int ripple_budget = RAIN_RIPPLE_BUDGET; // the water sim takes 8/frame total
    for (int i = 0; i < splashes; ++i) {
        const float ang = rand01() * glm::two_pi<float>();
        const float r = std::sqrt(rand01()) * SPLASH_RADIUS; // uniform over the disc
        const float wx = player_pos_.x + std::cos(ang) * r;
        const float wz = player_pos_.z + std::sin(ang) * r;
        // River first — the surface is a known plane, no ray needed.
        if (river_active_ && std::abs(wx - river_center_x(wz)) < RIVER_HALF_WIDTH &&
            heightmap_.get_height_world(wx, wz) < RIVER_WATER_LEVEL) {
            if (ripple_budget > 0) {
                --ripple_budget;
                pending_ripples_.push_back(
                    {{wx, RIVER_WATER_LEVEL, wz}, 5.0f + rand01() * 9.0f, 0.05f + 0.09f * rain});
            }
            continue;
        }
        glm::vec3 hit_pos{wx, get_terrain_height(wx, wz), wz};
        if (physics_ready_) {
            const auto hit =
                physics_.cast_ray({wx, SPLASH_RAY_TOP, wz}, {0.0f, -(SPLASH_RAY_TOP + 500.0f), 0.0f});
            if (hit.hit) {
                hit_pos = hit.position;
            }
        }
        pending_splashes_.push_back(hit_pos + glm::vec3(0.0f, 1.5f, 0.0f));
    }
}

// Sim-stage physics (the ONLY per-frame PhysicsWorld toucher): drive the
// pusher capsule to the character, land at most one kick per attack swing,
// step the world, then read the ball back into the render transform and the
// panel mirrors.
void PlaygroundGame::update_physics(float dt) {
    if (!physics_ready_) {
        return;
    }

    // The capsule shadows the heightmap-driven character (locomotion itself
    // stays non-physical); move_kinematic gives it the velocity that makes
    // Jolt's kinematic-vs-dynamic contact push the ball.
    physics_.move_kinematic(character_body_, player_pos_ + glm::vec3(0.0f, CHAR_CAPSULE_CENTER_Y, 0.0f),
                            glm::quat(1.0f, 0.0f, 0.0f, 0.0f), dt);

    // Kick: armed on the attack swing's rising edge (anim-state edge, not
    // per-frame), fired the first sim frame of the swing that finds the ball
    // in range and in front, disarmed either way when the swing ends.
    const bool attacking = attack_timer_ > 0.0f;
    if (attacking && !attack_was_active_) {
        kick_armed_ = true;
    }
    if (!attacking) {
        kick_armed_ = false;
    }
    attack_was_active_ = attacking;
    if (kick_armed_) {
        glm::vec3 fwd = camera().get_forward();
        fwd.y = 0.0f;
        fwd = glm::length(fwd) > 0.001f ? glm::normalize(fwd) : forward_from_yaw(facing_.current);
        glm::vec3 to_ball = physics_.position(ball_body_) - player_pos_;
        to_ball.y = 0.0f;
        const float d = glm::length(to_ball);
        if (d > 0.001f && d < KICK_RANGE + ball_radius_ && glm::dot(to_ball / d, fwd) > KICK_FACING_COS) {
            physics_.apply_impulse(ball_body_,
                                   fwd * KICK_IMPULSE_FLAT + glm::vec3(0.0f, KICK_IMPULSE_UP, 0.0f));
            kick_armed_ = false;
        }
    }

    physics_.update(dt);

    // The chrome mesh is origin-centred, so the body transform is the render
    // transform (rotation included — the mirror ball visibly rolls).
    chrome_transform_ = physics_.transform(ball_body_);
    ball_pos_ = physics_.position(ball_body_);
    ball_speed_ = glm::length(physics_.linear_velocity(ball_body_));
    ball_active_ = physics_.is_active(ball_body_);
}

// ---------------------------------------------------------------------------
// Frame description (the tail of on_simulate)
// ---------------------------------------------------------------------------

void PlaygroundGame::build_frame(SimFrame& sim) {
    // Describe the frame through the engine-owned FrameBuilder. Everything
    // below is copied or packet-allocated — nothing outlives this function on
    // the host side. The engine stamped the frame timing at kick and seals
    // the packet after its sim-stage tail; the host does neither.
    scene::FrameSink& fb = sim.frame();
    const auto& gs = graphics_settings();

    if (!engine_camera_) {
        fb.set_camera(camera().get_camera_state()); // --engine-camera=off only
    }
    // Stage the environment only when the game changed it (init, smoke-run
    // sun swings): a staged environment is adopted over the renderer's live
    // EnvironmentState, so staging on change — not every frame — keeps the
    // DevUI Environment tab's live edits working between host changes.
    // env_dirty_ is set at init/fence and cleared here (sim); the phases
    // never overlap, ordering rides the sim kick/join edges.
    // PLAYGROUND_SUN_SWEEP (--rt-bench only). A pure function of the frame
    // index at the pinned 1/60 timestep — never of wall clock — so two runs of
    // the same command produce the same sun on the same frame. Rotating the
    // direction TO the sun about world +Y keeps the elevation, and therefore
    // the intensity and the sky, exactly where the static run put them.
    if (sun_sweep_deg_per_s_ > 0.0f) {
        const float angle =
            glm::radians(sun_sweep_deg_per_s_ * static_cast<float>(frame_) / 60.0f);
        const glm::vec3 to_sun = -scene_sun_dir_;
        const float c = std::cos(angle);
        const float sn = std::sin(angle);
        const glm::vec3 rotated{to_sun.x * c + to_sun.z * sn, to_sun.y,
                                -to_sun.x * sn + to_sun.z * c};
        env_.sun_dir = -glm::normalize(rotated);
        env_dirty_ = true;
    }
    if (env_dirty_) {
        fb.set_environment(env_);
        env_dirty_ = false;
    }

    fb.set_draw_skybox(gs.skybox_enabled);
    fb.set_draw_ground(true);
    // The full scene covers the ground with instanced nature-pack grass
    // tufts; the engine's procedural blade grass only dresses the fallback.
    fb.set_draw_grass(gs.grass_enabled && !has_village_);

    // Motion ids for the velocity prepass: the player (skinned, id 1) and the
    // bobbing orbs (instanced, ids 100+) are the scene's moving objects; the
    // village + props stay id 0 (static: camera-only velocity).
    constexpr uint32_t MOTION_ID_PLAYER = 1;
    constexpr uint32_t MOTION_ID_ORB_BASE = 100;

    // The player's bone palette is computed into packet memory: no lifetime
    // contract on anim_player_ storage past this call.
    {
        const Model* model = models().get_model(player_model_);
        const auto joints = static_cast<uint32_t>(
            std::clamp<size_t>(model->skeleton.joints.size(), 1, anim::MAX_BONES));
        auto palette = fb.alloc_bone_palette(joints);
        if (palette.valid()) {
            anim_player_.write_palette(palette.matrices);
            fb.draw_skinned(player_model_, player_transform_, palette.id, {1.0f, 1.0f, 1.0f, 1.0f},
                            scene::DrawFlags::None, MOTION_ID_PLAYER);

            // The ANIMATED shadow proxy for that skinned draw
            // (docs/rt_lighting_plan.md §5.8). Emitted next to the draw it
            // belongs to, and only when the draw actually happened: the traced
            // any-hit passes skip this instance's bind-pose triangles, so
            // without these the character would cast no traced shadow at all.
            // Copied into packet memory, so shadow_capsules_ is free to be
            // rebuilt next frame. The orbs, props and village are not skinned
            // and are unaffected — their triangles are already the right shape.
            fb.add_shadow_capsules(shadow_capsules_);

            // Bone attachment: the lantern follows the hand through every
            // animation with no per-frame math on this side. One call.
            if (lantern_model_ != INVALID_MODEL_HANDLE && lantern_attach_.valid()) {
                const glm::mat4 held = lantern_attach_.world(player_transform_, anim_player_);
                fb.draw_model(lantern_model_, held, {1.0f, 0.9f, 0.65f, 1.0f});
            }
        }
    }

    // The pathfinding villager, and its route when debug draw is on.
    if (nav_.built() && orb_model_ != INVALID_MODEL_HANDLE) {
        const glm::mat4 t =
            glm::rotate(glm::translate(glm::mat4(1.0f),
                                       wanderer_.position + glm::vec3(0.0f, 60.0f, 0.0f)),
                        wanderer_.facing, glm::vec3(0, 1, 0));
        fb.draw_model(orb_model_, glm::scale(t, glm::vec3(1.4f)), {0.4f, 0.9f, 1.0f, 1.0f});

        if (nav_debug_draw_) {
            // The actual string-pulled path, so the smoothing is visible
            // rather than taken on trust.
            for (std::size_t i = 1; i < wanderer_.path.size(); ++i) {
                fb.add_debug_line(wanderer_.path[i - 1] + glm::vec3(0, 40, 0),
                                  wanderer_.path[i] + glm::vec3(0, 40, 0),
                                  bracken::core::Color{0xFF40E0FFu});
            }
        }
    }

    if (has_village_) {
        fb.draw_model(village_model_, village_transform_);
    }
    fb.draw_model(chrome_model_, chrome_transform_);

    for (const auto& prop : props_) {
        fb.draw_model(prop.model, prop.transform);
    }

    for (size_t orb_index = 0; orb_index < orbs_.size(); ++orb_index) {
        const auto& orb = orbs_[orb_index];
        glm::vec3 pos = orb.pos + glm::vec3(0, 10.0f * std::sin(time_ * 2.0f + orb.bob_phase), 0);
        fb.draw_model(orb_model_, glm::translate(glm::mat4(1.0f), pos), {1.0f, 1.0f, 1.0f, 1.0f},
                      scene::DrawFlags::None, MOTION_ID_ORB_BASE + static_cast<uint32_t>(orb_index));
        // casts_shadows: the orbs move (bob + respawn), constantly exercising
        // the local-shadow tile cache invalidation. Only the closest few win
        // the per-frame point budget; the rest fall back to unshadowed.
        // Intensity is radiometric (inverse-square in world units): a warm
        // pool directly beneath the orb that fades out well inside the 320
        // radius — anything standing in the pool throws a visible local shadow.
        // `source_radius` (R6): the orb IS the emitter, and its shell is a
        // radius-8 sphere, so authoring 8 here makes the traced penumbra the
        // one this bulb would really cast instead of the engine's 10-unit
        // default. Aggregate-initialized past stable_id, which stays 0 =
        // "engine assigns".
        fb.add_point_light({.position = pos,
                            .radius = 320.0f,
                            .color = {1.0f, 0.72f, 0.30f},
                            .intensity = 12000.0f,
                            .casts_shadows = true,
                            .stable_id = 0,
                            .source_radius = 8.0f});

        // Proximity bar above each orb — exercises 3D billboards.
        float d = glm::distance(glm::vec2(pos.x, pos.z), glm::vec2(player_pos_.x, player_pos_.z));
        float fill = std::clamp(1.0f - d / 1500.0f, 0.0f, 1.0f);
        fb.add_billboard_3d(pos.x, pos.y + 28.0f, pos.z, 46.0f, fill, 0xFFFFC040, 0x60000000, 0xFFFFD080);
    }

    if (flashlight_) {
        // Rig-light flashlight: the spot hangs above/right of the player
        // (camera-relative, so the framing is stable while the camera
        // orbits) and aims at the character's chest. The character stands in
        // the beam and throws a second, clearly visible ground shadow to the
        // camera-left — distinct from the sun's CSM shadow (casts_shadows).
        // The spot re-renders its atlas tile whenever the player/camera move
        // it; held still, the cached tile is reused. Intensity is radiometric
        // (inverse-square in world units): ~2x sun luminance at the beam
        // center, clearly visible against the low dusk sun.
        glm::vec3 fwd = camera().get_forward();
        fwd.y = 0.0f;
        fwd = glm::length(fwd) > 0.001f ? glm::normalize(fwd) : glm::vec3(0, 0, -1);
        const glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
        const glm::vec3 origin = player_pos_ + glm::vec3(0, 140, 0) + right * 160.0f - fwd * 40.0f;
        const glm::vec3 dir = glm::normalize(player_pos_ + glm::vec3(0, 40, 0) - origin);
        fb.add_spot_light({.position = origin,
                           .radius = 1400.0f,
                           .direction = dir,
                           .inner_angle = core::Degrees{18.0f},
                           .outer_angle = core::Degrees{32.0f},
                           .color = {0.9f, 0.95f, 1.0f},
                           .intensity = 200000.0f,
                           .casts_shadows = true,
                           .stable_id = 0,
                           // A wide softbox rather than a point: 60 units is
                           // 60 cm at this scale, which is what a rig light
                           // reads as and what makes the character's second
                           // ground shadow soften with distance instead of
                           // staying a hard edge. The A/B against the 10-unit
                           // default is visible at PLAYGROUND_TOD_HOUR=1.
                           .source_radius = 60.0f});
    }

    for (const auto& pos : pending_pickup_bursts_) {
        fb.spawn_effect(&pickup_burst_, pos, {0, 1, 0});
    }
    pending_pickup_bursts_.clear();
    if (pending_attack_arc_) {
        glm::vec3 fwd = forward_from_yaw(facing_.current);
        fb.spawn_effect(&attack_arc_, player_pos_ + fwd * 30.0f, fwd);
        pending_attack_arc_ = false;
    }
    if (pending_chimney_smoke_) {
        for (const auto& top : chimney_tops_) {
            fb.spawn_effect(&chimney_smoke_, top, {0, 1, 0});
        }
        pending_chimney_smoke_ = false;
    }
    if (pending_rain_spawn_) {
        fb.spawn_effect(&rain_layer_, player_pos_ + glm::vec3(0.0f, RAIN_LAYER_HEIGHT, 0.0f),
                        {0, 1, 0});
        pending_rain_spawn_ = false;
    }
    for (const auto& pos : pending_splashes_) {
        fb.spawn_effect(&rain_splash_, pos, {0, 1, 0});
    }
    pending_splashes_.clear();

    if (debug_draw_) {
        // Ray from the player toward the sun — lighting-direction ground
        // truth (from the host env copy; sim must not read the render-side
        // EnvironmentState while a frame is drawing).
        const glm::vec3 to_sun = -glm::normalize(env_.sun_dir);
        fb.add_debug_line(player_pos_ + glm::vec3(0, 100, 0),
                          player_pos_ + glm::vec3(0, 100, 0) + to_sun * 800.0f, 0xFF00FFFF);
        fb.add_debug_sphere(player_pos_ + glm::vec3(0, 95, 0), 65.0f, 0xFF30FF60);
        for (const auto& orb : orbs_) {
            fb.add_debug_sphere(orb.pos, ORB_PICKUP_RADIUS, 0xFF40A0FF);
        }
        // Skeleton overlay: one line per joint to its parent, post-IK.
        const Model* model = models().get_model(player_model_);
        auto world = anim_player_.world_transforms();
        for (size_t i = 0; i < model->skeleton.joints.size(); ++i) {
            int parent = model->skeleton.joints[i].parent_index;
            if (parent < 0) {
                continue;
            }
            glm::vec3 a = glm::vec3(player_transform_ * world[parent][3]);
            glm::vec3 b = glm::vec3(player_transform_ * world[i][3]);
            fb.add_debug_line(a, b, 0xFFFF5050);
        }
    }

    build_ui(sim, fb);
}

// The HUD, built through `bracken::ui` — the module's reference client.
//
// PORTING NOTE, because the smoke screenshot is a pinned baseline. The stat
// rows are declared with EXPLICIT fixed heights (24, then 20) rather than
// content heights, which reproduces the hand-written y offsets 20/44/64/84/
// 104/124 exactly. That is not a workaround: a fixed-height row is an ordinary
// thing to want, and doing it this way is what lets the port be checked against
// the previous frame pixel for pixel instead of "looks about right". The panel
// border is transparent for the same reason — the old HUD had none, and
// begin_panel skips a fully transparent border rather than emitting invisible
// quads.
//
// The ONE deliberate visual change is the keybind strip at the bottom, which
// stays on `UiLane::add_text_wrapped`: that call now measures against the real
// font instead of guessing 8 px per character, so the strip WRAPS onto a second
// line where it used to run off the right-hand edge of the screen.
void PlaygroundGame::build_ui(const SimFrame& sim, scene::FrameSink& fb) {
    ui::ContextConfig cfg;
    cfg.font = &sim.font_metrics();
    ui_.configure(cfg);
    ui_.style().panel_bg = 0xB0101418;
    ui_.style().panel_border = 0x00000000; // transparent -> not drawn at all

    // Built once and reused, because the demo menu's CANCEL verb is read out of
    // it below: `from_snapshot` is a pure mapping, but calling it twice would
    // invite the two copies to drift.
    const ui::InputFrame ui_input = ui::InputFrame::from_snapshot(sim.input());

    // F9 opens the demo menu. Never during a smoke run: that run has to render
    // a pure function of the frame index, and an open menu is state.
    if (smoke_frames_ <= 0 && sim.input().was_key_just_pressed(SDL_SCANCODE_F9)) {
        ui_menu_open_ = !ui_menu_open_;
    }

    ui_.begin(fb.ui(), ui_input,
              ui::Rect{0.0f, 0.0f, static_cast<float>(sim.screen_width()),
                       static_cast<float>(sim.screen_height())});

    ui::BoxStyle margin{};
    margin.padding = ui::Edges{10.0f, 10.0f, 0.0f, 0.0f};
    ui_.begin_column(margin);

    ui::BoxStyle panel{};
    panel.padding = ui::Edges{12.0f, 10.0f, 0.0f, 0.0f};
    panel.width = ui::Size::px(280.0f);
    panel.height = ui::Size::px(122.0f);
    ui_.begin_panel(panel);

    ui_.label("bracken playground", 0xFFFFD080, 1.0f, ui::Size::px(24.0f));
    // The smoke run must render a pure function of the frame index, so the
    // captured screenshot is comparable between runs. A live FPS readout is
    // wall-clock derived, so drawing it would put a run-to-run-varying string
    // on screen. The measured rate is still printed in the "[smoke] OK" line.
    if (smoke_frames_ > 0) {
        ui_.label("FPS --", 0xFFB8C4D0, 0.9f, ui::Size::px(20.0f));
    } else {
        ui_.label(std::format("FPS {:.0f}", fps()), 0xFFB8C4D0, 0.9f, ui::Size::px(20.0f));
    }
    ui_.label(std::format("Orbs collected: {}", orbs_collected_), 0xFFFFC040, 0.9f, ui::Size::px(20.0f));
    ui_.label(std::format("Anim state: {}", anim_sm_.current_state()), 0xFFB8C4D0, 0.9f, ui::Size::px(20.0f));
    ui_.label(std::format("Pos {:.0f}, {:.0f}", player_pos_.x, player_pos_.z), 0xFF8894A0, 0.9f,
              ui::Size::px(20.0f));
    if (smoke_frames_ > 0) {
        ui_.label(std::format("SMOKE {}/{}", frame_, smoke_frames_), 0xFFFF8060, 0.9f, ui::Size::px(20.0f));
    }
    ui_.end_panel();
    ui_.end_box();

    if (ui_menu_open_) {
        build_demo_menu(sim);
    }

    ui_.end();
    // CANCEL closes the demo menu (pad EAST / Escape / the menu-toggle button).
    // It is read here rather than inside build_demo_menu because the flag it
    // writes is checked at the TOP of the next frame — flipping it mid-
    // declaration would tear the frame currently being described. Without this
    // the demo's own comment about B closing the menu was simply untrue.
    if (ui_menu_open_ && ui_input.nav_cancel) {
        ui_menu_open_ = false;
    }
    // The capture the NEXT frame's input read is gated on. One frame late by
    // construction — the UI is described at the END of the frame, after the
    // game logic that would have to be gated — which is the same one-frame
    // rule ui::Context already documents for its own hit testing.
    ui_capture_ = ui_.capture();

    fb.ui().add_text_wrapped("WASD move | Shift run | mouse-drag orbit | wheel zoom | LMB attack | F flashlight | "
                             "F1 AO | F2 bloom | F4 fog | F5 shadows | F6 AA | F7 IBL | F8 local shadows | "
                             "F3 debug | F9 UI demo | F12 screenshot | Esc quit | F10 dev overlay",
                             10, static_cast<float>(sim.screen_height()) - 56.0f,
                             static_cast<float>(sim.screen_width()) - 20.0f, 0.9f, 0xFFC8D0DC);
}

// The worked example for the widget vocabulary: EVERY shipped widget, in one
// panel, navigable by keyboard and by gamepad. Never declared during a smoke
// run, so it cannot move the pinned screenshot baselines — which is also why
// it can afford to be exhaustive rather than tasteful.
//
// The pad path is the one worth calling out: the D-pad and the left stick move
// the focus, A activates, B closes, and the shoulder buttons walk the tab
// order — all of it while the game is still running, because those channels
// (`InputSnapshot::pad_nav_up` and friends) are edges that are NOT gated on
// set_game_input_enabled(false) the way `menu_*_pressed` is.
void PlaygroundGame::build_demo_menu(const SimFrame& sim) {
    ui::BoxStyle place{};
    place.padding = ui::Edges{static_cast<float>(sim.screen_width()) * 0.5f - 170.0f, 120.0f, 0.0f, 0.0f};
    ui_.begin_column(place);

    ui::BoxStyle panel{};
    panel.padding = ui::Edges::all(14.0f);
    panel.spacing = 8.0f;
    panel.width = ui::Size::px(340.0f);
    ui_.style().panel_border = 0xFF303840; // the demo panel DOES want a border
    ui_.begin_panel(panel);
    ui_.label("UI demo (F9 to close)", 0xFFFFD080);

    if (ui_.button(ui::id("demo.reset_orbs"), "Reset orb count", ui::Size::flex(1.0f))) {
        orbs_collected_ = 0;
    }
    (void)ui_.toggle(ui::id("demo.debug"), "Debug overlay", debug_draw_, ui::Size::flex(1.0f));

    ui_.label("Master volume", 0xFF8894A0, 0.9f);
    if (ui_.slider(ui::id("demo.volume"), ui_demo_volume_, 0.0f, 1.0f, ui::Size::flex(1.0f))) {
        audio().set_master_gain(ui_demo_volume_);
    }
    // The same value again as a read-only bar, so the two are visibly one
    // number — a progress bar is not focusable and takes no id.
    ui_.progress_bar(ui_demo_volume_, ui::Size::flex(1.0f));

    ui_.separator();

    // A radio group: one selection across four widgets, each with its own id.
    // The ordinals ARE the WeatherMode enumerators, which is the whole reason
    // radio takes an int rather than owning an enum it cannot know about.
    ui_.label("Weather", 0xFF8894A0, 0.9f);
    ui::BoxStyle row{};
    row.spacing = 6.0f;
    ui_.begin_row(row);
    bool weather_changed = ui_.radio(ui::id("demo.wx.auto"), "Auto", ui_demo_weather_, 0);
    weather_changed = ui_.radio(ui::id("demo.wx.clear"), "Clear", ui_demo_weather_, 1) || weather_changed;
    weather_changed = ui_.radio(ui::id("demo.wx.cloud"), "Cloud", ui_demo_weather_, 2) || weather_changed;
    weather_changed = ui_.radio(ui::id("demo.wx.rain"), "Rain", ui_demo_weather_, 3) || weather_changed;
    ui_.end_box();
    if (weather_changed) {
        day_weather_.mode = static_cast<WeatherMode>(std::clamp(ui_demo_weather_, 0, 3));
    }

    ui_.separator();

    // A text field. It commits on ENTER, and while it has focus the Context
    // reports keyboard capture, so handle_input's masked snapshot keeps the
    // character from walking off while the player types.
    ui_.label("Marker name (Enter to place)", 0xFF8894A0, 0.9f);
    if (ui_.text_field(ui::id("demo.marker"), ui_demo_marker_, 24, ui::Size::flex(1.0f))) {
        BRACKEN_LOG_INFO("playground", "UI demo: marker '{}' placed at {:.0f}, {:.0f}", ui_demo_marker_,
                         player_pos_.x, player_pos_.z);
    }

    // A scrolling list, to exercise the clip command and the wheel.
    ui_.label("Scroll me", 0xFF8894A0, 0.9f);
    ui::BoxStyle list{};
    list.padding = ui::Edges::all(4.0f);
    list.spacing = 2.0f;
    list.width = ui::Size::flex(1.0f);
    list.height = ui::Size::px(90.0f);
    ui_.begin_scroll(ui::id("demo.list"), list);
    for (int i = 0; i < 24; ++i) {
        ui_.label(std::format("row {}", i), 0xFFB8C4D0, 0.9f);
    }
    ui_.end_scroll();

    if (ui_.button(ui::id("demo.close"), "Close", ui::Size::flex(1.0f))) {
        ui_menu_open_ = false;
    }
    ui_.end_panel();
    ui_.end_box();
}

// ---------------------------------------------------------------------------
// Frame fence (main thread, no sim in flight)
// ---------------------------------------------------------------------------

void PlaygroundGame::on_main_frame(bracken::MainFrame& mf) {
    sync_dev_panel();

    // Forward the ripples the joined sim frame queued into the render-side
    // wave sim, before that frame renders (matching the old sequential
    // update-then-render timing).
    if (!pending_ripples_.empty()) {
        if (auto* water = scene_renderer().water()) {
            for (const auto& r : pending_ripples_) {
                water->add_ripple(r.pos, r.radius, r.strength);
            }
        }
        pending_ripples_.clear();
    }

    if (rt_bench_) {
        // Deliberately INSTEAD of update_smoke_fence: its scripted flips
        // (volumetric fog on at 80, AA at 110/160, AO at 140, the sun swings)
        // would each change what the passes below are timing, halfway through
        // the measurement.
        update_rt_bench_fence();
    } else if (smoke_frames_ > 0) {
        update_smoke_fence(mf);
    }
}

// --rt-bench, fence side. One sample per frame; the report is printed once.
void PlaygroundGame::update_rt_bench_fence() {
    if (smoke_start_ns_ == 0) {
        smoke_start_ns_ = SDL_GetTicksNS();
    }
    // Discard the warm-up: the first frames pay pipeline creation, denoiser
    // history invalidation and the initial TLAS build, none of which recur.
    if (frame_ > RT_BENCH_WARMUP_FRAMES) {
        if (const std::optional<bracken::RenderStats> stats = render_stats()) {
            for (const auto& pass : stats->pass_times_ms) {
                if (pass.cpu_ms) {
                    rt_bench_pass_ms_[std::string(pass.name)].push_back(*pass.cpu_ms);
                }
            }
        }
        rt_bench_sim_ms_.push_back(sim_ms());
        rt_bench_render_ms_.push_back(render_ms());
        rt_bench_fence_ms_.push_back(fence_ms());
        rt_bench_present_wait_ms_.push_back(present_wait_ms());
        // Only when a readback has landed for that signal: an invalid entry is
        // absence of evidence, and averaging a zero into the window would
        // report a collapse the run never had.
        const auto& denoise = scene_renderer().raytracing_stats().denoise;
        if (denoise.shadow_valid) {
            rt_bench_shadow_accum_.push_back(denoise.shadow_mean_accum);
            rt_bench_shadow_fire_.push_back(denoise.shadow_gradient_fire);
        }
        if (denoise.local_valid) {
            rt_bench_local_accum_.push_back(denoise.local_mean_accum);
            rt_bench_local_fire_.push_back(denoise.local_gradient_fire);
        }
        if (denoise.di_noise_valid) {
            rt_bench_di_noise_.push_back(denoise.di_noise_mean_delta);
            rt_bench_di_level_.push_back(denoise.di_noise_mean_level);
        }
        ++rt_bench_sampled_frames_;
    }

    if (frame_ >= smoke_frames_) {
        print_rt_bench_report();
        quit();
    }
}

namespace {

/** Mean and p95 of a sample set. p95 is nearest-rank on the sorted copy —
 *  no interpolation, so the number printed is one the run actually produced. */
struct BenchSummary {
    float mean = 0.0f;
    float p95 = 0.0f;
};

BenchSummary summarize(std::vector<float> samples) {
    BenchSummary out;
    if (samples.empty()) {
        return out;
    }
    double sum = 0.0;
    for (float v : samples) {
        sum += v;
    }
    out.mean = static_cast<float>(sum / static_cast<double>(samples.size()));
    std::sort(samples.begin(), samples.end());
    const std::size_t rank =
        std::min(samples.size() - 1, static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(samples.size()))) - 1);
    out.p95 = samples[rank];
    return out;
}

} // namespace

// Machine-readable on purpose: one `[rtbench] <kind> <name> ...` record per
// line, fixed field order, so a diff between two builds is a diff between two
// files. This is the gate docs/rt_lighting_plan.md §5.0 asks for before any
// traversal change may claim a win.
void PlaygroundGame::print_rt_bench_report() {
    const double elapsed_s =
        static_cast<double>(SDL_GetTicksNS() - smoke_start_ns_) / static_cast<double>(SDL_NS_PER_SECOND);
    const auto& g = graphics_settings();
    std::cout << std::format("[rtbench] run frames={} sampled={} warmup={} fps={:.1f} "
                             "rt_shadows={} rt_reflections={} rt_local_shadows={} "
                             "rt_shadow_res={} rt_reflection_res={} rt_local_shadow_res={} "
                             "rt_range={} driver={}\n",
                             frame_, rt_bench_sampled_frames_, RT_BENCH_WARMUP_FRAMES,
                             elapsed_s > 0.0 ? frame_ / elapsed_s : 0.0, g.rt_shadows ? 1 : 0,
                             g.rt_reflections ? 1 : 0, static_cast<int>(g.rt_local_shadows),
                             static_cast<int>(g.rt_shadow_resolution),
                             static_cast<int>(g.rt_reflection_resolution),
                             static_cast<int>(g.rt_local_shadow_resolution),
                             static_cast<int>(g.rt_max_instance_range), gpu_driver_name());
    // CPU RECORD time, not GPU time — SDL3 has no timestamp API, so a pass can
    // record in 40 us and cost 4 ms on the device. present_wait below is the
    // CPU-bound-vs-GPU-bound signal.
    std::cout << "[rtbench] pass name mean_ms p95_ms samples\n";
    for (const auto& [name, samples] : rt_bench_pass_ms_) {
        const BenchSummary s = summarize(samples);
        std::cout << std::format("[rtbench] pass {} {:.4f} {:.4f} {}\n", name, s.mean, s.p95,
                                 samples.size());
    }
    struct Row {
        const char* name;
        const std::vector<float>* samples;
    };
    const Row rows[] = {{"sim", &rt_bench_sim_ms_},
                        {"render", &rt_bench_render_ms_},
                        {"fence", &rt_bench_fence_ms_},
                        {"present_wait", &rt_bench_present_wait_ms_}};
    for (const Row& row : rows) {
        const BenchSummary s = summarize(*row.samples);
        std::cout << std::format("[rtbench] frame {} {:.4f} {:.4f} {}\n", row.name, s.mean, s.p95,
                                 row.samples->size());
    }
    const auto& rt = scene_renderer().raytracing_stats();
    // R5b: what the two visibility denoisers actually did, averaged over the
    // run. `sun_sweep` is echoed because the whole table means the opposite
    // thing at 0 — there, a fire fraction near zero and a full accumulation
    // window are the PASS condition, and anything else is a regression.
    {
        const BenchSummary sa = summarize(rt_bench_shadow_accum_);
        const BenchSummary sf = summarize(rt_bench_shadow_fire_);
        const BenchSummary la = summarize(rt_bench_local_accum_);
        const BenchSummary lf = summarize(rt_bench_local_fire_);
        std::cout << std::format(
            "[rtbench] denoise sun_sweep_deg_s={:.2f} "
            "sun_accum={:.2f} sun_grad_fire={:.4f} sun_samples={} "
            "local_accum={:.2f} local_grad_fire={:.4f} local_samples={} "
            "sun_grad_rays={} local_grad_rays={}\n",
            sun_sweep_deg_per_s_, sa.mean, sf.mean, rt_bench_shadow_accum_.size(), la.mean,
            lf.mean, rt_bench_local_accum_.size(), rt.shadow_gradient_rays,
            rt.local_shadow_gradient_rays);
    }
    // R7's temporal-noise probe: the number the smoke test structurally cannot
    // produce, and the one the whole package is judged on. Under a pinned
    // camera and a static sun this is temporal NOISE; under PLAYGROUND_SUN_SWEEP
    // it also contains the legitimate change, so the two conditions are only
    // ever compared against themselves.
    {
        const BenchSummary noise = summarize(rt_bench_di_noise_);
        const BenchSummary level = summarize(rt_bench_di_level_);
        std::cout << std::format(
            "[rtbench] di noise={:.5f} p95={:.5f} level={:.5f} relative={:.5f} samples={}\n",
            noise.mean, noise.p95, level.mean,
            level.mean > 1e-6f ? noise.mean / level.mean : 0.0f, rt_bench_di_noise_.size());
    }
    std::cout << std::format("[rtbench] rt shadows_traced={} reflections_traced={} "
                             "local_shadows_traced={} shadow_rays={} "
                             "reflection_rays={} local_shadow_rays={} "
                             "instances={} tlas_builds={} tlas_refits={} "
                             "blas_mb={:.2f} "
                             "attrs_mb={:.2f} tlas_mb={:.3f} accel_mb={:.2f}\n",
                             rt.shadows_traced, rt.reflections_traced, rt.local_shadows_traced,
                             rt.shadow_rays, rt.reflection_rays, rt.local_shadow_rays,
                             rt.instances, rt.tlas_builds, rt.tlas_refits,
                             static_cast<double>(rt.blas_bytes) / (1024.0 * 1024.0),
                             static_cast<double>(rt.attribute_bytes) / (1024.0 * 1024.0),
                             static_cast<double>(rt.tlas_bytes) / (1024.0 * 1024.0),
                             static_cast<double>(rt.acceleration_bytes) / (1024.0 * 1024.0));
}

void PlaygroundGame::sync_dev_panel() {
    // Apply the widget edits the overlay made since the last fence (the
    // overlay draws during rendering; panel_ itself is main-thread-only
    // data). Edge-detect the checkboxes against the value last shown so
    // sim-side toggles — the F key — aren't overwritten by a stale widget.
    if (panel_.flashlight != panel_applied_flashlight_) {
        flashlight_ = panel_.flashlight;
    }
    if (panel_.debug_draw != panel_applied_debug_draw_) {
        debug_draw_ = panel_.debug_draw;
    }
    for (; panel_.spawn_burst_clicks > 0; --panel_.spawn_burst_clicks) {
        pending_pickup_bursts_.push_back(player_pos_);
    }
    // Time of day + weather: same edge-detect discipline — the clock advances
    // and Auto mode rolls weather sim-side every frame, so only a widget the
    // user actually moved may write back. day_weather_ is sim-owned but the
    // fence is the one main-thread point with no sim in flight (the same
    // contract as flashlight_ above).
    if (panel_.tod_hour != panel_applied_tod_hour_) {
        day_weather_.hour = std::clamp(panel_.tod_hour, 0.0f, 23.99f);
    }
    if (panel_.tod_speed != panel_applied_tod_speed_) {
        day_weather_.speed = std::max(0.0f, panel_.tod_speed);
    }
    if (panel_.tod_animate != panel_applied_tod_animate_) {
        day_weather_.animate = panel_.tod_animate;
    }
    if (panel_.weather_mode != panel_applied_weather_mode_) {
        day_weather_.mode = static_cast<WeatherMode>(std::clamp(panel_.weather_mode, 0, 3));
    }
    if (panel_.rain_slider != panel_applied_rain_slider_) {
        day_weather_.rain_target = panel_.rain_slider;
    }
    // Refresh the display fields + widget baselines from live game state —
    // legal exactly here, with no sim in flight. The panel body reads ONLY
    // this copy while the next sim frame mutates the live state.
    panel_.flashlight = flashlight_;
    panel_applied_flashlight_ = flashlight_;
    panel_.debug_draw = debug_draw_;
    panel_applied_debug_draw_ = debug_draw_;
    panel_.orbs_collected = orbs_collected_;
    panel_.player_pos = player_pos_;
    panel_.anim_state = anim_sm_.current_state();
    panel_.village_loaded = has_village_;
    // Ball state comes from the sim-written mirrors, not PhysicsWorld —
    // physics is sim-stage and the fence never calls into it.
    panel_.ball_pos = ball_pos_;
    panel_.ball_speed = ball_speed_;
    panel_.ball_active = ball_active_;
    panel_.tod_hour = day_weather_.hour;
    panel_applied_tod_hour_ = panel_.tod_hour;
    panel_.tod_speed = day_weather_.speed;
    panel_applied_tod_speed_ = panel_.tod_speed;
    panel_.tod_animate = day_weather_.animate;
    panel_applied_tod_animate_ = panel_.tod_animate;
    panel_.weather_mode = static_cast<int>(day_weather_.mode);
    panel_applied_weather_mode_ = panel_.weather_mode;
    panel_.rain_slider = day_weather_.rain_target;
    panel_applied_rain_slider_ = panel_.rain_slider;
    panel_.overcast_now = day_weather_.overcast();
    panel_.rain_now = day_weather_.rain();
    panel_.weather_label = day_weather_.weather_label();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float PlaygroundGame::rand01() {
    rng_ = rng_ * 1664525u + 1013904223u;
    return static_cast<float>(rng_ >> 8) / 16777216.0f;
}

glm::vec3 PlaygroundGame::random_ground_pos(float margin) {
    float x = margin + rand01() * (WORLD - 2.0f * margin);
    float z = margin + rand01() * (WORLD - 2.0f * margin);
    return {x, heightmap_.get_height_world(x, z), z};
}

} // namespace playground
