#include "playground.hpp"

#include "character.hpp"
#include "engine/animation/ik_solver.hpp"
#include "engine/procedural/rock_generator.hpp"
#include "engine/procedural/tree_generator.hpp"
#include "engine/render/lighting/light.hpp"
#include "engine/systems/camera_controller.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <SDL3/SDL.h>

namespace playground {

using namespace mmo::engine;
namespace anim = mmo::engine::animation;
namespace lighting = mmo::engine::render::lighting;

namespace {
constexpr float WORLD = 8000.0f;
constexpr float WORLD_MARGIN = 250.0f;
constexpr int ORB_COUNT = 15;
constexpr float ORB_PICKUP_RADIUS = 60.0f;
constexpr float ORB_HOVER = 26.0f;
constexpr float WALK_SPEED = 140.0f;
constexpr float RUN_SPEED = 300.0f;
constexpr float ATTACK_DURATION = 0.45f;

glm::vec3 forward_from_yaw(float yaw) {
    // The character model faces -Z; rotate_y(yaw) maps that to this direction.
    return {-std::sin(yaw), 0.0f, -std::cos(yaw)};
}
} // namespace

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

bool PlaygroundGame::on_init() {
    if (!init_renderer(1600, 900, "mmo_engine playground", WORLD, WORLD)) {
        return false;
    }

    build_heightmap();
    set_heightmap(heightmap_);

    register_models();
    setup_state_machine();
    setup_effects();
    scatter_props();

    for (int i = 0; i < ORB_COUNT; ++i) {
        orbs_.push_back({random_ground_pos(600.0f) + glm::vec3(0, ORB_HOVER, 0), rand01() * 6.28f});
    }

    player_pos_ = {WORLD * 0.5f, 0.0f, WORLD * 0.5f};
    player_pos_.y = heightmap_.get_height_world(player_pos_.x, player_pos_.z);

    auto& cam = camera();
    cam.set_terrain_height_func([this](float x, float z) { return get_terrain_height(x, z); });
    cam.set_screen_size(screen_width(), screen_height());
    auto cfg = cam.get_config();
    cfg.distance = 380.0f;
    cfg.height_offset = 120.0f;
    cam.set_config(cfg);
    cam.set_target(player_pos_ + glm::vec3(0, 110, 0));

    if (smoke_frames_ > 0) {
        set_collect_render_stats(true);
    }
    return true;
}

void PlaygroundGame::on_shutdown() {
    shutdown_renderer();
}

void PlaygroundGame::build_heightmap() {
    constexpr uint32_t RES = 257;
    heightmap_.resolution = RES;
    heightmap_.world_origin_x = 0.0f;
    heightmap_.world_origin_z = 0.0f;
    heightmap_.world_size = WORLD;
    heightmap_.min_height = -100.0f;
    heightmap_.max_height = 150.0f;
    heightmap_.height_data.resize(RES * RES);
    for (uint32_t z = 0; z < RES; ++z) {
        for (uint32_t x = 0; x < RES; ++x) {
            float wx = WORLD * static_cast<float>(x) / (RES - 1);
            float wz = WORLD * static_cast<float>(z) / (RES - 1);
            float h = 40.0f * std::sin(wx * 0.0011f + 0.5f) * std::cos(wz * 0.0008f) +
                      15.0f * std::sin(wx * 0.0040f + 2.0f) * std::sin(wz * 0.0035f) +
                      5.0f * std::sin(wx * 0.0130f) * std::cos(wz * 0.0110f + 1.0f);
            float normalized = (h - heightmap_.min_height) / (heightmap_.max_height - heightmap_.min_height);
            heightmap_.height_data[z * RES + x] = static_cast<uint16_t>(normalized * 65535.0f);
        }
    }
}

void PlaygroundGame::register_models() {
    auto& mm = models();
    player_model_ = mm.register_model("player", build_character_model());
    orb_model_ = mm.register_model("orb", build_orb_model());
    mm.register_model("orb_shard", build_shard_model()); // referenced by name from effects

    mm.register_model("tree_oak", procedural::TreeGenerator::generate_oak(7));
    mm.register_model("tree_pine", procedural::TreeGenerator::generate_pine(21));
    mm.register_model("tree_birch", procedural::TreeGenerator::generate_birch(33));
    mm.register_model("rock_boulder", procedural::RockGenerator::generate_boulder(11));
    mm.register_model("rock_slate", procedural::RockGenerator::generate_slate(29));
    mm.register_model("rock_cluster", procedural::RockGenerator::generate_cluster(43));
}

void PlaygroundGame::scatter_props() {
    auto& mm = models();
    const ModelHandle trees[] = {mm.get_handle("tree_oak"), mm.get_handle("tree_pine"), mm.get_handle("tree_birch")};
    const ModelHandle rocks[] = {mm.get_handle("rock_boulder"), mm.get_handle("rock_slate"),
                                 mm.get_handle("rock_cluster")};

    auto place = [this](ModelHandle handle, float scale, float sink) {
        glm::vec3 pos = random_ground_pos(400.0f);
        // Keep the spawn clearing prop-free.
        if (glm::distance(glm::vec2(pos.x, pos.z), glm::vec2(WORLD * 0.5f)) < 400.0f) {
            return;
        }
        glm::mat4 t = glm::translate(glm::mat4(1.0f), pos - glm::vec3(0, sink, 0));
        t = glm::rotate(t, rand01() * 6.28f, glm::vec3(0, 1, 0));
        t = glm::scale(t, glm::vec3(scale));
        props_.push_back({handle, t});
    };

    for (int i = 0; i < 50; ++i) {
        place(trees[i % 3], 4.0f + rand01() * 3.0f, 2.0f);
    }
    for (int i = 0; i < 25; ++i) {
        float s = 8.0f + rand01() * 14.0f;
        place(rocks[i % 3], s, s * 0.4f);
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
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

void PlaygroundGame::on_update(float dt) {
    last_dt_ = dt;
    time_ += dt;
    ++frame_;

    if (smoke_frames_ > 0) {
        update_smoke_driver(dt);
    } else {
        handle_input(dt);
    }
    update_player(dt);
    update_camera(dt);
    update_orbs(dt);
}

void PlaygroundGame::handle_input(float dt) {
    (void)dt;
    auto& in = input();

    if (in.was_key_just_pressed(SDL_SCANCODE_ESCAPE)) {
        quit();
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F)) {
        flashlight_ = !flashlight_;
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F3)) {
        debug_draw_ = !debug_draw_;
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F12)) {
        request_screenshot(std::format("screenshot_{:03d}.png", screenshot_counter_++));
    }

    // Graphics-settings toggles, applied through the same path a settings menu
    // would use.
    auto& gs = graphics_settings();
    bool gs_dirty = false;
    if (in.was_key_just_pressed(SDL_SCANCODE_F1)) {
        gs.ao_mode = static_cast<AOMode>((static_cast<int>(gs.ao_mode) + 1) % 3);
        gs_dirty = true;
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F2)) {
        gs.bloom_enabled = !gs.bloom_enabled;
        gs_dirty = true;
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F4)) {
        gs.volumetric_fog = !gs.volumetric_fog;
        gs.god_rays = gs.volumetric_fog;
        gs_dirty = true;
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F5)) {
        gs.shadow_mode = static_cast<ShadowMode>((static_cast<int>(gs.shadow_mode) + 1) % 3);
        gs_dirty = true;
    }
    if (in.was_key_just_pressed(SDL_SCANCODE_F6)) {
        gs.aa_mode = gs.aa_mode == AAMode::Off ? AAMode::FXAA : AAMode::Off;
        gs_dirty = true;
    }
    if (gs_dirty) {
        apply_all_graphics_settings(gs);
    }

    // Camera-relative movement.
    glm::vec3 fwd = camera().get_forward();
    fwd.y = 0.0f;
    fwd = glm::length(fwd) > 0.001f ? glm::normalize(fwd) : glm::vec3(0, 0, -1);
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
    glm::vec3 dir{0.0f};
    if (in.move_forward()) dir += fwd;
    if (in.move_backward()) dir -= fwd;
    if (in.move_right()) dir += right;
    if (in.move_left()) dir -= right;
    move_dir_ = glm::length(dir) > 0.001f ? glm::normalize(dir) : glm::vec3(0.0f);

    const bool* keys = SDL_GetKeyboardState(nullptr);
    running_input_ = keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT];

    if (in.mouse_left_pressed() && attack_timer_ <= 0.0f) {
        attack_timer_ = ATTACK_DURATION;
        pending_attack_arc_ = true;
        camera().add_directional_shake(forward_from_yaw(facing_.current), 6.0f, 0.15f);
    }
    in.clear_mouse_edges();
}

void PlaygroundGame::update_smoke_driver(float dt) {
    // Self-drive: orbit the camera, run toward the nearest orb (real pickups),
    // attack periodically, and flip heavyweight settings mid-run so the whole
    // post-processing matrix gets exercised.
    camera().rotate_yaw(35.0f * dt);

    const Orb* nearest = nullptr;
    float best = 1e30f;
    for (const auto& orb : orbs_) {
        float d = glm::distance(glm::vec2(orb.pos.x, orb.pos.z), glm::vec2(player_pos_.x, player_pos_.z));
        if (d < best) {
            best = d;
            nearest = &orb;
        }
    }
    if (nearest) {
        glm::vec3 d = nearest->pos - player_pos_;
        d.y = 0.0f;
        move_dir_ = glm::length(d) > 1.0f ? glm::normalize(d) : glm::vec3(0.0f);
    }
    running_input_ = true;

    if (frame_ % 120 == 60 && attack_timer_ <= 0.0f) {
        attack_timer_ = ATTACK_DURATION;
        pending_attack_arc_ = true;
    }

    auto& gs = graphics_settings();
    if (frame_ == 80) {
        gs.volumetric_fog = true;
        gs.god_rays = true;
        apply_all_graphics_settings(gs);
    }
    if (frame_ == 140) {
        gs.ao_mode = AOMode::GTAO;
        apply_all_graphics_settings(gs);
    }
    if (frame_ == 200) {
        gs.shadow_mode = ShadowMode::PCF;
        apply_all_graphics_settings(gs);
    }

    // Late enough that all the settings flips above are visible in the image.
    if (frame_ == std::max(1, smoke_frames_ - 20)) {
        request_screenshot(SMOKE_SCREENSHOT_PATH);
    }

    if (frame_ >= smoke_frames_) {
        const auto& stats = render_stats();
        std::cout << std::format(
            "[smoke] OK frames={} fps={:.0f} draw_calls={} triangles={} entities={} orbs_collected={} driver={}\n",
            frame_, fps(), stats.draw_calls, stats.triangle_count, stats.entities_rendered, orbs_collected_,
            gpu_driver_name());
        quit();
    }
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
    player_pos_ += player_vel_ * dt;
    player_pos_.x = std::clamp(player_pos_.x, WORLD_MARGIN, WORLD - WORLD_MARGIN);
    player_pos_.z = std::clamp(player_pos_.z, WORLD_MARGIN, WORLD - WORLD_MARGIN);
    player_pos_.y = get_terrain_height(player_pos_.x, player_pos_.z);

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
                            model->skeleton, ik, player_transform_, 1.0f, foot_smoother_.smoothed_left_offset,
                            foot_smoother_.smoothed_right_offset);
    }
    float forward_lean = std::min(player_speed_ * 0.0004f, 0.15f);
    float lateral_lean = std::clamp(-facing_.turn_rate * 0.04f, -0.12f, 0.12f);
    anim::apply_body_lean(anim_player_.mutable_bone_matrices(), anim_player_.mutable_world_transforms(),
                          model->skeleton, ik.spine, forward_lean, lateral_lean);
}

void PlaygroundGame::update_camera(float dt) {
    auto& cam = camera();
    cam.set_screen_size(screen_width(), screen_height());
    if (smoke_frames_ <= 0) {
        cam.set_yaw(input().get_camera_yaw());
        cam.set_pitch(input().get_camera_pitch());
        cam.adjust_zoom(input().get_camera_zoom_delta());
    }
    cam.set_target(player_pos_ + glm::vec3(0, 110, 0));
    cam.set_target_velocity(player_vel_);
    cam.update(dt);
    input().set_camera_forward(cam.get_forward().x, cam.get_forward().z);
}

void PlaygroundGame::update_orbs(float dt) {
    (void)dt;
    for (auto& orb : orbs_) {
        float d = glm::distance(glm::vec2(orb.pos.x, orb.pos.z), glm::vec2(player_pos_.x, player_pos_.z));
        if (d < ORB_PICKUP_RADIUS) {
            ++orbs_collected_;
            pending_pickup_bursts_.push_back(orb.pos);
            camera().add_shake(systems::ShakeType::Impact, 4.0f, 0.25f);
            orb.pos = random_ground_pos(600.0f) + glm::vec3(0, ORB_HOVER, 0);
            orb.bob_phase = rand01() * 6.28f;
        }
    }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void PlaygroundGame::on_render() {
    scene_.clear();
    ui_.clear();
    const auto& gs = graphics_settings();

    scene_.set_draw_skybox(gs.skybox_enabled);
    scene_.set_draw_ground(true);
    scene_.set_draw_grass(gs.grass_enabled);

    scene_.add_skinned_model(player_model_, player_transform_, anim_player_.bone_matrices());

    for (const auto& prop : props_) {
        scene_.add_model(prop.model, prop.transform);
    }

    for (const auto& orb : orbs_) {
        glm::vec3 pos = orb.pos + glm::vec3(0, 10.0f * std::sin(time_ * 2.0f + orb.bob_phase), 0);
        scene_.add_model(orb_model_, glm::translate(glm::mat4(1.0f), pos));
        scene_.add_point_light({pos, 320.0f, {1.0f, 0.72f, 0.30f}, 8.0f});

        // Proximity bar above each orb — exercises 3D billboards.
        float d = glm::distance(glm::vec2(pos.x, pos.z), glm::vec2(player_pos_.x, player_pos_.z));
        float fill = std::clamp(1.0f - d / 1500.0f, 0.0f, 1.0f);
        scene_.add_billboard_3d(pos.x, pos.y + 28.0f, pos.z, 46.0f, fill, 0xFFFFC040, 0x60000000, 0xFFFFD080);
    }

    if (flashlight_) {
        glm::vec3 dir = camera().get_forward();
        scene_.add_spot_light({player_pos_ + glm::vec3(0, 150, 0), 1400.0f, dir,
                               std::cos(glm::radians(14.0f)), {0.9f, 0.95f, 1.0f}, std::cos(glm::radians(26.0f)),
                               14.0f, {}});
    }

    for (const auto& pos : pending_pickup_bursts_) {
        scene_.add_particle_effect_spawn(&pickup_burst_, pos, {0, 1, 0});
    }
    pending_pickup_bursts_.clear();
    if (pending_attack_arc_) {
        glm::vec3 fwd = forward_from_yaw(facing_.current);
        scene_.add_particle_effect_spawn(&attack_arc_, player_pos_ + fwd * 30.0f, fwd);
        pending_attack_arc_ = false;
    }

    if (debug_draw_) {
        scene_.add_debug_sphere(player_pos_ + glm::vec3(0, 95, 0), 65.0f, 0xFF30FF60);
        for (const auto& orb : orbs_) {
            scene_.add_debug_sphere(orb.pos, ORB_PICKUP_RADIUS, 0xFF40A0FF);
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
            scene_.add_debug_line(a, b, 0xFFFF5050);
        }
    }

    build_ui();
    render_frame(scene_, ui_, camera().get_camera_state(), last_dt_);
}

void PlaygroundGame::build_ui() {
    ui_.add_filled_rect(10, 10, 280, 122, 0xB0101418);
    ui_.add_text("mmo_engine playground", 22, 20, 1.0f, 0xFFFFD080);
    ui_.add_text(std::format("FPS {:.0f}", fps()), 22, 44, 0.9f, 0xFFB8C4D0);
    ui_.add_text(std::format("Orbs collected: {}", orbs_collected_), 22, 64, 0.9f, 0xFFFFC040);
    ui_.add_text(std::format("Anim state: {}", anim_sm_.current_state()), 22, 84, 0.9f, 0xFFB8C4D0);
    ui_.add_text(std::format("Pos {:.0f}, {:.0f}", player_pos_.x, player_pos_.z), 22, 104, 0.9f, 0xFF8894A0);
    if (smoke_frames_ > 0) {
        ui_.add_text(std::format("SMOKE {}/{}", frame_, smoke_frames_), 22, 124, 0.9f, 0xFFFF8060);
    }
    ui_.add_text_wrapped("WASD move | Shift run | mouse-drag orbit | wheel zoom | LMB attack | F flashlight | "
                         "F1 AO | F2 bloom | F4 fog | F5 shadows | F6 FXAA | F3 debug | F12 screenshot | Esc quit",
                         10, static_cast<float>(screen_height()) - 56.0f, static_cast<float>(screen_width()) - 20.0f,
                         0.9f, 0xFFC8D0DC);
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
