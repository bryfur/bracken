#include "character.hpp"

#include "bracken/animation/animation_types.hpp"
#include <cmath>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace playground {

using bracken::Mesh;
using bracken::Model;
using bracken::gpu::SkinnedVertex;
using bracken::gpu::Vertex3D;
namespace anim = bracken::animation;

namespace {

constexpr float TWO_PI = glm::two_pi<float>();

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

// Append an axis-aligned box with per-face normals, fully weighted to `joint`.
void add_skinned_box(Mesh& mesh, const glm::vec3& center, const glm::vec3& half, uint8_t joint,
                     const glm::vec4& color) {
    struct Face {
        glm::vec3 normal, u, v;
    };
    static const Face faces[6] = {
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},  {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}}, {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},  {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},  {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},
    };
    for (const auto& f : faces) {
        uint32_t base = static_cast<uint32_t>(mesh.skinned_vertices.size());
        glm::vec3 n_half = f.normal * half;
        glm::vec3 u_half = f.u * half;
        glm::vec3 v_half = f.v * half;
        const glm::vec3 corners[4] = {
            center + n_half - u_half - v_half,
            center + n_half + u_half - v_half,
            center + n_half + u_half + v_half,
            center + n_half - u_half + v_half,
        };
        for (const auto& c : corners) {
            SkinnedVertex vert;
            vert.position = c;
            vert.normal = f.normal;
            vert.color = color;
            vert.joints[0] = joint;
            vert.weights[0] = 1.0f;
            mesh.skinned_vertices.push_back(vert);
        }
        mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
}

// Append a UV sphere (static Vertex3D geometry).
void add_sphere(Mesh& mesh, float radius, int stacks, int sectors, const glm::vec4& color) {
    for (int i = 0; i <= stacks; ++i) {
        float phi = glm::pi<float>() * static_cast<float>(i) / static_cast<float>(stacks);
        for (int j = 0; j <= sectors; ++j) {
            float theta = TWO_PI * static_cast<float>(j) / static_cast<float>(sectors);
            glm::vec3 n{std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta)};
            Vertex3D v;
            v.position = n * radius;
            v.normal = n;
            v.color = color;
            mesh.vertices.push_back(v);
        }
    }
    uint32_t ring = static_cast<uint32_t>(sectors) + 1;
    for (uint32_t i = 0; i < static_cast<uint32_t>(stacks); ++i) {
        for (uint32_t j = 0; j < static_cast<uint32_t>(sectors); ++j) {
            uint32_t a = i * ring + j;
            uint32_t b = a + ring;
            // CCW from outside (engine front face); the flat-shaded look of
            // the emissive orbs hid an inverted winding here for a long time —
            // the chrome sphere's mirror made it visible (N pointed inward).
            mesh.indices.insert(mesh.indices.end(), {a, a + 1, b, a + 1, b + 1, b});
        }
    }
}

// ---------------------------------------------------------------------------
// Skeleton helpers
// ---------------------------------------------------------------------------

int add_joint(anim::Skeleton& skel, std::vector<glm::vec3>& global_pos, const std::string& name, int parent,
              const glm::vec3& local_t) {
    anim::Joint j;
    j.name = name;
    j.parent_index = parent;
    j.local_translation = local_t;
    j.local_rotation = glm::quat(1, 0, 0, 0);
    j.local_scale = glm::vec3(1.0f);
    glm::vec3 global = parent >= 0 ? global_pos[parent] + local_t : local_t;
    // Bind pose is pure translation, so the inverse bind is a negative translate.
    j.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), -global);
    skel.joints.push_back(std::move(j));
    global_pos.push_back(global);
    skel.joint_node_indices.push_back(static_cast<int>(skel.joints.size()) - 1);
    return static_cast<int>(skel.joints.size()) - 1;
}

// ---------------------------------------------------------------------------
// Animation helpers — clips are authored by sampling closures over the cycle.
// ---------------------------------------------------------------------------

anim::AnimationChannel& new_channel(anim::AnimationClip& clip, int bone) {
    clip.channels.emplace_back();
    clip.channels.back().bone_index = bone;
    return clip.channels.back();
}

// phase runs 0..1 across the clip; the last key repeats phase 0 for clean loops.
void sample_rotation(anim::AnimationClip& clip, int bone, int keys,
                     const std::function<glm::quat(float)>& f) {
    auto& ch = new_channel(clip, bone);
    for (int i = 0; i <= keys; ++i) {
        float t = clip.duration * static_cast<float>(i) / static_cast<float>(keys);
        float phase = (i == keys) ? 0.0f : static_cast<float>(i) / static_cast<float>(keys);
        ch.rotation_times.push_back(t);
        ch.rotations.push_back(f(phase));
    }
}

void sample_position(anim::AnimationClip& clip, int bone, int keys,
                     const std::function<glm::vec3(float)>& f) {
    auto& ch = new_channel(clip, bone);
    for (int i = 0; i <= keys; ++i) {
        float t = clip.duration * static_cast<float>(i) / static_cast<float>(keys);
        float phase = (i == keys) ? 0.0f : static_cast<float>(i) / static_cast<float>(keys);
        ch.position_times.push_back(t);
        ch.positions.push_back(f(phase));
    }
}

glm::quat rot_x(float a) { return glm::angleAxis(a, glm::vec3(1, 0, 0)); }
glm::quat rot_y(float a) { return glm::angleAxis(a, glm::vec3(0, 1, 0)); }

// Joint indices (fixed by construction order in build_character_model).
enum : int {
    J_HIPS = 0,
    J_SPINE,
    J_HEAD,
    J_LEFT_UPPER_LEG,
    J_LEFT_LOWER_LEG,
    J_LEFT_FOOT,
    J_RIGHT_UPPER_LEG,
    J_RIGHT_LOWER_LEG,
    J_RIGHT_FOOT,
    J_LEFT_ARM,
    J_RIGHT_ARM,
    JOINT_COUNT
};

constexpr float HIPS_Y = 95.0f;

// A symmetric locomotion cycle: legs/arms swing in opposite phase.
anim::AnimationClip make_locomotion_clip(const std::string& name, float duration, float leg_swing, float knee_bend,
                                         float arm_swing, float bob, float forward_pitch) {
    anim::AnimationClip clip;
    clip.name = name;
    clip.duration = duration;
    const int KEYS = 16;

    auto swing = [](float amplitude, float phase_offset) {
        return [=](float p) { return rot_x(amplitude * std::sin(TWO_PI * p + phase_offset)); };
    };
    // Knee flexes once per stride, only while the leg swings backward.
    auto knee = [=](float phase_offset) {
        return [=](float p) {
            float s = std::sin(TWO_PI * p + phase_offset);
            return rot_x(knee_bend * std::max(0.0f, s));
        };
    };

    sample_rotation(clip, J_LEFT_UPPER_LEG, KEYS, swing(leg_swing, 0.0f));
    sample_rotation(clip, J_RIGHT_UPPER_LEG, KEYS, swing(leg_swing, glm::pi<float>()));
    sample_rotation(clip, J_LEFT_LOWER_LEG, KEYS, knee(glm::half_pi<float>()));
    sample_rotation(clip, J_RIGHT_LOWER_LEG, KEYS, knee(glm::half_pi<float>() + glm::pi<float>()));
    sample_rotation(clip, J_LEFT_FOOT, KEYS, swing(-leg_swing * 0.4f, 0.0f));
    sample_rotation(clip, J_RIGHT_FOOT, KEYS, swing(-leg_swing * 0.4f, glm::pi<float>()));
    sample_rotation(clip, J_LEFT_ARM, KEYS, swing(arm_swing, glm::pi<float>()));
    sample_rotation(clip, J_RIGHT_ARM, KEYS, swing(arm_swing, 0.0f));
    sample_rotation(clip, J_SPINE, KEYS, [=](float p) {
        return rot_x(forward_pitch) * rot_y(0.08f * std::sin(TWO_PI * 2.0f * p));
    });
    // Hips bob twice per stride (once per footfall).
    sample_position(clip, J_HIPS, KEYS, [=](float p) {
        return glm::vec3(0.0f, HIPS_Y + bob * 0.5f * (1.0f - std::cos(TWO_PI * 2.0f * p)), 0.0f);
    });
    return clip;
}

anim::AnimationClip make_idle_clip() {
    anim::AnimationClip clip;
    clip.name = "Idle";
    clip.duration = 3.0f;
    const int KEYS = 16;
    sample_rotation(clip, J_SPINE, KEYS,
                    [](float p) { return rot_x(0.02f + 0.04f * std::sin(TWO_PI * p)); });
    sample_rotation(clip, J_LEFT_ARM, KEYS,
                    [](float p) { return rot_x(0.06f * std::sin(TWO_PI * p + 0.4f)); });
    sample_rotation(clip, J_RIGHT_ARM, KEYS,
                    [](float p) { return rot_x(0.06f * std::sin(TWO_PI * p + 2.1f)); });
    sample_position(clip, J_HIPS, KEYS, [](float p) {
        return glm::vec3(0.0f, HIPS_Y + 1.5f * std::sin(TWO_PI * p), 0.0f);
    });
    return clip;
}

// Overhead swing with the right arm; explicit keys, non-looping.
anim::AnimationClip make_attack_clip() {
    anim::AnimationClip clip;
    clip.name = "Attack";
    clip.duration = 0.45f;

    auto& arm = new_channel(clip, J_RIGHT_ARM);
    arm.rotation_times = {0.0f, 0.12f, 0.24f, 0.45f};
    arm.rotations = {rot_x(0.0f), rot_x(-2.2f), rot_x(1.1f), rot_x(0.0f)};

    auto& spine = new_channel(clip, J_SPINE);
    spine.rotation_times = {0.0f, 0.12f, 0.24f, 0.45f};
    spine.rotations = {rot_y(0.0f), rot_y(-0.35f), rot_y(0.30f), rot_y(0.0f)};
    return clip;
}

} // namespace

// ---------------------------------------------------------------------------

std::unique_ptr<Model> build_character_model() {
    auto model = std::make_unique<Model>();

    // Skeleton — parents strictly before children (AnimationPlayer requirement).
    // Names match the engine's default FootIKBoneNames so foot IK binds.
    std::vector<glm::vec3> global;
    add_joint(model->skeleton, global, "Hips", -1, {0, HIPS_Y, 0});          // 0
    add_joint(model->skeleton, global, "Spine", J_HIPS, {0, 35, 0});         // 1  (y 130)
    add_joint(model->skeleton, global, "Head", J_SPINE, {0, 32, 0});         // 2  (y 162)
    add_joint(model->skeleton, global, "LeftUpperLeg", J_HIPS, {-12, -5, 0});  // 3  (y 90)
    add_joint(model->skeleton, global, "LeftLowerLeg", J_LEFT_UPPER_LEG, {0, -45, 0});  // 4  (y 45)
    add_joint(model->skeleton, global, "LeftFoot", J_LEFT_LOWER_LEG, {0, -40, 0});      // 5  (y 5)
    add_joint(model->skeleton, global, "RightUpperLeg", J_HIPS, {12, -5, 0}); // 6
    add_joint(model->skeleton, global, "RightLowerLeg", J_RIGHT_UPPER_LEG, {0, -45, 0}); // 7
    add_joint(model->skeleton, global, "RightFoot", J_RIGHT_LOWER_LEG, {0, -40, 0});     // 8
    add_joint(model->skeleton, global, "LeftArm", J_SPINE, {-24, 22, 0});    // 9  (y 152)
    add_joint(model->skeleton, global, "RightArm", J_SPINE, {24, 22, 0});    // 10
    model->has_skeleton = true;
    model->foot_ik.init(model->skeleton);

    // Body — one box per joint, bind-pose positions in model space, feet at
    // y=0. Villager palette: russet wool tunic, undyed hose, weathered boots.
    Mesh mesh;
    mesh.is_skinned = true;
    const glm::vec4 body{0.16f, 0.30f, 0.15f, 1.0f}; // dyed-wool green, can't read as skin
    const glm::vec4 dark{0.28f, 0.24f, 0.19f, 1.0f};
    const glm::vec4 head{0.86f, 0.72f, 0.52f, 1.0f};
    add_skinned_box(mesh, {0, 97, 0}, {14, 10, 8}, J_HIPS, dark);
    add_skinned_box(mesh, {0, 134, 0}, {15, 20, 9}, J_SPINE, body);
    add_skinned_box(mesh, {0, 172, 0}, {9, 10, 9}, J_HEAD, head);
    add_skinned_box(mesh, {-12, 68, 0}, {5.5f, 23, 5.5f}, J_LEFT_UPPER_LEG, dark);
    add_skinned_box(mesh, {12, 68, 0}, {5.5f, 23, 5.5f}, J_RIGHT_UPPER_LEG, dark);
    add_skinned_box(mesh, {-12, 25, 0}, {5, 21, 5}, J_LEFT_LOWER_LEG, dark);
    add_skinned_box(mesh, {12, 25, 0}, {5, 21, 5}, J_RIGHT_LOWER_LEG, dark);
    add_skinned_box(mesh, {-12, 4, -4}, {5, 4, 9}, J_LEFT_FOOT, dark);
    add_skinned_box(mesh, {12, 4, -4}, {5, 4, 9}, J_RIGHT_FOOT, dark);
    add_skinned_box(mesh, {-26, 128, 0}, {4.5f, 25, 4.5f}, J_LEFT_ARM, body);
    add_skinned_box(mesh, {26, 128, 0}, {4.5f, 25, 4.5f}, J_RIGHT_ARM, body);
    mesh.material.roughness_factor = 0.8f;
    model->meshes.push_back(std::move(mesh));

    // Clips — indices bound by name via AnimationStateMachine::bind_clips.
    model->animations.push_back(make_idle_clip());
    model->animations.push_back(make_locomotion_clip("Walk", 0.9f, 0.55f, 0.5f, 0.45f, 4.0f, 0.03f));
    model->animations.push_back(make_locomotion_clip("Run", 0.55f, 0.85f, 0.9f, 0.7f, 8.0f, 0.14f));
    model->animations.push_back(make_attack_clip());

    // Set by hand, not left for register_model to scan: this is a SKINNED
    // model, so the vertex AABB is the bind pose and an animation that reaches
    // outside it would be frustum-culled. These carry the headroom.
    model->bounds = {{-31, 0, -18}, {31, 182, 18}};
    return model;
}

std::unique_ptr<Model> build_orb_model() {
    auto model = std::make_unique<Model>();
    Mesh mesh;
    add_sphere(mesh, 8.0f, 10, 14, {1.0f, 0.85f, 0.45f, 1.0f});
    mesh.material.roughness_factor = 0.35f;
    // HDR emissive so the orbs feed the bloom pass (bloom_threshold defaults to 1.1).
    mesh.material.emissive_factor = {2.4f, 1.7f, 0.55f};
    // The orb is a light BULB: its point light sits at the mesh center, so if
    // this shell cast shadows it would occlude its own light entirely
    // (local-light shadow maps render from inside it). Emissive bulb geometry
    // must opt out of shadow casting.
    mesh.cast_shadows = false;
    model->meshes.push_back(std::move(mesh));
    model->bounds = {glm::vec3(-8.0f), glm::vec3(8.0f)};
    return model;
}

std::unique_ptr<Model> build_chrome_sphere_model() {
    auto model = std::make_unique<Model>();
    Mesh mesh;
    add_sphere(mesh, 45.0f, 24, 32, {1.0f, 1.0f, 1.0f, 1.0f});
    // Polished metal: with IBL on, the ambient specular term mirrors the
    // prefiltered sky (mip 0 at this roughness), sun highlight included.
    mesh.material.metallic_factor = 1.0f;
    mesh.material.roughness_factor = 0.06f;
    model->meshes.push_back(std::move(mesh));
    model->bounds = {glm::vec3(-45.0f), glm::vec3(45.0f)};
    return model;
}

std::unique_ptr<Model> build_shard_model() {
    auto model = std::make_unique<Model>();
    Mesh mesh;
    add_sphere(mesh, 2.5f, 4, 6, {1.0f, 0.75f, 0.35f, 1.0f});
    mesh.material.emissive_factor = {2.0f, 1.2f, 0.4f};
    model->meshes.push_back(std::move(mesh));
    model->bounds = {glm::vec3(-2.5f), glm::vec3(2.5f)};
    return model;
}

std::unique_ptr<Model> build_rain_streak_model() {
    auto model = std::make_unique<Model>();
    Mesh mesh;
    // Two crossed vertical quads, each emitted with both windings so the
    // streak is visible from every yaw without a double-sided pipeline.
    // Centered on the origin; the fall direction is baked by the effect.
    constexpr float HALF_W = 0.9f;
    constexpr float HALF_H = 26.0f;
    const glm::vec4 color{0.42f, 0.52f, 0.72f, 1.0f}; // dim: particles blend additively
    const glm::vec3 axes[2] = {{1, 0, 0}, {0, 0, 1}};
    for (const glm::vec3& axis : axes) {
        const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        const glm::vec3 corners[4] = {
            -axis * HALF_W + glm::vec3(0, -HALF_H, 0),
            axis * HALF_W + glm::vec3(0, -HALF_H, 0),
            axis * HALF_W + glm::vec3(0, HALF_H, 0),
            -axis * HALF_W + glm::vec3(0, HALF_H, 0),
        };
        const glm::vec3 normal = glm::cross(axis, glm::vec3(0, 1, 0));
        for (const glm::vec3& c : corners) {
            Vertex3D v;
            v.position = c;
            v.normal = normal;
            v.color = color;
            mesh.vertices.push_back(v);
        }
        mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        mesh.indices.insert(mesh.indices.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
    }
    mesh.cast_shadows = false;
    model->meshes.push_back(std::move(mesh));
    model->bounds = {{-HALF_W, -HALF_H, -HALF_W}, {HALF_W, HALF_H, HALF_W}};
    return model;
}

std::unique_ptr<Model> build_splash_drop_model() {
    auto model = std::make_unique<Model>();
    Mesh mesh;
    add_sphere(mesh, 1.6f, 3, 5, {0.55f, 0.65f, 0.85f, 1.0f});
    mesh.cast_shadows = false;
    model->meshes.push_back(std::move(mesh));
    model->bounds = {glm::vec3(-1.6f), glm::vec3(1.6f)};
    return model;
}

std::unique_ptr<Model> build_smoke_puff_model() {
    auto model = std::make_unique<Model>();
    Mesh mesh;
    // Low-poly so the faceting reads as billowing; the effect scales it up
    // over each particle's lifetime.
    add_sphere(mesh, 9.0f, 5, 7, {0.74f, 0.72f, 0.70f, 1.0f});
    mesh.material.roughness_factor = 1.0f;
    // Smoke should never write into the shadow atlas or the sun cascades.
    mesh.cast_shadows = false;
    model->meshes.push_back(std::move(mesh));
    model->bounds = {glm::vec3(-9.0f), glm::vec3(9.0f)};
    return model;
}

} // namespace playground
