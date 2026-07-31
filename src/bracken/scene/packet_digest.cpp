#include "bracken/scene/packet_digest.hpp"

#include "bracken/contracts/camera_state.hpp"
#include "bracken/contracts/environment.hpp"
#include "bracken/contracts/light.hpp"
#include "bracken/effects/effect_definition.hpp"
#include "bracken/scene/frame_packet.hpp"

#include <bit>
#include <cstring>
#include <string_view>

namespace bracken::scene {
namespace {

// FNV-1a 64. Chosen over a stronger hash deliberately: the digest is a
// change-detector for a regression gate, not a security primitive, and a
// short, dependency-free, trivially re-implementable mixer keeps the gate
// something a human can reason about and reproduce by hand if they need to.
constexpr uint64_t k_fnv_offset = 1469598103934665603ULL;
constexpr uint64_t k_fnv_prime = 1099511628211ULL;

void hash_bytes(uint64_t& h, const void* data, std::size_t len) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= k_fnv_prime;
    }
}

// Floats go in by bit pattern, never by value: that keeps -0.0 distinct from
// +0.0 and makes the digest stable regardless of NaN payloads, both of which a
// value comparison would blur.
void hash_f32(uint64_t& h, float v) {
    const uint32_t bits = std::bit_cast<uint32_t>(v);
    hash_bytes(h, &bits, sizeof(bits));
}

void hash_u32(uint64_t& h, uint32_t v) { hash_bytes(h, &v, sizeof(v)); }

void hash_floats(uint64_t& h, const float* v, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        hash_f32(h, v[i]);
    }
}

// A struct made purely of 4-byte members has no padding, so reading it as a
// float run is exact and order-stable. Each caller static_asserts its size, so
// adding a member of a different width turns into a compile error here rather
// than a silently weakened gate.
template <typename T> void hash_pod_floats(uint64_t& h, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) % sizeof(float) == 0);
    alignas(T) unsigned char buf[sizeof(T)];
    std::memcpy(buf, &v, sizeof(T));
    const float* f = std::bit_cast<const float*>(&buf[0]);
    hash_floats(h, f, sizeof(T) / sizeof(float));
}

void hash_str(uint64_t& h, std::string_view s) {
    hash_u32(h, static_cast<uint32_t>(s.size()));
    hash_bytes(h, s.data(), s.size());
}

void hash_mat4(uint64_t& h, const glm::mat4& m) { hash_floats(h, &m[0][0], 16); }

void hash_vec4(uint64_t& h, const glm::vec4& v) { hash_floats(h, &v.x, 4); }

void hash_vec3(uint64_t& h, const glm::vec3& v) { hash_floats(h, &v.x, 3); }

} // namespace

uint64_t packet_digest(const FramePacket& packet) {
    uint64_t h = k_fnv_offset;

    // Each section is prefixed with its element count, so a section that
    // empties out is distinguishable from one that was removed entirely.
    hash_u32(h, static_cast<uint32_t>(packet.model_commands().size()));
    for (const auto& c : packet.model_commands()) {
        hash_u32(h, c.model_handle);
        hash_u32(h, c.motion_id);
        hash_mat4(h, c.transform);
        hash_vec4(h, c.tint);
        // Bit by named bit, not the packed word: DrawFlags replaced two bools
        // whose 0/1 words are what the pre-flags baseline digest was taken
        // over, and a re-encoding of the same frame must not move the gate.
        // A NEW flag is therefore invisible here until it is added below.
        hash_u32(h, has(c.flags, DrawFlags::ForceNonInstanced) ? 1u : 0u);
        hash_u32(h, has(c.flags, DrawFlags::NoFog) ? 1u : 0u);
    }

    hash_u32(h, static_cast<uint32_t>(packet.skinned_commands().size()));
    for (const auto& c : packet.skinned_commands()) {
        hash_u32(h, c.model_handle);
        hash_u32(h, c.motion_id);
        hash_mat4(h, c.transform);
        hash_vec4(h, c.tint);
        hash_u32(h, c.palette.value);
        hash_u32(h, has(c.flags, DrawFlags::NoFog) ? 1u : 0u);
    }

    // Palette CONTENTS, not the arena addresses they live at: the pointer is
    // packet memory and varies per run, the matrices are the actual pose.
    hash_u32(h, static_cast<uint32_t>(packet.palettes().size()));
    for (const auto& p : packet.palettes()) {
        hash_u32(h, p.joint_count);
        hash_u32(h, p.matrix_offset);
        if (p.matrices != nullptr) {
            hash_floats(h, &p.matrices[0][0][0], std::size_t{p.joint_count} * 16);
        }
    }

    hash_u32(h, static_cast<uint32_t>(packet.billboards().size()));
    for (const auto& b : packet.billboards()) {
        static_assert(sizeof(BillboardDrawCommand) == 8 * sizeof(float));
        hash_pod_floats(h, b);
    }

    hash_u32(h, static_cast<uint32_t>(packet.debug_lines().size()));
    for (const auto& d : packet.debug_lines()) {
        hash_vec3(h, d.start);
        hash_vec3(h, d.end);
        hash_u32(h, d.color.argb);
    }

    // Effect spawns carry a host-owned definition POINTER. Addresses move with
    // ASLR on every launch, so the asset's name is what gets hashed.
    hash_u32(h, static_cast<uint32_t>(packet.effect_spawns().size()));
    for (const auto& e : packet.effect_spawns()) {
        hash_str(h, e.definition != nullptr ? std::string_view{e.definition->name} : std::string_view{});
        hash_vec3(h, e.position);
        hash_vec3(h, e.direction);
    }

    hash_u32(h, static_cast<uint32_t>(packet.point_lights().size()));
    for (const auto& l : packet.point_lights()) {
        hash_vec3(h, l.position);
        hash_f32(h, l.radius);
        hash_vec3(h, l.color);
        hash_f32(h, l.intensity);
        hash_u32(h, l.casts_shadows ? 1u : 0u);
        // shadow_slot is engine-written during render prepare, i.e. AFTER the
        // digest is taken; it is excluded so the value cannot depend on
        // whether the packet has been rendered yet.
    }

    hash_u32(h, static_cast<uint32_t>(packet.spot_lights().size()));
    for (const auto& l : packet.spot_lights()) {
        hash_vec3(h, l.position);
        hash_f32(h, l.radius);
        hash_vec3(h, l.direction);
        hash_f32(h, l.inner_cos);
        hash_vec3(h, l.color);
        hash_f32(h, l.outer_cos);
        hash_f32(h, l.intensity);
        hash_u32(h, l.casts_shadows ? 1u : 0u);
    }

    hash_u32(h, static_cast<uint32_t>(packet.ui_commands().size()));
    for (const auto& c : packet.ui_commands()) {
        // The variant index participates: two shapes with identical numbers
        // are still different commands.
        hash_u32(h, static_cast<uint32_t>(c.data.index()));
        std::visit(
            [&h](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                if constexpr (std::is_same_v<T, UiTextCommand>) {
                    hash_str(h, cmd.text);
                    hash_f32(h, cmd.x);
                    hash_f32(h, cmd.y);
                    hash_f32(h, cmd.scale);
                    hash_u32(h, cmd.color.argb);
                } else if constexpr (std::is_same_v<T, UiButtonCommand>) {
                    hash_str(h, cmd.label);
                    hash_f32(h, cmd.x);
                    hash_f32(h, cmd.y);
                    hash_f32(h, cmd.w);
                    hash_f32(h, cmd.h);
                    hash_u32(h, cmd.color.argb);
                    hash_u32(h, cmd.selected ? 1u : 0u);
                } else {
                    // The remaining UI commands are pure 4-byte-member PODs.
                    hash_pod_floats(h, cmd);
                }
            },
            c.data);
    }

    hash_u32(h, packet.has_camera() ? 1u : 0u);
    if (packet.has_camera()) {
        static_assert(sizeof(CameraState) == 51 * sizeof(float));
        hash_pod_floats(h, packet.camera());
    }

    hash_u32(h, packet.has_environment() ? 1u : 0u);
    if (packet.has_environment()) {
        // 16 float + 13 glm::vec3 = 55 floats, all 4-byte aligned, no padding.
        // If this fires, a member of another width was added: hash it
        // explicitly rather than relaxing the assert.
        static_assert(sizeof(render::EnvironmentState) == 55 * sizeof(float));
        hash_pod_floats(h, packet.environment());
    }

    hash_u32(h, packet.should_draw_skybox() ? 1u : 0u);
    hash_u32(h, packet.should_draw_ground() ? 1u : 0u);
    hash_u32(h, packet.should_draw_grass() ? 1u : 0u);

    // Engine sim-tail section: the effect-instance snapshot the renderer
    // actually draws, plus the sim-advanced clocks.
    hash_u32(h, packet.has_engine_frame() ? 1u : 0u);
    if (packet.has_engine_frame()) {
        hash_u32(h, static_cast<uint32_t>(packet.effect_emitters().size()));
        for (const auto& e : packet.effect_emitters()) {
            hash_str(h, e.definition != nullptr ? std::string_view{e.definition->name} : std::string_view{});
            hash_u32(h, e.first_particle);
            hash_u32(h, e.particle_count);
        }
        hash_u32(h, static_cast<uint32_t>(packet.effect_particles().size()));
        for (const auto& p : packet.effect_particles()) {
            static_assert(sizeof(EffectParticleSnapshot) == 12 * sizeof(float));
            hash_pod_floats(h, p);
        }
        hash_f32(h, packet.scene_time());
        hash_f32(h, packet.sky_time());
    }

    return h;
}


} // namespace bracken::scene
