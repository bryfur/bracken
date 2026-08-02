#pragma once

// Shared implementation details of the SceneRenderer translation units.
//
// SceneRenderer's implementation is split across six .cpp files —
// scene_renderer.cpp (lifecycle + the render_frame script),
// scene_renderer_upload.cpp, scene_renderer_prepass.cpp,
// scene_renderer_draw.cpp, scene_renderer_shadow.cpp and
// scene_renderer_capture.cpp. This header carries what more than one of them
// needs and no host ever should. It is included ONLY by those files.
//
// It deliberately does NOT hold the class's private member declarations:
// those are member declarations and C++ requires them inside the class
// definition, so scene_renderer.hpp keeps them (see docs/refactor_plan.md
// Phase 6 — the plan's wording on that point is not implementable).

#include "bracken/model_loader.hpp"

namespace bracken::scene {

// One log tag for every SceneRenderer translation unit (the logger-discipline
// test requires BRACKEN_LOG_* with a tag; keeping the spelling single keeps
// the channel greppable).
inline constexpr const char* LOG_TAG = "scene";

// Fill the per-mesh material fields shared by ModelLightingUniforms and
// InstancedLightingUniforms (identical field names in both structs).
//
// Called from the draw, prepass and shadow TUs, so it cannot stay TU-local.
// KNOWN, ACCEPTED: this weakens what used to be internal linkage
// (`static inline` in scene_renderer.cpp) to an inline function with external
// linkage. That is the price of the split; nothing else about it changed.
template <typename U>
inline void apply_mesh_material(U& u, const Mesh& mesh, float rt_reflection_max_roughness,
                                float rt_reflection_fade_start) {
    const PBRMaterial& mat = mesh.material;
    u.hasTexture = (mesh.has_texture && mesh.texture) ? 1 : 0;
    u.hasNormalMap = mat.normal_texture ? 1 : 0;
    u.normalScale = mat.normal_scale;
    u.baseColorFactor = mat.base_color_factor;
    u.metallicFactor = mat.metallic_factor;
    u.roughnessFactor = mat.roughness_factor;
    u.occlusionStrength = mat.occlusion_strength;
    u.hasMRMap = mat.metallic_roughness_texture ? 1.0f : 0.0f;
    u.emissiveFactor = mat.emissive_factor;
    u.hasEmissiveMap = mat.emissive_texture ? 1.0f : 0.0f;
    u.hasPackedOcclusion = mat.has_packed_occlusion ? 1.0f : 0.0f;
    // glTF alphaMode: only MASK materials alpha-test; OPAQUE/BLEND must not
    // discard, which a cutoff of 0 guarantees in the shader.
    u.alphaCutoff = (mat.alpha_mode == AlphaMode::Mask) ? mat.alpha_cutoff : 0.0f;
    // Screen-space AO comes from the depth prepass. OPAQUE meshes are always
    // in it, and MASK meshes join via the alpha-tested prepass variants (so
    // foliage both receives and casts screen AO — leaf-on-leaf occlusion is
    // what separates a canopy from a flat blob). BLEND meshes write no
    // prepass depth, so the AO texel under them describes the geometry
    // behind them; they skip the AO term. Skinned MASK meshes are the one
    // gap (no masked skinned prepass variant) — they'd sample the AO of
    // whatever is behind them, which for a moving character is benign.
    u.screenAOEnabled = (mat.alpha_mode != AlphaMode::Blend) ? 1.0f : 0.0f;
    // The reflection prepass deliberately marks MASK foliage fully rough and
    // omits BLEND geometry. Their forward draws must therefore keep the full
    // IBL specular lobe; handing it off to a trace that never ran creates dark
    // foliage. OPAQUE meshes get the frame's exact handover window.
    const bool rt_eligible = mat.alpha_mode == AlphaMode::Opaque;
    u.rtReflectionMaxRoughness = rt_eligible ? rt_reflection_max_roughness : 0.0f;
    u.rtReflectionFadeStart = rt_eligible ? rt_reflection_fade_start : 0.0f;
}

} // namespace bracken::scene
