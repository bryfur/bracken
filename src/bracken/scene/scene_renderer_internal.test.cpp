#include "scene_renderer_internal.hpp"

#include "bracken/gpu/gpu_uniforms.hpp"
#include <gtest/gtest.h>

namespace bracken::scene {
namespace {

TEST(RtReflectionMaterialGate, OpaqueMaterialReceivesFrameHandoverWindow) {
    Mesh mesh;
    mesh.material.alpha_mode = AlphaMode::Opaque;
    gpu::ModelLightingUniforms u{};

    apply_mesh_material(u, mesh, 0.55f, 0.25f);

    EXPECT_FLOAT_EQ(u.rtReflectionMaxRoughness, 0.55f);
    EXPECT_FLOAT_EQ(u.rtReflectionFadeStart, 0.25f);
}

TEST(RtReflectionMaterialGate, MaskAndBlendKeepTheirIblSpecular) {
    for (AlphaMode mode : {AlphaMode::Mask, AlphaMode::Blend}) {
        Mesh mesh;
        mesh.material.alpha_mode = mode;
        gpu::ModelLightingUniforms u{};

        apply_mesh_material(u, mesh, 0.55f, 0.25f);

        EXPECT_FLOAT_EQ(u.rtReflectionMaxRoughness, 0.0f);
        EXPECT_FLOAT_EQ(u.rtReflectionFadeStart, 0.0f);
    }
}

} // namespace
} // namespace bracken::scene
