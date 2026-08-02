#include "bracken/render/water_renderer.hpp"

#include <gtest/gtest.h>

namespace bracken::render {
namespace {

// The CPU uniform mirrors must match the HLSL cbuffers byte-for-byte
// (water_sim.comp.hlsl / water.vert.hlsl / water.frag.hlsl). These pins
// catch accidental reordering or padding drift on either side.

TEST(WaterUniformLayout, SimUniformsMatchShader) {
    EXPECT_EQ(alignof(WaterRenderer::SimUniforms), 16u);
    EXPECT_EQ(sizeof(WaterRenderer::SimUniforms), 160u);
    EXPECT_EQ(offsetof(WaterRenderer::SimUniforms, resolution), 0u);
    EXPECT_EQ(offsetof(WaterRenderer::SimUniforms, flowDir), 16u);
    EXPECT_EQ(offsetof(WaterRenderer::SimUniforms, impulseCount), 24u);
    EXPECT_EQ(offsetof(WaterRenderer::SimUniforms, impulses), 32u);
}

TEST(WaterUniformLayout, VertexUniformsMatchShader) {
    EXPECT_EQ(alignof(WaterRenderer::VertexUniforms), 16u);
    EXPECT_EQ(sizeof(WaterRenderer::VertexUniforms), 96u);
    EXPECT_EQ(offsetof(WaterRenderer::VertexUniforms, simWindow), 64u);
    EXPECT_EQ(offsetof(WaterRenderer::VertexUniforms, params), 80u);
}

TEST(WaterUniformLayout, FragmentUniformsMatchShader) {
    EXPECT_EQ(alignof(WaterRenderer::FragmentUniforms), 16u);
    EXPECT_EQ(sizeof(WaterRenderer::FragmentUniforms), 272u);
    EXPECT_EQ(offsetof(WaterRenderer::FragmentUniforms, screenDepthAB), 112u);
    EXPECT_EQ(offsetof(WaterRenderer::FragmentUniforms, misc), 160u);
    EXPECT_EQ(offsetof(WaterRenderer::FragmentUniforms, viewProjSSR), 176u);
    EXPECT_EQ(offsetof(WaterRenderer::FragmentUniforms, ssrParams), 240u);
    EXPECT_EQ(offsetof(WaterRenderer::FragmentUniforms, rtParams), 256u);
}

TEST(WaterGrid, SquareRegionUsesMaxCellsBothAxes) {
    EXPECT_EQ(WaterRenderer::grid_cells_for(1000.0f, 1000.0f, 160), glm::ivec2(160, 160));
}

TEST(WaterGrid, AspectScalesShorterAxis) {
    // A 1:8 river strip: the short axis gets 1/8 of the cells.
    EXPECT_EQ(WaterRenderer::grid_cells_for(1000.0f, 8000.0f, 160), glm::ivec2(20, 160));
    EXPECT_EQ(WaterRenderer::grid_cells_for(8000.0f, 1000.0f, 160), glm::ivec2(160, 20));
}

TEST(WaterGrid, DegenerateRegionsClampToOneCell) {
    EXPECT_EQ(WaterRenderer::grid_cells_for(0.0f, 1000.0f, 160), glm::ivec2(1, 160));
    EXPECT_EQ(WaterRenderer::grid_cells_for(1000.0f, 0.5f, 160), glm::ivec2(160, 1));
}

TEST(WaterSimConstants, CourantStable) {
    // The sim clamps c^2*dt^2/dx^2 at 0.24 (< 0.25 stability bound for the
    // 5-point stencil). Verify the default region tuning stays under the
    // clamp so the clamp is headroom, not the operating point.
    WaterRegion region;
    const float texel = region.sim_extent / static_cast<float>(WaterRenderer::SIM_RESOLUTION);
    const float courant = region.wave_speed * WaterRenderer::SIM_SUBSTEP / texel;
    EXPECT_LT(courant * courant, 0.24f);
}

} // namespace
} // namespace bracken::render
