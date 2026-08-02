// ProbeVolume without a GPU.
//
// The interesting half (addressing, scrolling, invalidation) is covered by
// gi_math.test.cpp against the device-free ProbeCascade. What is left here is
// the owner's contract: an uninitialized volume must be inert rather than
// crash, and the memory figure the smoke line and the plan's budget both quote
// must be derivable from the constants rather than being a number someone typed.

#include "bracken/render/gi/probe_volume.hpp"

#include <gtest/gtest.h>

using namespace bracken::render::gi;

TEST(ProbeVolume, UninitializedIsInert) {
    ProbeVolume volume;
    EXPECT_FALSE(volume.is_ready());
    EXPECT_FALSE(volume.stats().active);
    EXPECT_EQ(volume.stats().probes, 0u);
    EXPECT_EQ(volume.stats().atlas_bytes, 0u);
    EXPECT_EQ(volume.irradiance(0), nullptr);
    EXPECT_EQ(volume.visibility(0), nullptr);
    EXPECT_EQ(volume.probe_state(0), nullptr);
    // update() on a volume that was never init()ed is a no-op, not a null
    // deref: SceneRenderer only calls it when the field exists, but a host
    // driving one directly must not be able to fault the engine with it.
    volume.update(nullptr, glm::vec3(0.0f));
    EXPECT_EQ(volume.stats().probes_invalidated, 0u);
}

TEST(ProbeVolume, CascadeAccessorClampsRatherThanFaults) {
    ProbeVolume volume;
    EXPECT_EQ(volume.irradiance(-1), nullptr);
    EXPECT_EQ(volume.visibility(CASCADE_COUNT), nullptr);
    EXPECT_EQ(volume.probe_state(999), nullptr);
    // cascade() saturates so a dev-overlay slider cannot walk off the array.
    EXPECT_EQ(volume.cascade(-5).cascade(), volume.cascade(0).cascade());
    EXPECT_EQ(volume.cascade(999).cascade(), volume.cascade(CASCADE_COUNT - 1).cascade());
}

// The memory story, pinned. docs/architecture.md and the [smoke] gi line both
// quote these; docs/rt_lighting_plan.md §6 budgets ~35 MB for the field.
TEST(ProbeVolume, AtlasFootprintMatchesTheDocumentedFigure) {
    const std::size_t irradiance = static_cast<std::size_t>(atlas_width(IRRADIANCE_PROBE_TEXELS)) *
                                   static_cast<std::size_t>(atlas_height(IRRADIANCE_PROBE_TEXELS)) * 8;
    const std::size_t visibility = static_cast<std::size_t>(atlas_width(VISIBILITY_PROBE_TEXELS)) *
                                   static_cast<std::size_t>(atlas_height(VISIBILITY_PROBE_TEXELS)) * 4;
    const std::size_t state = static_cast<std::size_t>(PROBES_PER_ATLAS_ROW) *
                              static_cast<std::size_t>(ATLAS_ROWS) * 1;

    EXPECT_EQ(irradiance, 3538944u); // RGBA16F, 3.375 MB
    EXPECT_EQ(visibility, 7077888u); // RG16F,   6.75 MB
    EXPECT_EQ(state, 6912u);         // R8_UINT, one texel per probe

    const std::size_t per_cascade = irradiance + visibility + state;
    const std::size_t total = per_cascade * CASCADE_COUNT;
    EXPECT_EQ(per_cascade, 10623744u);
    EXPECT_EQ(total, 31871232u); // ~30.4 MB for the whole field
    EXPECT_LT(total, 35u * 1024u * 1024u) << "the field outgrew the plan's ~35 MB budget";
}
