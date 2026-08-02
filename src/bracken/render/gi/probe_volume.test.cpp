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
    EXPECT_EQ(volume.irradiance(), nullptr);
    EXPECT_EQ(volume.visibility(), nullptr);
    EXPECT_EQ(volume.probe_state(), nullptr);
    EXPECT_EQ(volume.probe_offset(), nullptr);
    EXPECT_EQ(volume.linear_sampler(), nullptr);
    EXPECT_EQ(volume.point_sampler(), nullptr);
    // Same for the update pass: no device, no pipelines, nothing to trace.
    volume.trace(nullptr, ProbeUpdateParams{});
    EXPECT_EQ(volume.stats().rays_this_frame, 0u);
    EXPECT_EQ(volume.stats().probes_updated, 0u);
    // update() on a volume that was never init()ed is a no-op, not a null
    // deref: SceneRenderer only calls it when the field exists, but a host
    // driving one directly must not be able to fault the engine with it.
    volume.update(nullptr, glm::vec3(0.0f));
    EXPECT_EQ(volume.stats().probes_invalidated, 0u);
}

TEST(ProbeVolume, CascadeAccessorClampsRatherThanFaults) {
    ProbeVolume volume;
    // Every cascade lives in one banded texture since R1c, so the accessors no
    // longer take an index at all — what is left to clamp is the CPU-side
    // scroll state the dev overlay reads.
    for (const glm::ivec3& base : volume.cascade_bases()) {
        EXPECT_EQ(base, glm::ivec3(0));
    }
    // cascade() saturates so a dev-overlay slider cannot walk off the array.
    EXPECT_EQ(volume.cascade(-5).cascade(), volume.cascade(0).cascade());
    EXPECT_EQ(volume.cascade(999).cascade(), volume.cascade(CASCADE_COUNT - 1).cascade());
}

// The memory story, pinned. docs/architecture.md and the [smoke] gi line both
// quote these; docs/rt_lighting_plan.md §6 budgets ~35 MB for the field.
TEST(ProbeVolume, AtlasFootprintMatchesTheDocumentedFigure) {
    // Per CASCADE BAND — the textures themselves are CASCADE_COUNT times these.
    const std::size_t irradiance = static_cast<std::size_t>(atlas_width(IRRADIANCE_PROBE_TEXELS)) *
                                   static_cast<std::size_t>(atlas_band_height(IRRADIANCE_PROBE_TEXELS)) * 8;
    const std::size_t visibility = static_cast<std::size_t>(atlas_width(VISIBILITY_PROBE_TEXELS)) *
                                   static_cast<std::size_t>(atlas_band_height(VISIBILITY_PROBE_TEXELS)) * 4;
    const std::size_t state = static_cast<std::size_t>(PROBES_PER_ATLAS_ROW) *
                              static_cast<std::size_t>(ATLAS_ROWS) * 1;
    // R1b's addition: one RGBA16F texel per probe carrying the relocation
    // offset (xyz) and the classification verdict (w).
    const std::size_t offset = static_cast<std::size_t>(PROBES_PER_ATLAS_ROW) *
                               static_cast<std::size_t>(ATLAS_ROWS) * 8;

    EXPECT_EQ(irradiance, 3538944u); // RGBA16F, 3.375 MB
    EXPECT_EQ(visibility, 7077888u); // RG16F,   6.75 MB
    EXPECT_EQ(state, 6912u);         // R8_UINT, one texel per probe
    EXPECT_EQ(offset, 55296u);       // RGBA16F, one texel per probe

    const std::size_t per_cascade = irradiance + visibility + state + offset;
    const std::size_t total = per_cascade * CASCADE_COUNT;
    EXPECT_EQ(per_cascade, 10679040u);
    EXPECT_EQ(total, 32037120u); // ~30.6 MB for the whole field
    EXPECT_LT(total, 35u * 1024u * 1024u) << "the field outgrew the plan's ~35 MB budget";

    // R1c banded the cascades into one texture per quantity. The TEXELS are
    // identical — the same probes in the same tiling, stacked — so the total
    // above must be reproducible from the whole-texture dimensions too, which
    // is what proves the repack moved no memory.
    const std::size_t banded =
        static_cast<std::size_t>(atlas_width(IRRADIANCE_PROBE_TEXELS)) *
            static_cast<std::size_t>(atlas_height(IRRADIANCE_PROBE_TEXELS)) * 8 +
        static_cast<std::size_t>(atlas_width(VISIBILITY_PROBE_TEXELS)) *
            static_cast<std::size_t>(atlas_height(VISIBILITY_PROBE_TEXELS)) * 4 +
        static_cast<std::size_t>(PROBES_PER_ATLAS_ROW) *
            static_cast<std::size_t>(ATLAS_ROWS_ALL_CASCADES) * (1 + 8);
    EXPECT_EQ(banded, total);
}

// The transient buffers the update passes need, which are NOT part of the
// field's resident footprint but are allocated alongside it.
TEST(ProbeVolume, TraceBufferFootprintIsBoundedByTheWorstQuality) {
    const std::size_t rays = static_cast<std::size_t>(gi_max_rays_per_dispatch()) * 16; // float4
    const std::size_t lists = static_cast<std::size_t>(gi_max_probes_per_dispatch()) * 32 * CASCADE_COUNT;
    EXPECT_EQ(rays, 2654208u); // 2.53 MB — one cascade's worst-case turn
    EXPECT_EQ(lists, 165888u); // 162 KB — one update list per cascade
    EXPECT_LT(rays + lists, 4u * 1024u * 1024u);
}
