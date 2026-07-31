// PipelineRegistry pre-creation contract tests.
//
// Real pipeline creation needs a GPU device + window, which unit tests do not
// have — the smoke test covers that path. What CAN be pinned here without a
// device is the enumeration logic the pre-creation contract rests on:
// preload_all_pipelines() walks [0, Count) of both enums, so the static
// enumeration being complete is exactly what guarantees get_pipeline() never
// hits the guarded creation path on the hot path. The creation_count() hook
// additionally lets us verify that no creation is ever attempted through an
// uninitialized registry.

#include "bracken/gpu/pipeline_registry.hpp"

#include "bracken/core/assert.hpp"

#include <gtest/gtest.h>
#include <set>
#include <string>

namespace bracken::gpu {
namespace {

TEST(PipelineEnumeration, EveryGraphicsTypeHasAName) {
    // pipeline_type_to_string is maintained alongside the creation switch in
    // create_pipeline_uncached — a new PipelineType that returns "Unknown"
    // here is a type the enumeration (and therefore preload) forgot.
    std::set<std::string> names;
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        auto type = static_cast<PipelineType>(i);
        std::string name = pipeline_type_to_string(type);
        EXPECT_NE(name, "Unknown") << "PipelineType " << i << " missing from the static enumeration";
        // Names must be distinct, otherwise the table has a copy-paste hole.
        EXPECT_TRUE(names.insert(name).second) << "duplicate pipeline name: " << name;
    }
    EXPECT_EQ(names.size(), static_cast<size_t>(PipelineType::Count));
}

TEST(PipelineEnumeration, EveryComputeTypeHasAName) {
    std::set<std::string> names;
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        auto type = static_cast<ComputePipelineType>(i);
        std::string name = compute_pipeline_type_to_string(type);
        EXPECT_NE(name, "Unknown") << "ComputePipelineType " << i << " missing from the static enumeration";
        EXPECT_TRUE(names.insert(name).second) << "duplicate compute pipeline name: " << name;
    }
    EXPECT_EQ(names.size(), static_cast<size_t>(ComputePipelineType::Count));
}

TEST(PipelineEnumeration, OutOfRangeTypesAreUnknown) {
    EXPECT_STREQ(pipeline_type_to_string(PipelineType::Count), "Unknown");
    EXPECT_STREQ(compute_pipeline_type_to_string(ComputePipelineType::Count), "Unknown");
}

TEST(PipelineRegistryContract, UninitializedRegistryNeverCreates) {
    // These tests run on the process main thread; the main-thread asserts in
    // the guarded creation path bind to it on first use.
    core::bind_main_thread();

    PipelineRegistry registry;
    EXPECT_FALSE(registry.is_initialized());
    EXPECT_FALSE(registry.is_preloaded());
    EXPECT_EQ(registry.creation_count(), 0u);
    EXPECT_EQ(registry.cached_pipeline_count(), 0u);

    // A representative full get_pipeline sweep: without a device every call
    // returns nullptr from the guard, and — the actual contract under test —
    // the creation counter proves the create path was never entered.
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        EXPECT_EQ(registry.get_pipeline(static_cast<PipelineType>(i)), nullptr);
    }
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        EXPECT_EQ(registry.get_compute_pipeline(static_cast<ComputePipelineType>(i)), nullptr);
    }
    EXPECT_EQ(registry.creation_count(), 0u);
    EXPECT_EQ(registry.cached_pipeline_count(), 0u);
}

TEST(PipelineRegistryContract, PreloadBeforeInitFailsWithoutCreating) {
    core::bind_main_thread();

    PipelineRegistry registry;
    EXPECT_FALSE(registry.preload_all_pipelines());
    // Preload must not flip the preloaded latch (or create anything) when the
    // registry has no device — otherwise a later real preload would be
    // misreported as an enumeration gap.
    EXPECT_FALSE(registry.is_preloaded());
    EXPECT_EQ(registry.creation_count(), 0u);
    EXPECT_EQ(registry.cached_pipeline_count(), 0u);
}

TEST(PipelineRegistryContract, ShutdownWithoutInitIsIdempotent) {
    core::bind_main_thread();

    PipelineRegistry registry;
    registry.shutdown();
    registry.shutdown();
    EXPECT_FALSE(registry.is_initialized());
    EXPECT_EQ(registry.creation_count(), 0u);
}

} // namespace
} // namespace bracken::gpu
