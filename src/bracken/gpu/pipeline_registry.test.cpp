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
#include <utility>
#include <vector>

namespace bracken::gpu {
namespace {

TEST(PipelineEnumeration, EveryGraphicsTypeHasAName) {
    // The name comes from the type's catalogue row (gpu/pipeline_catalog.hpp),
    // so "Unknown" here means the row is missing and preload would fail on it.
    // The row's *contents* are checked in gpu/pipeline_catalog.test.cpp.
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

// =============================================================================
// Host-registered pipelines (gpu/host_pipeline.hpp)
//
// Same limitation as above — no device, so no pipeline can actually be built.
// What CAN be pinned device-free is every refusal path, the search-root
// ordering, and the fact that registering does not perturb the enumeration the
// catalogue's golden walks.
// =============================================================================

HostGraphicsPipelineDesc valid_graphics_desc(std::string name) {
    HostGraphicsPipelineDesc desc;
    desc.name = std::move(name);
    desc.vertex_shader = "host.vert.spv";
    desc.fragment_shader = "host.frag.spv";
    return desc;
}

TEST(PipelineRegistryHostContract, RegistrationWithoutADeviceReturnsInvalidAndCreatesNothing) {
    core::bind_main_thread();

    PipelineRegistry registry;
    const HostPipelineHandle graphics = registry.register_graphics_pipeline(valid_graphics_desc("host_gfx"));
    EXPECT_FALSE(graphics.is_valid());

    HostComputePipelineDesc compute_desc;
    compute_desc.name = "host_compute";
    compute_desc.shader = "host.comp.spv";
    const HostComputePipelineHandle compute = registry.register_compute_pipeline(std::move(compute_desc));
    EXPECT_FALSE(compute.is_valid());

    // Exactly the contract UninitializedRegistryNeverCreates pins for the enum
    // path: no device means no creation attempt at all, and nothing lands in
    // the shared cache.
    EXPECT_EQ(registry.creation_count(), 0u);
    EXPECT_EQ(registry.cached_pipeline_count(), 0u);
    EXPECT_EQ(registry.host_pipeline_count(), 0u);
}

TEST(PipelineRegistryHostContract, MalformedDescriptionsAreRefused) {
    core::bind_main_thread();

    PipelineRegistry registry;
    // Empty name, and a name with no shaders. Both refused before the device
    // check would even matter — asserted here through host_pipeline_count,
    // which stays zero whatever the refusal reason.
    EXPECT_FALSE(registry.register_graphics_pipeline(valid_graphics_desc("")).is_valid());
    HostGraphicsPipelineDesc no_shaders;
    no_shaders.name = "no_shaders";
    EXPECT_FALSE(registry.register_graphics_pipeline(std::move(no_shaders)).is_valid());
    HostComputePipelineDesc no_shader;
    no_shader.name = "no_shader";
    EXPECT_FALSE(registry.register_compute_pipeline(std::move(no_shader)).is_valid());
    EXPECT_EQ(registry.host_pipeline_count(), 0u);
    EXPECT_EQ(registry.creation_count(), 0u);
}

TEST(PipelineRegistryHostContract, LookupByAnInvalidOrStaleHandleIsNull) {
    core::bind_main_thread();

    PipelineRegistry registry;
    // A default-constructed handle, and a fabricated one. Both must be nullptr
    // rather than an out-of-range read into whatever the host table holds.
    EXPECT_EQ(registry.get_pipeline(HostPipelineHandle{}), nullptr);
    EXPECT_EQ(registry.get_compute_pipeline(HostComputePipelineHandle{}), nullptr);
    EXPECT_EQ(registry.get_pipeline(HostPipelineHandle{7, 3}), nullptr);
    EXPECT_EQ(registry.get_compute_pipeline(HostComputePipelineHandle{7, 3}), nullptr);
    EXPECT_EQ(registry.host_pipeline_desc(HostPipelineHandle{}), nullptr);
    EXPECT_EQ(registry.host_compute_pipeline_desc(HostComputePipelineHandle{}), nullptr);

    // Unregistering a handle that was never live is a logged no-op, not an
    // abort: a host tearing down in an unknown order is not a contract breach.
    registry.unregister_pipeline(HostPipelineHandle{});
    registry.unregister_pipeline(HostComputePipelineHandle{4, 1});
    EXPECT_EQ(registry.creation_count(), 0u);
}

TEST(PipelineRegistryHostContract, ShaderSearchPathsKeepTheEngineRootFirstAndDeduplicate) {
    core::bind_main_thread();

    PipelineRegistry registry;
    ASSERT_EQ(registry.shader_search_paths().size(), 1u);
    const std::string engine_root = registry.shader_path();
    EXPECT_EQ(registry.shader_search_paths()[0], engine_root);

    EXPECT_TRUE(registry.add_shader_search_path("shaders_game"));
    // Normalized with a trailing separator, so "dir" and "dir/" are one root.
    ASSERT_EQ(registry.shader_search_paths().size(), 2u);
    EXPECT_EQ(registry.shader_search_paths()[1], "shaders_game/");
    EXPECT_FALSE(registry.add_shader_search_path("shaders_game/"));
    EXPECT_FALSE(registry.add_shader_search_path("shaders_game"));
    EXPECT_EQ(registry.shader_search_paths().size(), 2u);

    EXPECT_FALSE(registry.add_shader_search_path("")) << "an empty root is a mistake, not a no-op";
    EXPECT_EQ(registry.shader_search_paths().size(), 2u);

    // Ordering is the whole safety property: if a host root could come first, a
    // game shipping its own model.frag.spv would replace PipelineType::Model's.
    EXPECT_TRUE(registry.add_shader_search_path("second_root"));
    ASSERT_EQ(registry.shader_search_paths().size(), 3u);
    EXPECT_EQ(registry.shader_search_paths()[0], engine_root);
    EXPECT_EQ(registry.shader_search_paths()[1], "shaders_game/");
    EXPECT_EQ(registry.shader_search_paths()[2], "second_root/");
    EXPECT_EQ(registry.shader_path(), engine_root) << "shader_path() is element 0, always";
}

TEST(PipelineRegistryHostContract, TheHostSurfaceDoesNotPerturbTheEnumeration) {
    core::bind_main_thread();

    PipelineRegistry registry;
    // Capture the catalogue's answers, exercise the whole host surface, and
    // demand the answers are unchanged. This is the registry-side statement of
    // what gpu/pipeline_catalog.test.cpp's golden depends on: the catalogue is
    // a pure function of the enums and nothing a host does can reach it.
    std::vector<std::string> graphics_names;
    std::vector<bool> targets_swapchain;
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const auto type = static_cast<PipelineType>(i);
        graphics_names.emplace_back(pipeline_type_to_string(type));
        targets_swapchain.push_back(pipeline_targets_swapchain(type));
    }
    std::vector<std::string> compute_names;
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        compute_names.emplace_back(compute_pipeline_type_to_string(static_cast<ComputePipelineType>(i)));
    }

    EXPECT_TRUE(registry.add_shader_search_path("host_shaders"));
    for (int i = 0; i < 8; ++i) {
        (void)registry.register_graphics_pipeline(valid_graphics_desc("host_" + std::to_string(i)));
    }
    registry.set_swapchain_format(SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM);

    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const auto type = static_cast<PipelineType>(i);
        EXPECT_EQ(std::string(pipeline_type_to_string(type)), graphics_names[static_cast<size_t>(i)]);
        EXPECT_EQ(pipeline_targets_swapchain(type), targets_swapchain[static_cast<size_t>(i)]);
    }
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        EXPECT_EQ(std::string(compute_pipeline_type_to_string(static_cast<ComputePipelineType>(i))),
                  compute_names[static_cast<size_t>(i)]);
    }
    EXPECT_EQ(registry.creation_count(), 0u) << "no device: nothing was built by any of the above";
    EXPECT_EQ(registry.cached_pipeline_count(), 0u);
}

} // namespace
} // namespace bracken::gpu
