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

#include "bracken/core/asset/file_watcher.hpp"
#include "bracken/core/assert.hpp"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iterator>
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

// =============================================================================
// The ADDITIVE claim, path by path.
//
// The host surface widened five shared lifecycle paths — preload,
// invalidate_all, set_swapchain_format, enable_hot_reload and shutdown — plus
// the shader-path member the catalogue resolves against. Each of those is a
// place where a host registration could, if the widening were sloppy, change
// what the ENUM path does. The three tests below are the ones that can be made
// device-free; the sixth path (create_pipeline_uncached, which needs a device)
// is covered structurally by CatalogueShaderResolutionNeverConsultsHostRoots.
// =============================================================================

TEST(PipelineRegistryLifecycle, EveryLifecyclePathIsInertWithoutADeviceOrAHost) {
    core::bind_main_thread();

    PipelineRegistry registry;
    const std::string engine_root = registry.shader_path();

    // Walk every lifecycle entry point on a registry that has neither a device
    // nor a host registration. None of them may create, cache, or latch.
    EXPECT_FALSE(registry.preload_all_pipelines());
    EXPECT_FALSE(registry.is_preloaded()) << "preload before init must not latch (see the guard in preload)";
    registry.invalidate_all();
    registry.set_swapchain_format(SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM);
    EXPECT_EQ(registry.get_swapchain_format(), SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM);
    registry.reload_all_pipelines();

    // enable_hot_reload before init is a warned no-op — it must not consume the
    // watcher, because the retry after the FileWatcher comes up is the whole
    // reason the latch is per-root rather than global.
    core::asset::FileWatcher watcher;
    registry.enable_hot_reload(watcher);

    registry.shutdown();

    EXPECT_EQ(registry.creation_count(), 0u);
    EXPECT_EQ(registry.cached_pipeline_count(), 0u);
    EXPECT_EQ(registry.host_pipeline_count(), 0u);
    EXPECT_FALSE(registry.is_preloaded());
    EXPECT_EQ(registry.shader_path(), engine_root);
    ASSERT_EQ(registry.shader_search_paths().size(), 1u) << "no host root was added, so there is exactly one";
}

TEST(PipelineRegistryLifecycle, ShutdownReleasesHostRegistrationsAndKeepsTheSearchRoots) {
    core::bind_main_thread();

    PipelineRegistry registry;
    const std::string engine_root = registry.shader_path();
    ASSERT_TRUE(registry.add_shader_search_path("game_shaders"));
    ASSERT_EQ(registry.shader_search_paths().size(), 2u);

    registry.shutdown();

    // The DECISION this pins: shutdown drops the registrations (they own GPU
    // objects that die with the device) but KEEPS the search roots (they are
    // configuration, not resources). A host that re-inits does not have to
    // re-add them, and element 0 is still the engine root — init() rewrites
    // that slot in place rather than appending, which is what stops a host root
    // ever being searched ahead of the engine's.
    EXPECT_EQ(registry.host_pipeline_count(), 0u);
    ASSERT_EQ(registry.shader_search_paths().size(), 2u);
    EXPECT_EQ(registry.shader_search_paths()[0], engine_root);
    EXPECT_EQ(registry.shader_search_paths()[1], "game_shaders/");
    EXPECT_EQ(registry.shader_path(), engine_root);
}

TEST(PipelineRegistryLifecycle, RegisteringAndUnregisteringLeavesTheCacheCountsWhereItFoundThem) {
    core::bind_main_thread();

    PipelineRegistry registry; // no device: registration is refused before it can cache
    const size_t cached_before = registry.cached_pipeline_count();
    const size_t created_before = registry.creation_count();

    std::vector<HostPipelineHandle> handles;
    for (int i = 0; i < 4; ++i) {
        handles.push_back(registry.register_graphics_pipeline(valid_graphics_desc("gfx_" + std::to_string(i))));
    }
    for (HostPipelineHandle h : handles) {
        registry.unregister_pipeline(h);
        // A stale handle must stay a nullptr answer rather than aliasing the
        // next registration into the same slot.
        EXPECT_EQ(registry.get_pipeline(h), nullptr);
        EXPECT_EQ(registry.host_pipeline_desc(h), nullptr);
    }
    EXPECT_EQ(registry.cached_pipeline_count(), cached_before);
    EXPECT_EQ(registry.creation_count(), created_before);
    EXPECT_EQ(registry.host_pipeline_count(), 0u);
}

// -----------------------------------------------------------------------------
// A source scan, because the claim is about a code path a device-free test
// cannot execute: catalogue rows resolve their shaders by concatenating
// shader_path() (search root 0, the engine's), NEVER through the host search
// roots. If that ever changed, a game shipping its own `model.frag.spv` could
// replace PipelineType::Model's — silently, and only in a shipped game.
// -----------------------------------------------------------------------------

/// Text of `pipeline_registry.cpp` between the first line containing `signature`
/// and the next line that is exactly "}" at column 0 — i.e. one function body,
/// given the file's house formatting. Empty when the signature is absent.
std::string function_body(const std::string& file, const std::string& signature) {
    const size_t start = file.find(signature);
    if (start == std::string::npos) {
        return {};
    }
    const size_t end = file.find("\n}\n", start);
    return file.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

TEST(PipelineRegistryLifecycle, CatalogueShaderResolutionNeverConsultsHostRoots) {
#ifndef BRACKEN_REPO_ROOT
    GTEST_SKIP() << "BRACKEN_REPO_ROOT not defined; cannot locate the source tree";
#else
    const std::filesystem::path path =
        std::filesystem::path(BRACKEN_REPO_ROOT) / "src" / "bracken" / "gpu" / "pipeline_registry.cpp";
    std::ifstream in(path);
    if (!in) {
        GTEST_SKIP() << "source not present at " << path.string();
    }
    const std::string source((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const std::string graphics = function_body(source, "PipelineRegistry::create_pipeline_uncached(PipelineType");
    ASSERT_FALSE(graphics.empty()) << "create_pipeline_uncached was renamed; update this test with the rename";
    EXPECT_NE(graphics.find("shader_path()"), std::string::npos)
        << "the catalogue path must concatenate shader_path(), the engine root";
    EXPECT_EQ(graphics.find("resolve_shader_path"), std::string::npos)
        << "a catalogue row resolved through the host search roots means a game can shadow an engine shader";

    const std::string compute =
        function_body(source, "PipelineRegistry::create_compute_pipeline_uncached(ComputePipelineType");
    ASSERT_FALSE(compute.empty()) << "create_compute_pipeline_uncached was renamed; update this test";
    EXPECT_NE(compute.find("shader_path()"), std::string::npos);
    EXPECT_EQ(compute.find("resolve_shader_path"), std::string::npos);

    // The other half of the same statement: the HOST path does go through the
    // roots. Without this the test above would also pass if host resolution had
    // quietly been changed to use the engine root only.
    const std::string host = function_body(source, "PipelineRegistry::create_host_pipeline_uncached(");
    ASSERT_FALSE(host.empty()) << "create_host_pipeline_uncached was renamed; update this test";
    EXPECT_NE(host.find("resolve_shader_path"), std::string::npos);
#endif
}

} // namespace
} // namespace bracken::gpu
