// Host-registered pipelines: the pure, device-free half.
//
// Everything here runs with no GPU and no window, because everything here is
// deliberately pure — the desc→PipelineConfig projection, the cache key, and
// the slot table behind the handles. What needs a device (actual pipeline
// creation, shader resolution across search roots) is covered by the registry's
// own contract tests and, end to end, by the playground.
//
// The claim these tests defend is the one the whole feature rests on: a host
// registration is a SECOND namespace beside the catalogue, never a widening of
// it. gpu/pipeline_catalog.test.cpp's golden asserts EXACTLY
// PipelineType::Count + ComputePipelineType::Count blocks, and it must keep
// doing so however many pipelines a game registers.

#include "bracken/gpu/host_pipeline.hpp"

#include "bracken/gpu/pipeline_catalog.hpp"

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <unordered_set>

namespace bracken::gpu {
namespace {

// The formats the catalogue's golden is captured with, repeated here so this
// file is readable on its own and so a drift in either copy shows up as a
// disagreement rather than as a silently shared mistake.
constexpr PipelineFormats k_formats{
    .swapchain = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
    .scene_color = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
    .velocity = SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT,
    .depth = SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
};

/// A minimal well-formed host graphics desc.
HostGraphicsPipelineDesc named(std::string name) {
    HostGraphicsPipelineDesc desc;
    desc.name = std::move(name);
    desc.vertex_shader = "x.vert.spv";
    desc.fragment_shader = "x.frag.spv";
    return desc;
}

/// Project a catalogue row onto the equivalent host desc. Used to prove the two
/// builders agree; if a field is added to PipelineDesc and not mirrored here,
/// EveryCatalogueRowRebuildsIdenticallyAsAHostDesc fails on that row.
HostGraphicsPipelineDesc mirror_of(const PipelineDesc& row) {
    HostGraphicsPipelineDesc desc;
    desc.name = row.name;
    desc.vertex_shader = row.vertex_shader;
    desc.fragment_shader = row.fragment_shader;
    desc.vertex_layout = row.vertex_layout;
    desc.primitive_type = row.primitive_type;
    desc.cull_mode = row.cull_mode;
    desc.blend_mode = row.blend_mode;
    desc.depth_test = row.depth_test;
    desc.depth_write = row.depth_write;
    desc.depth_compare = row.depth_compare;
    desc.depth_clip = row.depth_clip;
    desc.depth_bias = row.depth_bias;
    desc.color = row.color;
    desc.color2 = row.color2;
    desc.depth_target = row.depth_target;
    return desc;
}

/// Field-by-field equality of everything build_*_pipeline_config sets. Written
/// out rather than defaulted because PipelineConfig also carries the two shader
/// handles, which neither builder fills.
::testing::AssertionResult configs_match(const PipelineConfig& a, const PipelineConfig& b) {
#define BRACKEN_FIELD_EQ(field)                                                                                        \
    if (!(a.field == b.field)) {                                                                                       \
        return ::testing::AssertionFailure() << "config field differs: " #field;                                       \
    }
    BRACKEN_FIELD_EQ(primitive_type)
    BRACKEN_FIELD_EQ(fill_mode)
    BRACKEN_FIELD_EQ(cull_mode)
    BRACKEN_FIELD_EQ(front_face)
    BRACKEN_FIELD_EQ(depth_bias_constant)
    BRACKEN_FIELD_EQ(depth_bias_clamp)
    BRACKEN_FIELD_EQ(depth_bias_slope)
    BRACKEN_FIELD_EQ(enable_depth_bias)
    BRACKEN_FIELD_EQ(enable_depth_clip)
    BRACKEN_FIELD_EQ(depth_test_enable)
    BRACKEN_FIELD_EQ(depth_write_enable)
    BRACKEN_FIELD_EQ(depth_compare_op)
    BRACKEN_FIELD_EQ(stencil_test_enable)
    BRACKEN_FIELD_EQ(blend_mode)
    BRACKEN_FIELD_EQ(color_format)
    BRACKEN_FIELD_EQ(color_format2)
    BRACKEN_FIELD_EQ(has_depth_target)
    BRACKEN_FIELD_EQ(depth_format)
    BRACKEN_FIELD_EQ(sample_count)
#undef BRACKEN_FIELD_EQ
    if (a.vertex_buffers.size() != b.vertex_buffers.size()) {
        return ::testing::AssertionFailure() << "vertex_buffers size differs";
    }
    for (size_t i = 0; i < a.vertex_buffers.size(); ++i) {
        const auto& x = a.vertex_buffers[i];
        const auto& y = b.vertex_buffers[i];
        if (x.slot != y.slot || x.pitch != y.pitch || x.input_rate != y.input_rate ||
            x.instance_step_rate != y.instance_step_rate) {
            return ::testing::AssertionFailure() << "vertex buffer " << i << " differs";
        }
    }
    if (a.vertex_attributes.size() != b.vertex_attributes.size()) {
        return ::testing::AssertionFailure() << "vertex_attributes size differs";
    }
    for (size_t i = 0; i < a.vertex_attributes.size(); ++i) {
        const auto& x = a.vertex_attributes[i];
        const auto& y = b.vertex_attributes[i];
        if (x.location != y.location || x.buffer_slot != y.buffer_slot || x.format != y.format ||
            x.offset != y.offset) {
            return ::testing::AssertionFailure() << "vertex attribute " << i << " differs";
        }
    }
    return ::testing::AssertionSuccess();
}

// =============================================================================
// build_host_pipeline_config
// =============================================================================

TEST(HostPipeline, DefaultsMatchPipelineDescDefaults) {
    // Load-bearing, not tidy: a default-constructed host desc must be the same
    // back-face-culled, opaque, REVERSED-Z scene pipeline a default catalogue
    // row is. If these drift, a game copying the defaults silently gets
    // standard-Z depth testing and renders the scene inside out.
    const HostGraphicsPipelineDesc host{};
    const PipelineDesc row{};
    EXPECT_EQ(host.vertex_layout, row.vertex_layout);
    EXPECT_EQ(host.primitive_type, row.primitive_type);
    EXPECT_EQ(host.cull_mode, row.cull_mode);
    EXPECT_EQ(host.blend_mode, row.blend_mode);
    EXPECT_EQ(host.depth_test, row.depth_test);
    EXPECT_EQ(host.depth_write, row.depth_write);
    EXPECT_EQ(host.depth_compare, row.depth_compare);
    EXPECT_EQ(host.depth_compare, SDL_GPU_COMPAREOP_GREATER_OR_EQUAL) << "the engine renders reversed-Z";
    EXPECT_EQ(host.depth_clip, row.depth_clip);
    EXPECT_EQ(host.depth_bias.enabled, row.depth_bias.enabled);
    EXPECT_EQ(host.depth_bias.constant, row.depth_bias.constant);
    EXPECT_EQ(host.depth_bias.slope, row.depth_bias.slope);
    EXPECT_EQ(host.color, row.color);
    EXPECT_EQ(host.color2, row.color2);
    EXPECT_EQ(host.depth_target, row.depth_target);
    // And the escape hatches are OFF by default, so a default desc resolves its
    // formats exactly the way a catalogue row does.
    EXPECT_EQ(host.color_format_override, SDL_GPU_TEXTUREFORMAT_INVALID);
    EXPECT_EQ(host.color2_format_override, SDL_GPU_TEXTUREFORMAT_INVALID);
    EXPECT_EQ(host.depth_format_override, SDL_GPU_TEXTUREFORMAT_INVALID);
    EXPECT_TRUE(host.buffers.empty());
    EXPECT_TRUE(host.attributes.empty());
}

TEST(HostPipeline, EveryCatalogueRowRebuildsIdenticallyAsAHostDesc) {
    // The strongest available statement that the open half cannot drift from
    // the closed one: for all 60-odd rows, a host desc mirroring the row
    // produces a byte-equal PipelineConfig. Both builders go through
    // build_pipeline_config, so this is really a completeness check on the
    // projection — forget to copy cull_mode and the shadow rows fail here.
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const PipelineDesc& row = pipeline_desc(static_cast<PipelineType>(i));
        const PipelineConfig expected = build_pipeline_config(row, k_formats);
        const PipelineConfig actual = build_host_pipeline_config(mirror_of(row), k_formats);
        EXPECT_TRUE(configs_match(expected, actual)) << "row " << i << " (" << row.name << ")";
    }
}

TEST(HostPipeline, FormatOverridesWinOverTheResolvedTargetFormat) {
    HostGraphicsPipelineDesc desc;
    desc.color = TargetFormat::SceneColor;
    desc.color_format_override = SDL_GPU_TEXTUREFORMAT_R32_FLOAT;
    desc.color2_format_override = SDL_GPU_TEXTUREFORMAT_R8_UNORM;
    desc.depth_format_override = SDL_GPU_TEXTUREFORMAT_D24_UNORM;

    const PipelineConfig config = build_host_pipeline_config(desc, k_formats);
    EXPECT_EQ(config.color_format, SDL_GPU_TEXTUREFORMAT_R32_FLOAT);
    EXPECT_EQ(config.color_format2, SDL_GPU_TEXTUREFORMAT_R8_UNORM);
    EXPECT_EQ(config.depth_format, SDL_GPU_TEXTUREFORMAT_D24_UNORM);

    // INVALID means "keep what the TargetFormat resolved to" — it is not a
    // request for an invalid format.
    HostGraphicsPipelineDesc plain;
    plain.color = TargetFormat::Swapchain;
    const PipelineConfig resolved = build_host_pipeline_config(plain, k_formats);
    EXPECT_EQ(resolved.color_format, k_formats.swapchain);
    EXPECT_EQ(resolved.depth_format, k_formats.depth);
}

TEST(HostPipeline, ExplicitVertexBuffersReplaceTheNamedLayout) {
    HostGraphicsPipelineDesc desc;
    desc.vertex_layout = VertexLayout::Vertex3D; // would produce one buffer + five attributes
    desc.buffers = {SDL_GPUVertexBufferDescription{
        .slot = 0, .pitch = 24, .input_rate = SDL_GPU_VERTEXINPUTRATE_INSTANCE, .instance_step_rate = 1}};
    desc.attributes = {SDL_GPUVertexAttribute{
        .location = 7, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = 4}};

    const PipelineConfig config = build_host_pipeline_config(desc, k_formats);
    ASSERT_EQ(config.vertex_buffers.size(), 1u);
    EXPECT_EQ(config.vertex_buffers[0].pitch, 24u);
    EXPECT_EQ(config.vertex_buffers[0].input_rate, SDL_GPU_VERTEXINPUTRATE_INSTANCE);
    ASSERT_EQ(config.vertex_attributes.size(), 1u) << "the named layout's attributes must be REPLACED, not appended";
    EXPECT_EQ(config.vertex_attributes[0].location, 7u);
}

TEST(HostPipeline, EmptyBuffersKeepTheNamedLayout) {
    HostGraphicsPipelineDesc desc;
    desc.vertex_layout = VertexLayout::Vertex2D;
    const PipelineConfig config = build_host_pipeline_config(desc, k_formats);
    const PipelineConfig reference =
        build_pipeline_config(PipelineDesc{.vertex_layout = VertexLayout::Vertex2D}, k_formats);
    EXPECT_TRUE(configs_match(reference, config));
    EXPECT_FALSE(config.vertex_buffers.empty());
}

TEST(HostPipeline, NoDepthTargetForcesTestAndWriteOff) {
    // Mirrors the catalogue's own rule (pipeline_catalog.test.cpp): a config
    // with no depth attachment must not claim to test or write depth, whatever
    // the desc says.
    HostGraphicsPipelineDesc desc;
    desc.depth_target = false;
    desc.depth_test = true;
    desc.depth_write = true;
    const PipelineConfig config = build_host_pipeline_config(desc, k_formats);
    EXPECT_FALSE(config.has_depth_target);
    EXPECT_FALSE(config.depth_test_enable);
    EXPECT_FALSE(config.depth_write_enable);
}

// =============================================================================
// host_pipeline_desc_error — the refusal reasons, device-free
//
// PipelineRegistry checks this BEFORE it checks whether it has a device, which
// is what makes these reachable at all: a registry with no device refuses
// everything, so a registry-level "is_valid() == false" cannot tell a malformed
// desc apart from a missing device.
// =============================================================================

TEST(HostPipelineValidation, AWellFormedDescHasNoError) {
    HostGraphicsPipelineDesc desc = named("ok");
    EXPECT_EQ(host_pipeline_desc_error(desc), nullptr);

    HostComputePipelineDesc compute;
    compute.name = "ok";
    compute.shader = "x.comp.spv";
    EXPECT_EQ(host_compute_pipeline_desc_error(compute), nullptr);
}

TEST(HostPipelineValidation, EveryMalformedFieldIsNamed) {
    // Each case is asserted individually, and each starts from a desc that is
    // otherwise well-formed — so a test that passes because SOMETHING else was
    // wrong is not possible.
    HostGraphicsPipelineDesc no_name = named("");
    EXPECT_NE(host_pipeline_desc_error(no_name), nullptr) << "empty name";

    HostGraphicsPipelineDesc no_vs = named("no_vs");
    no_vs.vertex_shader.clear();
    EXPECT_NE(host_pipeline_desc_error(no_vs), nullptr) << "empty vertex shader";

    HostGraphicsPipelineDesc no_fs = named("no_fs");
    no_fs.fragment_shader.clear();
    EXPECT_NE(host_pipeline_desc_error(no_fs), nullptr) << "empty fragment shader";

    // An empty entry point is not a default, it is a mistake: SDL would look up
    // a function called "" and the failure would surface as a shader load
    // error naming the file rather than the field.
    HostGraphicsPipelineDesc no_entry = named("no_entry");
    no_entry.vertex_entry.clear();
    EXPECT_NE(host_pipeline_desc_error(no_entry), nullptr) << "empty vertex entry point";

    // attributes without buffers: the explicit vertex layout REPLACES the named
    // one only when `buffers` is non-empty, so these attributes would be
    // dropped on the floor.
    HostGraphicsPipelineDesc orphan_attrs = named("orphan_attrs");
    orphan_attrs.attributes = {SDL_GPUVertexAttribute{
        .location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = 0}};
    EXPECT_NE(host_pipeline_desc_error(orphan_attrs), nullptr) << "attributes with no buffers";

    // …and buffers with no attributes is LEGAL (a vertex shader may read none).
    HostGraphicsPipelineDesc buffers_only = named("buffers_only");
    buffers_only.buffers = {SDL_GPUVertexBufferDescription{
        .slot = 0, .pitch = 12, .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX, .instance_step_rate = 0}};
    EXPECT_EQ(host_pipeline_desc_error(buffers_only), nullptr) << "buffers without attributes is legal";

    HostComputePipelineDesc compute_no_name;
    compute_no_name.shader = "x.comp.spv";
    EXPECT_NE(host_compute_pipeline_desc_error(compute_no_name), nullptr);

    HostComputePipelineDesc compute_no_shader;
    compute_no_shader.name = "c";
    EXPECT_NE(host_compute_pipeline_desc_error(compute_no_shader), nullptr);

    HostComputePipelineDesc compute_no_entry;
    compute_no_entry.name = "c";
    compute_no_entry.shader = "x.comp.spv";
    compute_no_entry.entry_point.clear();
    EXPECT_NE(host_compute_pipeline_desc_error(compute_no_entry), nullptr);
}

// =============================================================================
// PipelineCacheKey — the reason the two halves can share one cache
// =============================================================================

TEST(PipelineCacheKeyTest, CatalogAndHostOrdinalsAreDisjoint) {
    // The mechanical proof that a host registration can never alias a catalogue
    // cache entry: the two ordinal spaces overlap completely, and the origin
    // tag is the only thing keeping them apart.
    const auto limit = static_cast<uint32_t>(PipelineType::Count);
    for (uint32_t i = 0; i < limit; ++i) {
        const PipelineCacheKey from_enum = PipelineCacheKey::catalog(static_cast<PipelineType>(i));
        const PipelineCacheKey from_host = PipelineCacheKey::host(i);
        EXPECT_EQ(from_enum.index, from_host.index) << "the test is only meaningful if the ordinals collide";
        EXPECT_FALSE(from_enum == from_host) << "host slot " << i << " aliases PipelineType(" << i << ")";
    }
}

TEST(PipelineCacheKeyTest, KeysAreHashableAndDistinct) {
    std::unordered_set<PipelineCacheKey> keys;
    for (uint32_t i = 0; i < 64; ++i) {
        EXPECT_TRUE(keys.insert(PipelineCacheKey::catalog(static_cast<PipelineType>(i))).second);
        EXPECT_TRUE(keys.insert(PipelineCacheKey::host(i)).second);
    }
    EXPECT_EQ(keys.size(), 128u);
    // Re-inserting an equal key must not grow the set — i.e. hash and equality
    // agree, which is what an unordered_map key needs.
    EXPECT_FALSE(keys.insert(PipelineCacheKey::host(3)).second);
}

TEST(PipelineCacheKeyTest, CatalogOverloadsCoverBothEnums) {
    // Graphics and compute ordinals live in SEPARATE maps, so the same key is
    // correct for both and is deliberately not disambiguated.
    EXPECT_EQ(PipelineCacheKey::catalog(PipelineType::Count).origin, PipelineCacheKey::Origin::Catalog);
    EXPECT_EQ(PipelineCacheKey::catalog(ComputePipelineType::Count).origin, PipelineCacheKey::Origin::Catalog);
    EXPECT_EQ(PipelineCacheKey::host(0).origin, PipelineCacheKey::Origin::Host);
}

// =============================================================================
// HostPipelineTable — handle stability, reuse and generation
// =============================================================================

using Table = HostPipelineTable<HostGraphicsPipelineTag, HostGraphicsPipelineDesc>;

TEST(HostPipelineTableTest, HandlesAreStableAcrossFurtherRegistrations) {
    Table table;
    const auto first = table.add(named("first"));
    const Table::Entry* first_entry = table.find(first);
    ASSERT_NE(first_entry, nullptr);
    const char* name_address = first_entry->desc.name.c_str();

    for (int i = 0; i < 64; ++i) {
        (void)table.add(named("filler" + std::to_string(i)));
    }

    // Both halves matter. The entry pointer stays valid because entries are
    // heap-allocated, and the NAME's address stays valid because of the same
    // thing — which is what the GPU debug-label API requires, since it stores
    // the raw pointer and reads it later in the frame.
    EXPECT_EQ(table.find(first), first_entry);
    EXPECT_EQ(table.find(first)->desc.name.c_str(), name_address);
    EXPECT_EQ(table.live_count(), 65u);
}

TEST(HostPipelineTableTest, RemovedHandlesGoStaleAndSlotsAreReused) {
    Table table;
    const auto a = table.add(named("a"));
    const auto b = table.add(named("b"));
    ASSERT_TRUE(table.remove(a));
    EXPECT_EQ(table.find(a), nullptr) << "a released handle must not resolve";
    EXPECT_FALSE(table.remove(a)) << "double release is a no-op, not a second removal";
    EXPECT_NE(table.find(b), nullptr) << "removing one registration must not disturb its neighbours";

    // The freed slot is reused, and the generation bump is what stops the stale
    // handle from resolving to the NEW registration in it.
    const auto c = table.add(named("c"));
    EXPECT_EQ(c.index, a.index);
    EXPECT_NE(c.generation, a.generation);
    EXPECT_EQ(table.find(a), nullptr);
    ASSERT_NE(table.find(c), nullptr);
    EXPECT_EQ(table.find(c)->desc.name, "c");
}

TEST(HostPipelineTableTest, DefaultHandleAndOutOfRangeIndexResolveToNull) {
    Table table;
    EXPECT_EQ(table.find(core::Handle<HostGraphicsPipelineTag>{}), nullptr);
    EXPECT_EQ(table.find(core::Handle<HostGraphicsPipelineTag>{99, 1}), nullptr);
    (void)table.add(named("only"));
    // Right index, wrong generation.
    EXPECT_EQ(table.find(core::Handle<HostGraphicsPipelineTag>{0, 999}), nullptr);
}

TEST(HostPipelineTableTest, LiveNamesAreDetectedAndFreedNamesAreNot) {
    Table table;
    const auto a = table.add(named("shared"));
    EXPECT_TRUE(table.has_live_name("shared"));
    EXPECT_FALSE(table.has_live_name("other"));
    ASSERT_TRUE(table.remove(a));
    EXPECT_FALSE(table.has_live_name("shared")) << "a released name must be reusable";
}

TEST(HostPipelineTableTest, ClearReleasesEverythingAndKeepsSlotStorage) {
    Table table;
    const auto a = table.add(named("a"));
    const auto b = table.add(named("b"));
    table.clear();
    EXPECT_EQ(table.live_count(), 0u);
    EXPECT_EQ(table.find(a), nullptr);
    EXPECT_EQ(table.find(b), nullptr);
    // Slot storage survives, so the host ordinals a cache may still hold keep
    // meaning what they meant.
    EXPECT_EQ(table.slots().size(), 2u);
    EXPECT_FALSE(table.slots()[0]->live);
}

TEST(HostPipelineTableTest, SlotsWalkInRegistrationOrder) {
    // Preload, hot reload and the swapchain rebuild all iterate slots(), so the
    // order they rebuild in is this order — deterministic, never a hash order.
    Table table;
    (void)table.add(named("zero"));
    (void)table.add(named("one"));
    (void)table.add(named("two"));
    ASSERT_EQ(table.slots().size(), 3u);
    EXPECT_EQ(table.slots()[0]->desc.name, "zero");
    EXPECT_EQ(table.slots()[1]->desc.name, "one");
    EXPECT_EQ(table.slots()[2]->desc.name, "two");
}

} // namespace
} // namespace bracken::gpu
