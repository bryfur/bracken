// GPU resource lifetime under failure — the ~2,100 lines of buffer/texture/
// device create/release code that sit under every renderer and had no test.
//
// Scope, stated honestly: these tests drive the paths that DO NOT need a
// device — the parameter checks and the "device never came up" branch. That is
// the branch that matters most in the field: `RenderContext::init` failing is
// how a game boots to a black screen, and every renderer's init() then calls
// create_* on a device that is not there. Every one of those calls must return
// nullptr, log, and leave nothing half-constructed, and shutdown must be
// idempotent so the teardown path does not turn a boot failure into a crash.
//
// Deliberately NOT covered here (would need a real SDL_GPUDevice + window, and
// no other test in this binary creates either — the tree keeps unit tests
// headless and puts GPU coverage in the playground smoke run):
//   * GPUBuffer::update / Mapping::upload SIZE-OVERRUN rejection. Both are
//     guarded by an earlier `!transfer_buffer_` / `!buffer_` check, and a
//     dynamic buffer cannot exist without a device, so the overrun branch is
//     structurally unreachable from a headless test. The reachable half of
//     each chain IS pinned below.
//   * the SDL_CreateGPU* success paths.

#include <gtest/gtest.h>

#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"

#include <SDL3/SDL_gpu.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

using namespace bracken;
using bracken::core::LogLevel;
using bracken::core::Logger;
using bracken::core::LogSink;

namespace {

/// Captures everything the engine logs while a test runs, so "returns nullptr"
/// and "says why" can be asserted together — a silent nullptr is the failure
/// mode that produces a black screen with no explanation.
class RecordingSink final : public LogSink {
public:
    void write(LogLevel level, const char* category, std::string_view msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.push_back(Entry{level, std::string(category ? category : ""), std::string(msg)});
    }

    bool saw(LogLevel level, std::string_view fragment) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const Entry& e : entries_) {
            if (e.level == level && e.msg.find(fragment) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    std::size_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
    }

private:
    struct Entry {
        LogLevel level;
        std::string category;
        std::string msg;
    };
    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

/// A GPUDevice that is never init()ed: `is_initialized()` is false and every
/// create_* short-circuits, which is exactly the shape of a failed boot.
struct GpuLifetimeFixture : ::testing::Test {
    void SetUp() override {
        Logger::instance().clear_sinks();
        Logger::instance().set_min_level(LogLevel::Debug);
        sink = std::make_shared<RecordingSink>();
        handle = Logger::instance().add_sink(sink);
    }
    void TearDown() override {
        Logger::instance().remove_sink(handle);
        Logger::instance().clear_sinks();
        Logger::instance().add_sink(std::make_shared<bracken::core::StdoutSink>());
        Logger::instance().set_min_level(LogLevel::Info);
    }

    gpu::GPUDevice device;
    std::shared_ptr<RecordingSink> sink;
    Logger::SinkHandle handle = 0;
};

} // namespace

// ============================================================================
// GPUDevice
// ============================================================================

TEST_F(GpuLifetimeFixture, UninitializedDeviceIsInertAndHonest) {
    EXPECT_FALSE(device.is_initialized());
    EXPECT_EQ(device.handle(), nullptr);
    EXPECT_EQ(device.window(), nullptr);
    // Accessors must answer without touching the null handle.
    EXPECT_EQ(device.width(), 0);
    EXPECT_EQ(device.height(), 0);
    EXPECT_EQ(device.driver_name(), "none");
    EXPECT_EQ(device.swapchain_format(), SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM);
    EXPECT_FALSE(device.supports_present_mode(SDL_GPU_PRESENTMODE_VSYNC));
    EXPECT_FALSE(device.supports_format(SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTURETYPE_2D,
                                        SDL_GPU_TEXTUREUSAGE_SAMPLER));
    EXPECT_FALSE(device.set_swapchain_parameters(SDL_GPU_PRESENTMODE_VSYNC));
    EXPECT_EQ(device.begin_frame(), nullptr);
    EXPECT_FALSE(device.end_frame_and_wait(nullptr));
    // Not one of the above may reach SDL, so none of them may log either.
    EXPECT_EQ(sink->count(), 0u);
}

TEST_F(GpuLifetimeFixture, EveryFactoryReturnsNullWithoutADevice) {
    SDL_GPUBufferCreateInfo buffer_info{};
    buffer_info.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
    buffer_info.size = 64;
    EXPECT_EQ(device.create_buffer(buffer_info), nullptr);

    SDL_GPUTransferBufferCreateInfo transfer_info{};
    transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    transfer_info.size = 64;
    EXPECT_EQ(device.create_transfer_buffer(transfer_info), nullptr);

    SDL_GPUTextureCreateInfo texture_info{};
    texture_info.type = SDL_GPU_TEXTURETYPE_2D;
    texture_info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    texture_info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    texture_info.width = 4;
    texture_info.height = 4;
    texture_info.layer_count_or_depth = 1;
    texture_info.num_levels = 1;
    EXPECT_EQ(device.create_texture(texture_info), nullptr);

    SDL_GPUSamplerCreateInfo sampler_info{};
    EXPECT_EQ(device.create_sampler(sampler_info), nullptr);

    SDL_GPUGraphicsPipelineCreateInfo pipeline_info{};
    EXPECT_EQ(device.create_graphics_pipeline(pipeline_info), nullptr);

    SDL_GPUShaderCreateInfo shader_info{};
    EXPECT_EQ(device.create_shader(shader_info), nullptr);

    EXPECT_EQ(device.map_transfer_buffer(nullptr, true), nullptr);
}

// Every renderer's shutdown() runs on the failure path too, often twice (once
// explicitly, once from the destructor). Releasing into a dead device must be
// a no-op, not a null deref.
TEST_F(GpuLifetimeFixture, ReleaseIntoADeadDeviceIsANoOp) {
    device.release_buffer(nullptr);
    device.release_transfer_buffer(nullptr);
    device.release_texture(nullptr);
    device.release_sampler(nullptr);
    device.release_graphics_pipeline(nullptr);
    device.release_shader(nullptr);
    device.release_compute_pipeline(nullptr);
    device.unmap_transfer_buffer(nullptr);
    // Also with non-null (stale) handles: the device check must come first, so
    // no SDL call is made with a destroyed device.
    auto* stale_buffer = reinterpret_cast<SDL_GPUBuffer*>(0x1);
    device.release_buffer(stale_buffer);
    EXPECT_FALSE(device.is_initialized());
}

TEST_F(GpuLifetimeFixture, ShutdownIsIdempotent) {
    device.shutdown();
    device.shutdown();
    device.shutdown();
    EXPECT_FALSE(device.is_initialized());
    EXPECT_EQ(device.handle(), nullptr);
    // Nothing was ever created, so nothing may be logged as destroyed.
    EXPECT_FALSE(sink->saw(LogLevel::Info, "Device destroyed"));
}

TEST_F(GpuLifetimeFixture, PresentWaitStartsAtZeroAndResets) {
    EXPECT_FLOAT_EQ(device.present_wait_ms(), 0.0f);
    device.reset_present_wait();
    EXPECT_FLOAT_EQ(device.present_wait_ms(), 0.0f);
    // end_frame(nullptr) returns before the timer, so the counter stays clean.
    device.end_frame(nullptr);
    EXPECT_FLOAT_EQ(device.present_wait_ms(), 0.0f);
}

// ============================================================================
// GPUBuffer
// ============================================================================

TEST_F(GpuLifetimeFixture, BufferFactoriesRejectBadParametersAndSayWhy) {
    const uint32_t payload[4] = {1, 2, 3, 4};

    EXPECT_EQ(gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Vertex, nullptr, sizeof(payload)), nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::create_static: Invalid parameters"));

    sink->clear();
    EXPECT_EQ(gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Vertex, payload, 0), nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::create_static: Invalid parameters"));

    sink->clear();
    EXPECT_EQ(gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, 0), nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::create_dynamic: Invalid size"));
}

// Valid parameters, dead device: the SDL create fails, and the factory must
// report THAT rather than the parameter check — the two failures need
// different fixes and the log is the only thing that distinguishes them.
TEST_F(GpuLifetimeFixture, BufferFactoriesReportDeviceFailureDistinctly) {
    const uint32_t payload[4] = {1, 2, 3, 4};

    EXPECT_EQ(gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Vertex, payload, sizeof(payload)), nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::create_static: Failed to create buffer"));
    EXPECT_FALSE(sink->saw(LogLevel::Error, "GPUBuffer::create_static: Invalid parameters"));

    sink->clear();
    EXPECT_EQ(gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::ComputeStorage, 256), nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::create_dynamic: Failed to create buffer"));
}

// Every buffer Type must funnel through the same guarded path — a Type added
// without a usage-flag case would otherwise fall through silently.
TEST_F(GpuLifetimeFixture, EveryBufferTypeFailsCleanlyWithoutADevice) {
    const gpu::GPUBuffer::Type types[] = {
        gpu::GPUBuffer::Type::Vertex, gpu::GPUBuffer::Type::Index, gpu::GPUBuffer::Type::Uniform,
        gpu::GPUBuffer::Type::Storage, gpu::GPUBuffer::Type::ComputeStorage,
        gpu::GPUBuffer::Type::Indirect,
    };
    for (gpu::GPUBuffer::Type type : types) {
        SCOPED_TRACE(static_cast<int>(type));
        sink->clear();
        EXPECT_EQ(gpu::GPUBuffer::create_dynamic(device, type, 128), nullptr);
        EXPECT_TRUE(sink->saw(LogLevel::Error, "Failed to create buffer"));
    }
}

// The buffer Type -> SDL usage-flag mapping, which is a pure function and
// therefore the one part of the indirect-dispatch path (R4a) that IS checkable
// without a device. SDL_DispatchGPUComputeIndirect itself is exercised first by
// R4b; ComputePass::dispatch_indirect says so in its doc comment rather than
// implying coverage that does not exist.
TEST(GpuBufferTypes, IndirectCarriesTheIndirectAndComputeWriteBits) {
    const SDL_GPUBufferUsageFlags indirect = gpu::GPUBuffer::usage_flags(gpu::GPUBuffer::Type::Indirect);
    EXPECT_TRUE((indirect & SDL_GPU_BUFFERUSAGE_INDIRECT) != 0)
        << "SDL_DispatchGPUComputeIndirect requires SDL_GPU_BUFFERUSAGE_INDIRECT";
    // The arguments are produced by a compute pass — that is the entire point
    // of dispatching indirectly — so the write bit is part of the contract.
    EXPECT_TRUE((indirect & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE) != 0);
    // And it is not a vertex/index/graphics-storage buffer wearing a new name.
    EXPECT_EQ(indirect & (SDL_GPU_BUFFERUSAGE_VERTEX | SDL_GPU_BUFFERUSAGE_INDEX |
                          SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ),
              0u);
}

// No OTHER type may carry the indirect bit: a buffer that can be dispatched
// from without having been declared as such is how an argument-layout mistake
// becomes a driver hang instead of a validation message.
TEST(GpuBufferTypes, OnlyIndirectCarriesTheIndirectBit) {
    const gpu::GPUBuffer::Type others[] = {
        gpu::GPUBuffer::Type::Vertex, gpu::GPUBuffer::Type::Index, gpu::GPUBuffer::Type::Uniform,
        gpu::GPUBuffer::Type::Storage, gpu::GPUBuffer::Type::ComputeStorage,
    };
    for (gpu::GPUBuffer::Type type : others) {
        SCOPED_TRACE(static_cast<int>(type));
        EXPECT_EQ(gpu::GPUBuffer::usage_flags(type) & SDL_GPU_BUFFERUSAGE_INDIRECT, 0u);
    }
}

// The command layout dispatch_indirect documents, pinned against SDL's own
// struct: three consecutive Uint32 group counts, 12 bytes, no stride field.
TEST(GpuBufferTypes, IndirectDispatchCommandIsThreeUint32GroupCounts) {
    static_assert(sizeof(SDL_GPUIndirectDispatchCommand) == 12, "three Uint32s, nothing else");
    EXPECT_EQ(offsetof(SDL_GPUIndirectDispatchCommand, groupcount_x), 0u);
    EXPECT_EQ(offsetof(SDL_GPUIndirectDispatchCommand, groupcount_y), 4u);
    EXPECT_EQ(offsetof(SDL_GPUIndirectDispatchCommand, groupcount_z), 8u);
}

// A default-constructed Mapping owns nothing: destroying it must not unmap,
// and upload() must refuse instead of dereferencing its null buffer.
TEST_F(GpuLifetimeFixture, EmptyMappingIsSafeAndRefusesUpload) {
    gpu::GPUBuffer::Mapping mapping;
    EXPECT_FALSE(static_cast<bool>(mapping));
    EXPECT_EQ(mapping.data(), nullptr);
    mapping.upload(nullptr, 16);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::Mapping::upload: Not mapped"));

    // Moving an empty mapping leaves both ends empty and destructible.
    gpu::GPUBuffer::Mapping moved = std::move(mapping);
    EXPECT_FALSE(static_cast<bool>(moved));
    gpu::GPUBuffer::Mapping assigned;
    assigned = std::move(moved);
    EXPECT_FALSE(static_cast<bool>(assigned));
}

// ============================================================================
// GPUTexture / GPUSampler
// ============================================================================

TEST_F(GpuLifetimeFixture, TextureFactoriesReturnNullAndLogWithoutADevice) {
    struct Case {
        const char* fragment;
        std::unique_ptr<gpu::GPUTexture> (*make)(gpu::GPUDevice&);
    };
    const Case cases[] = {
        {"GPUTexture::create_2d: Failed to create texture",
         [](gpu::GPUDevice& d) {
             return gpu::GPUTexture::create_2d(d, 4, 4, gpu::TextureFormat::RGBA8, nullptr, false);
         }},
        {"GPUTexture::create_render_target: Failed to create texture",
         [](gpu::GPUDevice& d) {
             return gpu::GPUTexture::create_render_target(d, 4, 4, gpu::TextureFormat::RGBA8);
         }},
        {"GPUTexture::create_depth: Failed to create texture",
         [](gpu::GPUDevice& d) { return gpu::GPUTexture::create_depth(d, 4, 4); }},
        {"GPUTexture::create_depth_array: Failed to create texture",
         [](gpu::GPUDevice& d) { return gpu::GPUTexture::create_depth_array(d, 4, 4, 2); }},
        {"GPUTexture::create_2d_array: Failed to create texture",
         [](gpu::GPUDevice& d) {
             return gpu::GPUTexture::create_2d_array(d, 4, 4, 2, gpu::TextureFormat::RGBA8, nullptr, false);
         }},
        {"GPUTexture::create_cube: Failed to create texture",
         [](gpu::GPUDevice& d) {
             return gpu::GPUTexture::create_cube(d, 4, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                                                SDL_GPU_TEXTUREUSAGE_SAMPLER, false);
         }},
        {"GPUTexture::create_depth_stencil: Failed to create texture",
         [](gpu::GPUDevice& d) { return gpu::GPUTexture::create_depth_stencil(d, 4, 4); }},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.fragment);
        sink->clear();
        EXPECT_EQ(c.make(device), nullptr);
        EXPECT_TRUE(sink->saw(LogLevel::Error, c.fragment));
    }
}

TEST_F(GpuLifetimeFixture, TextureLoadFromMissingFileFailsLoudly) {
    EXPECT_EQ(gpu::GPUTexture::load_from_file(device, "no-such-texture-file.png"), nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUTexture::load_from_file: Failed to load"));
}

TEST_F(GpuLifetimeFixture, SamplerCreationFailsCleanlyWithoutADevice) {
    for (const gpu::SamplerConfig& config :
         {gpu::SamplerConfig::linear_repeat(), gpu::SamplerConfig::linear_clamp(),
          gpu::SamplerConfig::nearest_repeat(), gpu::SamplerConfig::nearest_clamp(),
          gpu::SamplerConfig::anisotropic(8.0f)}) {
        sink->clear();
        EXPECT_EQ(gpu::GPUSampler::create(device, config), nullptr);
        EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUSampler::create: Failed to create sampler"));
    }
}

// ============================================================================
// Compute pipelines — the other half of the boot chain
// ============================================================================

TEST_F(GpuLifetimeFixture, ComputePipelineRejectsEmptyAndMissingBytecode) {
    EXPECT_EQ(gpu::GPUComputePipeline::create_from_spirv(device, std::vector<uint8_t>{}, "CSMain"), nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUComputePipeline::create_from_spirv: Empty bytecode"));

    sink->clear();
    EXPECT_EQ(gpu::GPUComputePipeline::load_spirv(device, "no-such-shader.comp.spv", "CSMain"), nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "Failed to open"));
}
