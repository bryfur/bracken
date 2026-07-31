// GPU-layer diagnostics must reach a host-installed LogSink.
//
// Before the log-channel migration every failure below went to SDL_Log at SDL's
// default INFO priority: a game that installed a `core::LogSink` saw none of
// them, and `Logger::set_min_level` — advertised as *the* level knob — silenced
// none of them either. Those are exactly the errors that produce a black
// screen, so this file pins both halves of the fix:
//   * a forced GPU resource-creation failure is delivered to the sink, and
//   * it arrives at ERROR, distinguishable from the boot chatter next to it.
//
// The failures used here are the parameter-validation early-outs, which are
// reached before any SDL handle is touched — so the test needs no GPU, no
// window and no display, and runs in the normal headless test lane.

#include <gtest/gtest.h>

#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_shader.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace bracken;
using bracken::core::LogLevel;
using bracken::core::Logger;
using bracken::core::LogSink;

namespace {

class CapturingSink final : public LogSink {
public:
    struct Entry {
        LogLevel level;
        std::string category;
        std::string msg;
    };

    void write(LogLevel level, const char* category, std::string_view msg) override {
        std::lock_guard<std::mutex> lock(mutex);
        entries.push_back(Entry{level, std::string(category ? category : ""), std::string(msg)});
    }

    bool saw(LogLevel level, std::string_view fragment) const {
        for (const Entry& e : entries) {
            if (e.level == level && e.msg.find(fragment) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    mutable std::mutex mutex;
    std::vector<Entry> entries;
};

// Installs a capturing sink as the *only* sink, restoring the default afterwards.
struct GpuLogCaptureFixture : ::testing::Test {
    void SetUp() override {
        Logger::instance().clear_sinks();
        Logger::instance().set_min_level(LogLevel::Info);
        sink = std::make_shared<CapturingSink>();
        handle = Logger::instance().add_sink(sink);
    }
    void TearDown() override {
        Logger::instance().remove_sink(handle);
        Logger::instance().clear_sinks();
        Logger::instance().add_sink(std::make_shared<bracken::core::StdoutSink>());
        Logger::instance().set_min_level(LogLevel::Info);
    }

    // Never initialized: every call below returns on its parameter check before
    // reaching the SDL handle.
    gpu::GPUDevice device;
    std::shared_ptr<CapturingSink> sink;
    Logger::SinkHandle handle = 0;
};

} // namespace

TEST_F(GpuLogCaptureFixture, BufferCreationFailureReachesTheSinkAtError) {
    auto buffer = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Vertex, nullptr, 0);
    EXPECT_EQ(buffer, nullptr);

    ASSERT_FALSE(sink->entries.empty()) << "a host LogSink saw nothing when GPU buffer creation failed";
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::create_static: Invalid parameters"));
    EXPECT_EQ(sink->entries.front().category, "gpu");
}

TEST_F(GpuLogCaptureFixture, DynamicBufferCreationFailureReachesTheSinkAtError) {
    auto buffer = gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, 0);
    EXPECT_EQ(buffer, nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::create_dynamic: Invalid size"));
}

TEST_F(GpuLogCaptureFixture, ShaderCreationFailureReachesTheSinkAtError) {
    auto shader = gpu::GPUShader::create_from_spirv(device, std::vector<uint8_t>{}, gpu::ShaderStage::Vertex, "VSMain");
    EXPECT_EQ(shader, nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUShader::create_from_spirv: Empty bytecode"));
}

TEST_F(GpuLogCaptureFixture, ComputePipelineCreationFailureReachesTheSinkAtError) {
    auto pipeline = gpu::GPUComputePipeline::create_from_spirv(device, std::vector<uint8_t>{}, "CSMain");
    EXPECT_EQ(pipeline, nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUComputePipeline::create_from_spirv: Empty bytecode"));
}

// The other half of the complaint: set_min_level was documented as the single
// level knob but could not reach anything the GPU layer emitted.
TEST_F(GpuLogCaptureFixture, MinLevelNowSilencesGpuDiagnostics) {
    Logger::instance().set_min_level(LogLevel::Off);
    auto buffer = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Vertex, nullptr, 0);
    EXPECT_EQ(buffer, nullptr);
    EXPECT_TRUE(sink->entries.empty());

    Logger::instance().set_min_level(LogLevel::Error);
    buffer = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Vertex, nullptr, 0);
    EXPECT_EQ(buffer, nullptr);
    EXPECT_TRUE(sink->saw(LogLevel::Error, "GPUBuffer::create_static: Invalid parameters"));
}
