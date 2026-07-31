#pragma once

// TEMPORARY — refactor phase 6c, commit 1 of 4.
//
// Captures the exact GPU state every hand-written pipeline creator in
// pipeline_registry.cpp produces, so that the tabulated catalogue introduced in
// commit 2 can be diffed field-by-field against it. This whole translation unit
// (and the three call sites that feed it) is deleted in commit 3, once the
// creators are gone; only the captured text file survives, as the permanent
// regression golden for gpu/pipeline_catalog.cpp.
//
// The hooks are null in every normal run — installing them is opt-in and
// happens only inside PipelineRegistry's dump pass.

#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include <string>

namespace bracken::gpu::golden {

/// Called with every PipelineConfig handed to GPUPipeline::create.
extern void (*g_config_hook)(const PipelineConfig&);

/// Called with every shader request that reaches ShaderManager::get (cache hits
/// included — the creators are what we are recording, not the cache).
extern void (*g_shader_hook)(const std::string& path, ShaderStage stage, const ShaderResources& resources);

/// Called with every compute module handed to GPUComputePipeline::load_spirv.
extern void (*g_compute_hook)(const std::string& path, const std::string& entry_point,
                              const ComputeResources& resources);

/// Render one PipelineConfig as stable, reviewable text. Deliberately spells
/// every SDL enum symbolically: the point of the golden is that a human can
/// read the depth-compare column.
std::string describe(const PipelineConfig& config);

/// Render a shader request line.
std::string describe_shader(const std::string& path, ShaderStage stage, const ShaderResources& resources);

/// Render a compute module line.
std::string describe_compute(const std::string& path, const std::string& entry_point,
                             const ComputeResources& resources);

} // namespace bracken::gpu::golden
