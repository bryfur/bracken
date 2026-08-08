// CPU uniform struct <-> compiled shader agreement, checked against ground truth.
//
// `CLAUDE.md`: "CPU uniform structs in gpu/gpu_uniforms.hpp must match their
// HLSL cbuffer counterparts byte-for-byte. This is the most common way to
// silently break rendering." Until this file existed, nothing checked that:
// gpu_uniforms.test.cpp pinned the CPU side against hand-typed numbers, and
// the HLSL side was pinned by comments. Editing one side alone compiled,
// linked, ran, and rendered garbage.
//
// The check reads the **compiled SPIR-V** the build already produces and
// extracts each uniform block's declared byte extent from the explicit
// Offset/ArrayStride/MatrixStride decorations DXC emitted. That is ground
// truth for what the shader will actually read — no HLSL packing rules are
// reimplemented here, which is precisely why the survey's proposed HLSL text
// scanner was rejected (docs/refactor_plan.md §6).
//
// Three tests:
//   * BlockSizesMatchCpuStructs  — every (shader, set, binding) in the table
//     declares exactly sizeof(T) bytes.
//   * EveryShaderBlockIsAccountedFor — the reverse direction: a uniform block
//     in ANY staged shader that no table row and no allowlist entry covers is
//     a failure, so a new cbuffer cannot quietly appear unchecked.
//   * StructuredBufferStridesMatchCpuRecords — storage-buffer element strides
//     (the per-instance records) equal sizeof of their CPU record.
//
// Headless: reads files only, no GPU device, no SDL_shadercross. Skips (not
// fails) when the staged shader directory is absent, e.g. a relocated test
// binary.

#include <gtest/gtest.h>

#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/render/grass_renderer.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
#include "bracken/render/terrain_renderer.hpp"
#include "bracken/render/water_renderer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ============================================================================
// Minimal SPIR-V reader: enough to answer "how many bytes does this uniform
// block declare?" and "what is this StructuredBuffer's element stride?".
//
// Everything needed is an explicit decoration in the module (SPIR-V requires
// explicit layout for Block/BufferBlock types), so this walks the type graph
// and reads the numbers DXC wrote down. It never infers a layout rule.
// ============================================================================

class SpirvModule {
public:
    struct Block {
        uint32_t set = 0;
        uint32_t binding = 0;
        std::string type_name;   // e.g. "type.CompositeUniforms"
        std::size_t extent = 0;  // bytes covered by declared members
        bool storage = false;    // BufferBlock / StorageBuffer (StructuredBuffer)
        std::optional<std::size_t> element_stride; // runtime-array stride, storage only
    };

    /// Parse `path`. Returns false if the file is missing or not SPIR-V.
    bool load(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            return false;
        }
        const std::streamsize byte_size = file.tellg();
        if (byte_size < 20 || (byte_size % 4) != 0) {
            return false;
        }
        file.seekg(0, std::ios::beg);
        words_.resize(static_cast<std::size_t>(byte_size) / 4);
        if (!file.read(reinterpret_cast<char*>(words_.data()), byte_size)) {
            return false;
        }
        // Only native-endian SPIR-V is produced by this build; a byte-swapped
        // module is a broken artifact, not something to silently tolerate.
        if (words_[0] != SPIRV_MAGIC) {
            return false;
        }
        parse();
        return true;
    }

    const std::vector<Block>& blocks() const { return blocks_; }

    const Block* find(uint32_t set, uint32_t binding) const {
        for (const Block& b : blocks_) {
            if (b.set == set && b.binding == binding) {
                return &b;
            }
        }
        return nullptr;
    }

private:
    static constexpr uint32_t SPIRV_MAGIC = 0x07230203u;

    // Op codes / decorations / storage classes used below (SPIR-V 1.0 spec).
    enum Op : uint32_t {
        OpName = 5,
        OpTypeInt = 21,
        OpTypeFloat = 22,
        OpTypeVector = 23,
        OpTypeMatrix = 24,
        OpTypeArray = 28,
        OpTypeRuntimeArray = 29,
        OpTypeStruct = 30,
        OpTypePointer = 32,
        OpConstant = 43,
        OpVariable = 59,
        OpDecorate = 71,
        OpMemberDecorate = 72,
    };
    enum Decoration : uint32_t {
        DecBlock = 2,
        DecBufferBlock = 3,
        DecArrayStride = 6,
        DecMatrixStride = 7,
        DecBinding = 33,
        DecDescriptorSet = 34,
        DecOffset = 35,
    };
    enum StorageClass : uint32_t {
        ScUniform = 2,
        ScStorageBuffer = 12,
    };

    struct TypeInfo {
        Op kind = OpTypeInt;
        uint32_t a = 0; // int/float: bit width | vector/matrix/array/rtarray/pointer: component or pointee
        uint32_t b = 0; // vector/matrix: component count | array: length constant id | pointer: pointee
        std::vector<uint32_t> members; // struct only
    };

    void parse() {
        std::size_t i = 5; // skip the 5-word header
        while (i < words_.size()) {
            const uint32_t word_count = words_[i] >> 16;
            const uint32_t op = words_[i] & 0xFFFFu;
            if (word_count == 0 || i + word_count > words_.size()) {
                break; // malformed; stop rather than read past the end
            }
            // data() + arithmetic, not &words_[i + 1]: a trailing 1-word
            // instruction makes i + 1 == size(), and forming that reference
            // trips hardened-libstdc++ operator[] even though a 1-word op
            // never dereferences ops (n == 0).
            const uint32_t* ops = words_.data() + i + 1;
            const uint32_t n = word_count - 1;
            switch (op) {
                case OpName:
                    if (n >= 2) {
                        names_[ops[0]] = decode_string(ops + 1, n - 1);
                    }
                    break;
                case OpDecorate:
                    if (n >= 2) {
                        decorations_[ops[0]][ops[1]] = (n >= 3) ? ops[2] : 0u;
                    }
                    break;
                case OpMemberDecorate:
                    if (n >= 3) {
                        member_decorations_[{ops[0], ops[1]}][ops[2]] = (n >= 4) ? ops[3] : 0u;
                    }
                    break;
                case OpTypeInt:
                case OpTypeFloat:
                    if (n >= 2) {
                        types_[ops[0]] = TypeInfo{static_cast<Op>(op), ops[1], 0, {}};
                    }
                    break;
                case OpTypeVector:
                case OpTypeMatrix:
                case OpTypeArray:
                    if (n >= 3) {
                        types_[ops[0]] = TypeInfo{static_cast<Op>(op), ops[1], ops[2], {}};
                    }
                    break;
                case OpTypeRuntimeArray:
                    if (n >= 2) {
                        types_[ops[0]] = TypeInfo{OpTypeRuntimeArray, ops[1], 0, {}};
                    }
                    break;
                case OpTypeStruct: {
                    if (n >= 1) {
                        TypeInfo t{OpTypeStruct, 0, 0, {}};
                        t.members.assign(ops + 1, ops + n);
                        types_[ops[0]] = std::move(t);
                    }
                    break;
                }
                case OpTypePointer:
                    if (n >= 3) {
                        types_[ops[0]] = TypeInfo{OpTypePointer, ops[1], ops[2], {}};
                    }
                    break;
                case OpConstant:
                    if (n >= 3) {
                        constants_[ops[1]] = ops[2];
                    }
                    break;
                case OpVariable:
                    if (n >= 3) {
                        variables_.push_back({ops[0], ops[1], ops[2]});
                    }
                    break;
                default:
                    break;
            }
            i += word_count;
        }
        collect_blocks();
    }

    static std::string decode_string(const uint32_t* words, uint32_t count) {
        std::string s;
        s.reserve(count * 4);
        for (uint32_t w = 0; w < count; ++w) {
            for (int byte = 0; byte < 4; ++byte) {
                const char c = static_cast<char>((words[w] >> (byte * 8)) & 0xFFu);
                if (c == '\0') {
                    return s;
                }
                s.push_back(c);
            }
        }
        return s;
    }

    std::optional<uint32_t> decoration(uint32_t id, uint32_t dec) const {
        const auto it = decorations_.find(id);
        if (it == decorations_.end()) {
            return std::nullopt;
        }
        const auto d = it->second.find(dec);
        return d == it->second.end() ? std::nullopt : std::optional<uint32_t>(d->second);
    }

    std::optional<uint32_t> member_decoration(uint32_t id, uint32_t member, uint32_t dec) const {
        const auto it = member_decorations_.find({id, member});
        if (it == member_decorations_.end()) {
            return std::nullopt;
        }
        const auto d = it->second.find(dec);
        return d == it->second.end() ? std::nullopt : std::optional<uint32_t>(d->second);
    }

    /// Byte size of `type`, using the module's own explicit strides.
    /// `matrix_stride` is the enclosing struct member's MatrixStride, if any.
    std::size_t type_size(uint32_t type, std::optional<uint32_t> matrix_stride, int depth = 0) const {
        if (depth > 16) {
            return 0; // cycle guard; SPIR-V type graphs are acyclic in practice
        }
        const auto it = types_.find(type);
        if (it == types_.end()) {
            return 0;
        }
        const TypeInfo& t = it->second;
        switch (t.kind) {
            case OpTypeInt:
            case OpTypeFloat:
                return t.a / 8u;
            case OpTypeVector:
                return t.b * type_size(t.a, std::nullopt, depth + 1);
            case OpTypeMatrix:
                // Columns are laid out at MatrixStride; without the decoration
                // the module is not explicitly laid out and the answer is 0
                // (which surfaces as a size mismatch rather than a guess).
                return matrix_stride ? t.b * *matrix_stride : 0;
            case OpTypeArray: {
                const auto stride = decoration(type, DecArrayStride);
                const auto len = constants_.find(t.b);
                if (!stride || len == constants_.end()) {
                    return 0;
                }
                return *stride * len->second;
            }
            case OpTypeStruct:
                return struct_extent(type, depth + 1);
            default:
                return 0;
        }
    }

    /// Bytes covered by a struct's declared members: max(offset + size).
    /// This deliberately EXCLUDES trailing padding the shader does not
    /// declare — the whole point is to catch a CPU struct that declares pad
    /// its cbuffer does not (see the BloomDownsampleUniforms note in the
    /// table below).
    std::size_t struct_extent(uint32_t type, int depth = 0) const {
        const auto it = types_.find(type);
        if (it == types_.end() || it->second.kind != OpTypeStruct) {
            return 0;
        }
        std::size_t extent = 0;
        const std::vector<uint32_t>& members = it->second.members;
        for (std::size_t m = 0; m < members.size(); ++m) {
            const auto offset = member_decoration(type, static_cast<uint32_t>(m), DecOffset);
            if (!offset) {
                continue; // not explicitly laid out
            }
            const auto stride = member_decoration(type, static_cast<uint32_t>(m), DecMatrixStride);
            extent = std::max<std::size_t>(extent, *offset + type_size(members[m], stride, depth));
        }
        return extent;
    }

    void collect_blocks() {
        for (const Variable& v : variables_) {
            if (v.storage_class != ScUniform && v.storage_class != ScStorageBuffer) {
                continue;
            }
            const auto ptr = types_.find(v.type);
            if (ptr == types_.end() || ptr->second.kind != OpTypePointer) {
                continue;
            }
            const uint32_t struct_type = ptr->second.b;
            const auto st = types_.find(struct_type);
            if (st == types_.end() || st->second.kind != OpTypeStruct) {
                continue;
            }
            const bool is_block = decoration(struct_type, DecBlock).has_value();
            const bool is_buffer_block = decoration(struct_type, DecBufferBlock).has_value();
            if (!is_block && !is_buffer_block) {
                continue;
            }
            Block b;
            b.set = decoration(v.id, DecDescriptorSet).value_or(0u);
            b.binding = decoration(v.id, DecBinding).value_or(0u);
            const auto name = names_.find(struct_type);
            b.type_name = name == names_.end() ? std::string{} : name->second;
            b.storage = is_buffer_block || v.storage_class == ScStorageBuffer;
            b.extent = struct_extent(struct_type);
            for (uint32_t member : st->second.members) {
                const auto mt = types_.find(member);
                if (mt != types_.end() && mt->second.kind == OpTypeRuntimeArray) {
                    if (const auto stride = decoration(member, DecArrayStride)) {
                        b.element_stride = *stride;
                    }
                }
            }
            blocks_.push_back(std::move(b));
        }
    }

    struct Variable {
        uint32_t type = 0;
        uint32_t id = 0;
        uint32_t storage_class = 0;
    };

    std::vector<uint32_t> words_;
    std::map<uint32_t, std::string> names_;
    std::map<uint32_t, std::map<uint32_t, uint32_t>> decorations_;
    std::map<std::pair<uint32_t, uint32_t>, std::map<uint32_t, uint32_t>> member_decorations_;
    std::map<uint32_t, TypeInfo> types_;
    std::map<uint32_t, uint32_t> constants_;
    std::vector<Variable> variables_;
    std::vector<Block> blocks_;
};

// ============================================================================
// The table: which CPU struct backs which (shader, descriptor set, binding).
//
// A struct declared by several shaders gets a row per shader — they are
// separate compilations and can drift independently. Sets follow the SDL3 GPU
// convention (vertex uniforms space1, fragment uniforms space3, compute
// uniforms space2); see src/bracken/shaders/README.md.
// ============================================================================

struct UniformRow {
    const char* shader;      // file name in the staged shader directory
    uint32_t set;
    uint32_t binding;
    const char* cpu_type;    // for the failure message
    std::size_t cpu_size;
};

#define ROW(shader, set, binding, T) UniformRow{shader, set, binding, #T, sizeof(T)}

using namespace bracken;

const UniformRow UNIFORM_TABLE[] = {
    // ---- gpu/gpu_uniforms.hpp (all 37 structs; the three storage-buffer
    // records are covered by StructuredBufferStridesMatchCpuRecords below) ----
    ROW("model.vert.spv", 1, 0, gpu::ModelTransformUniforms),
    ROW("skinned_model.vert.spv", 1, 0, gpu::ModelTransformUniforms),
    ROW("model.frag.spv", 3, 0, gpu::ModelLightingUniforms),
    ROW("model.frag.spv", 3, 1, gpu::ShadowDataUniforms),
    ROW("model_instanced.frag.spv", 3, 1, gpu::ShadowDataUniforms),
    ROW("terrain.frag.spv", 3, 1, gpu::ShadowDataUniforms),
    ROW("grass.frag.spv", 3, 1, gpu::ShadowDataUniforms),
    ROW("skybox.frag.spv", 3, 0, gpu::SkyboxFragmentUniforms),
    ROW("sky_capture.frag.spv", 3, 0, gpu::SkyCaptureUniforms),
    ROW("ibl_prefilter.frag.spv", 3, 0, gpu::IblFilterUniforms),
    ROW("ibl_irradiance.frag.spv", 3, 0, gpu::IblFilterUniforms),
    ROW("grid.vert.spv", 1, 0, gpu::GridVertexUniforms),
    ROW("ui.vert.spv", 1, 0, gpu::UIScreenUniforms),
    ROW("ui.frag.spv", 3, 0, gpu::UIFragmentUniforms),
    ROW("model_instanced.vert.spv", 1, 0, gpu::InstancedCameraUniforms),
    ROW("particle.vert.spv", 1, 0, gpu::InstancedCameraUniforms),
    ROW("particle.vert.spv", 1, 1, gpu::ParticleBatchUniforms),
    ROW("particle.frag.spv", 3, 0, gpu::ParticleFragmentUniforms),
    ROW("model_instanced.frag.spv", 3, 0, gpu::InstancedLightingUniforms),
    ROW("shadow_depth_instanced.vert.spv", 1, 0, gpu::InstancedShadowUniforms),
    ROW("prepass_velocity_masked.frag.spv", 3, 0, gpu::PrepassMaskedUniforms),
    ROW("prepass_model.vert.spv", 1, 0, gpu::PrepassModelUniforms),
    ROW("prepass_model_masked.vert.spv", 1, 0, gpu::PrepassModelUniforms),
    ROW("prepass_skinned.vert.spv", 1, 0, gpu::PrepassModelUniforms),
    ROW("prepass_instanced.vert.spv", 1, 0, gpu::PrepassCameraUniforms),
    ROW("prepass_instanced_masked.vert.spv", 1, 0, gpu::PrepassCameraUniforms),
    ROW("prepass_terrain.vert.spv", 1, 0, gpu::PrepassCameraUniforms),
    // Depth-prepass G-buffer variants (ray-traced reflections). Same bodies as
    // the rows above, compiled with BRACKEN_PREPASS_GBUFFER, so they must back
    // onto the same CPU structs — that is exactly what this table pins.
    ROW("prepass_model_gbuffer.vert.spv", 1, 0, gpu::PrepassModelUniforms),
    ROW("prepass_skinned_gbuffer.vert.spv", 1, 0, gpu::PrepassModelUniforms),
    ROW("prepass_instanced_gbuffer.vert.spv", 1, 0, gpu::PrepassCameraUniforms),
    ROW("prepass_terrain_gbuffer.vert.spv", 1, 0, gpu::PrepassCameraUniforms),
    // The MASK G-buffer variants, split out of the velocity-only ones on
    // 2026-08-08 so they can carry a real normal (see
    // prepass_gbuffer_masked.frag.hlsl). Same bodies, same CPU structs.
    ROW("prepass_model_masked_gbuffer.vert.spv", 1, 0, gpu::PrepassModelUniforms),
    ROW("prepass_instanced_masked_gbuffer.vert.spv", 1, 0, gpu::PrepassCameraUniforms),
    ROW("prepass_gbuffer.frag.spv", 3, 0, gpu::PrepassGBufferUniforms),
    ROW("prepass_gbuffer_masked.frag.spv", 3, 0, gpu::PrepassMaskedUniforms),
    ROW("shadow_depth.vert.spv", 1, 0, gpu::ShadowTransformUniforms),
    ROW("shadow_skinned.vert.spv", 1, 0, gpu::ShadowSkinnedTransformUniforms),
    ROW("shadow_terrain.vert.spv", 1, 0, gpu::ShadowTerrainUniforms),
    ROW("shadow_depth.frag.spv", 3, 0, gpu::ShadowFragUniforms),
    ROW("gtao.frag.spv", 3, 0, gpu::GTAOUniforms),
    ROW("gtao_temporal.frag.spv", 3, 0, gpu::GtaoTemporalUniforms),
    ROW("gtao_upsample.frag.spv", 3, 0, gpu::GtaoUpsampleUniforms),
    ROW("blur_ao.frag.spv", 3, 0, gpu::BlurUniforms),
    ROW("composite.frag.spv", 3, 0, gpu::CompositeUniforms),
    ROW("fxaa.frag.spv", 3, 0, gpu::FxaaUniforms),
    ROW("taa_resolve.frag.spv", 3, 0, gpu::TaaResolveUniforms),
    ROW("taa_sharpen.frag.spv", 3, 0, gpu::TaaSharpenUniforms),
    // BloomDownsampleUniforms pads to 32 on both sides: the HLSL cbuffer
    // declares three trailing scalar pads so the declared extent matches
    // sizeof exactly (this test found the missing pads).
    ROW("bloom_downsample.frag.spv", 3, 0, gpu::BloomDownsampleUniforms),
    ROW("bloom_upsample.frag.spv", 3, 0, gpu::BloomUpsampleUniforms),
    ROW("volumetric_fog.frag.spv", 3, 0, gpu::VolumetricFogUniforms),
    ROW("exposure_histogram.comp.spv", 2, 0, gpu::ExposureHistogramUniforms),
    ROW("exposure_adapt.comp.spv", 2, 0, gpu::ExposureAdaptUniforms),
    // Probe-field GI (render/gi/). GiProbeTraceUniforms's leading 336 bytes are
    // ShadowDataUniforms field for field, which is what lets the trace shader
    // include shadow_common.hlsli and call calcShadow() instead of
    // reimplementing the standard-Z cascade math — so this row also pins that
    // embedding, not just the tail.
    ROW("gi_probe_trace.comp.spv", 2, 0, gpu::GiProbeTraceUniforms),
    // Raster GI (render/gi/voxel_*): the voxel radiance clipmap.
    ROW("voxel_voxelize.comp.spv", 2, 0, gpu::VoxelVoxelizeUniforms),
    ROW("gi_probe_blend.comp.spv", 2, 0, gpu::GiProbeBlendUniforms),
    ROW("gi_probe_blend_visibility.comp.spv", 2, 0, gpu::GiProbeBlendUniforms),
    ROW("gi_probe_classify.comp.spv", 2, 0, gpu::GiProbeClassifyUniforms),
    // Hierarchical-Z pyramid (render/hzb.hpp).
    ROW("hzb_copy.comp.spv", 2, 0, gpu::HzbCopyUniforms),
    ROW("hzb_reduce.comp.spv", 2, 0, gpu::HzbReduceUniforms),
    // Screen-probe gather (render/gi/screen_probes.hpp, R4b). Six passes, six
    // cbuffers. GiScreenProbeTraceUniforms's leading 336 bytes are
    // ShadowDataUniforms field for field, for the same reason
    // GiProbeTraceUniforms's are — so this row pins that embedding too.
    ROW("gi_screen_probe_place.comp.spv", 2, 0, gpu::GiScreenProbePlaceUniforms),
    ROW("gi_screen_probe_march.comp.spv", 2, 0, gpu::GiScreenProbeMarchUniforms),
    ROW("gi_screen_probe_args.comp.spv", 2, 0, gpu::GiScreenProbeArgsUniforms),
    ROW("gi_screen_probe_trace.comp.spv", 2, 0, gpu::GiScreenProbeTraceUniforms),
    ROW("gi_screen_probe_filter.comp.spv", 2, 0, gpu::GiScreenProbeFilterUniforms),
    ROW("gi_screen_probe_project.comp.spv", 2, 0, gpu::GiScreenProbeProjectUniforms),

    // ---- renderer-owned mirrors (same contract, different header) ----
    ROW("model.frag.spv", 3, 2, render::lighting::ClusterParams),
    ROW("model_instanced.frag.spv", 3, 2, render::lighting::ClusterParams),
    ROW("terrain.frag.spv", 3, 2, render::lighting::ClusterParams),
    ROW("grass.frag.spv", 3, 2, render::lighting::ClusterParams),
    // The compute mirror of the same cbuffer — cluster_compute.hlsli binds it
    // on set 2 (compute uniforms) instead of set 3 (fragment uniforms), at
    // whatever slot the including shader hands it. R2b's local-shadow trace is
    // its first and only real consumer; the R2a placeholder that existed purely
    // to make the include a compiled artifact is gone with it.
    ROW("rt_local_shadow.comp.spv", 2, 1, render::lighting::ClusterParams),
    // R5b's local gradient replay takes the same include and therefore the same
    // b1: it needs `clusterParams.view` for a view depth that agrees with the
    // CPU binner's, exactly as the trace does.
    ROW("rt_local_shadow_gradient.comp.spv", 2, 1, render::lighting::ClusterParams),
    ROW("terrain.vert.spv", 1, 0, render::TerrainTransformUniforms),
    ROW("terrain.frag.spv", 3, 0, render::TerrainLightingUniforms),
    ROW("grass.vert.spv", 1, 0, render::GrassVertexUniforms),
    ROW("prepass_grass.vert.spv", 1, 0, render::GrassVertexUniforms),
    ROW("grass.frag.spv", 3, 0, render::GrassLightingUniforms),
    ROW("water_sim.comp.spv", 2, 0, render::WaterRenderer::SimUniforms),
    ROW("rt_shadow.comp.spv", 2, 0, gpu::RtShadowUniforms),
    ROW("rt_shadow_denoise_classify.comp.spv", 2, 0, gpu::RtShadowClassifyUniforms),
    ROW("rt_shadow_denoise_temporal.comp.spv", 2, 0, gpu::RtShadowTemporalUniforms),
    ROW("rt_shadow_denoise_filter.comp.spv", 2, 0, gpu::RtShadowFilterUniforms),
    ROW("rt_reflection_denoise_temporal.comp.spv", 2, 0, gpu::RtReflectionTemporalUniforms),
    // R5c's Full-DI denoiser, the two passes of it.
    ROW("rt_di_denoise_temporal.comp.spv", 2, 0, gpu::RtDiTemporalUniforms),
    ROW("rt_di_denoise_spatial.comp.spv", 2, 0, gpu::RtDiSpatialUniforms),
    ROW("rt_di_noise.comp.spv", 2, 0, gpu::RtDiNoiseUniforms),
    ROW("rt_reflection.comp.spv", 2, 0, gpu::RtReflectionUniforms),
    // b1 of the same shader is ClusterGrid's own ClusterParams, pushed straight
    // from ClusterGrid::params() — its row is with the renderer-owned mirrors
    // above.
    ROW("rt_local_shadow.comp.spv", 2, 0, gpu::RtLocalShadowUniforms),
    // R5b's A-SVGF gradient chain. The reconstruct pass is dispatched once per
    // enabled signal against that signal's own targets, so it has one row.
    ROW("rt_shadow_gradient.comp.spv", 2, 0, gpu::RtShadowGradientUniforms),
    ROW("rt_local_shadow_gradient.comp.spv", 2, 0, gpu::RtLocalShadowGradientUniforms),
    ROW("rt_gradient_reconstruct.comp.spv", 2, 0, gpu::RtGradientReconstructUniforms),
    ROW("rt_reflection_resolve.frag.spv", 3, 0, gpu::RtReflectionResolveUniforms),
    ROW("water.vert.spv", 1, 0, render::WaterRenderer::VertexUniforms),
    ROW("water.frag.spv", 3, 0, render::WaterRenderer::FragmentUniforms),

    // ---- text: the CPU side pushes bare glm types (text_renderer.cpp) ----
    ROW("text.vert.spv", 1, 0, glm::mat4),
    ROW("text.frag.spv", 3, 0, glm::vec4),
};

#undef ROW

std::filesystem::path shader_dir() {
#ifdef BRACKEN_SHADER_SPV_DIR
    return std::filesystem::path(BRACKEN_SHADER_SPV_DIR);
#else
    return {};
#endif
}

bool shaders_available() {
    std::error_code ec;
    const std::filesystem::path dir = shader_dir();
    return !dir.empty() && std::filesystem::is_directory(dir, ec);
}

} // namespace

// Every CPU struct in the table declares exactly as many bytes as the shader
// that consumes it. This is the check CLAUDE.md asks for, against the compiled
// artifact rather than against a comment.
TEST(UniformLayout, BlockSizesMatchCpuStructs) {
    if (!shaders_available()) {
        GTEST_SKIP() << "compiled shader directory not available";
    }
    for (const UniformRow& row : UNIFORM_TABLE) {
        SCOPED_TRACE(::testing::Message() << row.cpu_type << " vs " << row.shader << " (set " << row.set << ", b"
                                          << row.binding << ")");
        SpirvModule module;
        ASSERT_TRUE(module.load(shader_dir() / row.shader)) << "could not read compiled shader";
        const SpirvModule::Block* block = module.find(row.set, row.binding);
        ASSERT_NE(block, nullptr) << "shader declares no uniform block at this set/binding";
        EXPECT_EQ(block->extent, row.cpu_size)
            << "CPU struct and HLSL cbuffer '" << block->type_name
            << "' disagree — edit BOTH sides (CLAUDE.md: the most common way to silently break rendering)";
    }
}

// The reverse direction. Without this, adding a cbuffer to a shader and
// forgetting the table row leaves it unchecked forever — the net would have a
// hole exactly where the newest code is.
TEST(UniformLayout, EveryShaderBlockIsAccountedFor) {
    if (!shaders_available()) {
        GTEST_SKIP() << "compiled shader directory not available";
    }
    // (shader, set, binding) triples covered by the table.
    std::set<std::string> covered;
    for (const UniformRow& row : UNIFORM_TABLE) {
        covered.insert(std::string(row.shader) + "|" + std::to_string(row.set) + "|" + std::to_string(row.binding));
    }
    // Storage buffers are element-stride checked separately, and the shared
    // set-0 StructuredBuffer bindings carry no cbuffer contract.
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(shader_dir(), ec)) {
        if (entry.path().extension() != ".spv") {
            continue;
        }
        const std::string name = entry.path().filename().string();
        SpirvModule module;
        ASSERT_TRUE(module.load(entry.path())) << name;
        for (const SpirvModule::Block& block : module.blocks()) {
            if (block.storage) {
                continue;
            }
            const std::string key = name + "|" + std::to_string(block.set) + "|" + std::to_string(block.binding);
            EXPECT_TRUE(covered.count(key) == 1)
                << "uniform block '" << block.type_name << "' in " << name << " (set " << block.set << ", b"
                << block.binding << ", " << block.extent
                << " B) is not in UNIFORM_TABLE — add a row naming the CPU struct that backs it";
        }
    }
    ASSERT_FALSE(ec) << "could not enumerate " << shader_dir();
}

// StructuredBuffer element strides: the per-instance records the renderer
// packs into a mapped transfer buffer. A stride disagreement shifts every
// instance after the first, which reads as "the whole scene is in the wrong
// place" rather than as a layout bug.
TEST(UniformLayout, StructuredBufferStridesMatchCpuRecords) {
    if (!shaders_available()) {
        GTEST_SKIP() << "compiled shader directory not available";
    }
    struct StrideRow {
        const char* shader;
        uint32_t set;
        uint32_t binding;
        const char* cpu_type;
        std::size_t cpu_size;
    };
    const StrideRow rows[] = {
        {"model_instanced.vert.spv", 0, 0, "gpu::InstanceData", sizeof(bracken::gpu::InstanceData)},
        {"prepass_instanced.vert.spv", 0, 0, "gpu::InstanceData", sizeof(bracken::gpu::InstanceData)},
        {"shadow_depth_instanced.vert.spv", 0, 0, "gpu::ShadowInstanceData",
         sizeof(bracken::gpu::ShadowInstanceData)},
        {"particle.vert.spv", 0, 0, "gpu::ParticleInstanceData", sizeof(bracken::gpu::ParticleInstanceData)},
        // The bone palette SSBO is a flat float4x4 stream (MAX_BONES caps the
        // CPU-side arrays; the GPU side has no compile-time bone count).
        {"skinned_model.vert.spv", 0, 0, "glm::mat4", sizeof(glm::mat4)},
        {"prepass_skinned.vert.spv", 0, 0, "glm::mat4", sizeof(glm::mat4)},
        {"shadow_skinned.vert.spv", 0, 0, "glm::mat4", sizeof(glm::mat4)},
        // The GI update list: the CPU writes one of these per probe scheduled
        // this frame, and three compute passes read it. Its slot differs per
        // shader because the trace binds the whole acceleration structure ahead
        // of it and the blend passes bind almost nothing.
        {"gi_probe_trace.comp.spv", 0, 12, "gpu::GiProbeUpdateRecord",
         sizeof(bracken::gpu::GiProbeUpdateRecord)},
        {"gi_probe_blend.comp.spv", 0, 2, "gpu::GiProbeUpdateRecord",
         sizeof(bracken::gpu::GiProbeUpdateRecord)},
        {"gi_probe_classify.comp.spv", 0, 1, "gpu::GiProbeUpdateRecord",
         sizeof(bracken::gpu::GiProbeUpdateRecord)},
        // The screen-probe header, written by the placement pass and read by
        // the four after it. Every one of them indexes the same array by probe
        // id, so a stride disagreement would not be a wrong value, it would be
        // a different probe.
        {"gi_screen_probe_place.comp.spv", 1, 0, "gpu::GiScreenProbeHeader",
         sizeof(bracken::gpu::GiScreenProbeHeader)},
        {"gi_screen_probe_march.comp.spv", 0, 2, "gpu::GiScreenProbeHeader",
         sizeof(bracken::gpu::GiScreenProbeHeader)},
        {"gi_screen_probe_trace.comp.spv", 0, 11, "gpu::GiScreenProbeHeader",
         sizeof(bracken::gpu::GiScreenProbeHeader)},
        {"gi_screen_probe_filter.comp.spv", 0, 3, "gpu::GiScreenProbeHeader",
         sizeof(bracken::gpu::GiScreenProbeHeader)},
        {"gi_screen_probe_project.comp.spv", 0, 1, "gpu::GiScreenProbeHeader",
         sizeof(bracken::gpu::GiScreenProbeHeader)},
    };
    for (const StrideRow& row : rows) {
        SCOPED_TRACE(::testing::Message() << row.cpu_type << " vs " << row.shader);
        SpirvModule module;
        ASSERT_TRUE(module.load(shader_dir() / row.shader));
        const SpirvModule::Block* block = module.find(row.set, row.binding);
        ASSERT_NE(block, nullptr) << "shader declares no storage buffer at this set/binding";
        ASSERT_TRUE(block->element_stride.has_value()) << "storage buffer has no runtime-array stride";
        EXPECT_EQ(*block->element_stride, row.cpu_size);
    }
}

// The reader itself must be able to fail: a truncated / non-SPIR-V file must
// be rejected rather than silently reflecting as "zero blocks", which would
// make every check above vacuously pass.
TEST(UniformLayout, ReaderRejectsNonSpirvInput) {
    SpirvModule module;
    EXPECT_FALSE(module.load(std::filesystem::path("this-file-does-not-exist.spv")));
#ifdef BRACKEN_REPO_ROOT
    // A real file that is not SPIR-V (wrong magic).
    const std::filesystem::path source = std::filesystem::path(BRACKEN_REPO_ROOT) / "CMakeLists.txt";
    std::error_code ec;
    if (std::filesystem::exists(source, ec)) {
        SpirvModule not_spirv;
        EXPECT_FALSE(not_spirv.load(source));
    }
#endif
}
