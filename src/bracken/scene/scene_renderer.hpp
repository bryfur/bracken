#pragma once

#include "bracken/contracts/graphics_settings.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_debug.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
// Needed in full: the inline instance-batch walk dereferences ModelManager/Model.
#include "bracken/contracts/camera_state.hpp"
#include "bracken/contracts/environment.hpp"
#include "bracken/contracts/frustum.hpp"
#include "bracken/contracts/occlusion.hpp"
#include "bracken/contracts/render_stats.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
#include "bracken/render/raytracing/emissive_clusters.hpp"
#include "bracken/render/gi/gi_bindings.hpp"
#include "bracken/render/gi/probe_volume.hpp"
#include "bracken/render/gi/voxel_gi.hpp"
#include "bracken/render/gi/screen_probes.hpp"
#include "bracken/render/hzb.hpp"
#include "bracken/render/rt_renderer.hpp"
#include "bracken/render/lighting/local_shadows.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/motion_history.hpp"
#include "bracken/scene/scene_prep.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Forward declarations for types only used in private implementation
namespace bracken::render {
class RenderContext;
class TerrainRenderer;
class WorldRenderer;
class UIRenderer;
class EffectRenderer;
class GrassRenderer;
class ShadowMap;
class PostProcessStack;
class GTAO;
class Bloom;
class VolumetricFog;
class TAA;
class AutoExposure;
class IBL;
class WaterRenderer;
} // namespace bracken::render

namespace bracken::gpu {
class PipelineRegistry;
}

namespace bracken::core::jobs {
class JobSystem;
}

namespace bracken {
class ModelManager;
struct Heightmap;
struct HeightmapGrid;
struct EffectDefinition;
struct Model;
struct Mesh;
} // namespace bracken

namespace bracken::scene {

namespace gpu = bracken::gpu;
namespace render = bracken::render;

/**
 * SceneRenderer consumes a sealed FramePacket and produces a rendered frame.
 * Owns all GPU resources, sub-renderers, and rendering state.
 * The host describes a frame through a FrameBuilder; SceneRenderer draws it.
 */
class SceneRenderer {
public:
    SceneRenderer();
    ~SceneRenderer();

    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;
    SceneRenderer(SceneRenderer&&) = delete;
    SceneRenderer& operator=(SceneRenderer&&) = delete;

    /**
     * Initialize all rendering subsystems.
     * @param context RenderContext owned by the application (must outlive this)
     * @param world_width World width for terrain/world renderers
     * @param world_height World height for terrain/world renderers
     */
    [[nodiscard]] bool init(render::RenderContext& context, float world_width = 32000.0f, float world_height = 32000.0f);

    void shutdown();

    /**
     * Render a complete frame from a SEALED (or Consuming) FramePacket — the
     * frame's immutable description: 3D commands, packet-owned bone palettes,
     * lights, billboards, debug lines, UI commands, camera, environment,
     * timing (dt = frame.dt()) and the engine section (sim-advanced
     * scene/sky clocks + the effect-instance snapshot the particle pass
     * consumes — the renderer never touches the live, sim-stage-owned
     * EffectSystem; effect SPAWNS are applied on the sim side before seal).
     * The packet is genuinely const: nothing in it is mutated or retained,
     * and nothing in it references host storage (the old bone-matrix span
     * lifetime contract is gone). The caller retires the packet wholesale
     * after this returns.
     *
     * Camera/environment are per-frame optional: has_environment() adopts the
     * packet snapshot into env_ (otherwise env_ — live-editable via
     * environment(), e.g. the dev overlay — is used unchanged). Without a
     * camera this frame the previous frame's camera is reused; if no camera
     * was ever set the 3D passes are skipped (UI still renders).
     */
    void render_frame(const FramePacket& frame);

    // ========== Configuration ==========

    /// The single-map world. Since paged terrain landed this is the 1x1
    /// adapter over set_heightmap_grid and nothing else.
    void set_heightmap(const Heightmap& heightmap);
    /**
     * An N x M grid of heightmap PAGES (docs/gap_streaming.md). Installs the
     * per-page terrain meshes and rewires the grass height binding — which
     * for a paged grid is a COARSE world-spanning proxy, so grass follows a
     * smoothed terrain (a known limitation of this landing, see
     * TerrainRenderer::heightmap_texture).
     *
     * A grid that fails HeightmapGrid::validate() is REFUSED with a logged
     * error and the previous terrain is kept — see heightmap_grid.hpp for the
     * seam-quantization trap that check exists for.
     */
    void set_heightmap_grid(const HeightmapGrid& grid);
    void set_anisotropic_filter(int level);
    void set_graphics_settings(const GraphicsSettings& settings);
    void set_vsync_mode(int mode);
    int max_vsync_mode() const;
    void set_screen_size(int width, int height);

    // ========== Debug Stats ==========

    void set_collect_stats(bool enabled) { collect_stats_ = enabled; }
    bool collect_stats() const { return collect_stats_; }
    const bracken::RenderStats& render_stats() const { return render_stats_; }

    /**
     * Everything the last rendered frame threw away (unknown model handles,
     * skinned draws with no skeleton, over-budget palettes, dropped lights,
     * a missing camera). Unlike render_stats() these are accumulated
     * unconditionally — a drop is never something a host opted out of
     * knowing about.
     */
    const bracken::FrameDiagnostics& frame_diagnostics() const { return frame_diagnostics_; }

    /**
     * Diagnostic predicate: a frame carrying 3D content that is rendered
     * before any camera was ever set draws nothing at all. Device-free static
     * so the check is testable headless.
     */
    static bool frame_needs_camera(const FramePacket& frame, bool have_camera) {
        return frame.has_3d_content() && !have_camera;
    }

    // ========== Screenshot ==========

    /**
     * Capture the next rendered frame (final swapchain image, UI included) to
     * a PNG at `path`. One-shot; the capturing frame submits with a blocking
     * GPU wait, so expect a hitch — intended for integration tests and debug.
     */
    void request_screenshot(std::string path) { pending_screenshot_path_ = std::move(path); }

    // ========== Accessors ==========

    render::RenderContext* context();
    render::TerrainRenderer& terrain();
    ModelManager& models();
    render::GrassRenderer* grass();

    /** Water surface + wave simulation. Configure via water()->set_region()
     *  (nothing renders until a region is set); inject ripples with
     *  water()->add_ripple(). Null until init() succeeds. */
    render::WaterRenderer* water();
    gpu::PipelineRegistry* pipeline_registry() { return pipeline_registry_.get(); }
    float get_terrain_height(float x, float z);

    /** Measure text at `scale` in the UI font (pixels). 0 before init() —
     *  hosts laying out UI (e.g. wrapped text emission) query these. */
    int text_width(std::string_view text, float scale = 1.0f) const;
    int text_height(float scale = 1.0f) const;

    /** Live-editable environment (sun/sky/fog — single source of truth).
     *  Mutations take effect next frame: render_frame re-propagates env_ to
     *  every sub-renderer at the top of each frame. */
    render::EnvironmentState& environment() { return env_; }

    /**
     * Build the ray-tracing acceleration structure over the loaded models.
     *
     * Call once after models are loaded, and only if rt_shadows or
     * rt_reflections will be used — nothing is built otherwise. `want_attributes`
     * must be true if reflections may be enabled this session; see
     * render::RtRenderer.
     *
     * Models release their CPU geometry at GPU upload, so this needs models
     * that still have it (the same constraint load_model_with_lods documents).
     */
    void build_raytracing_scene(const ModelManager& models, bool want_attributes);

    /**
     * Forget that the acceleration structure was ever built, so the next frame
     * that needs one rebuilds it from scratch.
     *
     * THE ONLY CORRECT ANSWER TO AN EVICTION. `RtScene` rebuilds its TLAS when
     * the instance SET changes — a hash of the count and the handle sequence —
     * and unloading a model changes NEITHER, so nothing fires on its own and
     * the BLAS keeps pointing at the CPU geometry `ModelManager::unload_model`
     * just freed. `ResidencyManager` calls this through its geometry
     * invalidator; anything else that unloads or reuploads a model must too.
     *
     * Cheap to call when ray tracing is off (it clears a bool); expensive when
     * it is on (the next frame pays a full BLAS build, ~1 s for a village).
     */
    void invalidate_raytracing_scene() { rt_scene_built_ = false; }

    /// Per-frame instance list for the TLAS. Static scenes can call this once.
    void set_raytracing_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances);

    [[nodiscard]] const render::RtStats& raytracing_stats() const;

    /**
     * R7 tuning for the traced LOCAL-light tier, and the noise probe's switch.
     *
     * A struct of its own rather than four setters, and here rather than on
     * `GraphicsSettings`, because none of it is a quality LEVEL a player would
     * ever be shown: two of the fields exist so a measurement can turn the R7
     * package off and compare against itself, and the third costs a dispatch to
     * produce a number the frame does not use. `GraphicsSettings` is the
     * player-facing contract and these are a developer's instruments.
     *
     * The defaults are the shipped configuration, so a host that never touches
     * this gets the tier as documented.
     */
    struct RtLocalTuning {
        /// k — spatial reservoir-reuse neighbours (R7a). 0 reproduces the
        /// pre-R7a sampling bit for bit, in BOTH tiers.
        uint32_t spatial_neighbors = 3;
        /// Radius of the neighbour ring, in traced texels.
        float spatial_radius = 8.0f;
        /// A-trous iterations of the Full-DI spatial filter (R7c). 1 is the R5c
        /// chain; 2 adds a stride-2 pass that skips converged texels.
        uint32_t di_spatial_iterations = 2;
        /// Measure the Full-DI tier's frame-to-frame noise (R7). Off by
        /// default: it is a whole dispatch spent on a diagnostic.
        bool di_noise_probe = false;
        /// M_e — emissive-mesh candidates drawn per pixel (R6). **0 turns the
        /// emissive candidate stream off entirely**, which is the A/B control
        /// the §5.7 cost figures are measured against: with no records sent,
        /// every RIS weight in the trace collapses to its R5c form bit for bit.
        uint32_t emissive_candidates = 2;
        /// Fraction of each emissive candidate draw taken from the pixel's
        /// PER-FROXEL list rather than from the scene-wide power alias table
        /// (R8, docs/rt_lighting_plan.md §5.7's follow-up 2). **0 reproduces
        /// R6's purely global draw exactly** — the mixture collapses to its
        /// second component — which is the before column of that measurement.
        float emissive_froxel_share = 1.0f;
    };
    void set_rt_local_tuning(const RtLocalTuning& tuning) { rt_local_tuning_ = tuning; }
    [[nodiscard]] const RtLocalTuning& rt_local_tuning() const { return rt_local_tuning_; }

    /** Probe-field GI accounting — cascade count, probe count, how many probes
     *  the scroll invalidated this frame, and the atlas bytes. All zero (and
     *  `active == false`) while `gi_mode == GiMode::Off`, which is the default.
     *  Read-only, main-thread; mirrors raytracing_stats(). */
    [[nodiscard]] const render::gi::GiStats& gi_stats() const;

    /** The hierarchical-Z pyramid, or null when it is not built.
     *
     *  Built only for `gi_mode == GiMode::ProbesScreen` (R4a) — the pass is
     *  gated on its future consumer so that no configuration that exists today
     *  allocates it or spends a dispatch on it. Exposed so a host can report
     *  `hzb=WxH mips=N`; the pyramid's texture is renderer-internal. */
    [[nodiscard]] const render::HzbPyramid* hzb() const { return hzb_.get(); }

    /** The screen-probe gather, or null when it is not allocated.
     *
     *  Built only for `gi_mode == GiMode::ProbesScreen` (R4b). Exposed for the
     *  smoke line and the dev overlay; the atlases are renderer-internal. */
    [[nodiscard]] const render::gi::ScreenProbeGather* screen_probes() const {
        return screen_probes_.get();
    }

    /** Effect/particle counts of the LAST CONSUMED packet's effect-instance
     *  snapshot — render-side data, safe for the dev overlay to read while
     *  the next frame's sim is in flight (the live EffectSystem is
     *  sim-stage-owned and must never be read from the render side). */
    uint32_t frame_effect_count() const { return frame_effect_count_; }
    uint32_t frame_particle_count() const { return frame_particle_count_; }

    /** Read-only view of the sky IBL prefilter chain (refresh stats for the
     *  dev overlay / smoke logging). Null until init() succeeds. */
    const render::IBL* ibl() const { return ibl_.get(); }

    /** Histogram auto-exposure system — adapted-EV debug readback for the
     *  dev overlay / smoke logging, plus metering tunables (percentiles,
     *  target gray). Null until init() succeeds. */
    render::AutoExposure* auto_exposure() { return auto_exposure_.get(); }
    const render::AutoExposure* auto_exposure() const { return auto_exposure_.get(); }

    /** Read-only view of the local (point/spot) shadow system — selection /
     *  tile-cache stats for the dev overlay and smoke logging. Null until
     *  init() succeeds. */
    const render::lighting::LocalShadowMap* local_shadows() const { return local_shadows_.get(); }

    // ========== Motion vectors / TAA foundation ==========

    /** Whether the TAA sub-pixel camera jitter (Halton 2,3) was applied to
     *  this frame's draw projection. Derived per frame from the graphics
     *  settings: ON exactly when `aa_mode == AAMode::TAA` and the TAA resolve
     *  can actually run (AO/offscreen path + TAA history initialized) — any
     *  other AA mode renders unjittered. Velocity output is unaffected either
     *  way (it always uses unjittered matrices). */
    bool taa_jitter_enabled() const { return taa_jitter_enabled_; }

    /** Screen-size RG16F motion-vector target written by the depth prepass
     *  (UV-space delta, current minus previous — see
     *  scene/velocity_math.hpp for the exact convention). Null until the
     *  first prepass frame; contents are only valid for frames where
     *  depth_prepass_active() was true. */
    gpu::GPUTexture* velocity_texture() { return velocity_texture_.get(); }

    /** True when this frame ran the depth prepass (velocity + depth laid
     *  down before the forward pass; opaque forward draws used EQUAL). */
    bool depth_prepass_active() const { return frame_prepass_active_; }

    /** This frame's / previous frame's UNJITTERED view-projection. TAA and
     *  any depth-reprojection consumer must use these, never the jittered
     *  draw matrices. */
    const glm::mat4& current_view_projection_unjittered() const { return frame_cam_.view_proj_unjittered; }
    const glm::mat4& previous_view_projection_unjittered() const { return prev_view_proj_unjittered_; }

    // Post-UI callback: called after UI rendering, before end_frame.
    // Receives (command_buffer, swapchain_texture) for additional render passes (e.g. ImGui).
    using PostUICallback = std::function<void(SDL_GPUCommandBuffer*, SDL_GPUTexture*)>;
    void set_post_ui_callback(PostUICallback cb) { post_ui_callback_ = std::move(cb); }

    // ========== CPU Prepare fan-out (docs/multithreading_plan.md §4.3) ==========
    // The prep cores themselves live in scene/scene_prep.hpp as
    // ScenePrep — device-free statics on a type that structurally cannot
    // reach a main-thread API, which is what makes the renderer's only
    // thread-tier boundary a compile-time fact rather than a comment. What
    // stays here is the three thin adapters that touch renderer members:
    // build_instance_batches / pack_bone_palettes gather the params and
    // outputs, upload_bone_palettes maps the transfer buffer and hands
    // ScenePrep::prep_copy_palettes a plain pointer.

    // Cascade count of the CSM atlas, defined by ScenePrep (whose scratch
    // pools are sized by it) and mirrored here because the renderer's
    // per-cascade member arrays use it too.
    static constexpr int MAX_SHADOW_CASCADES = ScenePrep::MAX_SHADOW_CASCADES;

    // ========== The instance-batch walk: ONE definition of the order ==========
    /**
     * Walk the frame's non-empty instance bins in PACKING order — ascending
     * ModelHandle, with a running `base_instance` — and return the offset one
     * past the last bin (what the shadow packer carries into the next
     * cascade).
     *
     * upload_instance_buffers() packs the bins into the instance storage
     * buffer in exactly this order, and three draw walks — forward
     * (render_instanced_models), depth prepass (prepass_draw_instanced) and
     * shadow (render_instanced_shadow_models) — index into that packed buffer
     * with the same running offset. All four must agree exactly, and they now
     * live in four different translation units, so the order has exactly one
     * definition: this function.
     *
     * The subtle half is a bin whose handle no longer resolves to a Model. It
     * was still PACKED — the pack loop never looks a model up — so a draw
     * walk must skip the draw yet still ADVANCE past its range
     * (for_each_drawable_instance_batch does exactly that). Getting it wrong
     * makes every later bin render with another model's transforms, which is
     * visible only in a scene carrying both an unresolvable handle and >= 2
     * model types. Device-free and static so that contract is testable
     * headless (scene/instance_batch_walk.test.cpp).
     *
     * fn(ModelHandle, const std::vector<Instance>&, uint32_t base_instance)
     */
    template <typename Instance, typename Fn>
    static uint32_t for_each_instance_batch(const std::vector<std::vector<Instance>>& bins, uint32_t base_instance,
                                            Fn&& fn) {
        for (ModelHandle handle = 0; handle < bins.size(); ++handle) {
            const std::vector<Instance>& instances = bins[handle];
            if (instances.empty()) {
                continue;
            }
            fn(handle, instances, base_instance);
            base_instance += static_cast<uint32_t>(instances.size());
        }
        return base_instance;
    }

private:
    // Frame lifecycle
    void begin_frame();
    void end_frame();
    // Records a swapchain download, fence-submits the frame, writes the PNG.
    void capture_screenshot_and_end_frame();
    void begin_main_pass();
    void end_main_pass();
    void begin_ui();
    void end_ui();

    // Rendering
    void render_3d_scene(const FramePacket& frame, const CameraState& camera);
    void render_model_command(const ModelDrawCommand& cmd, const CameraState& camera);
    void render_model_command_inner(const ModelDrawCommand& cmd, const CameraState& camera);
    void render_skinned_model_command_inner(const SkinnedDrawCommand& cmd, Model& model, const CameraState& camera,
                                            uint32_t palette_offset);
    void build_instance_batches(const FramePacket& frame, const CameraState& camera, const Frustum& frustum,
                                int active_cascades);
    void upload_instance_buffers(SDL_GPUCopyPass* copy_pass);
    void pack_bone_palettes(const FramePacket& frame, const CameraState& camera);
    void upload_bone_palettes(SDL_GPUCopyPass* copy_pass, const FramePacket& frame);
    void render_instanced_models(const CameraState& camera);
    void render_instanced_shadow_models(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                        const glm::mat4& light_view_projection, int cascade);

    // Drawing form of for_each_instance_batch (below): bins whose handle does
    // not resolve to a Model are skipped but still consume their
    // base_instance range — see the base walk's contract.
    // fn(Model&, const std::vector<Instance>&, uint32_t base_instance)
    template <typename Instance, typename Fn>
    uint32_t for_each_drawable_instance_batch(const std::vector<std::vector<Instance>>& bins, uint32_t base_instance,
                                              Fn&& fn) {
        return for_each_instance_batch(
            bins, base_instance, [&](ModelHandle handle, const std::vector<Instance>& instances, uint32_t base) {
                Model* model = (handle != INVALID_MODEL_HANDLE) ? model_manager_->get_model(handle) : nullptr;
                if (!model) {
                    return; // dropped draw, but the range is still packed
                }
                fn(*model, instances, base);
            });
    }

    // Depth prepass: velocity (RG16F) + scene depth for every opaque
    // (alphaMode OPAQUE) draw, before the forward pass. Returns true when the
    // pass was recorded (forward opaques then depth-test EQUAL, writes off).
    // `draw_camera` carries the jittered projection; velocity uses the
    // frame's unjittered matrices.
    [[nodiscard]] bool render_depth_prepass(const FramePacket& frame, const CameraState& draw_camera);
    /// `base` when this frame's prepass writes velocity only, its G-buffer
    /// twin when the oct-normal/roughness/metallic attachment is live.
    [[nodiscard]] gpu::PipelineType prepass_pipeline(gpu::PipelineType base) const;
    /// Bind the mesh's metallicRoughness map + push PrepassGBufferUniforms for
    /// one G-buffer prepass draw. Shared by all three model walks so the
    /// prepass cannot classify a surface differently from the forward pass.
    void push_prepass_gbuffer_material(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const Mesh& mesh);
    void prepass_draw_instanced(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                const gpu::PrepassCameraUniforms& camera_uniforms);
    void prepass_draw_static_models(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                    const CameraState& draw_camera);
    void prepass_draw_skinned_models(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                     const CameraState& draw_camera);

    // Bone-palette double buffer: [bone_palette_current_] is written/bound
    // this frame; the other slot still holds last frame's palettes for the
    // skinned prepass' previous-position path.
    gpu::GPUBuffer* current_bone_palette_buffer() { return bone_palette_buffers_[bone_palette_current_].get(); }
    gpu::GPUBuffer* prev_bone_palette_buffer() { return bone_palette_buffers_[bone_palette_current_ ^ 1].get(); }

    // Transparent (glTF alphaMode BLEND) path: meshes queued during the opaque
    // walks, then drawn back-to-front after the skybox with depth writes off.
    void queue_transparent_mesh(const Mesh& mesh, const glm::mat4& transform, const glm::mat4& normal_matrix,
                                const glm::vec4& tint, bool no_fog, uint32_t palette_offset, const Model& model,
                                const CameraState& camera);
    void render_transparent_models(const CameraState& camera);

    // Bind the model/skinned pipeline variant (double-sided cull + post-
    // prepass EQUAL depth) and re-establish the pass-level fragment state
    // that follows a bind.
    bool bind_model_pipeline(bool double_sided, bool depth_equal = false);
    bool bind_skinned_pipeline(bool double_sided, bool depth_equal = false);
    void render_ui_commands(const FramePacket& frame);
    void draw_billboard_3d(const BillboardDrawCommand& cmd, const CameraState& camera);
    void upload_debug_lines(const FramePacket& frame, SDL_GPUCopyPass* copy_pass);
    void render_debug_lines(const CameraState& camera);

    // Shadow rendering (casters come from the per-frame resolved scratch:
    // instance batches + non_instanced_commands_ + skinned_resolved_).
    void render_shadow_passes();
    void bind_shadow_data(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, int sampler_slot);

    // Local (point/spot) light shadows: draws the tiles the selector marked
    // for re-render this frame into the local atlas (casters = model +
    // skinned commands within each light's radius; terrain and grass do not
    // cast local shadows). Runs after the CSM shadow passes.
    void render_local_shadow_passes(const FramePacket& frame, const CameraState& camera);
    // Bind the local-shadow atlas + comparison sampler at the given fragment
    // sampler slot of the current main pass (falls back to the CSM atlas as a
    // typed filler when the local atlas is unavailable — the slot is always
    // declared but only sampled for lights with shadow_slot >= 0).
    void bind_local_shadow_atlas(uint32_t slot);

    // Model-pipeline material state. Lighting uniforms and per-slot texture
    // bindings persist across draws within a pass, so both are pushed lazily;
    // reset_model_bind_state() invalidates after every model-family pipeline bind.
    void reset_model_bind_state();
    void bind_material_texture(uint32_t slot, SDL_GPUTexture* texture);
    void bind_mesh_material_textures(const Mesh& mesh);
    gpu::ModelLightingUniforms make_model_lighting(const CameraState& camera, bool fog_active) const;

    // Bind clustered-lighting storage buffers + ClusterParams uniform on the
    // current main pass. Slot 2 is reserved for ClusterParams across all main-pass
    // fragment shaders (model/instanced/terrain/grass).
    void bind_cluster_lights(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, int uniform_slot);

    // Bind the frame's screen-space AO texture (or the 1x1 white dummy when
    // `enabled` is false / AO didn't run) at the given fragment sampler slot
    // of the current main pass. Every model-family pipeline bind must include
    // this — the shaders always declare the slot.
    void bind_screen_ao(uint32_t slot, bool enabled = true);

    // Bind the frame's ray-traced sun-visibility mask (or the 1x1 white dummy
    // when ray tracing is off) at the given fragment sampler slot. Every
    // model-family pipeline bind must include this — model.frag /
    // model_instanced.frag always DECLARE the slot, and an undeclared-but-
    // unbound sampler descriptor is undefined behaviour in SDL3 GPU. The white
    // dummy reads 1.0 = fully lit from any UV, which is the correct no-op.
    void bind_rt_shadow_mask(uint32_t slot);
    void bind_rt_local_shadow_mask(uint32_t slot);
    /// The Full-DI tier's two demodulated targets as fragment READ-ONLY STORAGE
    /// textures (R5c). `first_slot` is a storage-texture slot, numbered from 0
    /// independently of the sampler slots — SDL3 GPU binds the three classes
    /// through three separate calls even though the shader sees one contiguous
    /// set-2 run. Always called, with the 1x1 stand-in outside Full mode.
    void bind_local_di(uint32_t first_slot);
    /// `GraphicsSettings::rt_local_shadows` as the float every scene cbuffer
    /// carries: 2 in Full, 1 in Ratio, 0 otherwise. Derived from what actually
    /// came out of the renderer this frame, not from the setting, so a frame
    /// that could not produce the DI pair renders as Ratio rather than as
    /// nothing.
    [[nodiscard]] float local_rt_shadow_mode_value() const {
        if (frame_rt_local_di_active_) {
            return 2.0f;
        }
        return frame_rt_local_shadows_active_ ? 1.0f : 0.0f;
    }

    // Fill the 3-entry sky IBL binding array (irradiance cube, prefiltered
    // specular cube, BRDF LUT — real chain when the IBL renderer is ready,
    // cube-typed dummies otherwise). Returns false if no valid binding can be
    // built (no dummies either — only possible before init completes).
    bool fill_ibl_bindings(SDL_GPUTextureSamplerBinding out[3]);

    // Bind the sky IBL chain at [base_slot, base_slot + 3) of the current
    // main pass. Every model-family pipeline bind must include this — the
    // shaders always declare the slots (terrain/grass receive the same
    // bindings through their render() parameters instead).
    void bind_ibl(uint32_t base_slot);
    /// Bind the probe field's three atlases as one contiguous run starting at
    /// `base_slot` (model family; terrain/grass take `frame_gi_` by pointer).
    void bind_gi(uint32_t base_slot);
    /// Rebuild `frame_gi_` from the probe volume + screen-probe gather, or fill
    /// it with dummies.
    void refresh_gi_bindings();

    // Setup
    void init_pipelines();

    // ========== Sub-renderers ==========
    render::RenderContext* context_ = nullptr;
    std::unique_ptr<gpu::PipelineRegistry> pipeline_registry_;
    std::unique_ptr<render::TerrainRenderer> terrain_;
    std::unique_ptr<render::WorldRenderer> world_;
    std::unique_ptr<render::UIRenderer> ui_;
    std::unique_ptr<render::EffectRenderer> effects_;
    std::unique_ptr<ModelManager> model_manager_;
    std::unique_ptr<render::GrassRenderer> grass_renderer_;
    std::unique_ptr<render::ShadowMap> shadow_map_;
    std::unique_ptr<render::PostProcessStack> post_;
    std::unique_ptr<render::GTAO> gtao_;
    std::unique_ptr<render::Bloom> bloom_;
    std::unique_ptr<render::VolumetricFog> volumetric_fog_;
    std::unique_ptr<render::TAA> taa_;
    std::unique_ptr<render::AutoExposure> auto_exposure_;
    std::unique_ptr<render::IBL> ibl_;
    std::unique_ptr<render::WaterRenderer> water_;
    // Snapshots of the HDR color + depth taken between the opaque pass and
    // the water pass (what water refraction / shoreline softness sample —
    // the live targets stay bound for the actual draw). Screen-sized; lazily
    // (re)created, only while a water region is active.
    std::unique_ptr<gpu::GPUTexture> water_scene_copy_;
    std::unique_ptr<gpu::GPUTexture> water_depth_copy_;
    render::lighting::ClusterGrid cluster_grid_;
    bool cluster_grid_ready_ = false;
    // R8's second froxel binner (docs/rt_lighting_plan.md §5.7 follow-up 2):
    // emissive light records into the SAME froxel grid the lights were binned
    // into, so the traced local-light pass draws its emissive candidates from a
    // local list instead of a scene-wide alias table. Owned here rather than by
    // RtRenderer because it borrows `cluster_grid_.cluster_bounds()` — the two
    // binners must agree about where a froxel is, and the only way to guarantee
    // that is to test the same boxes.
    render::rt::EmissiveClusterGrid emissive_clusters_;
    bool emissive_clusters_ready_ = false;
    std::unique_ptr<render::lighting::LocalShadowMap> local_shadows_;

    // ========== GPU Resources ==========
    std::unique_ptr<gpu::GPUBuffer> debug_line_vertex_buffer_;
    size_t debug_line_buffer_capacity_ = 0; // max line count the buffer can hold
    size_t debug_line_vertex_count_ = 0;    // vertices uploaded this frame (consolidated copy pass)
    std::vector<float> debug_line_scratch_; // reused per frame to avoid alloc
    std::unique_ptr<gpu::GPUTexture> depth_texture_;
    SDL_GPUSampler* default_sampler_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> dummy_white_texture_;    // 1x1 white for binding slots that need a texture

    /// Probe-field GI. Created LAZILY on the first frame with
    /// `gi_mode != GiMode::Off` and DESTROYED the moment it goes back to Off —
    /// the same policy as normal_rough_texture_ above, and for the same reason:
    /// the field is ~30 MB of atlases, and a game that never enables GI must
    /// never pay for it. The cost of destroying rather than parking it is that
    /// toggling GI back on restarts from a cold field (every probe
    /// uninitialized), which is exactly what a fresh allocation would be
    /// anyway — the atlases hold no valid contents across a shutdown.
    std::unique_ptr<render::gi::ProbeVolume> probe_volume_;
    /// The RASTER ray source (GiMode::VoxelProbes). Allocated on the same
    /// lazy/latched policy as the probe field and destroyed the moment the mode
    /// leaves VoxelProbes — ~35 MB of volumes is not something to keep resident
    /// for a mode the host is not using.
    std::unique_ptr<render::gi::VoxelGi> voxel_gi_;
    bool voxel_gi_init_failed_ = false;
    /// Model set the voxel scene was built from, so the (expensive) gather runs
    /// when the models change rather than every frame.
    std::size_t voxel_gi_model_count_ = 0;
    /// Latched allocation failure, so a device that cannot give us the atlases
    /// costs one attempt rather than one per frame. Cleared when gi_mode goes
    /// back to Off, which is the only way to ask for a retry.
    bool gi_init_failed_ = false;

    /// Hierarchical-Z pyramid over the prepass depth. Same lazy-create /
    /// destroy-on-off policy as probe_volume_ and normal_rough_texture_, on the
    /// same argument: it is a screen-sized mip chain, and R4b's screen-probe
    /// gather is the only thing that will ever read it, so it is built only for
    /// GiMode::ProbesScreen. Nothing consumes it in R4a.
    std::unique_ptr<render::HzbPyramid> hzb_;
    /// Latched init failure (the device may not support R32_FLOAT with
    /// SIMULTANEOUS read-write) — same one-attempt policy as gi_init_failed_.
    bool hzb_init_failed_ = false;

    // ---- Hi-Z draw occlusion culling (GraphicsSettings::occlusion_culling) --
    //
    // A SECOND pyramid, and not an oversight. hzb_ retains the CLOSEST surface
    // because the screen-probe march needs that; culling needs the FARTHEST
    // (render/hzb.hpp hzb_reduce_far). They are opposite reductions of the same
    // buffer, so neither can be recovered from the other and one texture cannot
    // serve both. Same lazy-create / latched-failure / destroy-on-off policy as
    // hzb_.
    std::unique_ptr<render::HzbPyramid> hzb_far_;
    bool hzb_far_init_failed_ = false;
    /// Destination of the coarse-mip download, sized to the level.
    SDL_GPUTransferBuffer* occlusion_transfer_ = nullptr;
    std::size_t occlusion_transfer_bytes_ = 0;
    /// Fence of the readback submitted at the END of the previous frame, waited
    /// on DETERMINISTICALLY at the start of this one. Deterministically, not
    /// "if it happens to be signalled": a cull whose outcome depended on GPU
    /// timing would make the frame irreproducible run to run and quietly retire
    /// every pinned screenshot baseline. It is a full frame old and normally
    /// already signalled, so the wait is expected to cost nothing.
    SDL_GPUFence* occlusion_fence_ = nullptr;
    /// The downloaded level, owned for the whole frame so the span the Prepare
    /// stage reads cannot dangle.
    std::vector<float> occlusion_depth_;
    /// What the Prepare stage sees. Refilled each frame; `usable()` decides
    /// whether it is consulted at all.
    OcclusionField occlusion_field_;
    /// A download has completed at least once since the last resize/enable.
    bool occlusion_depth_ready_ = false;
    /// Mip the CPU reads. 4 => 100x57 floats at 1600x900, 23 KB: coarse enough
    /// that the download and the footprint walk are both trivial, fine enough
    /// that a prop-sized box still covers a texel or two.
    static constexpr int k_occlusion_level = 4;
    /// Camera state at the moment the depth in `occlusion_depth_` was
    /// rasterized — the inputs to the field's travel/rotation gates.
    glm::mat4 occlusion_view_proj_{1.0f};
    glm::vec3 occlusion_camera_position_{0.0f};
    glm::vec3 occlusion_camera_forward_{0.0f, 0.0f, -1.0f};
    /// Pending readback's camera state, promoted to the three above when the
    /// download it belongs to is adopted.
    glm::mat4 pending_occlusion_view_proj_{1.0f};
    glm::vec3 pending_occlusion_camera_position_{0.0f};
    glm::vec3 pending_occlusion_camera_forward_{0.0f, 0.0f, -1.0f};
    /// Camera of the pyramid build RECORDED this frame, held until the readback
    /// is submitted after the frame's command buffer (submit_occlusion_readback).
    glm::mat4 occlusion_build_view_proj_{1.0f};
    glm::vec3 occlusion_build_camera_position_{0.0f};
    glm::vec3 occlusion_build_camera_forward_{0.0f, 0.0f, -1.0f};
    /// A pyramid build was recorded on this frame's command buffer, so a
    /// download is owed once that buffer has been submitted.
    bool occlusion_build_recorded_ = false;
    /// Per-frame answer from RenderPlan::occlusion_active.
    bool frame_occlusion_active_ = false;

    /// Wait the previous frame's readback, adopt it into occlusion_depth_ and
    /// refill the geometric half of occlusion_field_. Main-thread.
    void resolve_occlusion_readback();
    /// Build the farthest-surface pyramid over this frame's prepass depth.
    /// RECORDS only; the download is submitted separately. Main-thread; no pass
    /// may be open on `cmd`.
    void build_occlusion_pyramid(SDL_GPUCommandBuffer* cmd, const CameraState& camera);
    /**
     * Request the coarse-mip download of the pyramid this frame built.
     *
     * MUST be called after the frame's command buffer has been submitted. SDL3
     * GPU executes command buffers in submission order and a readback rides its
     * OWN buffer, submitted the instant it is recorded — so requesting it beside
     * the build, which is only RECORDED on the frame's buffer, captures the
     * PREVIOUS frame's pyramid and pairs two-frame-old depth with a
     * one-frame-old matrix. Main-thread.
     */
    void submit_occlusion_readback();
    /// Drop the pyramid, the transfer buffer and any in-flight fence.
    void release_occlusion_resources();

    /// The R4b screen-probe gather. Same lazy-create / destroy-on-off policy as
    /// the pyramid it marches, and gated on the same mode: it is the quality
    /// tier of GiMode::ProbesScreen and nothing else allocates it.
    std::unique_ptr<render::gi::ScreenProbeGather> screen_probes_;
    /// Latched allocation failure — same one-attempt policy as gi_init_failed_.
    bool screen_probes_init_failed_ = false;

    // Compute-shader ray tracing. Null-checked everywhere: both features are
    // off by default and nothing is built unless one is on.
    std::unique_ptr<render::RtRenderer> rt_renderer_; // created in init()
    bool frame_rt_shadows_active_ = false;
    /// Whether the froxel-RIS local-light shadow pass produced a mask this
    /// frame. Also the reason `frame_local_shadows_active_` is false: in Ratio
    /// mode the traced signal OWNS local shadowing and the rasterized atlas is
    /// not updated at all.
    bool frame_rt_local_shadows_active_ = false;
    /// Whether that pass ran in the FULL-DI tier this frame (R5c). When it did,
    /// the four scene shaders drop their analytic point/spot loop entirely and
    /// read the two demodulated DI storage textures below instead — so this is
    /// not "an extra effect on top of Ratio", it is which of two mutually
    /// exclusive code paths the frame's local lighting came out of.
    bool frame_rt_local_di_active_ = false;
    RtLocalTuning rt_local_tuning_{};
    /// Normal+roughness G-buffer for ray-traced reflections. Allocated only
    /// while reflections are on, and released the moment they go off — a game
    /// that never enables them never pays for a screen-sized target.
    std::unique_ptr<gpu::GPUTexture> normal_rough_texture_;
    /// Whether THIS frame's prepass attached the G-buffer, and therefore
    /// whether the prepass draws must use the *GBuffer pipeline variants: a
    /// pass and its pipelines must agree on the attachment count.
    bool frame_prepass_gbuffer_ = false;
    /// The roughness handover window this frame's reflection pass used, or 0
    /// when it did not run (max_roughness == 0 is the shaders' "reflections
    /// off" encoding). The forward shaders scale their IBL specular lobe by
    /// one minus the traced weight computed from these two, using the same
    /// function the compute pass writes into its alpha
    /// (shaders/reflection_gate.hlsli), so the two sum to one.
    float frame_rt_reflection_max_roughness_ = 0.0f;
    float frame_rt_reflection_fade_start_ = 0.0f;
    /// Whether the BLAS set has been built, and whether it was built with the
    /// per-triangle attributes reflections need. Both drive the lazy (re)build
    /// in ensure_raytracing_scene.
    bool rt_scene_built_ = false;
    bool rt_scene_has_attributes_ = false;
    /// Reused across frames so rebuilding the TLAS from the packet does not
    /// allocate every frame.
    std::vector<std::pair<ModelHandle, glm::mat4>> rt_instances_scratch_;
    /// The same list for the RASTER GI path, gathered independently because
    /// `rt_instances_scratch_` is filled inside the ray-tracing block that
    /// GiMode::VoxelProbes deliberately skips. Sharing it would have made the
    /// voxel clipmap silently empty whenever ray tracing was off — which is the
    /// only configuration the mode exists for.
    std::vector<std::pair<ModelHandle, glm::mat4>> voxel_instances_scratch_;

    /// Build (or rebuild) the acceleration structure if ray tracing needs one
    /// and the current one will not do. Called from the frame, so turning a
    /// toggle on mid-session works without the host doing anything.
    void ensure_raytracing_scene(bool want_attributes);
    /// Refresh the TLAS from THIS frame's draws — the packet is the source of
    /// truth for what is on screen, so the traced scene cannot desync from the
    /// rasterized one.
    /// @param cmd the frame's command buffer, so a TLAS that actually changed
    ///   is uploaded into it instead of through its own mid-frame submit.
    /// @param camera_pos the draw camera's world position, the origin the range
    ///   filter measures from.
    /// @param max_range GraphicsSettings::rt_max_instance_range in world units
    ///   (rt_range_units). Infinity — the default — skips the filter entirely.
    void refresh_raytracing_instances(const FramePacket& frame, SDL_GPUCommandBuffer* cmd,
                                      const glm::vec3& camera_pos, float max_range);
    std::unique_ptr<gpu::GPUTexture> default_normal_texture_; // 1x1 (128,128,255) flat tangent-space normal
    std::unique_ptr<gpu::GPUTexture> dummy_black_texture_;    // 1x1 black emissive fallback
    std::unique_ptr<gpu::GPUTexture> dummy_black_cube_;       // 1x1x6 black cube for the IBL slots when
                                                              // the IBL renderer is unavailable (a cube-
                                                              // declared slot must get a cube texture)

    // ========== Instanced Rendering ==========
    // Per-model instance bins indexed directly by ModelHandle (a dense index
    // into ModelManager's slot array). The touched lists make per-frame
    // clearing O(touched); packing/drawing iterate the dense vector in
    // ascending handle order (stable, texture-coherent), skipping empty bins.
    std::vector<std::vector<gpu::InstanceData>> instance_batches_;
    std::vector<ModelHandle> instance_batches_touched_;
    // Per-cascade shadow batches: each cascade has its own light-frustum-culled set of casters.
    // Only casters inside that cascade's light frustum are added, which dramatically
    // reduces shadow draw work for tight near cascades.
    std::array<std::vector<std::vector<gpu::ShadowInstanceData>>, MAX_SHADOW_CASCADES> shadow_instance_batches_;
    std::array<std::vector<ModelHandle>, MAX_SHADOW_CASCADES> shadow_batches_touched_;
    std::vector<const ModelDrawCommand*> non_instanced_commands_; // individual draw fallback
    std::unique_ptr<gpu::GPUBuffer> instance_storage_buffer_;
    size_t instance_storage_capacity_ = 0;
    std::unique_ptr<gpu::GPUBuffer> shadow_instance_storage_buffer_;
    size_t shadow_instance_storage_capacity_ = 0;

    // Per-cascade base_instance / count into shadow_instance_storage_buffer_
    // after packing all cascades contiguously in upload_instance_buffers().
    std::array<uint32_t, MAX_SHADOW_CASCADES> shadow_cascade_base_instance_{};

    // Per-cascade light frustums extracted from cascade.light_view_projection.
    std::array<Frustum, MAX_SHADOW_CASCADES> shadow_cascade_frustums_{};

    // ========== Transparent Queue ==========
    // glTF alphaMode BLEND meshes collected during the opaque walks (instanced
    // draws fall back to one queued entry per instance) and drawn sorted
    // back-to-front after the skybox. No OIT; sort granularity is one mesh per
    // command/instance (see docs/architecture.md).
    struct TransparentDraw {
        const Mesh* mesh;
        glm::mat4 transform;
        glm::mat4 normal_matrix;
        glm::vec4 tint;
        uint32_t palette_offset; // INVALID_PALETTE_OFFSET => static (Vertex3D) draw
        bool no_fog;
        float view_dist_sq;
    };
    std::vector<TransparentDraw> transparent_draws_;
    static constexpr uint32_t INVALID_PALETTE_OFFSET = 0xFFFFFFFFu;
    // Cull + depth variant currently bound for the model / skinned pipeline
    // families (only meaningful while that family is the active pipeline).
    //
    // ENTRY-ORDER DEPENDENCY (survey S16). These four are written ONLY by
    // bind_model_pipeline / bind_skinned_pipeline and are never reset at a
    // pass or frame boundary, so on their own they describe the last bind of
    // any earlier pass. What makes them correct is that the first model /
    // skinned draw of every frame enters through render_3d_scene, which binds
    // unconditionally before the walk — after that the flags track the live
    // pipeline. The readers (render_model_command_inner,
    // render_skinned_model_command_inner) live in a different translation
    // unit from the writers, where that coupling stops being visible, so the
    // *_bound_ flags below turn it into a debug assertion instead of a
    // comment. Resetting the variant flags themselves would change behaviour
    // (an extra pipeline bind per frame) and is deliberately not done here.
    bool model_pipeline_double_sided_ = false;
    bool skinned_pipeline_double_sided_ = false;
    bool model_pipeline_depth_equal_ = false;
    bool skinned_pipeline_depth_equal_ = false;
    // Diagnostic only (asserted, never branched on): a model / skinned
    // pipeline bind has happened since render_3d_scene cleared these at the
    // top of the pass, i.e. the variant flags above describe THIS pass.
    bool model_pipeline_bound_ = false;
    bool skinned_pipeline_bound_ = false;

    // ========== Bone Palettes ==========
    // Bone palettes live in the FramePacket (allocated by the host through
    // FrameBuilder::alloc_bone_palette, budgeted at build time by
    // FramePacket::MAX_BONE_PALETTES): the frame's packed palette stream is
    // uploaded palettes()-in-order into one storage buffer, shared by the
    // main pass and every shadow cascade. Draws carry only a uint palette
    // offset (BonePalette::matrix_offset, prefix-summed at seal) instead of
    // an 8 KB uniform push per pass.
    // DOUBLE-BUFFERED: uploads alternate between the two slots each rendered
    // frame so the previous frame's palettes stay bindable — the skinned
    // depth prepass reads them (via MotionHistory palette offsets) to compute
    // per-bone motion vectors.
    std::array<std::unique_ptr<gpu::GPUBuffer>, 2> bone_palette_buffers_;
    std::array<size_t, 2> bone_palette_capacities_{};         // bytes; grown geometrically
    int bone_palette_current_ = 0;                            // slot written/bound this frame
    bool prev_palette_buffer_valid_ = false;                  // prev slot holds last frame's palettes
    size_t bone_palette_matrix_count_ = 0;                    // total matrices uploaded this frame

    // Skinned commands resolved once per frame by pack_bone_palettes() (see
    // ScenePrep::ResolvedSkinnedEntry).
    std::vector<ScenePrep::ResolvedSkinnedEntry> skinned_resolved_;

    // Per-view / per-chunk scratch pools for the CPU Prepare fan-out. Owned
    // by the renderer, not by ScenePrep, which is stateless — capacity is
    // stable across frames.
    ScenePrep::PrepScratch prep_scratch_;

    // Per-frame scratch for shadow-pass static entity resolution. Cleared per
    // render_shadow_passes call but capacity is preserved across frames.
    // Held as a member rather than a thread_local local so it participates
    // in normal object lifetime / can be reset by tests.
    struct ResolvedStaticEntry {
        const ModelDrawCommand* cmd;
        Model* model;
    };
    std::vector<ResolvedStaticEntry> shadow_static_scratch_;

    // ========== Render State ==========
    // Frame camera adopted from the packet when the host set one
    // (frame.has_camera()); a camera-less frame reuses the previous camera,
    // and until any camera was ever set the 3D passes are skipped entirely
    // (have_frame_camera_ false).
    CameraState frame_camera_state_{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(0.0f)};
    bool have_frame_camera_ = false;
    Frustum frame_frustum_{}; // Extracted once per frame, reused across passes
    SDL_GPURenderPass* main_render_pass_ = nullptr;
    SDL_GPUTexture* current_swapchain_ = nullptr;
    bool had_main_pass_this_frame_ = false;
    render::EnvironmentState env_;         // single source of truth for sun/sky/fog
    glm::vec3 light_dir_ = env_.sun_dir;   // refreshed from env_ every frame
    // Scene clock (grass wind / water shading). Advanced by the SIM stage and
    // adopted from the packet's engine section each frame (has_engine_frame);
    // the renderer never integrates dt itself. Hand-built test packets
    // without an engine section leave it unchanged.
    float scene_time_ = 0.0f;
    // Effect-instance snapshot counts of the last consumed packet (dev
    // overlay stats — render-side data only).
    uint32_t frame_effect_count_ = 0;
    uint32_t frame_particle_count_ = 0;
    GraphicsSettings* graphics_ = nullptr;
    GraphicsSettings default_graphics_;

    // Last-bound fragment texture per material slot (0..5) and last-pushed model
    // lighting uniforms; both invalidated by reset_model_bind_state().
    std::array<SDL_GPUTexture*, 6> last_frag_tex_{};
    gpu::ModelLightingUniforms last_model_lighting_{};
    bool model_lighting_valid_ = false;

    // Cluster params published to shaders this frame; zeroed (gridDim.w == 0)
    // when the scene has no dynamic lights so stale light counts never leak.
    render::lighting::ClusterParams frame_cluster_params_{};

    // ========== Local (point/spot) light shadows ==========
    // Per-frame copies of the scene's lights: the local-shadow selector
    // writes each selected light's shadow_slot into these before they feed
    // the cluster grid (scene commands are immutable to the renderer).
    // Capacity persists across frames.
    std::vector<render::lighting::PointLight> frame_point_lights_;
    std::vector<render::lighting::SpotLight> frame_spot_lights_;
    // Moving casters this frame (motion_id != 0 / skinned): cached local
    // shadow tiles they intersect must re-render.
    std::vector<render::lighting::LocalShadowDynamicCaster> local_dynamic_casters_;
    // Per-light caster scratch for render_local_shadow_passes.
    struct LocalShadowCaster {
        const ModelDrawCommand* cmd;
        Model* model;
        glm::vec3 center;
        float radius;
    };
    std::vector<LocalShadowCaster> local_static_casters_;
    std::vector<const ScenePrep::ResolvedSkinnedEntry*> local_skinned_casters_;
    // Set per frame: local shadows setting on, LocalShadowMap alive, and the
    // scene has dynamic lights (so the selector ran and slots are current).
    bool frame_local_shadows_active_ = false;

    // Cached frame state (computed once in render_frame, reused everywhere)
    gpu::ShadowDataUniforms frame_shadow_uniforms_{};
    bool frame_fog_active_ = false;
    float frame_draw_dist_sq_ = 0.0f;
    bool frame_do_frustum_cull_ = false;

    // Cached camera matrix inverses — recomputed only when the source matrix changes.
    // glm::inverse on a 4x4 is ~80 flops; running it every frame for AO/fog is waste.
    glm::mat4 cached_projection_{0.0f};
    glm::mat4 cached_inv_projection_{1.0f};

    // ========== Motion Vectors / TAA Foundation ==========
    // Per-frame camera matrices: the unjittered copies are captured BEFORE the
    // optional TAA sub-pixel jitter is applied to the draw projection.
    // Velocity math and frustum culling use the unjittered set exclusively.
    struct FrameCameraMatrices {
        glm::mat4 view{1.0f};
        glm::mat4 proj_unjittered{1.0f};
        glm::mat4 proj_jittered{1.0f};       // == unjittered when jitter is off
        glm::mat4 view_proj_unjittered{1.0f};
        glm::mat4 view_proj_jittered{1.0f};
        glm::vec2 jitter_pixels{0.0f};
    };
    FrameCameraMatrices frame_cam_{};
    glm::mat4 prev_view_proj_unjittered_{1.0f}; // last rendered frame's unjittered VP
    /// Last rendered frame's VIEW matrix, latched in lockstep with the one
    /// above. The local A-SVGF gradient replay needs it because the reservoir
    /// records a true view depth rather than a linearized device depth, so it
    /// cannot re-derive "where was this point last frame" from the VP pair.
    glm::mat4 prev_view_{1.0f};

    // ---- R5b: the analytic sun gradient (docs/rt_lighting_plan.md §5.6) ----
    //
    // The renderer knows the sun moved before any ray is cast, and it is the
    // ONLY place that knows it once: `EnvironmentState` is rebuilt from the
    // packet every frame and carries no history. Measuring the delta here and
    // handing one number to every temporal accumulator is what makes a
    // time-of-day scrub collapse the shadow denoisers, the local-light
    // denoiser and the probe field's blend on the same frame, rather than each
    // discovering it separately and at a different lag.
    glm::vec3 prev_sun_dir_{0.0f};
    /// sun_color * sun_intensity — the product, because dimming by scaling the
    /// intensity and dimming by darkening the colour are the same event.
    glm::vec3 prev_sun_radiance_{0.0f};
    bool sun_state_valid_ = false;
    /// This frame's reset floor in [0, 1]. 0 on every frame the sun held still,
    /// which is what keeps the pre-R5b path bit-identical.
    float lighting_reset_ = 0.0f;
    bool have_prev_view_proj_ = false;          // first frame: prev == current (zero camera velocity)
    uint64_t frame_index_ = 0;                  // rendered-frame counter (drives the jitter sequence)
    bool taa_jitter_enabled_ = false;           // derived per frame: aa_mode == TAA && resolve can run

    // Velocity (motion-vector) render target: RG16F, screen-sized, written by
    // the depth prepass, cleared to zero (sky/unmoving pixels).
    std::unique_ptr<gpu::GPUTexture> velocity_texture_;
    // Set per frame: the depth prepass ran, so opaque forward draws bind the
    // *DepthEqual pipeline variants (EQUAL compare, depth writes off).
    bool frame_prepass_active_ = false;

    // The AO texture forward draws sample this frame: gtao_'s full-res RGBA8
    // output when AO ran, otherwise the 1x1 white dummy. Reset at the top of
    // every render_frame (never dangles across resizes/mode switches).
    //
    // This means EXACTLY "GTAO or white" again. The RT shadow pass used to
    // overwrite it with an RG8 mask that carried a copy of AO in R, which made
    // "the AO texture" mean two different things depending on a setting; see
    // frame_rt_shadow_texture_.
    SDL_GPUTexture* frame_ao_texture_ = nullptr;

    // The traced sun-visibility mask forward draws sample this frame (R8, at
    // the RT pass's own resolution), or null when ray tracing is off — in
    // which case the model shaders get the white dummy and never read it
    // (rtShadowEnabled == 0). Reset at the top of every render_frame.
    SDL_GPUTexture* frame_rt_shadow_texture_ = nullptr;

    // The traced LOCAL (point/spot) visibility ratio forward draws multiply
    // their clustered subtotal by this frame (full-res R8), or null in Off mode
    // — in which case every scene shader gets the white dummy and never reads
    // it (rtLocalShadowMode == 0). Reset at the top of every render_frame.
    SDL_GPUTexture* frame_rt_local_shadow_texture_ = nullptr;

    // The Full-DI tier's two demodulated targets (R5c), at the traced
    // resolution, bound to the four scene shaders as READ-ONLY STORAGE textures
    // — never as samplers, because those shaders sit at SDL's 16-sampler
    // ceiling exactly. Null outside Full mode, where the 1x1 stand-in below
    // holds the slots and `rtLocalShadowMode < 2` is what stops them being read.
    SDL_GPUTexture* frame_rt_local_di_diffuse_ = nullptr;
    SDL_GPUTexture* frame_rt_local_di_specular_ = nullptr;
    /// R7b's third DI target: the winning light's octahedral direction (RG16F).
    SDL_GPUTexture* frame_rt_local_di_direction_ = nullptr;
    /// 1x1 RGBA16F zero, GRAPHICS_STORAGE_READ. Zeroed rather than left
    /// undefined for the reason every dummy here is: alpha 0 is the "no answer"
    /// the reconstruction rejects, so even a shader that ignored the mode flag
    /// would read no contribution rather than garbage.
    std::unique_ptr<gpu::GPUTexture> dummy_di_texture_;
    /// The RG16F half of that stand-in, for R7b's direction slot. A separate
    /// texture because the slot's channel count differs and a storage image's
    /// does not get to be approximate.
    std::unique_ptr<gpu::GPUTexture> dummy_di_dir_texture_;

    // Sky IBL is active this frame: settings flag on AND the prefilter chain
    // is alive. Drives the iblEnabled uniform flag of every scene shader.
    bool frame_ibl_active_ = false;

    // The probe field's three atlas bindings + per-cascade window corners for
    // this frame's forward draws and the fog march, or dummies with
    // `enabled == 0` when GI is off. Rebuilt once per frame, right after the
    // update passes, so every consumer in the frame reads ONE snapshot — a
    // pixel shaded against cascade bases the volume has since scrolled past
    // would sample the wrong probes.
    render::gi::GiBindings frame_gi_{};

    // motion_id -> previous-frame state (model matrix, palette offset).
    MotionHistory motion_history_;

    // ========== Debug Stats ==========
    bool collect_stats_ = false;
    bracken::RenderStats render_stats_;
    // Reset every render_frame regardless of collect_stats_ (see
    // frame_diagnostics()).
    bracken::FrameDiagnostics frame_diagnostics_;
    gpu::GPUTimestampPool gpu_timer_pool_;

    // ========== Post-UI Callback ==========
    PostUICallback post_ui_callback_;

    // ========== Screenshot ==========
    std::string pending_screenshot_path_; // non-empty = capture next frame
};

} // namespace bracken::scene
