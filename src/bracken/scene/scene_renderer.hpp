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
#include "bracken/contracts/render_stats.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
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

    void set_heightmap(const Heightmap& heightmap);
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

    /// Per-frame instance list for the TLAS. Static scenes can call this once.
    void set_raytracing_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances);

    [[nodiscard]] const render::RtStats& raytracing_stats() const;

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
    std::unique_ptr<render::lighting::LocalShadowMap> local_shadows_;

    // ========== GPU Resources ==========
    std::unique_ptr<gpu::GPUBuffer> debug_line_vertex_buffer_;
    size_t debug_line_buffer_capacity_ = 0; // max line count the buffer can hold
    size_t debug_line_vertex_count_ = 0;    // vertices uploaded this frame (consolidated copy pass)
    std::vector<float> debug_line_scratch_; // reused per frame to avoid alloc
    std::unique_ptr<gpu::GPUTexture> depth_texture_;
    SDL_GPUSampler* default_sampler_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> dummy_white_texture_;    // 1x1 white for binding slots that need a texture

    // Compute-shader ray tracing. Null-checked everywhere: both features are
    // off by default and nothing is built unless one is on.
    std::unique_ptr<render::RtRenderer> rt_renderer_; // created in init()
    bool frame_rt_shadows_active_ = false;
    SDL_GPUTexture* frame_rt_reflection_texture_ = nullptr;
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
    bool have_prev_view_proj_ = false;          // first frame: prev == current (zero camera velocity)
    uint64_t frame_index_ = 0;                  // rendered-frame counter (drives the jitter sequence)
    bool taa_jitter_enabled_ = false;           // derived per frame: aa_mode == TAA && resolve can run

    // Velocity (motion-vector) render target: RG16F, screen-sized, written by
    // the depth prepass, cleared to zero (sky/unmoving pixels).
    std::unique_ptr<gpu::GPUTexture> velocity_texture_;
    // Set per frame: the depth prepass ran, so opaque forward draws bind the
    // *DepthEqual pipeline variants (EQUAL compare, depth writes off).
    bool frame_prepass_active_ = false;

    // The AO texture forward draws sample this frame: gtao_'s full-res R8
    // output when AO ran, otherwise the 1x1 white dummy. Reset at the top of
    // every render_frame (never dangles across resizes/mode switches).
    SDL_GPUTexture* frame_ao_texture_ = nullptr;

    // Sky IBL is active this frame: settings flag on AND the prefilter chain
    // is alive. Drives the iblEnabled uniform flag of every scene shader.
    bool frame_ibl_active_ = false;

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
