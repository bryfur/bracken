#pragma once

#include "engine/gpu/gpu_buffer.hpp"
#include "engine/gpu/gpu_debug.hpp"
#include "engine/gpu/gpu_texture.hpp"
#include "engine/gpu/gpu_uniforms.hpp"
#include "engine/graphics_settings.hpp"
#include "engine/render/environment.hpp"
#include "engine/render/lighting/light_cluster.hpp"
#include "engine/render_stats.hpp"
#include "engine/scene/camera_state.hpp"
#include "engine/scene/frustum.hpp"
#include "engine/scene/render_scene.hpp"
#include "engine/scene/ui_scene.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Forward declarations for types only used in private implementation
namespace mmo::engine::render {
class RenderContext;
class TerrainRenderer;
class WorldRenderer;
class UIRenderer;
class EffectRenderer;
class GrassRenderer;
class ShadowMap;
class AmbientOcclusion;
class Bloom;
class VolumetricFog;
} // namespace mmo::engine::render

namespace mmo::engine::gpu {
class PipelineRegistry;
}

namespace mmo::engine::systems {
class EffectSystem;
}

namespace mmo::engine {
class ModelManager;
struct Heightmap;
struct EffectDefinition;
struct Model;
struct Mesh;
} // namespace mmo::engine

namespace mmo::engine::scene {

namespace gpu = mmo::engine::gpu;
namespace render = mmo::engine::render;

/**
 * SceneRenderer consumes RenderScene + UIScene and produces a rendered frame.
 * Owns all GPU resources, sub-renderers, and rendering state.
 * Game builds scenes; SceneRenderer draws them.
 */
class SceneRenderer {
public:
    SceneRenderer();
    ~SceneRenderer();

    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    /**
     * Initialize all rendering subsystems.
     * @param context RenderContext owned by the application (must outlive this)
     * @param world_width World width for terrain/world renderers
     * @param world_height World height for terrain/world renderers
     */
    bool init(render::RenderContext& context, float world_width = 32000.0f, float world_height = 32000.0f);

    void shutdown();

    /**
     * Render a complete frame from scene descriptions.
     * @param scene 3D world scene commands
     * @param ui_scene 2D UI commands
     * @param camera Camera state for the frame
     * @param dt Delta time for animations/effects
     */
    // RenderScene is taken by non-const reference because the renderer
    // drains particle-effect spawn commands from it as part of consumption.
    void render_frame(RenderScene& scene, const UIScene& ui_scene, const CameraState& camera, float dt);

    // ========== Configuration ==========

    void set_heightmap(const Heightmap& heightmap);
    void set_anisotropic_filter(int level);
    void set_graphics_settings(const GraphicsSettings& settings);
    void set_vsync_mode(int mode);
    int max_vsync_mode() const;
    void set_screen_size(int width, int height);

    // ========== Debug Stats ==========

    void set_collect_stats(bool enabled) { collect_stats_ = enabled; }
    const engine::RenderStats& render_stats() const { return render_stats_; }

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
    gpu::PipelineRegistry* pipeline_registry() { return pipeline_registry_.get(); }
    float get_terrain_height(float x, float z);

    // Post-UI callback: called after UI rendering, before end_frame.
    // Receives (command_buffer, swapchain_texture) for additional render passes (e.g. ImGui).
    using PostUICallback = std::function<void(SDL_GPUCommandBuffer*, SDL_GPUTexture*)>;
    void set_post_ui_callback(PostUICallback cb) { post_ui_callback_ = std::move(cb); }

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
    void render_3d_scene(const RenderScene& scene, const CameraState& camera);
    void render_model_command(const ModelCommand& cmd, const CameraState& camera);
    void render_model_command_inner(const ModelCommand& cmd, const CameraState& camera);
    void render_skinned_model_command_inner(const SkinnedModelCommand& cmd, const CameraState& camera,
                                            uint32_t palette_offset);
    void build_instance_batches(const RenderScene& scene, const CameraState& camera, const Frustum& frustum);
    void upload_instance_buffers(SDL_GPUCopyPass* copy_pass);
    void pack_bone_palettes(const RenderScene& scene);
    void upload_bone_palettes(SDL_GPUCopyPass* copy_pass);
    void render_instanced_models(const CameraState& camera);
    void render_instanced_shadow_models(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                        const glm::mat4& light_view_projection, int cascade);

    // Transparent (glTF alphaMode BLEND) path: meshes queued during the opaque
    // walks, then drawn back-to-front after the skybox with depth writes off.
    void queue_transparent_mesh(const Mesh& mesh, const glm::mat4& transform, const glm::mat4& normal_matrix,
                                const glm::vec4& tint, bool no_fog, uint32_t palette_offset, const Model& model,
                                const CameraState& camera);
    void render_transparent_models(const CameraState& camera);

    // Bind the (double-sided or default-cull) model/skinned pipeline variant
    // and re-establish the pass-level fragment state that follows a bind.
    bool bind_model_pipeline(bool double_sided);
    bool bind_skinned_pipeline(bool double_sided);
    void render_ui_commands(const UIScene& ui_scene, const CameraState& camera);
    void draw_billboard_3d(const Billboard3DCommand& cmd, const CameraState& camera);
    void upload_debug_lines(const RenderScene& scene, SDL_GPUCopyPass* copy_pass);
    void render_debug_lines(const CameraState& camera);

    // Shadow rendering
    void render_shadow_passes(const RenderScene& scene, const CameraState& camera);
    void bind_shadow_data(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, int sampler_slot);

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

    // Depth pre-pass (renders depth-only before main pass to eliminate overdraw)

    // Setup
    void init_pipelines();

    // ========== Sub-renderers ==========
    render::RenderContext* context_ = nullptr;
    std::unique_ptr<gpu::PipelineRegistry> pipeline_registry_;
    std::unique_ptr<render::TerrainRenderer> terrain_;
    std::unique_ptr<render::WorldRenderer> world_;
    std::unique_ptr<render::UIRenderer> ui_;
    std::unique_ptr<render::EffectRenderer> effects_;
    std::unique_ptr<mmo::engine::systems::EffectSystem> effect_system_;
    std::unique_ptr<ModelManager> model_manager_;
    std::unique_ptr<render::GrassRenderer> grass_renderer_;
    std::unique_ptr<render::ShadowMap> shadow_map_;
    std::unique_ptr<render::AmbientOcclusion> ao_;
    std::unique_ptr<render::Bloom> bloom_;
    std::unique_ptr<render::VolumetricFog> volumetric_fog_;
    render::lighting::ClusterGrid cluster_grid_;
    bool cluster_grid_ready_ = false;

    // ========== GPU Resources ==========
    std::unique_ptr<gpu::GPUBuffer> debug_line_vertex_buffer_;
    size_t debug_line_buffer_capacity_ = 0; // max line count the buffer can hold
    size_t debug_line_vertex_count_ = 0;    // vertices uploaded this frame (consolidated copy pass)
    std::vector<float> debug_line_scratch_; // reused per frame to avoid alloc
    std::unique_ptr<gpu::GPUTexture> depth_texture_;
    SDL_GPUSampler* default_sampler_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> dummy_white_texture_;    // 1x1 white for binding slots that need a texture
    std::unique_ptr<gpu::GPUTexture> default_normal_texture_; // 1x1 (128,128,255) flat tangent-space normal
    std::unique_ptr<gpu::GPUTexture> dummy_black_texture_;    // 1x1 black emissive fallback

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
    std::array<std::vector<std::vector<gpu::ShadowInstanceData>>, 4> shadow_instance_batches_;
    std::array<std::vector<ModelHandle>, 4> shadow_batches_touched_;
    std::vector<const ModelCommand*> non_instanced_commands_;         // individual draw fallback
    std::vector<const SkinnedModelCommand*> shadow_skinned_commands_; // pre-collected for shadow passes
    std::unique_ptr<gpu::GPUBuffer> instance_storage_buffer_;
    size_t instance_storage_capacity_ = 0;
    std::unique_ptr<gpu::GPUBuffer> shadow_instance_storage_buffer_;
    size_t shadow_instance_storage_capacity_ = 0;

    // Per-cascade base_instance / count into shadow_instance_storage_buffer_
    // after packing all cascades contiguously in upload_instance_buffers().
    std::array<uint32_t, 4> shadow_cascade_base_instance_{};

    // Per-cascade light frustums extracted from cascade.light_view_projection.
    std::array<Frustum, 4> shadow_cascade_frustums_{};

    // Per-frame packing scratch (cleared each frame, capacity preserved).
    std::vector<gpu::InstanceData> packed_instances_;
    std::vector<gpu::ShadowInstanceData> packed_shadow_instances_;

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
    // Cull variant currently bound for the model / skinned pipeline families
    // (only meaningful while that family is the active pipeline).
    bool model_pipeline_double_sided_ = false;
    bool skinned_pipeline_double_sided_ = false;

    // ========== Bone Palettes ==========
    // All skinned commands' bone matrices packed tightly (joint_count matrices
    // each, not MAX_BONES) into one storage buffer, uploaded once per frame and
    // shared by the main pass and every shadow cascade. Draws carry only a
    // uint palette offset instead of an 8 KB uniform push per pass.
    static constexpr uint32_t MAX_BONE_PALETTES = 1024;       // per-frame cap; draws beyond are dropped + logged
    static constexpr uint32_t INVALID_PALETTE_OFFSET = 0xFFFFFFFFu;
    std::unique_ptr<gpu::GPUBuffer> bone_palette_buffer_;
    size_t bone_palette_capacity_ = 0;                        // bytes; grown geometrically
    std::vector<glm::mat4> bone_palette_scratch_;             // CPU pack scratch, reused per frame
    std::vector<uint32_t> skinned_palette_offsets_;           // parallel to scene.skinned_commands()
    // Dedupe shared bone arrays (e.g. the static identity palette submitted for
    // unbound skeletons): bone-array pointer -> (offset, packed joint count).
    // Flat vector with linear lookup: skinned counts are small and clear()
    // keeps capacity, so the per-frame path stays allocation-free (a
    // node-based map would re-allocate one node per palette every frame).
    struct PaletteDedupeEntry {
        const void* key;
        uint32_t offset;
        uint32_t joint_count;
    };
    std::vector<PaletteDedupeEntry> palette_dedupe_;

    // Per-frame scratch for shadow-pass entity resolution. Cleared per
    // render_shadow_passes call but capacity is preserved across frames.
    // Held as members rather than thread_local locals so they participate
    // in normal object lifetime / can be reset by tests.
    struct ResolvedSkinnedEntry {
        const SkinnedModelCommand* cmd;
        Model* model;
        glm::vec3 world_center;
        float world_radius;
        uint32_t palette_offset;
    };
    struct ResolvedStaticEntry {
        const ModelCommand* cmd;
        Model* model;
    };
    std::vector<ResolvedSkinnedEntry> shadow_skinned_scratch_;
    std::vector<ResolvedStaticEntry> shadow_static_scratch_;

    // ========== Render State ==========
    Frustum frame_frustum_{}; // Extracted once per frame, reused across passes
    SDL_GPURenderPass* main_render_pass_ = nullptr;
    SDL_GPUTexture* current_swapchain_ = nullptr;
    bool had_main_pass_this_frame_ = false;
    EnvironmentState env_;                 // single source of truth for sun/sky/fog
    glm::vec3 light_dir_ = env_.sun_dir;   // refreshed from env_ every frame
    float scene_time_ = 0.0f;              // accumulated dt, drives grass animation
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

    // Cached frame state (computed once in render_frame, reused everywhere)
    gpu::ShadowDataUniforms frame_shadow_uniforms_{};
    bool frame_fog_active_ = false;
    float frame_draw_dist_sq_ = 0.0f;
    bool frame_do_frustum_cull_ = false;

    // Cached camera matrix inverses — recomputed only when the source matrix changes.
    // glm::inverse on a 4x4 is ~80 flops; running it every frame for AO/fog is waste.
    glm::mat4 cached_projection_{0.0f};
    glm::mat4 cached_inv_projection_{1.0f};

    // ========== Debug Stats ==========
    bool collect_stats_ = false;
    engine::RenderStats render_stats_;
    gpu::GPUTimestampPool gpu_timer_pool_;

    // ========== Post-UI Callback ==========
    PostUICallback post_ui_callback_;

    // ========== Screenshot ==========
    std::string pending_screenshot_path_; // non-empty = capture next frame
};

} // namespace mmo::engine::scene
