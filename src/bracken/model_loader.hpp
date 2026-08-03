#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <span>
#include <vector>

// The model identity contract: ModelHandle / INVALID_MODEL_HANDLE. Kept in the
// leaf `contracts/` layer so the frame packet can name a model without pulling
// the GPU resource headers below in behind it; re-exported here so every
// existing `#include "bracken/model_loader.hpp"` still sees both names.
#include "bracken/contracts/model_handle.hpp"
#include "bracken/core/result.hpp"

// Include GPU abstraction headers for proper type definitions
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/texture_cache.hpp"
#include "bracken/gpu/gpu_types.hpp"

// Animation types from the dedicated animation module
#include "bracken/animation/animation_types.hpp"

// Forward declarations for SDL3 GPU types
struct SDL_GPUDevice;
struct SDL_GPUCommandBuffer;

namespace bracken::core::asset {
class FileWatcher;
}

namespace bracken {

// ModelHandle / INVALID_MODEL_HANDLE: contracts/model_handle.hpp.

// Use canonical vertex types from gpu module
using gpu::SkinnedVertex;
using gpu::Vertex3D;

// glTF alphaMode: how base-color alpha is interpreted at draw time.
enum class AlphaMode : uint8_t {
    Opaque,
    Mask, // alpha-test against alpha_cutoff
    Blend,
};

// PBR material parameters mirroring glTF pbrMetallicRoughness.
// Textures are uploaded on demand; scalar factors act as fallbacks.
struct PBRMaterial {
    glm::vec4 base_color_factor{1.0f, 1.0f, 1.0f, 1.0f};
    float metallic_factor = 0.0f;
    float roughness_factor = 0.85f;
    float normal_scale = 1.0f;
    float occlusion_strength = 1.0f;
    // Includes the KHR_materials_emissive_strength multiplier when present.
    glm::vec3 emissive_factor{0.0f, 0.0f, 0.0f};
    AlphaMode alpha_mode = AlphaMode::Opaque;
    float alpha_cutoff = 0.5f;
    bool double_sided = false;
    // True when occlusionTexture shares the metallic-roughness image (glTF ORM
    // packing): AO lives in the R channel of metallic_roughness_texture. This
    // is the ONLY supported occlusion layout — a standalone (non-ORM) AO map
    // has no sampler binding in the model shaders, so the loader drops it with
    // a warning. See docs/skinned_mesh_format.md.
    bool has_packed_occlusion = false;
    // CPU-side raw pixels for deferred GPU upload; freed after upload_to_gpu.
    std::vector<uint8_t> metallic_roughness_pixels;
    int metallic_roughness_width = 0;
    int metallic_roughness_height = 0;
    std::vector<uint8_t> normal_pixels;
    int normal_width = 0;
    int normal_height = 0;
    std::vector<uint8_t> emissive_pixels;
    int emissive_width = 0;
    int emissive_height = 0;
    // Non-owning: owned by the ModelManager's TextureCache, which uploads one
    // copy per distinct image and outlives every model borrowing from it.
    gpu::GPUTexture* metallic_roughness_texture = nullptr;
    gpu::GPUTexture* normal_texture = nullptr;
    gpu::GPUTexture* emissive_texture = nullptr;
};

struct Mesh {
    std::vector<Vertex3D> vertices;
    std::vector<SkinnedVertex> skinned_vertices; // Used if model has skeleton
    std::vector<uint32_t> indices;

    // SDL3 GPU resources
    std::unique_ptr<gpu::GPUBuffer> vertex_buffer;
    std::unique_ptr<gpu::GPUBuffer> index_buffer;
    // Non-owning, like the material maps above — see texture_cache.hpp.
    gpu::GPUTexture* texture = nullptr;
    PBRMaterial material;
    bool uploaded = false;

    bool has_texture = false;
    bool is_skinned = false; // True if using skinned vertices

    /** The albedo to bind; null when there is none.
     *
     *  There is no longer a separate `shared_texture` slot. It existed so a
     *  generated LOD level could borrow its base level's albedo instead of
     *  uploading a second copy; now that every texture comes from the
     *  content-addressed TextureCache, an LOD level asking for the same pixels
     *  is simply a cache hit and needs no special case. */
    [[nodiscard]] SDL_GPUTexture* albedo_handle() const {
        return texture != nullptr ? texture->handle() : nullptr;
    }

    /** Should this mesh sample an albedo texture at all? Collapses the
     *  `has_texture && texture` pair that used to be repeated at every bind
     *  site — and which silently meant "no texture" if either half was wrong. */
    [[nodiscard]] bool has_albedo() const { return has_texture && albedo_handle() != nullptr; }

    // Perf flags.
    //
    // cast_shadows=false : shadow passes skip this mesh (e.g. transparent leaves
    //   that would cause expensive alpha-test overdraw in shadow depth rasterization).
    // shadow_only=true   : main pass skips this mesh; only shadow passes render it.
    //   Used for canopy shadow proxies — a low-poly ellipsoid that approximates the
    //   leaf silhouette for shadows, replacing thousands of individual leaf quads.
    bool cast_shadows = true;
    bool shadow_only = false;

    // Texture data for deferred upload (before GPU upload)
    std::vector<uint8_t> texture_pixels;
    int texture_width = 0;
    int texture_height = 0;

    /**
     * Compact base-colour proxy for software RT closest-hit shading.
     *
     * Base-colour pixels are released after GPU upload, long before the RT
     * BLAS is built. When CPU geometry retention is enabled, upload_to_gpu()
     * samples the authored sRGB texture at each triangle centroid, multiplies
     * vertex colour + baseColorFactor in linear space, and keeps only this
     * 16-byte-per-triangle result. Reflection hits therefore preserve material
     * colour without retaining every source texture or requiring a bindless
     * texture array. Empty means the RT builder uses the untextured factors.
     */
    std::vector<glm::vec4> rt_triangle_base_colors;

    /**
     * Compact EMISSIVE proxy for software RT hits, the same shape and the same
     * lifetime rule as `rt_triangle_base_colors` above (R6, §5.7 of
     * docs/rt_lighting_plan.md).
     *
     * `emissive_factor` multiplied by the emissive texture's centroid-UV sample
     * where one exists, in LINEAR space. Kept separate from the base-colour
     * proxy rather than widening it, because the two have different existence
     * conditions: a mesh may be emissive with no texture at all (the
     * playground's orbs are exactly that), while the base-colour bake requires
     * one. Empty means "no triangle of this mesh emits", which is the common
     * case and costs nothing.
     */
    std::vector<glm::vec3> rt_triangle_emissive;

    // Cached index count (valid after GPU upload, when CPU-side data is freed)
    uint32_t cached_index_count = 0;

    // Convenience accessors
    uint32_t vertex_count() const {
        return is_skinned ? static_cast<uint32_t>(skinned_vertices.size()) : static_cast<uint32_t>(vertices.size());
    }
    uint32_t index_count() const {
        return cached_index_count > 0 ? cached_index_count : static_cast<uint32_t>(indices.size());
    }

    // Bind vertex and index buffers for rendering (SDL3 GPU API)
    void bind_buffers(struct SDL_GPURenderPass* pass) const;
};

/** Axis-aligned bounding box in model-local space. */
struct AABB {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};

    [[nodiscard]] glm::vec3 size() const { return max - min; }
    [[nodiscard]] glm::vec3 center() const { return (min + max) * 0.5f; }
    /** No extent in any axis — which is also the default-constructed "nobody
     *  filled this in" state ModelManager::register_model derives from. */
    [[nodiscard]] bool empty() const { return !(max.x > min.x || max.y > min.y || max.z > min.z); }
};

struct Model {
    std::vector<Mesh> meshes;
    /** Model-local bounds. Set by the loader/generator, or derived from mesh
     *  vertices by register_model when left empty. Skinned models usually want
     *  them set by hand with animation headroom: the vertex AABB is the BIND
     *  pose, and a pose that reaches outside it would be culled. */
    AABB bounds;
    bool loaded = false;

    // Skeletal animation data
    animation::Skeleton skeleton;
    std::vector<animation::AnimationClip> animations;
    bool has_skeleton = false;
    animation::FootIKData foot_ik;

    // Pre-computed bounding sphere for fast frustum culling (avoids per-entity recomputation)
    glm::vec3 bounding_center = {0, 0, 0};
    float bounding_half_diag = 0.0f;

    /**
     * Level-of-detail chain. `lod_handles[k]` is the handle of level k+1 —
     * coarser than this model, which is always level 0.
     *
     * Levels are separate MODELS rather than extra mesh sets inside this one,
     * which is what keeps LOD out of the renderer's hot path entirely: the
     * Prepare stage swaps the handle before binning and every stage after it
     * (binning, upload, draw) is unchanged and unaware. It also means a
     * project that authors its own LOD meshes registers them like any other
     * model and links them here, with no special asset path.
     *
     * `lod_thresholds[k]` is the projected screen radius below which level
     * k+1 takes over — bounding radius divided by distance, so it is
     * resolution-independent and a small distant prop and a huge far one
     * switch at the same apparent size. Descending.
     */
    std::vector<ModelHandle> lod_handles;
    std::vector<float> lod_thresholds;

    float width() const { return bounds.size().x; }
    float height() const { return bounds.size().y; }
    float depth() const { return bounds.size().z; }
    float max_dimension() const { return std::max({width(), height(), depth()}); }

    void compute_bounding_sphere() {
        bounding_center = bounds.center();
        bounding_half_diag = glm::length(bounds.size()) * 0.5f;
    }

    // Animation helpers
    int find_animation(const std::string& name) const {
        for (size_t i = 0; i < animations.size(); i++) {
            if (animations[i].name == name) {
                {
                    return static_cast<int>(i);
                }
            }
        }
        return -1;
    }
};

/**
 * What a .glb load quietly could not use.
 *
 * glTF animation channels that target a node outside the skin, use a path
 * other than translation/rotation/scale (i.e. morph-target `weights`), or
 * name an out-of-range sampler are dropped by the loader. Each drop is
 * legitimate; dropping them ALL in silence is not, because the visible
 * symptom is "my animation does nothing" with no explanation anywhere. The
 * loader counts them and logs one summary per model.
 */
struct ModelLoadReport {
    uint32_t channels_non_skin_node = 0;
    uint32_t channels_non_trs_path = 0;
    uint32_t channels_bad_sampler = 0;

    [[nodiscard]] uint32_t total() const {
        return channels_non_skin_node + channels_non_trs_path + channels_bad_sampler;
    }
    [[nodiscard]] bool any() const { return total() != 0; }
};

/** Why ModelManager::register_model refused a model. */
enum class ModelRegisterError : uint8_t {
    /// No meshes, or no mesh with any geometry to draw or upload.
    EmptyModel,
    /// Another model already owns that name. Registering anyway would orphan
    /// the previous handle: it would stay valid but unreachable by name.
    NameAlreadyRegistered,
    /// A device is set but at least one mesh failed to reach the GPU.
    UploadFailed,
};

[[nodiscard]] const char* to_string(ModelRegisterError error);

class ModelLoader {
public:
    /// Parse a .glb into `model`. Pure CPU work on caller-owned data — no
    /// shared state, thread-agnostic (Phase 6 streaming will run it on jobs).
    [[nodiscard]] static bool load_glb(const std::string& path, Model& model);

    /**
     * Parse a .glb from bytes already in memory — from an AssetPack entry, a
     * downloaded buffer, or anything else.
     *
     * Worker-safe: every piece of parser state is per-call, so any number of
     * threads may decode different models concurrently. It produces a Model
     * with CPU-side geometry only; ModelLoader::upload_to_gpu still has to run
     * on the main thread, which is exactly the split AsyncModelLoader uses.
     *
     * `name` is used only for diagnostics.
     */
    [[nodiscard]] static bool load_glb_from_memory(std::span<const std::byte> bytes,
                                                   const std::string& name, Model& model);

private:
    [[nodiscard]] static bool load_glb_impl(const std::string& path,
                                            std::span<const std::byte> bytes, Model& model);

public:
    /// Create GPU buffers/textures for every un-uploaded mesh and free the
    /// CPU-side copies. Main-thread tier (asserted): GPU resource creation.
    /**
     * Upload a model's geometry and textures.
     *
     * @param retain_cpu_geometry keep the vertex/index arrays after upload.
     *        Ray tracing needs them (its BVH is built from them) and so does
     *        LOD generation; everything else does not, and a scene's vertex
     *        data is usually the largest CPU allocation in the process — hence
     *        opt-in rather than default.
     */
    /// @param textures owns the uploaded images; one upload per distinct
    ///        image across every model that shares it (texture_cache.hpp).
    static void upload_to_gpu(gpu::GPUDevice& device, TextureCache& textures, Model& model,
                              bool retain_cpu_geometry = false);
    /// Release the model's GPU resources. Main-thread tier (asserted).
    static void free_gpu_resources(Model& model);
};

/**
 * @brief Owns all loaded models and their GPU resources.
 *
 * Thread-safety tiers (see docs/multithreading_plan.md §6.1):
 * - **main-thread (asserted)**: every mutation path — load_model,
 *   register_model, reload_model, enable_hot_reload, unload_all (set_device
 *   belongs to init and is main-thread by convention).
 * - **worker-safe after load**: get_model/get_handle lookups are const and
 *   safe from any thread once loading is complete and no mutation (including
 *   a hot-reload apply, which rewrites model contents in place) is in flight.
 *   The hot-reload fence guarantees that quiescence in the pipelined frame.
 */
/**
 * Choose which level of `model`'s LOD chain to draw.
 *
 * `projected_radius` is the model's world bounding radius divided by its
 * distance to the camera — an apparent-size measure, so the same threshold
 * behaves identically for a pebble up close and a tower far away, and does not
 * change meaning when the window is resized.
 *
 * Returns `base` when the model has no chain, so callers need no branch.
 *
 * Pure: the result depends only on its arguments, which is what lets the
 * Prepare stage call it inside the parallel binning fan-out and stay
 * bit-identical at any worker count.
 *
 * NOTE: there is deliberately no hysteresis. A stable switch band needs
 * per-instance state that persists across frames, and the FramePacket carries
 * no such identity for static instances (motion_id is opt-in and only set for
 * movers). Thresholds are therefore spaced far enough apart that a switch is
 * a small change in a small object; if a project needs guaranteed
 * pop-free transitions it wants dithered cross-fade, which is a renderer
 * feature rather than a selection one.
 */
[[nodiscard]] inline ModelHandle select_lod(const Model& model, ModelHandle base,
                                            float projected_radius) noexcept {
    for (std::size_t k = 0; k < model.lod_thresholds.size(); ++k) {
        if (projected_radius >= model.lod_thresholds[k]) {
            return k == 0 ? base : model.lod_handles[k - 1];
        }
    }
    return model.lod_handles.empty() ? base : model.lod_handles.back();
}

class ModelManager {
public:
    ModelManager();
    ~ModelManager();

    // Non-copyable, non-movable: owns GPU-backed models; handles and raw
    // Model* handed to callers must stay stable.
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    ModelManager(ModelManager&&) = delete;
    ModelManager& operator=(ModelManager&&) = delete;

    void set_device(gpu::GPUDevice* device) { device_ = device; }

    /// The cache backing every model texture. Exposed so code that uploads a
    /// model directly (EffectRenderer) shares the same images as everyone else.
    [[nodiscard]] TextureCache& texture_cache() noexcept { return textures_; }

    /// Load a model and return its handle (INVALID_MODEL_HANDLE on failure).
    /// Main-thread (asserted).
    ModelHandle load_model(const std::string& name, const std::string& path);

    /**
     * Load a model and generate its LOD chain in one step.
     *
     * This has to be one call rather than load_model() followed by
     * build_lods(): uploading frees the model's CPU-side vertices and indices,
     * and simplification needs them. Doing it in the wrong order is a silent
     * no-op, so build_lods() logs an error in that case rather than quietly
     * producing nothing.
     *
     * See build_lods() for the meaning of `ratios` and `thresholds`.
     */
    ModelHandle load_model_with_lods(const std::string& name, const std::string& path,
                                     std::span<const float> ratios,
                                     std::span<const float> thresholds);

    /// O(1) lookup by handle. Const-safe after load (worker-safe when no
    /// mutation is in flight).
    Model* get_model(ModelHandle handle) const;

    /// Name-based lookup (backward compat, uses hash map). Const-safe after
    /// load.
    Model* get_model(const std::string& name) const;

    /**
     * Generate a LOD chain for `handle` by simplifying its meshes, register
     * each level as its own model, and link them.
     *
     * `ratios` are triangle-count fractions of the ORIGINAL, descending
     * (e.g. {0.4f, 0.12f}); `thresholds` are the projected screen radii at
     * which each level takes over, also descending and one shorter — see
     * Model::lod_thresholds.
     *
     * Generated rather than authored because most projects have no LOD assets
     * and never will: an engine whose LOD support requires an art pipeline is
     * an engine whose LOD support goes unused. A project that DOES author
     * levels can skip this and populate Model::lod_handles directly.
     *
     * Returns the number of levels created. Main-thread, at load time — it
     * simplifies and uploads geometry. Skinned meshes are skipped: clustering
     * would collapse vertices carrying different bone weights and tear the
     * mesh apart under animation.
     */
    std::size_t build_lods(ModelHandle handle, std::span<const float> ratios,
                           std::span<const float> thresholds);

    /// Resolve a name to a handle (INVALID_MODEL_HANDLE if not found).
    /// Const-safe after load.
    ModelHandle get_handle(const std::string& name) const;

    /// Number of registered models. Handles are dense indices below this, so
    /// this is also the bound for iterating every model — which is what
    /// building an acceleration structure over the whole scene needs.
    [[nodiscard]] std::size_t model_count() const noexcept;

    /** Keep CPU-side geometry on every subsequently loaded model. Set before
     *  loading if ray tracing will be used — its acceleration structure is
     *  built from those arrays, and they are gone by the time it asks. */
    void set_retain_cpu_geometry(bool retain) noexcept { retain_cpu_geometry_ = retain; }
    [[nodiscard]] bool retains_cpu_geometry() const noexcept { return retain_cpu_geometry_; }

    /**
     * Register a pre-built model (e.g. procedurally generated). Takes
     * ownership, uploads to the GPU when a device is set, and finishes the
     * model the host used to finish by hand: `bounds` are derived from the
     * mesh vertices when left empty, the bounding sphere is computed from
     * them, and `loaded` is set. Forgetting compute_bounding_sphere() used to
     * leave `bounding_half_diag == 0`, which frustum-culls the model into
     * invisibility with no message anywhere.
     *
     * Main-thread (asserted). See ModelRegisterError for the refusals — the
     * previous signature validated nothing and answered a name collision by
     * silently orphaning the older handle.
     */
    [[nodiscard]] core::Result<ModelHandle, ModelRegisterError> register_model(const std::string& name,
                                                                              std::unique_ptr<Model> model);

    /// Re-parse the source file and replace the model's contents in place:
    /// the ModelHandle and any raw Model* previously returned by get_model
    /// stay valid across the reload. Pointers/spans INTO the model (meshes,
    /// animations, skeleton joints) are invalidated — re-derive them from the
    /// Model each frame. Returns false if the file failed to load (the
    /// existing model is left untouched). Main-thread (asserted).
    [[nodiscard]] bool reload_model(ModelHandle handle);

    /// Enable hot-reload: re-load any model whose source .glb file is
    /// modified on disk (the loader is GLB-binary-only; there are no .gltf
    /// models to watch). Main-thread (asserted).
    void enable_hot_reload(core::asset::FileWatcher& watcher);

    /// Release every model and its GPU resources. Main-thread (asserted).
    void unload_all();

private:
    gpu::GPUDevice* device_ = nullptr;
    bool retain_cpu_geometry_ = false;
    // Owns every model texture. Declared BEFORE models_ so it is destroyed
    // after them: meshes hold non-owning pointers into it.
    TextureCache textures_;
    // Slot 0 is reserved (INVALID_MODEL_HANDLE), models start at index 1
    std::vector<std::unique_ptr<Model>> models_;
    std::vector<std::string> model_paths_; // parallel to models_; "" for procedural
    std::unordered_map<std::string, ModelHandle> name_to_handle_;
    core::asset::FileWatcher* watcher_ = nullptr;
};

} // namespace bracken
