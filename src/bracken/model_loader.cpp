#include "bracken/model_loader.hpp"

#include "bracken/mesh_simplify.hpp"
#include "bracken/gpu/gpu_types.hpp"
#include "bracken/model_utils.hpp"
#include "bracken/internal/model_material_proxy.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_iostream.h"
#include "SDL3/SDL_pixels.h"
#include "SDL3/SDL_surface.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <SDL3/SDL_gpu.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"
#include <SDL3_image/SDL_image.h>

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/asset/file_watcher.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

namespace bracken {

// Implementation of Mesh::bind_buffers
void Mesh::bind_buffers(SDL_GPURenderPass* pass) const {
    if (vertex_buffer) {
        SDL_GPUBufferBinding vbuf = {vertex_buffer->handle(), 0};
        SDL_BindGPUVertexBuffers(pass, 0, &vbuf, 1);
    }
    if (index_buffer) {
        SDL_GPUBufferBinding ibuf = {index_buffer->handle(), 0};
        SDL_BindGPUIndexBuffer(pass, &ibuf, SDL_GPU_INDEXELEMENTSIZE_32BIT);
    }
}

} // namespace bracken

namespace {

// Custom image loader for tinygltf using SDL3_image
bool load_image_data(tinygltf::Image* image, const int image_idx, std::string* err, std::string* warn, int req_width,
                     int req_height, const unsigned char* bytes, int size, void* user_data) {
    (void)image_idx;
    (void)warn;
    (void)req_width;
    (void)req_height;
    (void)user_data;

    SDL_IOStream* io = SDL_IOFromConstMem(bytes, size);
    if (!io) {
        if (err) {
            *err = "Failed to create SDL IOStream from memory";
        }
        return false;
    }

    SDL_Surface* surface = IMG_Load_IO(io, true); // true = close IOStream when done
    if (!surface) {
        if (err) {
            *err = std::string("SDL_image failed to load: ") + SDL_GetError();
        }
        return false;
    }

    // Convert to RGBA format
    SDL_Surface* rgba_surface = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_RGBA32);
    SDL_DestroySurface(surface);

    if (!rgba_surface) {
        if (err) {
            *err = std::string("Failed to convert to RGBA: ") + SDL_GetError();
        }
        return false;
    }

    image->width = rgba_surface->w;
    image->height = rgba_surface->h;
    image->component = 4;
    image->bits = 8;
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;

    size_t data_size = static_cast<size_t>(rgba_surface->w * rgba_surface->h * 4);
    image->image.resize(data_size);
    memcpy(image->image.data(), rgba_surface->pixels, data_size);

    SDL_DestroySurface(rgba_surface);
    return true;
}

// Dummy write function (not used but required by tinygltf)
bool write_image_data(const std::string* basepath, const std::string* filename, const tinygltf::Image* image,
                      bool embedImages, const tinygltf::FsCallbacks* fs_cb, const tinygltf::URICallbacks* uri_cb,
                      std::string* out_uri, void* user_pointer) {
    (void)basepath;
    (void)filename;
    (void)image;
    (void)embedImages;
    (void)fs_cb;
    (void)uri_cb;
    (void)out_uri;
    (void)user_pointer;
    return false; // Writing not supported
}

} // anonymous namespace

namespace bracken {

bool ModelLoader::load_glb(const std::string& path, Model& model) {
    return load_glb_impl(path, {}, model);
}

bool ModelLoader::load_glb_from_memory(std::span<const std::byte> bytes, const std::string& name,
                                       Model& model) {
    if (bytes.empty()) {
        BRACKEN_LOG_ERROR("model", "load_glb_from_memory('{}'): no data", name);
        return false;
    }
    return load_glb_impl(name, bytes, model);
}

bool ModelLoader::load_glb_impl(const std::string& path, std::span<const std::byte> bytes,
                                Model& model) {
    BRACKEN_PROFILE_ZONE_TAG(path.c_str());
    tinygltf::Model gltf;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    // Set custom SDL3_image loader
    loader.SetImageLoader(load_image_data, nullptr);
    loader.SetImageWriter(write_image_data, nullptr);

    // Everything below is per-call local state, which is what makes parsing
    // worker-safe and lets the async loader decode many models at once.
    const bool ret =
        bytes.empty()
            ? loader.LoadBinaryFromFile(&gltf, &err, &warn, path)
            : loader.LoadBinaryFromMemory(&gltf, &err, &warn,
                                          reinterpret_cast<const unsigned char*>(bytes.data()),
                                          static_cast<unsigned int>(bytes.size()));

    if (!warn.empty()) {
        BRACKEN_LOG_WARN("model", "Warning loading {}: {}", path, warn);
    }
    if (!err.empty()) {
        BRACKEN_LOG_ERROR("model", "Error loading {}: {}", path, err);
        return false;
    }
    if (!ret) {
        BRACKEN_LOG_ERROR("model", "Failed to load GLB: {}", path);
        return false;
    }

    model.meshes.clear();
    model.bounds.min = glm::vec3(1e10f);
    model.bounds.max = glm::vec3(-1e10f);

    // Texture budget guard (see CLAUDE.md): assets shipping >2048² embedded
    // images should be resized in the optimization pass, not at runtime.
    for (size_t i = 0; i < gltf.images.size(); i++) {
        const auto& image = gltf.images[i];
        if (image.width > 2048 || image.height > 2048) {
            BRACKEN_LOG_WARN("model", "{}: embedded image {} is {}x{} (exceeds 2048x2048 budget)", path, i, image.width,
                            image.height);
        }
    }

    // Process all meshes
    for (const auto& mesh : gltf.meshes) {
        BRACKEN_PROFILE_ZONE("ModelLoader::parse_mesh");
        for (const auto& primitive : mesh.primitives) {
            Mesh out_mesh;

            // Get material info
            int texture_image_idx = -1;
            if (primitive.material >= 0 && primitive.material < static_cast<int>(gltf.materials.size())) {
                const auto& mat = gltf.materials[primitive.material];

                // Base color factor
                if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
                    out_mesh.material.base_color_factor = {
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[0]),
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[1]),
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[2]),
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[3]),
                    };
                }

                out_mesh.material.metallic_factor = static_cast<float>(mat.pbrMetallicRoughness.metallicFactor);
                out_mesh.material.roughness_factor = static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor);
                out_mesh.material.normal_scale = static_cast<float>(mat.normalTexture.scale);
                out_mesh.material.occlusion_strength = static_cast<float>(mat.occlusionTexture.strength);

                // Base color texture
                int tex_idx = mat.pbrMetallicRoughness.baseColorTexture.index;
                if (tex_idx >= 0 && tex_idx < static_cast<int>(gltf.textures.size())) {
                    texture_image_idx = gltf.textures[tex_idx].source;
                    out_mesh.has_texture = true;
                }

                // Metallic-roughness texture (glTF packs: G = roughness, B = metallic)
                int mr_src = -1;
                int mr_idx = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
                if (mr_idx >= 0 && mr_idx < static_cast<int>(gltf.textures.size())) {
                    int src = gltf.textures[mr_idx].source;
                    if (src >= 0 && src < static_cast<int>(gltf.images.size())) {
                        mr_src = src;
                        const auto& img = gltf.images[src];
                        out_mesh.material.metallic_roughness_pixels = img.image;
                        out_mesh.material.metallic_roughness_width = img.width;
                        out_mesh.material.metallic_roughness_height = img.height;
                    }
                }

                // Normal map (tangent-space, UNORM)
                int n_idx = mat.normalTexture.index;
                if (n_idx >= 0 && n_idx < static_cast<int>(gltf.textures.size())) {
                    int src = gltf.textures[n_idx].source;
                    if (src >= 0 && src < static_cast<int>(gltf.images.size())) {
                        const auto& img = gltf.images[src];
                        out_mesh.material.normal_pixels = img.image;
                        out_mesh.material.normal_width = img.width;
                        out_mesh.material.normal_height = img.height;
                    }
                }

                // Ambient occlusion (R channel). Only the glTF ORM packing —
                // occlusionTexture sharing the metallic-roughness image — is
                // supported; the model shaders have no separate AO sampler, so
                // a standalone occlusion map cannot be consumed. Drop it loudly
                // so asset authors know their baked AO is not applied.
                int ao_idx = mat.occlusionTexture.index;
                if (ao_idx >= 0 && ao_idx < static_cast<int>(gltf.textures.size())) {
                    int src = gltf.textures[ao_idx].source;
                    if (src >= 0 && src < static_cast<int>(gltf.images.size())) {
                        if (src == mr_src) {
                            out_mesh.material.has_packed_occlusion = true;
                        } else {
                            BRACKEN_LOG_WARN("model",
                                            "{}: standalone occlusion texture (image {}) is unsupported and was "
                                            "dropped — pack AO into the R channel of the metallic-roughness image "
                                            "(glTF ORM layout)",
                                            path, src);
                        }
                    }
                }

                // Emissive (sRGB texture; factor scaled by KHR_materials_emissive_strength)
                if (mat.emissiveFactor.size() == 3) {
                    out_mesh.material.emissive_factor = {static_cast<float>(mat.emissiveFactor[0]),
                                                         static_cast<float>(mat.emissiveFactor[1]),
                                                         static_cast<float>(mat.emissiveFactor[2])};
                }
                auto strength_it = mat.extensions.find("KHR_materials_emissive_strength");
                if (strength_it != mat.extensions.end() && strength_it->second.Has("emissiveStrength")) {
                    out_mesh.material.emissive_factor *=
                        static_cast<float>(strength_it->second.Get("emissiveStrength").GetNumberAsDouble());
                }
                int em_idx = mat.emissiveTexture.index;
                if (em_idx >= 0 && em_idx < static_cast<int>(gltf.textures.size())) {
                    int src = gltf.textures[em_idx].source;
                    if (src >= 0 && src < static_cast<int>(gltf.images.size())) {
                        const auto& img = gltf.images[src];
                        out_mesh.material.emissive_pixels = img.image;
                        out_mesh.material.emissive_width = img.width;
                        out_mesh.material.emissive_height = img.height;
                    }
                }

                // Alpha + culling state (data only; pipeline selection is the renderer's job)
                if (mat.alphaMode == "MASK") {
                    out_mesh.material.alpha_mode = AlphaMode::Mask;
                } else if (mat.alphaMode == "BLEND") {
                    out_mesh.material.alpha_mode = AlphaMode::Blend;
                }
                out_mesh.material.alpha_cutoff = static_cast<float>(mat.alphaCutoff);
                out_mesh.material.double_sided = mat.doubleSided;
            }

            // Helper to get accessor data
            auto get_buffer_data = [&](int accessor_idx) -> const uint8_t* {
                if (accessor_idx < 0) {
                    return nullptr;
                }
                const auto& accessor = gltf.accessors[accessor_idx];
                const auto& bufferView = gltf.bufferViews[accessor.bufferView];
                const auto& buffer = gltf.buffers[bufferView.buffer];
                return buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
            };

            // The decoders below assume tightly packed accessor data; interleaved
            // bufferViews (explicit byteStride) would need stride-aware reads.
            // Warn and treat the attribute as absent rather than decode garbage.
            auto get_packed_data = [&](int accessor_idx, const char* attr_name) -> const uint8_t* {
                if (accessor_idx < 0) {
                    return nullptr;
                }
                const auto& accessor = gltf.accessors[accessor_idx];
                const auto& bufferView = gltf.bufferViews[accessor.bufferView];
                int tight_stride = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(accessor.componentType)) *
                                   tinygltf::GetNumComponentsInType(static_cast<uint32_t>(accessor.type));
                if (bufferView.byteStride != 0 && static_cast<int>(bufferView.byteStride) != tight_stride) {
                    BRACKEN_LOG_WARN("model",
                                    "{}: {} accessor has unsupported byteStride {} (expected packed {}), "
                                    "attribute skipped",
                                    path, attr_name, bufferView.byteStride, tight_stride);
                    return nullptr;
                }
                return get_buffer_data(accessor_idx);
            };

            // Position
            auto pos_it = primitive.attributes.find("POSITION");
            if (pos_it == primitive.attributes.end()) {
                continue;
            }

            const auto& pos_accessor = gltf.accessors[pos_it->second];
            const float* positions = reinterpret_cast<const float*>(get_packed_data(pos_it->second, "POSITION"));
            if (!positions) {
                continue;
            }
            int vertex_count = static_cast<int>(pos_accessor.count);

            // Normal
            const float* normals = nullptr;
            auto norm_it = primitive.attributes.find("NORMAL");
            if (norm_it != primitive.attributes.end()) {
                normals = reinterpret_cast<const float*>(get_packed_data(norm_it->second, "NORMAL"));
            }

            // Texcoord
            const float* uvs = nullptr;
            auto uv_it = primitive.attributes.find("TEXCOORD_0");
            if (uv_it != primitive.attributes.end()) {
                uvs = reinterpret_cast<const float*>(get_packed_data(uv_it->second, "TEXCOORD_0"));
            }

            // Tangent (glTF: vec4 with bitangent sign in .w)
            const float* tangents = nullptr;
            auto tan_it = primitive.attributes.find("TANGENT");
            if (tan_it != primitive.attributes.end()) {
                tangents = reinterpret_cast<const float*>(get_packed_data(tan_it->second, "TANGENT"));
            }

            // Color
            const float* colors_f = nullptr;
            const uint8_t* colors_u8 = nullptr;
            auto color_it = primitive.attributes.find("COLOR_0");
            if (color_it != primitive.attributes.end()) {
                const auto& color_accessor = gltf.accessors[color_it->second];
                if (color_accessor.type != TINYGLTF_TYPE_VEC4) {
                    BRACKEN_LOG_WARN("model", "{}: COLOR_0 accessor type is not VEC4 (unsupported), attribute skipped",
                                    path);
                } else if (color_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    colors_f = reinterpret_cast<const float*>(get_packed_data(color_it->second, "COLOR_0"));
                } else {
                    colors_u8 = get_packed_data(color_it->second, "COLOR_0");
                }
            }

            // Build vertices
            out_mesh.vertices.reserve(vertex_count);
            for (int i = 0; i < vertex_count; i++) {
                Vertex3D v;
                v.position.x = positions[i * 3 + 0];
                v.position.y = positions[i * 3 + 1];
                v.position.z = positions[i * 3 + 2];

                model.bounds.min = glm::min(model.bounds.min, v.position);
                model.bounds.max = glm::max(model.bounds.max, v.position);

                if (normals) {
                    v.normal.x = normals[i * 3 + 0];
                    v.normal.y = normals[i * 3 + 1];
                    v.normal.z = normals[i * 3 + 2];
                } else {
                    v.normal = {0, 1, 0};
                }

                if (uvs) {
                    v.texcoord.x = uvs[i * 2 + 0];
                    v.texcoord.y = uvs[i * 2 + 1];
                } else {
                    v.texcoord = {0, 0};
                }

                if (colors_f) {
                    v.color.r = colors_f[i * 4 + 0];
                    v.color.g = colors_f[i * 4 + 1];
                    v.color.b = colors_f[i * 4 + 2];
                    v.color.a = colors_f[i * 4 + 3];
                } else if (colors_u8) {
                    v.color.r = colors_u8[i * 4 + 0] / 255.0f;
                    v.color.g = colors_u8[i * 4 + 1] / 255.0f;
                    v.color.b = colors_u8[i * 4 + 2] / 255.0f;
                    v.color.a = colors_u8[i * 4 + 3] / 255.0f;
                } else {
                    // No COLOR_0: bake white. baseColorFactor is applied in the
                    // fragment shader (it multiplies vertexColor on the
                    // untextured path), so baking it here would double-apply it.
                    v.color = {1.0f, 1.0f, 1.0f, 1.0f};
                }

                if (tangents) {
                    v.tangent.x = tangents[i * 4 + 0];
                    v.tangent.y = tangents[i * 4 + 1];
                    v.tangent.z = tangents[i * 4 + 2];
                    v.tangent.w = tangents[i * 4 + 3];
                } else {
                    // Filled in below after indices are known. Initialise to a
                    // safe value so unindexed/edge-case meshes still render.
                    v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                }

                out_mesh.vertices.push_back(v);
            }

            // Indices
            if (primitive.indices >= 0) {
                const auto& idx_accessor = gltf.accessors[primitive.indices];
                const uint8_t* idx_data = get_buffer_data(primitive.indices);
                int idx_count = static_cast<int>(idx_accessor.count);

                out_mesh.indices.reserve(idx_count);
                for (int i = 0; i < idx_count; i++) {
                    uint32_t index = 0;
                    switch (idx_accessor.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            index = reinterpret_cast<const uint16_t*>(idx_data)[i];
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            index = reinterpret_cast<const uint32_t*>(idx_data)[i];
                            break;
                        default:
                            index = idx_data[i];
                            break;
                    }
                    out_mesh.indices.push_back(index);
                }
            }

            // Fallback tangent generation when the glTF asset omits TANGENT.
            if (!tangents) {
                compute_tangents_from_uvs(out_mesh.vertices, out_mesh.indices);
            }

            // Store texture data directly in mesh for deferred GPU upload.
            // texture_image_idx is a raw glTF image index, so index gltf.images
            // directly (like every other texture type); an image can be present
            // but empty (e.g. an external-URI image that failed to resolve —
            // tinygltf downgrades that to a warning), and using a compacted
            // pixel vector here would shift every later mesh onto the wrong
            // image.
            if (out_mesh.has_texture) {
                const tinygltf::Image* image =
                    (texture_image_idx >= 0 && texture_image_idx < static_cast<int>(gltf.images.size()))
                        ? &gltf.images[texture_image_idx]
                        : nullptr;
                if (image && !image->image.empty()) {
                    out_mesh.texture_pixels = image->image;
                    out_mesh.texture_width = image->width;
                    out_mesh.texture_height = image->height;
                } else {
                    BRACKEN_LOG_WARN("model", "{}: base-color image {} has no pixel data; mesh renders untextured",
                                    path, texture_image_idx);
                    out_mesh.has_texture = false;
                }
            }

            // Check for skinning data (JOINTS_0 and WEIGHTS_0)
            auto joints_it = primitive.attributes.find("JOINTS_0");
            auto weights_it = primitive.attributes.find("WEIGHTS_0");

            if (joints_it != primitive.attributes.end() && weights_it != primitive.attributes.end()) {
                // The decode below assumes u8 joint indices and float weights;
                // other encodings are valid glTF but unsupported here. On a
                // mismatch the mesh ships unskinned rather than decode garbage.
                const uint8_t* joints_data = nullptr;
                const float* weights_data = nullptr;
                const auto& joints_accessor = gltf.accessors[joints_it->second];
                const auto& weights_accessor = gltf.accessors[weights_it->second];
                if (joints_accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    BRACKEN_LOG_WARN("model", "{}: JOINTS_0 componentType is not u8 (unsupported), skipping skinning",
                                    path);
                } else if (weights_accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    BRACKEN_LOG_WARN("model",
                                    "{}: WEIGHTS_0 componentType is not float (unsupported), skipping skinning", path);
                } else {
                    joints_data = get_packed_data(joints_it->second, "JOINTS_0");
                    weights_data = reinterpret_cast<const float*>(get_packed_data(weights_it->second, "WEIGHTS_0"));
                }

                if (joints_data && weights_data) {
                    out_mesh.is_skinned = true;

                    // Convert to skinned vertices
                    out_mesh.skinned_vertices.reserve(vertex_count);
                    for (int i = 0; i < vertex_count; i++) {
                        SkinnedVertex sv;
                        // Copy base vertex data
                        const auto& v = out_mesh.vertices[i];
                        sv.position = v.position;
                        sv.normal = v.normal;
                        sv.texcoord = v.texcoord;
                        sv.color = v.color;
                        sv.tangent = v.tangent;

                        // Copy joint indices and weights
                        for (int j = 0; j < 4; j++) {
                            sv.joints[j] = joints_data[i * 4 + j];
                            sv.weights[j] = weights_data[i * 4 + j];
                        }

                        out_mesh.skinned_vertices.push_back(sv);
                    }
                }
            }

            model.meshes.push_back(std::move(out_mesh));
        }
    }

    // Validated float-accessor fetch for skeleton and animation data,
    // mirroring the JOINTS_0/WEIGHTS_0 pattern: require FLOAT componentType,
    // the expected element type, and tight packing. Quantized exports
    // (normalized u8/u16 animation outputs are valid core glTF) and
    // interleaved views are unsupported — return null so callers drop the
    // data with a warning instead of reinterpreting it as garbage floats.
    auto get_float_accessor = [&](int accessor_idx, int expected_type, const char* what) -> const float* {
        if (accessor_idx < 0 || accessor_idx >= static_cast<int>(gltf.accessors.size())) {
            return nullptr;
        }
        const auto& accessor = gltf.accessors[accessor_idx];
        if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(gltf.bufferViews.size())) {
            BRACKEN_LOG_WARN("model", "{}: {} accessor has no bufferView (sparse accessors unsupported), data dropped",
                            path, what);
            return nullptr;
        }
        if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
            BRACKEN_LOG_WARN("model",
                            "{}: {} accessor componentType {} is not float (quantized data unsupported), data dropped",
                            path, what, accessor.componentType);
            return nullptr;
        }
        if (accessor.type != expected_type) {
            BRACKEN_LOG_WARN("model", "{}: {} accessor has unexpected element type {}, data dropped", path, what,
                            accessor.type);
            return nullptr;
        }
        const auto& bufferView = gltf.bufferViews[accessor.bufferView];
        int tight_stride =
            static_cast<int>(sizeof(float)) * tinygltf::GetNumComponentsInType(static_cast<uint32_t>(accessor.type));
        if (bufferView.byteStride != 0 && static_cast<int>(bufferView.byteStride) != tight_stride) {
            BRACKEN_LOG_WARN("model",
                            "{}: {} accessor has unsupported byteStride {} (expected packed {}), data dropped", path,
                            what, bufferView.byteStride, tight_stride);
            return nullptr;
        }
        const auto& buffer = gltf.buffers[bufferView.buffer];
        return reinterpret_cast<const float*>(buffer.data.data() + bufferView.byteOffset + accessor.byteOffset);
    };

    // Load skin data (skeleton)
    if (!gltf.skins.empty()) {
        BRACKEN_PROFILE_ZONE("ModelLoader::parse_skeleton");
        const auto& skin = gltf.skins[0];
        model.has_skeleton = true;

        // Load inverse bind matrices (identity fallback per glTF when absent
        // or when the accessor is unsupported — the latter warns above).
        std::vector<glm::mat4> inverse_bind_matrices;
        if (const float* data = get_float_accessor(skin.inverseBindMatrices, TINYGLTF_TYPE_MAT4, "inverseBindMatrices");
            data != nullptr) {
            const auto& accessor = gltf.accessors[skin.inverseBindMatrices];
            for (size_t i = 0; i < accessor.count; i++) {
                inverse_bind_matrices.push_back(glm::make_mat4(data + i * 16));
            }
        }

        // Create joints from skin.joints (node indices)
        model.skeleton.joint_node_indices = skin.joints;
        model.skeleton.joints.resize(skin.joints.size());

        // Build a map from node index to joint index
        std::unordered_map<int, int> node_to_joint;
        for (size_t i = 0; i < skin.joints.size(); i++) {
            node_to_joint[skin.joints[i]] = static_cast<int>(i);
        }

        // Load joint data
        for (size_t i = 0; i < skin.joints.size(); i++) {
            int node_idx = skin.joints[i];
            const auto& node = gltf.nodes[node_idx];
            animation::Joint& joint = model.skeleton.joints[i];

            joint.name = node.name;
            joint.parent_index = -1;

            // Find parent joint
            for (size_t j = 0; j < gltf.nodes.size(); j++) {
                const auto& potential_parent = gltf.nodes[j];
                for (int child : potential_parent.children) {
                    if (child == node_idx) {
                        auto parent_it = node_to_joint.find(static_cast<int>(j));
                        if (parent_it != node_to_joint.end()) {
                            joint.parent_index = parent_it->second;
                        }
                        break;
                    }
                }
            }

            // Get local transform
            if (node.translation.size() == 3) {
                joint.local_translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
            } else {
                joint.local_translation = glm::vec3(0.0f);
            }

            if (node.rotation.size() == 4) {
                joint.local_rotation = glm::quat(static_cast<float>(node.rotation[3]), // w
                                                 static_cast<float>(node.rotation[0]), // x
                                                 static_cast<float>(node.rotation[1]), // y
                                                 static_cast<float>(node.rotation[2])  // z
                );
            } else {
                joint.local_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            }

            if (node.scale.size() == 3) {
                joint.local_scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
            } else {
                joint.local_scale = glm::vec3(1.0f);
            }

            // Inverse bind matrix
            if (i < inverse_bind_matrices.size()) {
                joint.inverse_bind_matrix = inverse_bind_matrices[i];
            } else {
                joint.inverse_bind_matrix = glm::mat4(1.0f);
            }
        }

        // The animation/IK runtime propagates the joint hierarchy in a single
        // index-order pass, which requires parents-before-children order
        // (every parent_index < its joint's index). glTF does not guarantee
        // any ordering of skin.joints, so normalize out-of-order skins with a
        // stable topological reorder and remap the vertex joint indices that
        // referenced the old order.
        {
            const size_t joint_count = model.skeleton.joints.size();
            bool parents_first = true;
            for (size_t i = 0; i < joint_count; i++) {
                if (model.skeleton.joints[i].parent_index >= static_cast<int>(i)) {
                    parents_first = false;
                    break;
                }
            }
            if (!parents_first) {
                std::vector<int> new_of_old(joint_count, -1);
                size_t assigned = 0;
                bool progress = true;
                while (assigned < joint_count && progress) {
                    progress = false;
                    for (size_t i = 0; i < joint_count; i++) {
                        if (new_of_old[i] != -1) {
                            continue;
                        }
                        int parent = model.skeleton.joints[i].parent_index;
                        if (parent < 0 || (parent < static_cast<int>(joint_count) && new_of_old[parent] != -1)) {
                            new_of_old[i] = static_cast<int>(assigned++);
                            progress = true;
                        }
                    }
                }
                if (assigned < joint_count) {
                    BRACKEN_LOG_ERROR("model", "{}: skin joint hierarchy contains a cycle, cannot load", path);
                    return false;
                }

                std::vector<animation::Joint> reordered(joint_count);
                std::vector<int> reordered_nodes(joint_count);
                for (size_t old_idx = 0; old_idx < joint_count; old_idx++) {
                    auto new_idx = static_cast<size_t>(new_of_old[old_idx]);
                    reordered[new_idx] = std::move(model.skeleton.joints[old_idx]);
                    int parent = reordered[new_idx].parent_index;
                    reordered[new_idx].parent_index = parent < 0 ? -1 : new_of_old[parent];
                    reordered_nodes[new_idx] = model.skeleton.joint_node_indices[old_idx];
                }
                model.skeleton.joints = std::move(reordered);
                model.skeleton.joint_node_indices = std::move(reordered_nodes);

                for (auto& mesh : model.meshes) {
                    for (auto& sv : mesh.skinned_vertices) {
                        for (auto& joint_idx : sv.joints) {
                            if (joint_idx < joint_count) {
                                joint_idx = static_cast<uint8_t>(new_of_old[joint_idx]);
                            }
                        }
                    }
                }
                BRACKEN_LOG_WARN("model",
                                "{}: skin joints were not ordered parents-before-children; reordered {} joints "
                                "(vertex joint indices remapped)",
                                path, joint_count);
            }
        }

        BRACKEN_LOG_DEBUG("model", "  Loaded skeleton with {} joints", model.skeleton.joints.size());

        // Cache foot IK bone indices
        model.foot_ik.init(model.skeleton);
        if (model.foot_ik.valid) {
            BRACKEN_LOG_DEBUG("model", "  Foot IK bones found");
        }
    }

    // Load animations.
    //
    // Three kinds of glTF animation channel are silently unusable here, and
    // "silently" was the defect: an artist whose walk cycle does nothing gets
    // no signal at all. Count them per model and report once at the end —
    // per-channel logging on a 200-channel rig is worse than useless.
    ModelLoadReport report;
    for (const auto& gltf_anim : gltf.animations) {
        BRACKEN_PROFILE_ZONE("ModelLoader::parse_animation");
        animation::AnimationClip clip;
        clip.name = gltf_anim.name;
        clip.duration = 0.0f;

        // Build a map from node index to joint index
        std::unordered_map<int, int> node_to_joint;
        for (size_t i = 0; i < model.skeleton.joint_node_indices.size(); i++) {
            node_to_joint[model.skeleton.joint_node_indices[i]] = static_cast<int>(i);
        }

        // Group samplers by target node
        std::unordered_map<int, animation::AnimationChannel> channels_by_joint;
        bool warned_cubicspline = false;

        for (const auto& channel : gltf_anim.channels) {
            int node_idx = channel.target_node;
            auto joint_it = node_to_joint.find(node_idx);
            if (joint_it == node_to_joint.end()) {
                report.channels_non_skin_node++; // targets a node outside the skin
                continue;
            }
            int joint_idx = joint_it->second;

            // Only TRS paths are supported ("weights" = morph targets).
            const bool is_rotation = channel.target_path == "rotation";
            if (channel.target_path != "translation" && !is_rotation && channel.target_path != "scale") {
                report.channels_non_trs_path++;
                continue;
            }

            if (channel.sampler < 0 || channel.sampler >= static_cast<int>(gltf_anim.samplers.size())) {
                report.channels_bad_sampler++;
                continue;
            }
            const auto& sampler = gltf_anim.samplers[channel.sampler];

            // glTF sampler interpolation: LINEAR (default) and STEP are
            // honored. CUBICSPLINE stores 3 output elements per key
            // (in-tangent, value, out-tangent); cubic evaluation is not
            // implemented, so extract the in-place values (middle element)
            // and interpolate them linearly — poses at the keys are exact,
            // in-between frames lose the authored easing.
            animation::Interpolation interp = animation::Interpolation::Linear;
            size_t elems_per_key = 1;
            size_t value_elem = 0;
            if (sampler.interpolation == "STEP") {
                interp = animation::Interpolation::Step;
            } else if (sampler.interpolation == "CUBICSPLINE") {
                if (!warned_cubicspline) {
                    BRACKEN_LOG_WARN("model",
                                    "{}: animation '{}' uses CUBICSPLINE interpolation (unsupported); keyframe "
                                    "values are kept, tangents ignored, tracks play back linearly",
                                    path, gltf_anim.name);
                    warned_cubicspline = true;
                }
                elems_per_key = 3;
                value_elem = 1;
            }

            const float* times = get_float_accessor(sampler.input, TINYGLTF_TYPE_SCALAR, "animation sampler input");
            const float* values = get_float_accessor(
                sampler.output, is_rotation ? TINYGLTF_TYPE_VEC4 : TINYGLTF_TYPE_VEC3, "animation sampler output");
            if (!times || !values) {
                continue; // warning already logged by get_float_accessor
            }

            const auto& input_accessor = gltf.accessors[sampler.input];
            const auto& output_accessor = gltf.accessors[sampler.output];
            size_t count = input_accessor.count;
            if (output_accessor.count < count * elems_per_key) {
                BRACKEN_LOG_WARN("model",
                                "{}: animation '{}' sampler output has {} elements but {} keys need {}, "
                                "channel dropped",
                                path, gltf_anim.name, output_accessor.count, count, count * elems_per_key);
                continue;
            }

            auto& anim_channel = channels_by_joint[joint_idx];
            anim_channel.bone_index = joint_idx;

            if (channel.target_path == "translation") {
                anim_channel.position_interp = interp;
                for (size_t i = 0; i < count; i++) {
                    const float* v = values + (i * elems_per_key + value_elem) * 3;
                    anim_channel.position_times.push_back(times[i]);
                    anim_channel.positions.emplace_back(v[0], v[1], v[2]);
                    clip.duration = std::max(clip.duration, times[i]);
                }
            } else if (is_rotation) {
                anim_channel.rotation_interp = interp;
                for (size_t i = 0; i < count; i++) {
                    const float* v = values + (i * elems_per_key + value_elem) * 4;
                    anim_channel.rotation_times.push_back(times[i]);
                    anim_channel.rotations.emplace_back(v[3], v[0], v[1], v[2]);
                    clip.duration = std::max(clip.duration, times[i]);
                }
            } else { // scale
                anim_channel.scale_interp = interp;
                for (size_t i = 0; i < count; i++) {
                    const float* v = values + (i * elems_per_key + value_elem) * 3;
                    anim_channel.scale_times.push_back(times[i]);
                    anim_channel.scales.emplace_back(v[0], v[1], v[2]);
                    clip.duration = std::max(clip.duration, times[i]);
                }
            }
        }

        // Add channels to clip
        for (auto& [joint_idx, channel] : channels_by_joint) {
            clip.channels.push_back(std::move(channel));
        }

        if (!clip.channels.empty()) {
            model.animations.push_back(std::move(clip));
        }
    }

    if (!model.animations.empty()) {
        BRACKEN_LOG_DEBUG("model", "  Loaded {} animations:", model.animations.size());
        for (const auto& anim : model.animations) {
            BRACKEN_LOG_DEBUG("model", "    - {} ({}s)", anim.name, anim.duration);
        }
    }

    model.loaded = true;
    model.compute_bounding_sphere();
    BRACKEN_LOG_INFO("model", "Loaded GLB model: {} ({} meshes) bounds: Y=[{}, {}]", path, model.meshes.size(),
                    model.bounds.min.y, model.bounds.max.y);
    if (report.any()) {
        BRACKEN_LOG_WARN("model",
                        "{}: dropped {} animation channel(s) - {} targeting nodes outside the skin, {} on "
                        "unsupported paths (only translation/rotation/scale are read; morph 'weights' are not), "
                        "{} with an out-of-range sampler index",
                        path, report.total(), report.channels_non_skin_node, report.channels_non_trs_path,
                        report.channels_bad_sampler);
    }
    return true;
}

void ModelLoader::upload_to_gpu(gpu::GPUDevice& device, TextureCache& textures, Model& model,
                                bool retain_cpu_geometry) {
    BRACKEN_PROFILE_ZONE("ModelLoader::upload_to_gpu");
    // GPU resource creation is main-thread tier (buffers + textures).
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    for (auto& mesh : model.meshes) {
        if (mesh.uploaded) {
            continue;
        }
        if (mesh.vertices.empty() && mesh.skinned_vertices.empty()) {
            continue;
        }

        // Create vertex buffer
        if (mesh.is_skinned && !mesh.skinned_vertices.empty()) {
            mesh.vertex_buffer =
                gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Vertex, mesh.skinned_vertices.data(),
                                              mesh.skinned_vertices.size() * sizeof(SkinnedVertex));
        } else if (!mesh.vertices.empty()) {
            mesh.vertex_buffer = gpu::GPUBuffer::create_static(
                device, gpu::GPUBuffer::Type::Vertex, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex3D));
        }

        if (!mesh.vertex_buffer) {
            BRACKEN_LOG_ERROR("model", "Failed to create vertex buffer for mesh");
            continue;
        }

        // The source pixels are about to be released. Preserve only a compact
        // per-triangle linear-colour proxy when the caller retained geometry
        // for a later software-RT build; the normal raster path pays nothing.
        if (retain_cpu_geometry) {
            internal::bake_rt_triangle_base_colors(mesh);
            internal::bake_rt_triangle_emissive(mesh);
            internal::bake_rt_triangle_opacity(mesh);
        }

        // Create index buffer
        if (!mesh.indices.empty()) {
            mesh.index_buffer = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Index, mesh.indices.data(),
                                                              mesh.indices.size() * sizeof(uint32_t));

            if (!mesh.index_buffer) {
                BRACKEN_LOG_ERROR("model", "Failed to create index buffer for mesh");
            }
        }

        // Create texture if available. Base-color textures are sRGB-encoded per
        // glTF; sampler linearizes on read so lighting math stays in linear.
        if (mesh.has_texture && !mesh.texture_pixels.empty()) {
            mesh.texture = textures.acquire(device, mesh.texture_width, mesh.texture_height,
                                            gpu::TextureFormat::RGBA8_SRGB, mesh.texture_pixels,
                                            true); // Generate mipmaps

            if (!mesh.texture) {
                BRACKEN_LOG_ERROR("model", "Failed to create texture for mesh");
                mesh.has_texture = false;
            } else {
                // Clear CPU-side texture data to save memory
                mesh.texture_pixels.clear();
                mesh.texture_pixels.shrink_to_fit();
            }
        }

        // Normal map: stored UNORM (NOT sRGB) per glTF spec. Generate mipmaps so
        // distant fragments don't sample raw aliased high-frequency detail.
        if (!mesh.material.normal_pixels.empty()) {
            mesh.material.normal_texture =
                textures.acquire(device, mesh.material.normal_width, mesh.material.normal_height,
                                 gpu::TextureFormat::RGBA8, mesh.material.normal_pixels, true);
            if (mesh.material.normal_texture) {
                mesh.material.normal_pixels.clear();
                mesh.material.normal_pixels.shrink_to_fit();
            }
        }

        // Metallic-roughness: linear UNORM per glTF (G = roughness, B = metallic;
        // R carries AO when has_packed_occlusion).
        if (!mesh.material.metallic_roughness_pixels.empty()) {
            mesh.material.metallic_roughness_texture = textures.acquire(
                device, mesh.material.metallic_roughness_width, mesh.material.metallic_roughness_height,
                gpu::TextureFormat::RGBA8, mesh.material.metallic_roughness_pixels, true);
            if (mesh.material.metallic_roughness_texture) {
                mesh.material.metallic_roughness_pixels.clear();
                mesh.material.metallic_roughness_pixels.shrink_to_fit();
            }
        }

        // Emissive: sRGB-encoded like base color.
        if (!mesh.material.emissive_pixels.empty()) {
            mesh.material.emissive_texture =
                textures.acquire(device, mesh.material.emissive_width, mesh.material.emissive_height,
                                 gpu::TextureFormat::RGBA8_SRGB, mesh.material.emissive_pixels, true);
            if (mesh.material.emissive_texture) {
                mesh.material.emissive_pixels.clear();
                mesh.material.emissive_pixels.shrink_to_fit();
            }
        }

        mesh.uploaded = true;

        // Cache index count before freeing CPU-side geometry data
        mesh.cached_index_count = static_cast<uint32_t>(mesh.indices.size());

        // Free CPU-side vertex/index data - it's on the GPU now.
        //
        // Unless something still needs it: ray tracing builds its BVH from
        // these arrays, and LOD generation simplifies them. Retention is
        // opt-in because for most games this memory is pure waste — a scene's
        // vertex data is typically the largest CPU allocation there is.
        if (!retain_cpu_geometry) {
            mesh.vertices.clear();
            mesh.vertices.shrink_to_fit();
            mesh.skinned_vertices.clear();
            mesh.skinned_vertices.shrink_to_fit();
            mesh.indices.clear();
            mesh.indices.shrink_to_fit();
        }
    }
}

void ModelLoader::free_gpu_resources(Model& model) {
    // GPU resource destruction is main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    for (auto& mesh : model.meshes) {
        // Free SDL3 GPU resources
        mesh.vertex_buffer.reset();
        mesh.index_buffer.reset();
        // Textures are owned by the ModelManager's TextureCache and may still
        // be referenced by another model; only drop the borrow here.
        mesh.texture = nullptr;
        mesh.material.normal_texture = nullptr;
        mesh.material.metallic_roughness_texture = nullptr;
        mesh.material.emissive_texture = nullptr;
        mesh.uploaded = false;
        mesh.has_texture = false;
    }
}

void ModelLoader::free_gpu_resources(TextureCache& textures, Model& model) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    for (auto& mesh : model.meshes) {
        // One release per non-null slot, mirroring upload_to_gpu's one acquire
        // per slot. Two meshes referencing the same image took two references
        // and give back two; a null slot never took one. Doing this from the
        // Mesh rather than from a per-model list is what keeps the pairing
        // impossible to get out of step with the upload.
        textures.release(mesh.texture);
        textures.release(mesh.material.normal_texture);
        textures.release(mesh.material.metallic_roughness_texture);
        textures.release(mesh.material.emissive_texture);
    }
    free_gpu_resources(model);
}

ModelManager::ModelManager() {
    // Reserve slot 0 as INVALID_MODEL_HANDLE
    models_.push_back(nullptr);
    model_paths_.emplace_back();
}

ModelManager::~ModelManager() {
    unload_all();
}

ModelHandle ModelManager::load_model_with_lods(const std::string& name, const std::string& path,
                                               std::span<const float> ratios,
                                               std::span<const float> thresholds) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();

    auto model = std::make_unique<Model>();
    if (!ModelLoader::load_glb(path, *model)) {
        return INVALID_MODEL_HANDLE;
    }

    // Register FIRST but upload LAST: simplification reads the CPU-side
    // vertices and indices that upload_to_gpu() frees, so the LOD levels have
    // to be generated from the model while it is still only in system memory.
    const auto handle = static_cast<ModelHandle>(models_.size());
    models_.push_back(std::move(model));
    model_paths_.push_back(path);
    name_to_handle_[name] = handle;

    (void)build_lods(handle, ratios, thresholds);

    Model* base = get_model(handle);
    if (device_ && base != nullptr) {
        ModelLoader::upload_to_gpu(*device_, textures_, *base, retain_cpu_geometry_);
        // Levels were appended after the base, so uploading them here keeps
        // the "CPU geometry is freed at upload" rule uniform across all of them.
        for (ModelHandle lod : base->lod_handles) {
            if (Model* m = get_model(lod)) {
                ModelLoader::upload_to_gpu(*device_, textures_, *m, retain_cpu_geometry_);
            }
        }
        // No albedo sharing step: every level asked the cache for the same
        // pixels as the base and got the same texture back.
    }
    return handle;
}

ModelHandle ModelManager::load_model(const std::string& name, const std::string& path) {
    // Mutates the model table (and creates GPU resources): main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    auto model = std::make_unique<Model>();
    if (ModelLoader::load_glb(path, *model)) {
        // Upload to GPU if device is available
        if (device_) {
            ModelLoader::upload_to_gpu(*device_, textures_, *model, retain_cpu_geometry_);
        }

        ModelHandle handle = static_cast<ModelHandle>(models_.size());
        models_.push_back(std::move(model));
        model_paths_.push_back(path);
        name_to_handle_[name] = handle;

        if (watcher_) {
            // Watched for the lifetime of the app (never unwatched); reload logs itself.
            (void)watcher_->watch_file(path,
                                       [this, handle](const std::filesystem::path&) { (void)reload_model(handle); });
        }
        return handle;
    }
    return INVALID_MODEL_HANDLE;
}

std::size_t ModelManager::build_lods(ModelHandle handle, std::span<const float> ratios,
                                     std::span<const float> thresholds) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();

    Model* base = get_model(handle);
    if (base == nullptr || base->meshes.empty()) {
        return 0;
    }

    // upload_to_gpu() frees CPU-side vertices and indices once the geometry is
    // on the device, so a model that has already been uploaded has nothing
    // left to simplify. Say so instead of returning 0 and leaving the caller
    // to wonder why LOD "did nothing" — use load_model_with_lods(), which
    // generates the levels before the upload.
    bool has_cpu_geometry = false;
    for (const Mesh& m : base->meshes) {
        if (!m.indices.empty()) {
            has_cpu_geometry = true;
            break;
        }
    }
    if (!has_cpu_geometry) {
        BRACKEN_LOG_ERROR("model",
                          "build_lods(handle {}): the model's CPU geometry was released at "
                          "upload; call load_model_with_lods() instead of build_lods() after "
                          "load_model()",
                          handle);
        return 0;
    }
    if (thresholds.size() != ratios.size()) {
        BRACKEN_LOG_ERROR("model",
                          "build_lods: {} ratios but {} thresholds; they must match",
                          ratios.size(), thresholds.size());
        return 0;
    }
    if (base->has_skeleton) {
        // Clustering merges vertices by position alone, so two vertices bound
        // to different bones would collapse into one and the mesh would tear
        // apart the moment it animates. Authored skinned LODs are fine; a
        // generated one is not.
        BRACKEN_LOG_INFO("model", "build_lods: skipping skinned model (handle {})", handle);
        return 0;
    }

    std::size_t created = 0;
    std::size_t source_tris = 0;
    std::size_t lod_tris = 0;
    for (const Mesh& m : base->meshes) {
        source_tris += m.indices.size() / 3;
    }

    for (std::size_t level = 0; level < ratios.size(); ++level) {
        auto lod = std::make_unique<Model>();
        lod->bounds = base->bounds;
        lod->bounding_center = base->bounding_center;
        lod->bounding_half_diag = base->bounding_half_diag;
        lod->loaded = true;

        for (const Mesh& src : base->meshes) {
            SimplifiedMesh simplified =
                simplify_to_ratio(src.vertices, src.indices, ratios[level]);
            if (simplified.indices.size() < 3) {
                continue; // this mesh vanished at this ratio; drop it
            }
            Mesh dst;
            dst.vertices = std::move(simplified.vertices);
            dst.indices = std::move(simplified.indices);
            // Shading parameters carry over field by field: an LOD is the same
            // surface, so it must shade identically or the switch reads as a
            // lighting pop rather than a detail change. PBRMaterial owns its
            // texture objects and so cannot be copied wholesale — the images
            // are shared below instead.
            dst.material.base_color_factor = src.material.base_color_factor;
            dst.material.metallic_factor = src.material.metallic_factor;
            dst.material.roughness_factor = src.material.roughness_factor;
            dst.material.normal_scale = src.material.normal_scale;
            dst.material.occlusion_strength = src.material.occlusion_strength;
            dst.material.emissive_factor = src.material.emissive_factor;
            dst.material.alpha_cutoff = src.material.alpha_cutoff;
            dst.has_texture = src.has_texture;
            dst.cast_shadows = src.cast_shadows;
            lod->meshes.push_back(std::move(dst));
        }

        if (lod->meshes.empty()) {
            break; // nothing survived; coarser levels would be emptier still
        }
        if (level + 1 == ratios.size()) {
            for (const Mesh& m : lod->meshes) {
                lod_tris += m.indices.size() / 3;
            }
        }

        // Upload and texture sharing are the CALLER's job: at this point the
        // base may not be uploaded yet, so its GPUTexture does not exist to
        // share. load_model_with_lods() does both once the base is on the GPU.

        const auto lod_handle = static_cast<ModelHandle>(models_.size());
        models_.push_back(std::move(lod));
        model_paths_.emplace_back();
        base = get_model(handle); // models_ may have reallocated
        if (base == nullptr) {
            break; // unreachable: handle was resolved above and models_ only grows
        }
        base->lod_handles.push_back(lod_handle);
        base->lod_thresholds.push_back(thresholds[level]);
        ++created;
    }

    if (created > 0) {
        BRACKEN_LOG_INFO("model", "build_lods(handle {}): {} levels, {} -> {} triangles at the coarsest",
                         handle, created, source_tris, lod_tris);
    }
    return created;
}

std::size_t ModelManager::model_count() const noexcept {
    return models_.size();
}

Model* ModelManager::get_model(ModelHandle handle) const {
    if (handle == INVALID_MODEL_HANDLE || handle >= models_.size()) {
        return nullptr;
    }
    return models_[handle].get();
}

Model* ModelManager::get_model(const std::string& name) const {
    auto it = name_to_handle_.find(name);
    if (it == name_to_handle_.end()) {
        return nullptr;
    }
    return models_[it->second].get();
}

const char* to_string(ModelRegisterError error) {
    switch (error) {
    case ModelRegisterError::EmptyModel:
        return "model has no geometry";
    case ModelRegisterError::NameAlreadyRegistered:
        return "a model is already registered under that name";
    case ModelRegisterError::UploadFailed:
        return "one or more meshes failed to upload to the GPU";
    }
    return "unknown error";
}

core::Result<ModelHandle, ModelRegisterError> ModelManager::register_model(const std::string& name,
                                                                          std::unique_ptr<Model> model) {
    // Mutates the model table (and creates GPU resources): main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    BRACKEN_CONTRACT(model != nullptr, "register_model() with a null model");

    // A name collision used to overwrite the map entry and leave the previous
    // model in the table, reachable by handle but by nothing else — a leak
    // that presents as "my model didn't update".
    if (name_to_handle_.contains(name)) {
        BRACKEN_LOG_ERROR("model", "register_model('{}'): {}", name,
                          to_string(ModelRegisterError::NameAlreadyRegistered));
        return core::Err{ModelRegisterError::NameAlreadyRegistered};
    }

    // Which meshes carry geometry has to be answered BEFORE the upload: it
    // frees the CPU-side vertex arrays it consumed.
    bool any_geometry = false;
    for (const auto& mesh : model->meshes) {
        if (mesh.uploaded || !mesh.vertices.empty() || !mesh.skinned_vertices.empty()) {
            any_geometry = true;
            break;
        }
    }
    if (!any_geometry) {
        BRACKEN_LOG_ERROR("model", "register_model('{}'): {}", name, to_string(ModelRegisterError::EmptyModel));
        return core::Err{ModelRegisterError::EmptyModel};
    }

    // Derived state the host used to be responsible for remembering. Bounds
    // left empty are scanned off the mesh vertices (before the upload frees
    // them); a model that set its own — a skinned one with animation headroom
    // — keeps them.
    if (model->bounds.empty()) {
        glm::vec3 lo(std::numeric_limits<float>::max());
        glm::vec3 hi(std::numeric_limits<float>::lowest());
        for (const auto& mesh : model->meshes) {
            for (const auto& v : mesh.vertices) {
                lo = glm::min(lo, v.position);
                hi = glm::max(hi, v.position);
            }
            for (const auto& v : mesh.skinned_vertices) {
                lo = glm::min(lo, v.position);
                hi = glm::max(hi, v.position);
            }
        }
        if (lo.x <= hi.x) {
            model->bounds = {lo, hi};
        }
    }
    model->compute_bounding_sphere();
    model->loaded = true;

    if (device_) {
        ModelLoader::upload_to_gpu(*device_, textures_, *model, retain_cpu_geometry_);
        for (const auto& mesh : model->meshes) {
            if (!mesh.uploaded) {
                BRACKEN_LOG_ERROR("model", "register_model('{}'): {}", name,
                                  to_string(ModelRegisterError::UploadFailed));
                return core::Err{ModelRegisterError::UploadFailed};
            }
        }
    }

    ModelHandle handle = static_cast<ModelHandle>(models_.size());
    models_.push_back(std::move(model));
    model_paths_.emplace_back(); // procedural -- no source path
    name_to_handle_[name] = handle;
    return handle;
}

bool ModelManager::reload_model(ModelHandle handle) {
    // Hot-reload apply: replaces model contents in place — main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (handle == INVALID_MODEL_HANDLE || handle >= models_.size()) {
        return false;
    }
    if (handle >= model_paths_.size()) {
        return false;
    }
    const auto& path = model_paths_[handle];
    if (path.empty()) {
        return false; // procedural model, nothing to reload from
    }

    auto fresh = std::make_unique<Model>();
    if (!ModelLoader::load_glb(path, *fresh)) {
        BRACKEN_LOG_WARN("hot_reload", "ModelManager: reload failed for '{}'", path);
        return false;
    }
    if (device_) {
        ModelLoader::upload_to_gpu(*device_, textures_, *fresh);
        if (models_[handle]) {
            // Counted release: the outgoing contents took references at their
            // own upload, and a hot-reload that never gave them back would
            // leave every reloaded model's images pinned forever.
            ModelLoader::free_gpu_resources(textures_, *models_[handle]);
        }
    }
    if (models_[handle]) {
        // Move-assign into the existing object so raw Model* handed to
        // callers stay valid across reloads (the header promises pointer
        // stability; only the model's *contents* are replaced).
        *models_[handle] = std::move(*fresh);
    } else {
        models_[handle] = std::move(fresh);
    }
    BRACKEN_LOG_INFO("hot_reload", "ModelManager: reloaded '{}'", path);
    return true;
}

void ModelManager::enable_hot_reload(core::asset::FileWatcher& watcher) {
    // FileWatcher entry points are main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    watcher_ = &watcher;
    for (size_t i = 1; i < model_paths_.size(); ++i) {
        if (model_paths_[i].empty()) {
            continue;
        }
        ModelHandle h = static_cast<ModelHandle>(i);
        // Watched for the lifetime of the app (never unwatched); reload logs itself.
        (void)watcher_->watch_file(model_paths_[i], [this, h](const std::filesystem::path&) { (void)reload_model(h); });
    }
}

ModelHandle ModelManager::get_handle(const std::string& name) const {
    auto it = name_to_handle_.find(name);
    return it != name_to_handle_.end() ? it->second : INVALID_MODEL_HANDLE;
}

void ModelManager::unload_all() {
    // Mutates the model table and releases GPU resources: main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    for (size_t i = 1; i < models_.size(); ++i) {
        if (models_[i]) {
            ModelLoader::free_gpu_resources(*models_[i]);
        }
    }
    models_.clear();
    // Every borrow is dead now, so the images can go too.
    textures_.clear();
    models_.push_back(nullptr); // Re-reserve slot 0
    model_paths_.clear();
    model_paths_.emplace_back();
    name_to_handle_.clear();
}

std::size_t ModelManager::gpu_bytes(ModelHandle handle) const noexcept {
    if (handle == INVALID_MODEL_HANDLE || handle >= models_.size() || !models_[handle]) {
        return 0;
    }
    const Model& model = *models_[handle];
    std::size_t total = 0;
    for (const Mesh& mesh : model.meshes) {
        if (mesh.vertex_buffer) {
            total += mesh.vertex_buffer->size();
        }
        if (mesh.index_buffer) {
            total += mesh.index_buffer->size();
        }
        total += textures_.texture_bytes(mesh.texture);
        total += textures_.texture_bytes(mesh.material.normal_texture);
        total += textures_.texture_bytes(mesh.material.metallic_roughness_texture);
        total += textures_.texture_bytes(mesh.material.emissive_texture);
    }
    return total;
}

std::size_t ModelManager::unload_model(ModelHandle handle) {
    // Destroys GPU resources and mutates the model table: main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (handle == INVALID_MODEL_HANDLE || handle >= models_.size() || !models_[handle]) {
        return 0;
    }
    Model& model = *models_[handle];
    if (!model.loaded) {
        return 0;
    }
    // Measured BEFORE the free, because afterwards there is nothing to measure.
    const std::size_t released = gpu_bytes(handle);
    ModelLoader::free_gpu_resources(textures_, model);
    // Retained CPU geometry is the other half of an evicted model's footprint
    // — for a ray-traced scene it is the LARGER half — so it goes too. That is
    // also precisely why the acceleration structure has to be rebuilt: its
    // BLAS triangles were copied out of these arrays.
    for (Mesh& mesh : model.meshes) {
        mesh.vertices.clear();
        mesh.vertices.shrink_to_fit();
        mesh.skinned_vertices.clear();
        mesh.skinned_vertices.shrink_to_fit();
        mesh.indices.clear();
        mesh.indices.shrink_to_fit();
    }
    // The slot, the name, the path, the bounds and the LOD links all stay —
    // only `loaded` flips, which is the single flag every consumer tests.
    model.loaded = false;
    return released;
}

bool ModelManager::reupload_model(ModelHandle handle, std::unique_ptr<Model> decoded) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (handle == INVALID_MODEL_HANDLE || handle >= models_.size() || !models_[handle]) {
        BRACKEN_LOG_ERROR("model", "reupload_model: handle {} names no slot", handle);
        return false;
    }
    if (!decoded) {
        BRACKEN_LOG_ERROR("model", "reupload_model: null model for handle {}", handle);
        return false;
    }
    // "Replace" as well as "restore": a still-loaded slot gives its resources
    // back first, so the texture references stay balanced either way.
    (void)unload_model(handle);

    Model& slot = *models_[handle];
    // The LOD chain belongs to the SLOT, not to the freshly decoded bytes: a
    // decode knows nothing about the levels that were generated for this model
    // and would silently orphan them (they stay registered, unreachable, and
    // select_lod stops finding them).
    std::vector<ModelHandle> lod_handles = std::move(slot.lod_handles);
    std::vector<float> lod_thresholds = std::move(slot.lod_thresholds);

    // Move-assign into the existing object so raw Model* handed to callers
    // stay valid — the same contract reload_model keeps.
    slot = std::move(*decoded);
    slot.lod_handles = std::move(lod_handles);
    slot.lod_thresholds = std::move(lod_thresholds);

    if (device_ != nullptr) {
        ModelLoader::upload_to_gpu(*device_, textures_, slot, retain_cpu_geometry_);
    }
    slot.compute_bounding_sphere();
    slot.loaded = true;
    return true;
}

} // namespace bracken
