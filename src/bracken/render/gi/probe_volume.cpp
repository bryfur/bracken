#include "bracken/render/gi/probe_volume.hpp"

#include "bracken/core/logger.hpp"

namespace bracken::render::gi {

namespace {

constexpr const char* LOG_TAG = "gi";

/// Every atlas is both sampled and (from R1b) written by compute, so they all
/// carry the same usage set the traced targets do (`RtRenderer::ensure_target`).
constexpr SDL_GPUTextureUsageFlags ATLAS_USAGE = SDL_GPU_TEXTUREUSAGE_SAMPLER |
                                                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;

std::size_t texture_bytes(int width, int height, std::size_t bytes_per_texel) {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * bytes_per_texel;
}

} // namespace

ProbeVolume::~ProbeVolume() { shutdown(); }

bool ProbeVolume::init(gpu::GPUDevice& device) {
    shutdown();
    device_ = &device;

    const int irr_w = atlas_width(IRRADIANCE_PROBE_TEXELS);
    const int irr_h = atlas_height(IRRADIANCE_PROBE_TEXELS);
    const int vis_w = atlas_width(VISIBILITY_PROBE_TEXELS);
    const int vis_h = atlas_height(VISIBILITY_PROBE_TEXELS);

    stats_ = GiStats{};
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        Cascade& cascade = cascades_[static_cast<std::size_t>(c)];
        cascade.cpu.reset(c);
        cascade.irradiance = gpu::GPUTexture::create_2d(device, irr_w, irr_h,
                                                        SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, ATLAS_USAGE);
        cascade.visibility = gpu::GPUTexture::create_2d(device, vis_w, vis_h,
                                                        SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT, ATLAS_USAGE);
        cascade.state = gpu::GPUTexture::create_2d(device, PROBES_PER_ATLAS_ROW, ATLAS_ROWS,
                                                   SDL_GPU_TEXTUREFORMAT_R8_UINT, ATLAS_USAGE);
        if (!cascade.irradiance || !cascade.visibility || !cascade.state) {
            BRACKEN_LOG_ERROR(LOG_TAG, "ProbeVolume: could not allocate cascade {} atlases", c);
            shutdown();
            return false;
        }

        GiCascadeStats& cs = stats_.per_cascade[static_cast<std::size_t>(c)];
        cs.spacing = cascade_spacing(c);
        cs.probes = static_cast<std::uint32_t>(PROBES_PER_CASCADE);
        cs.atlas_bytes = texture_bytes(irr_w, irr_h, 8) + texture_bytes(vis_w, vis_h, 4) +
                         texture_bytes(PROBES_PER_ATLAS_ROW, ATLAS_ROWS, 1);
        stats_.atlas_bytes += cs.atlas_bytes;
        stats_.probes += cs.probes;
    }
    stats_.cascades = static_cast<std::uint32_t>(CASCADE_COUNT);
    stats_.active = true;
    ready_ = true;

    BRACKEN_LOG_INFO(LOG_TAG,
                     "probe field: {} cascades x {}x{}x{} probes, spacing {:.0f}..{:.0f} world units, "
                     "irradiance {}x{} visibility {}x{}, {:.1f} MB",
                     CASCADE_COUNT, GRID_X, GRID_Y, GRID_Z, cascade_spacing(0),
                     cascade_spacing(CASCADE_COUNT - 1), irr_w, irr_h, vis_w, vis_h,
                     static_cast<double>(stats_.atlas_bytes) / (1024.0 * 1024.0));
    return true;
}

void ProbeVolume::shutdown() {
    for (Cascade& cascade : cascades_) {
        cascade.irradiance.reset();
        cascade.visibility.reset();
        cascade.state.reset();
        cascade.cpu.reset(cascade.cpu.cascade());
    }
    device_ = nullptr;
    ready_ = false;
    stats_ = GiStats{};
}

void ProbeVolume::update(SDL_GPUCommandBuffer* cmd, const glm::vec3& camera_position) {
    if (!ready_) {
        return;
    }
    stats_.probes_invalidated = 0;
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        Cascade& cascade = cascades_[static_cast<std::size_t>(c)];
        const std::uint32_t invalidated = cascade.cpu.update(camera_position);

        GiCascadeStats& cs = stats_.per_cascade[static_cast<std::size_t>(c)];
        cs.base = cascade.cpu.base();
        cs.probes_invalidated = invalidated;
        stats_.probes_invalidated += invalidated;
        stats_.probes_invalidated_total += invalidated;

        // The whole state texture is 6912 bytes, so a scrolled cascade
        // re-uploads all of it rather than carrying a dirty-rect list. That is
        // a measured trade, not laziness: the worst case is all three cascades
        // scrolling in one frame, 20.25 KB of transfer — against which a rect
        // scheme would have to describe an L-shaped slab (a diagonal scroll
        // enters two overlapping arms), i.e. two or three partial copies plus
        // the code that computes them, to save a few kilobytes on a bus that
        // moves megabytes of instance data every frame.
        if (cmd != nullptr && cascade.cpu.state_dirty() && cascade.state) {
            cascade.state->upload(cmd, cascade.cpu.state().data(), PROBES_PER_ATLAS_ROW, ATLAS_ROWS);
            cascade.cpu.clear_state_dirty();
        }
    }
}

gpu::GPUTexture* ProbeVolume::irradiance(int cascade) const {
    if (cascade < 0 || cascade >= CASCADE_COUNT) {
        return nullptr;
    }
    return cascades_[static_cast<std::size_t>(cascade)].irradiance.get();
}

gpu::GPUTexture* ProbeVolume::visibility(int cascade) const {
    if (cascade < 0 || cascade >= CASCADE_COUNT) {
        return nullptr;
    }
    return cascades_[static_cast<std::size_t>(cascade)].visibility.get();
}

gpu::GPUTexture* ProbeVolume::probe_state(int cascade) const {
    if (cascade < 0 || cascade >= CASCADE_COUNT) {
        return nullptr;
    }
    return cascades_[static_cast<std::size_t>(cascade)].state.get();
}

const ProbeCascade& ProbeVolume::cascade(int index) const {
    const int clamped = index < 0 ? 0 : (index >= CASCADE_COUNT ? CASCADE_COUNT - 1 : index);
    return cascades_[static_cast<std::size_t>(clamped)].cpu;
}

} // namespace bracken::render::gi
