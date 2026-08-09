#include "bracken/render/raytracing/emissive_clusters.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace bracken::render::rt {

namespace {
constexpr const char* LOG_TAG = "rt";

// Fan-out grains, chosen the way light_cluster.cpp's are. Binning chunks the
// froxel range in whole Z SLICES so a record's row clipping stays chunk-local;
// the record-range precompute chunks the record array, whose entries are pure
// per-record functions with disjoint writes and therefore need no merge.
// ONE Z slice per chunk, not the light binner's three. The load here is wildly
// uneven — the near slices hold everything a village emits and the far ones hold
// nothing — so coarse chunks leave the pass waiting on one worker. Measured on
// the playground: 8 chunks saturated at 1.1 ms no matter how many workers were
// available; 24 chunks scale.
constexpr std::size_t BIN_SLICE_CHUNKS = 1;
// Below this many records the whole bin runs as ONE inline chunk (same body —
// granularity control, not a second path), which is what keeps a scene with a
// single glowing orb off the pool entirely.
constexpr std::size_t BIN_MIN_PARALLEL_RECORDS = 64;
constexpr std::size_t RANGE_GRAIN = 512;

/// NDC -> tile index, matching ClusterGrid's `tile_of`: index 0 starts at NDC -1
/// (left / bottom of screen). A second copy would be a second place for the two
/// binners to disagree about which tile a point is in, so this one is a
/// transliteration and nothing more.
uint32_t tile_of(float ndc, uint32_t dim) {
    const float t = std::floor((ndc * 0.5f + 0.5f) * static_cast<float>(dim));
    return static_cast<uint32_t>(std::clamp(t, 0.0f, static_cast<float>(dim) - 1.0f));
}

} // namespace

EmissiveClusterGrid::~EmissiveClusterGrid() {
    shutdown();
}

bool EmissiveClusterGrid::init(gpu::GPUDevice& device) {
    device_ = &device;
    return true;
}

void EmissiveClusterGrid::shutdown() {
    buffer_.reset();
    buffer_entries_ = 0;
    table_.clear();
    ranges_.clear();
    range_active_.clear();
    active_.clear();
    bin_index_.clear();
    bin_score_.clear();
    chunk_dropped_.clear();
    chunk_filled_.clear();
    chunk_entries_.clear();
    froxel_count_ = 0;
    projection_valid_ = false;
    live_ = false;
    dirty_ = false;
    stats_ = {};
    // The rebuild gate's memo. `last_records_` in particular must not outlive
    // the table it describes: a freed record vector's address is exactly the
    // kind of thing a later allocation reuses.
    last_records_ = nullptr;
    last_record_count_ = 0;
    last_generation_ = k_emissive_no_generation;
    last_view_ = glm::mat4(0.0f);
    last_inv_projection_ = glm::mat4(0.0f);
    last_z_planes_ = glm::vec4(0.0f);
    last_grid_dim_ = glm::uvec4(0u);
    device_ = nullptr;
}

std::span<const uint32_t> EmissiveClusterGrid::froxel_list(uint32_t froxel) const {
    if (froxel >= froxel_count_ || table_.empty()) {
        return {};
    }
    const std::size_t base = static_cast<std::size_t>(froxel) * k_emissive_froxel_stride;
    const uint32_t count = std::min(table_[base], k_emissive_froxel_capacity);
    return {table_.data() + base + 1, count};
}

void EmissiveClusterGrid::build(std::span<const EmissiveTriangle> records,
                                const lighting::ClusterParams& params,
                                std::span<const lighting::ClusterBounds> bounds) {
    build(core::jobs::JobSystem::instance(), records, params, bounds, k_emissive_no_generation);
}

void EmissiveClusterGrid::build(core::jobs::JobSystem& js, std::span<const EmissiveTriangle> records,
                                const lighting::ClusterParams& params,
                                std::span<const lighting::ClusterBounds> bounds) {
    build(js, records, params, bounds, k_emissive_no_generation);
}

void EmissiveClusterGrid::build(std::span<const EmissiveTriangle> records,
                                const lighting::ClusterParams& params,
                                std::span<const lighting::ClusterBounds> bounds,
                                uint32_t record_generation) {
    build(core::jobs::JobSystem::instance(), records, params, bounds, record_generation);
}

void EmissiveClusterGrid::build(core::jobs::JobSystem& js, std::span<const EmissiveTriangle> records,
                                const lighting::ClusterParams& params,
                                std::span<const lighting::ClusterBounds> bounds,
                                uint32_t record_generation) {
    BRACKEN_PROFILE_ZONE_COLOR("EmissiveClusterGrid::build", 0x27AE60);

    // THE REBUILD GATE, before anything is cleared. The table is a pure
    // function of the record contents, the view, the projection, the near/far
    // pair and the grid — so when none of them moved, the table already on hand
    // IS this frame's table and re-deriving it produces the same 235 KB. Every
    // input is compared: the caller's generation for the contents (see
    // `k_emissive_no_generation` for why that is not inferable from the span),
    // and `params`' four contributing fields, including `zPlanes` and `gridDim`
    // — the froxel BOUNDS are a function of those two as much as of the
    // projection, and comparing the view alone would memoize across a near-plane
    // change.
    //
    // `live_` gates the whole thing, which is what makes it correct across
    // invalidate() (it clears live_) and across a failed upload (that clears
    // live_ too). `dirty_` is deliberately left ALONE rather than forced false:
    // if the last upload could not allocate, the retry has to survive this.
    if (live_ && record_generation != k_emissive_no_generation &&
        record_generation == last_generation_ && records.data() == last_records_ &&
        records.size() == last_record_count_ && bounds.size() == froxel_count_ &&
        params.view == last_view_ && params.invProjection == last_inv_projection_ &&
        params.zPlanes == last_z_planes_ && params.gridDim == last_grid_dim_) {
        return;
    }

    stats_ = {};
    live_ = false;

    const uint32_t grid_x = params.gridDim.x;
    const uint32_t grid_y = params.gridDim.y;
    const uint32_t grid_z = params.gridDim.z;
    const uint64_t froxels = static_cast<uint64_t>(grid_x) * grid_y * grid_z;
    if (records.empty() || froxels == 0 || bounds.size() != froxels) {
        // No records, or a grid this table cannot describe. Leaving live_ false
        // is the whole degradation: the trace then sends share 0 and draws
        // exactly as R6 did.
        return;
    }
    froxel_count_ = static_cast<uint32_t>(froxels);

    const std::size_t table_size = emissive_cluster_table_size(froxel_count_);
    if (table_.size() != table_size) {
        table_.assign(table_size, 0u);
        bin_index_.assign(static_cast<std::size_t>(froxel_count_) * k_emissive_froxel_capacity, 0u);
        bin_score_.assign(static_cast<std::size_t>(froxel_count_) * k_emissive_froxel_capacity, 0.0f);
    }

    // The projection, memoized. ClusterParams carries only its inverse (the
    // cbuffer's layout is mirrored in HLSL and cannot grow a field for this), and
    // the conservative tile range below needs the forward matrix to project a
    // record's bounding box. One 4x4 inverse per projection change, not per
    // frame and not per record.
    if (!projection_valid_ || params.invProjection != last_inv_projection_) {
        projection_ = glm::inverse(params.invProjection);
        last_inv_projection_ = params.invProjection;
        projection_valid_ = true;
    }

    const float near_z = params.zPlanes.x;
    const float far_z = params.zPlanes.y;

    // The four side planes, in view space, from the same projection. Six dot
    // products reject a record that cannot touch the frustum at all, which is
    // most of a village's windows on any given frame — and it costs a fraction
    // of the eight matrix-vector products that bound its tile range.
    {
        const glm::mat4& p = projection_;
        const glm::vec4 row3(p[0][3], p[1][3], p[2][3], p[3][3]);
        const glm::vec4 row0(p[0][0], p[1][0], p[2][0], p[3][0]);
        const glm::vec4 row1(p[0][1], p[1][1], p[2][1], p[3][1]);
        side_planes_[0] = row3 + row0; // left
        side_planes_[1] = row3 - row0; // right
        side_planes_[2] = row3 + row1; // bottom
        side_planes_[3] = row3 - row1; // top
        for (glm::vec4& pl : side_planes_) {
            const float len = glm::length(glm::vec3(pl));
            if (len > 1e-8f) {
                pl /= len;
            }
        }
    }

    // ---- pass 1: each record's conservative froxel-index box ---------------
    //
    // A pure per-record function with disjoint writes, so the fan-out needs no
    // partials and no merge (docs/multithreading_plan.md §4.3's per-element
    // form). The RELEVANCE RADIUS is where the localization actually happens:
    // `power / d^2 >= threshold` inverts to `d <= sqrt(power / threshold)`, the
    // derived analogue of a point light's authored `radius`, capped at the
    // localization horizon.
    ranges_.resize(records.size());
    range_active_.assign(records.size(), 0u);
    core::jobs::parallel_for(
        js, 0, records.size(),
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                const EmissiveTriangle& r = records[i];
                if (!(r.power > 0.0f) || !(r.area > 0.0f)) {
                    continue;
                }
                const glm::vec3 centroid = r.v0 + (r.e1 + r.e2) * (1.0f / 3.0f);
                const glm::vec3 view_c = glm::vec3(params.view * glm::vec4(centroid, 1.0f));
                const float radius = std::min(std::sqrt(r.power / k_emissive_froxel_min_contribution),
                                              k_emissive_froxel_max_radius);
                if (!std::isfinite(radius) || radius <= 0.0f) {
                    continue;
                }
                const float depth_min = -view_c.z - radius;
                const float depth_max = -view_c.z + radius;
                if (depth_max < near_z || depth_min > far_z) {
                    continue;
                }
                bool outside = false;
                for (const glm::vec4& pl : side_planes_) {
                    if (glm::dot(glm::vec3(pl), view_c) + pl.w < -radius) {
                        outside = true;
                        break;
                    }
                }
                if (outside) {
                    continue;
                }

                const uint32_t z0 = lighting::cluster_z_slice(depth_min, params);
                const uint32_t z1 = lighting::cluster_z_slice(depth_max, params);

                uint32_t x0 = 0, x1 = grid_x - 1;
                uint32_t y0 = 0, y1 = grid_y - 1;
                if (depth_min >= near_z) {
                    const glm::vec3 bb_min = view_c - glm::vec3(radius);
                    const glm::vec3 bb_max = view_c + glm::vec3(radius);
                    glm::vec2 ndc_min(std::numeric_limits<float>::max());
                    glm::vec2 ndc_max(std::numeric_limits<float>::lowest());
                    bool projected = true;
                    for (int corner = 0; corner < 8; ++corner) {
                        const glm::vec4 clip =
                            projection_ * glm::vec4((corner & 1) ? bb_max.x : bb_min.x,
                                                    (corner & 2) ? bb_max.y : bb_min.y,
                                                    (corner & 4) ? bb_max.z : bb_min.z, 1.0f);
                        if (clip.w <= 1e-6f) {
                            projected = false;
                            break;
                        }
                        const glm::vec2 ndc = glm::vec2(clip) / clip.w;
                        ndc_min = glm::min(ndc_min, ndc);
                        ndc_max = glm::max(ndc_max, ndc);
                    }
                    if (projected) {
                        if (ndc_max.x < -1.0f || ndc_min.x > 1.0f || ndc_max.y < -1.0f ||
                            ndc_min.y > 1.0f) {
                            continue; // fully outside the frustum sideways
                        }
                        x0 = tile_of(ndc_min.x, grid_x);
                        x1 = tile_of(ndc_max.x, grid_x);
                        y0 = tile_of(ndc_min.y, grid_y);
                        y1 = tile_of(ndc_max.y, grid_y);
                    }
                }
                RecordRange& rr = ranges_[i];
                rr.view_center = view_c;
                rr.area = r.area;
                rr.power = r.power;
                rr.record = static_cast<uint32_t>(i);
                rr.x0 = static_cast<uint8_t>(x0);
                rr.x1 = static_cast<uint8_t>(x1);
                rr.y0 = static_cast<uint8_t>(y0);
                rr.y1 = static_cast<uint8_t>(y1);
                rr.z0 = static_cast<uint8_t>(z0);
                rr.z1 = static_cast<uint8_t>(z1);
                rr.min_c = static_cast<uint16_t>((z0 * grid_y + y0) * grid_x + x0);
                rr.max_c = static_cast<uint16_t>((z1 * grid_y + y1) * grid_x + x1);
                range_active_[i] = 1u;
            }
        },
        RANGE_GRAIN);

    // COMPACT, serially and in ascending record order — which is both the
    // cheapest correct thing (the flag array is one byte per record, so the scan
    // touches 8 KB where the ranges are 288 KB) and the thing that makes the
    // binning below deterministic: every chunk then walks the SAME sequence.
    active_.clear();
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (range_active_[i] != 0u) {
            active_.push_back(ranges_[i]);
        }
    }
    stats_.records_culled = static_cast<uint32_t>(records.size() - active_.size());

    // ---- pass 2: bin, chunked by FROXEL index -----------------------------
    //
    // Each chunk is the sole writer of its froxels' bins AND of their blocks in
    // the flat table (the stride is fixed, so no prefix sum forces a serial
    // pack). Records are walked in ascending index and the top-k insertion
    // replaces only on a STRICTLY greater score, so a tie keeps the lower record
    // index: bit-identical at any worker count, any grain (§6.2).
    const std::size_t bin_grain = static_cast<std::size_t>(grid_x) * grid_y * BIN_SLICE_CHUNKS;
    const std::size_t chunk_count = (froxel_count_ + bin_grain - 1) / bin_grain;
    chunk_dropped_.assign(chunk_count, 0u);
    chunk_filled_.assign(chunk_count, 0u);
    chunk_entries_.assign(chunk_count, 0u);

    auto bin_chunk = [&](std::size_t chunk_begin, std::size_t chunk_end) {
        const std::size_t slot = chunk_begin / bin_grain;
        uint32_t dropped_local = 0;
        uint32_t filled_local = 0;
        uint32_t entries_local = 0;

        // Per-froxel occupancy for this chunk's range only.
        for (std::size_t c = chunk_begin; c < chunk_end; ++c) {
            table_[c * k_emissive_froxel_stride] = 0u;
        }

        for (const RecordRange& rr : active_) {
            if (rr.max_c < chunk_begin || rr.min_c >= chunk_end) {
                continue;
            }
            bool past_chunk = false;
            for (uint32_t z = rr.z0; z <= rr.z1 && !past_chunk; ++z) {
                for (uint32_t y = rr.y0; y <= rr.y1; ++y) {
                    const std::size_t row = (static_cast<std::size_t>(z) * grid_y + y) * grid_x;
                    if (row + rr.x0 >= chunk_end) {
                        past_chunk = true; // rows only grow with (z, y)
                        break;
                    }
                    if (row + rr.x1 < chunk_begin) {
                        continue;
                    }
                    const std::size_t c_lo = std::max<std::size_t>(row + rr.x0, chunk_begin);
                    const std::size_t c_hi = std::min<std::size_t>(row + rr.x1, chunk_end - 1);
                    for (std::size_t c = c_lo; c <= c_hi; ++c) {
                        const lighting::ClusterBounds& cb = bounds[c];
                        const glm::vec3 centre = (cb.view_min + cb.view_max) * 0.5f;
                        const glm::vec3 d = centre - rr.view_center;
                        const float d2 = glm::dot(d, d);
                        // The same expression emissive_target() computes with
                        // both cosines at 1, evaluated AT THE FROXEL CENTRE — a
                        // sample of this record's p̂ inside the froxel on the
                        // analytic path's scale, NOT a bound on it. A point at
                        // the near face of a froxel of half-extent h is
                        // under-scored by up to (D / (D - h))^2, so a record
                        // can miss the list of the froxel that CONTAINS it once
                        // that froxel is deep enough (measured on the default
                        // 0.1/4000 grid: a power-4 record at ~182 units, a
                        // power-80 one at ~1064). At emissive_froxel_share = 1
                        // a nonempty list is the whole domain, so such a record
                        // is then reachable only where the list is empty.
                        //
                        // Scoring from the froxel's NEAREST POINT instead is the
                        // obvious repair and is a measured behaviour change, not
                        // a correctness one: it is strictly more permissive, and
                        // flipping a far froxel from empty to nonempty REMOVES
                        // its global fallback, which is what still carries the
                        // far-field fill. It does not belong in a comment fix.
                        const float score = rr.power / std::max(d2, rr.area);
                        if (!(score >= k_emissive_froxel_min_contribution)) {
                            continue;
                        }

                        const std::size_t bin = c * k_emissive_froxel_capacity;
                        uint32_t& count = table_[c * k_emissive_froxel_stride];
                        if (count < k_emissive_froxel_capacity) {
                            bin_index_[bin + count] = rr.record;
                            bin_score_[bin + count] = score;
                            ++count;
                            continue;
                        }
                        // Full: displace the weakest entry, and only on a
                        // STRICTLY greater score. The first minimum found
                        // scanning ascending is the one displaced, which is what
                        // makes the survivor set a function of the record order
                        // alone.
                        uint32_t weakest = 0;
                        float weakest_score = bin_score_[bin];
                        for (uint32_t k = 1; k < k_emissive_froxel_capacity; ++k) {
                            if (bin_score_[bin + k] < weakest_score) {
                                weakest_score = bin_score_[bin + k];
                                weakest = k;
                            }
                        }
                        ++dropped_local; // one record loses this froxel either way
                        if (score > weakest_score) {
                            bin_index_[bin + weakest] = rr.record;
                            bin_score_[bin + weakest] = score;
                        }
                    }
                }
            }
        }

        // Pack this chunk's froxels, ASCENDING BY RECORD INDEX. The order is
        // not cosmetic: the shader's membership scan (`rtEmissiveFroxelHolds`,
        // the `q_g` half of the mixture pdf) early-outs on it. Ties in the
        // top-k selection above already broke on the lower record index, so the
        // packed set is a pure function of the record order and identical at any
        // worker count.
        for (std::size_t c = chunk_begin; c < chunk_end; ++c) {
            const std::size_t base = c * k_emissive_froxel_stride;
            const uint32_t count = table_[base];
            if (count == 0) {
                continue;
            }
            ++filled_local;
            entries_local += count;
            const std::size_t bin = c * k_emissive_froxel_capacity;
            for (uint32_t k = 1; k < count; ++k) { // insertion sort, k <= capacity
                const uint32_t v = bin_index_[bin + k];
                uint32_t j = k;
                while (j > 0 && bin_index_[bin + j - 1] > v) {
                    bin_index_[bin + j] = bin_index_[bin + j - 1];
                    --j;
                }
                bin_index_[bin + j] = v;
            }
            for (uint32_t k = 0; k < count; ++k) {
                table_[base + 1 + k] = bin_index_[bin + k];
            }
        }

        // One write per CHUNK to chunk-indexed slots — never inside the per-item
        // loop (§7: no shared cache-line writes per item).
        chunk_dropped_[slot] = dropped_local;
        chunk_filled_[slot] = filled_local;
        chunk_entries_[slot] = entries_local;
    };

    if (js.is_initialized() && active_.size() >= BIN_MIN_PARALLEL_RECORDS) {
        core::jobs::parallel_for(js, 0, froxel_count_, bin_chunk, bin_grain);
    } else {
        bin_chunk(0, froxel_count_);
    }

    // Merge partials single-threaded in chunk-index order (§6.2).
    for (std::size_t k = 0; k < chunk_count; ++k) {
        stats_.entries_dropped += chunk_dropped_[k];
        stats_.froxels_filled += chunk_filled_[k];
        stats_.entries += chunk_entries_[k];
    }

    if (stats_.entries_dropped != 0 && !warned_dropped_) {
        warned_dropped_ = true;
        BRACKEN_LOG_DEBUG(LOG_TAG,
                          "EmissiveClusterGrid: {} froxel insertion(s) over the per-froxel cap of {}; "
                          "those records stay reachable through the global half of the mixture. Logged "
                          "once; RtStats::emissive_froxel_dropped tracks it per frame.",
                          stats_.entries_dropped, k_emissive_froxel_capacity);
    }

    // The gate's memo, recorded only on the path that actually produced a
    // table. Every early return above leaves it as it was and `live_` false,
    // so the next call rebuilds.
    last_records_ = records.data();
    last_record_count_ = records.size();
    last_generation_ = record_generation;
    last_view_ = params.view;
    last_z_planes_ = params.zPlanes;
    last_grid_dim_ = params.gridDim;

    live_ = true;
    dirty_ = true;
}

void EmissiveClusterGrid::upload(SDL_GPUCommandBuffer* cmd) {
    BRACKEN_PROFILE_ZONE_COLOR("EmissiveClusterGrid::upload", 0x27AE60);
    if (device_ == nullptr || cmd == nullptr || !dirty_ || table_.empty()) {
        return;
    }
    const std::size_t bytes = table_.size() * sizeof(uint32_t);
    if (!buffer_ || buffer_entries_ != table_.size()) {
        // ComputeStorage, not Storage: the trace reads this through a READ-WRITE
        // slot (its eight read-only ones were spent before R6 existed, let alone
        // R8), and SDL3 refuses a read-write binding on a buffer without
        // COMPUTE_STORAGE_WRITE. Nothing ever writes it.
        buffer_ = gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::ComputeStorage, bytes);
        buffer_entries_ = buffer_ ? table_.size() : 0;
        if (!buffer_) {
            BRACKEN_LOG_ERROR(LOG_TAG, "EmissiveClusterGrid: could not allocate the {} KB froxel table",
                              bytes / 1024);
            live_ = false;
            return;
        }
    }
    buffer_->update(cmd, table_.data(), bytes);
    dirty_ = false;
}

SDL_GPUBuffer* EmissiveClusterGrid::buffer() const {
    return buffer_ ? buffer_->handle() : nullptr;
}

} // namespace bracken::render::rt
