#include "bracken/gpu/gpu_debug.hpp"
#include "bracken/gpu/gpu_device.hpp"

namespace bracken::gpu {

GPUDebugGroup::GPUDebugGroup(SDL_GPUCommandBuffer* cb, const char* name) noexcept : cb_(cb) {
    if (cb_ && name) {
        SDL_PushGPUDebugGroup(cb_, name);
    } else {
        cb_ = nullptr;
    }
}

GPUDebugGroup::~GPUDebugGroup() noexcept {
    if (cb_) {
        SDL_PopGPUDebugGroup(cb_);
    }
}

bool GPUTimestampPool::init(GPUDevice& /*device*/, size_t pair_count) {
    capacity_ = pair_count;
    pass_names_.reserve(pair_count);
    supported_ = false;
    return true;
}

void GPUTimestampPool::shutdown() {
    pass_names_.clear();
    capacity_ = 0;
    supported_ = false;
}

int GPUTimestampPool::reserve_pair(const char* pass_name) {
    if (!pass_name) {
        return -1;
    }
    if (pass_names_.size() >= capacity_) {
        return -1;
    }
    const int idx = static_cast<int>(pass_names_.size());
    pass_names_.emplace_back(pass_name);
    timeline_.reserve(static_cast<std::size_t>(idx), pass_name);
    return idx;
}

void GPUTimestampPool::write_begin(SDL_GPUCommandBuffer* /*cb*/, int pair_idx) {
    // GPU-side: no-op until SDL3 exposes timestamp queries. CPU-side: this is
    // the start of the pass's record window.
    if (pair_idx >= 0) {
        timeline_.begin_pass(static_cast<std::size_t>(pair_idx));
    }
}

void GPUTimestampPool::write_end(SDL_GPUCommandBuffer* /*cb*/, int pair_idx) {
    if (pair_idx >= 0) {
        timeline_.end_pass(static_cast<std::size_t>(pair_idx));
    }
}

void GPUTimestampPool::read_results(std::vector<PassTiming>& out) {
    out.clear();
    out.reserve(pass_names_.size());
    for (std::size_t i = 0; i < pass_names_.size(); ++i) {
        PassTiming t;
        t.name = pass_names_[i]; // literal owned by the call site — view is safe
        if (timeline_.has_sample(i)) {
            t.cpu_ms = timeline_.pass_ms(i);
        }
        // t.gpu_ms stays nullopt while supported_ is false, which is always
        // today. When SDL grows a timestamp API, fill it here.
        out.push_back(t);
    }
}

void GPUTimestampPool::begin_frame() {
    pass_names_.clear();
    timeline_.begin_frame();
}

void GPUTimestampPool::end_frame(SDL_GPUCommandBuffer* /*cb*/) {
    // No resolve step until SDL3 exposes timestamp queries.
}

GPUTimerScope::GPUTimerScope(GPUTimestampPool& pool, SDL_GPUCommandBuffer* cb, const char* pass_name) noexcept
    : pool_(&pool),
      cb_(cb) {
    pair_idx_ = pool_->reserve_pair(pass_name);
    if (pair_idx_ >= 0) {
        pool_->write_begin(cb_, pair_idx_);
    }
}

GPUTimerScope::GPUTimerScope(GPUTimestampPool* pool, SDL_GPUCommandBuffer* cb,
                             const char* pass_name) noexcept
    : pool_(pool),
      cb_(cb) {
    if (pool_ == nullptr) {
        return; // pair_idx_ stays -1: the destructor is a no-op too
    }
    pair_idx_ = pool_->reserve_pair(pass_name);
    if (pair_idx_ >= 0) {
        pool_->write_begin(cb_, pair_idx_);
    }
}

GPUTimerScope::~GPUTimerScope() noexcept {
    if (pool_ && pair_idx_ >= 0) {
        pool_->write_end(cb_, pair_idx_);
    }
}

} // namespace bracken::gpu
