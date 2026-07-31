#pragma once

#include "bracken/memory/arena.hpp"

#include <cstddef>

namespace bracken::memory {

class FrameAllocator {
public:
    FrameAllocator() = default;
    explicit FrameAllocator(std::size_t initial_chunk_size)
        : arenas_{Arena(initial_chunk_size), Arena(initial_chunk_size)} {}

    FrameAllocator(const FrameAllocator&) = delete;
    FrameAllocator& operator=(const FrameAllocator&) = delete;

    FrameAllocator(FrameAllocator&&) = default;
    FrameAllocator& operator=(FrameAllocator&&) = default;

    Arena& this_frame() noexcept { return arenas_[current_]; }
    Arena& last_frame() noexcept { return arenas_[1 - current_]; }
    const Arena& this_frame() const noexcept { return arenas_[current_]; }
    const Arena& last_frame() const noexcept { return arenas_[1 - current_]; }

    // Flip the active arena and reset the new "this frame" so old contents are recycled.
    //
    // Owner handoff (debug builds): the newly-active arena may still be bound
    // to whichever thread filled it two frames ago, so swap() force-rebinds it
    // to the calling thread before reset(); reset() then re-opens the binding,
    // making the next allocating thread the new owner (see Arena's owner-thread
    // enforcement). Both calls are no-ops in release builds.
    void swap() {
        current_ = 1 - current_;
        arenas_[current_].rebind_owner();
        arenas_[current_].reset();
    }

private:
    Arena arenas_[2]{};
    int current_ = 0;
};

} // namespace bracken::memory
