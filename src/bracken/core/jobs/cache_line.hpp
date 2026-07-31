#pragma once

#include <cstddef>

namespace bracken::core::jobs {

// Cache-line padding constant for genuinely contended atomics/structures
// (deque top/bottom, TaskGroup counters, the JobSystem parking words).
//
// Deliberately a fixed 128 instead of std::hardware_destructive_interference_size:
// that constant is unavailable (or warns as ABI-unstable) on Apple clang, and
// Apple Silicon transfers/prefetches in 128-byte lines. 128 also covers the
// 64-byte lines of x86_64 — over-alignment is benign and only costs padding.
inline constexpr std::size_t k_cache_line = 128;

} // namespace bracken::core::jobs
