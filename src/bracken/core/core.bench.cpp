#include <benchmark/benchmark.h>

#include "bracken/core/hash.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/memory/arena.hpp"
#include "bracken/memory/frame_allocator.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace bracken::core;
using namespace bracken::core::jobs;
using namespace bracken::memory;

namespace {

std::string make_string(std::size_t len) {
    std::string s(len, '\0');
    for (std::size_t i = 0; i < len; ++i) {
        s[i] = static_cast<char>('a' + (i % 26));
    }
    return s;
}

// Init the shared JobSystem exactly once across all benchmark runs. The
// singleton does support shutdown()+init() re-initialization (pinned by the
// InitAfterShutdownReinitializes test); the static guard simply avoids
// redundant init calls across benchmark repetitions and multiple grain Args.
void ensure_job_system() {
    static bool initialized = [] {
        if (!JobSystem::instance().is_initialized()) {
            JobSystem::instance().init();
        }
        return true;
    }();
    (void)initialized;
}

} // namespace

// ---------------------------------------------------------------------------
// Hashing — fnv1a / fnv1a64 over runtime strings (DoNotOptimize on the input
// keeps the constexpr hash from being folded or hoisted out of the loop).
// ---------------------------------------------------------------------------

static void BM_Fnv1a(benchmark::State& state) {
    const std::size_t len = static_cast<std::size_t>(state.range(0));
    const std::string s = make_string(len);
    std::string_view sv(s);
    for (auto _ : state) {
        benchmark::DoNotOptimize(sv);
        uint32_t h = fnv1a(sv);
        benchmark::DoNotOptimize(h);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(len));
}
BENCHMARK(BM_Fnv1a)->Arg(8)->Arg(64)->Arg(512);

static void BM_Fnv1a64(benchmark::State& state) {
    const std::size_t len = static_cast<std::size_t>(state.range(0));
    const std::string s = make_string(len);
    std::string_view sv(s);
    for (auto _ : state) {
        benchmark::DoNotOptimize(sv);
        uint64_t h = fnv1a64(sv);
        benchmark::DoNotOptimize(h);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(len));
}
BENCHMARK(BM_Fnv1a64)->Arg(8)->Arg(64)->Arg(512);

// ---------------------------------------------------------------------------
// Arena — bump allocation of mixed small sizes, reset amortized over a batch
// (mirrors per-frame scratch usage: many allocs, occasional wholesale reset).
// ---------------------------------------------------------------------------

static void BM_ArenaAllocate(benchmark::State& state) {
    Arena arena(1 << 20);
    constexpr std::size_t k_sizes[] = {16, 32, 48, 64, 96, 128, 192, 256};
    constexpr std::size_t k_size_count = sizeof(k_sizes) / sizeof(k_sizes[0]);
    constexpr std::size_t k_reset_interval = 1024;
    std::size_t i = 0;
    for (auto _ : state) {
        void* p = arena.allocate(k_sizes[i % k_size_count], 16);
        benchmark::DoNotOptimize(p);
        if (++i % k_reset_interval == 0) {
            arena.reset();
        }
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ArenaAllocate);

// ---------------------------------------------------------------------------
// FrameAllocator — one iteration = one frame: a batch of allocations from
// this_frame() followed by swap() (flip + reset of the recycled arena).
// ---------------------------------------------------------------------------

static void BM_FrameAllocatorAllocSwap(benchmark::State& state) {
    FrameAllocator frames(1 << 20);
    constexpr std::size_t k_allocs_per_frame = 256;
    for (auto _ : state) {
        Arena& arena = frames.this_frame();
        for (std::size_t i = 0; i < k_allocs_per_frame; ++i) {
            void* p = arena.allocate(64, 16);
            benchmark::DoNotOptimize(p);
        }
        frames.swap();
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * k_allocs_per_frame);
}
BENCHMARK(BM_FrameAllocatorAllocSwap);

// ---------------------------------------------------------------------------
// parallel_for vs serial — sum 1M floats. Explicit grain bypasses the
// small-workload guard; per-chunk local sums are merged via a relaxed
// atomic add so chunks stay independent.
// ---------------------------------------------------------------------------

constexpr std::size_t k_sum_count = 1'000'000;

static void BM_SerialSum(benchmark::State& state) {
    std::vector<float> data(k_sum_count);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 97) * 0.5f;
    }
    for (auto _ : state) {
        benchmark::DoNotOptimize(data.data());
        float sum = 0.0f;
        for (float v : data) {
            sum += v;
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(k_sum_count));
}
BENCHMARK(BM_SerialSum);

static void BM_ParallelForSum(benchmark::State& state) {
    ensure_job_system();
    const std::size_t grain = static_cast<std::size_t>(state.range(0));
    std::vector<float> data(k_sum_count);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 97) * 0.5f;
    }
    for (auto _ : state) {
        std::atomic<float> total{0.0f};
        parallel_for(
            0, data.size(),
            [&](std::size_t b, std::size_t e) {
                float local = 0.0f;
                for (std::size_t i = b; i < e; ++i) {
                    local += data[i];
                }
                total.fetch_add(local, std::memory_order_relaxed);
            },
            grain);
        float sum = total.load(std::memory_order_relaxed);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(k_sum_count));
}
BENCHMARK(BM_ParallelForSum)->Arg(1024)->Arg(16384);

// ---------------------------------------------------------------------------
// parallel_for DISPATCH cost — the thing Phase 10a is about. The chunk body
// is deliberately near-empty (one dependent load per chunk) so what is timed
// is submit + wake + steal + join, not the work. Reported per CHUNK
// (SetItemsProcessed counts chunks, so items/s inverts to s/chunk): the
// per-node epoch bump + notify is a per-chunk cost, so a fixed total with a
// varying chunk count is the wrong axis. Run at BRACKEN_WORKERS=4 and 8.
// ---------------------------------------------------------------------------

static void BM_ParallelForDispatch(benchmark::State& state) {
    ensure_job_system();
    const std::size_t chunks = static_cast<std::size_t>(state.range(0));
    // Explicit grain: bypasses the small-total guard so chunk count is
    // exactly `chunks` regardless of the pool size.
    constexpr std::size_t grain = 64;
    const std::size_t total = chunks * grain;
    std::vector<std::uint32_t> data(total, 1u);

    for (auto _ : state) {
        parallel_for(
            0, total,
            [&](std::size_t b, std::size_t e) {
                std::uint32_t v = data[b] + data[e - 1];
                benchmark::DoNotOptimize(v);
            },
            grain);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(chunks));
    state.counters["chunks"] = static_cast<double>(chunks);
}
BENCHMARK(BM_ParallelForDispatch)->Arg(32)->Arg(64)->Arg(256)->Arg(1024);

// The same dispatch against a PARKED pool. The loop above is back-to-back, so
// workers never get past their spin window and the epoch notify is nearly
// free; between engine frames (~9 ms) they are parked on the futex, and a
// wake there costs a syscall. Both regimes are real — the frame's FIRST
// fan-out hits a parked pool, the rest of the frame hits a hot one — so both
// are measured. The idle gap is excluded from the timing (PauseTiming), and
// the iteration count is fixed to keep the wall time sane.
static void BM_ParallelForDispatchParked(benchmark::State& state) {
    ensure_job_system();
    const std::size_t chunks = static_cast<std::size_t>(state.range(0));
    constexpr std::size_t grain = 64;
    const std::size_t total = chunks * grain;
    std::vector<std::uint32_t> data(total, 1u);

    for (auto _ : state) {
        state.PauseTiming();
        // Comfortably past k_spin_rounds (64 steal sweeps): every worker is
        // on the futex when the dispatch below starts.
        std::this_thread::sleep_for(std::chrono::microseconds(400));
        state.ResumeTiming();
        parallel_for(
            0, total,
            [&](std::size_t b, std::size_t e) {
                std::uint32_t v = data[b] + data[e - 1];
                benchmark::DoNotOptimize(v);
            },
            grain);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(chunks));
    state.counters["chunks"] = static_cast<double>(chunks);
}
BENCHMARK(BM_ParallelForDispatchParked)->Arg(32)->Arg(64)->Arg(256)->Iterations(300);
