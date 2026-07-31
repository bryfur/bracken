#include <gtest/gtest.h>

#include "bracken/core/jobs/chase_lev_deque.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

using namespace bracken::core::jobs;

namespace {

// Every blocking drain loop in this file is bounded by this deadline so a
// lost element fails fast with a missing-item count instead of hanging until
// the ctest timeout.
[[nodiscard]] std::chrono::steady_clock::time_point drain_deadline() {
    return std::chrono::steady_clock::now() + std::chrono::seconds(30);
}

} // namespace

TEST(ChaseLevDequeTest, PopIsLifoForOwner) {
    ChaseLevDeque<int> d(4);
    int a = 1, b = 2, c = 3;
    d.push(&a);
    d.push(&b);
    d.push(&c);
    EXPECT_EQ(d.pop(), &c);
    EXPECT_EQ(d.pop(), &b);
    EXPECT_EQ(d.pop(), &a);
    EXPECT_EQ(d.pop(), nullptr);
    EXPECT_TRUE(d.empty());
}

TEST(ChaseLevDequeTest, StealIsFifoFromTop) {
    ChaseLevDeque<int> d(4);
    int a = 1, b = 2, c = 3;
    d.push(&a);
    d.push(&b);
    d.push(&c);
    EXPECT_EQ(d.steal(), &a);
    EXPECT_EQ(d.steal(), &b);
    EXPECT_EQ(d.steal(), &c);
    EXPECT_EQ(d.steal(), nullptr);
}

TEST(ChaseLevDequeTest, MixedPopAndStealMeetInTheMiddle) {
    ChaseLevDeque<int> d(8);
    int v[4] = {0, 1, 2, 3};
    for (int& x : v) d.push(&x);
    EXPECT_EQ(d.steal(), &v[0]); // top
    EXPECT_EQ(d.pop(), &v[3]);   // bottom
    EXPECT_EQ(d.steal(), &v[1]);
    EXPECT_EQ(d.pop(), &v[2]);
    EXPECT_EQ(d.pop(), nullptr);
    EXPECT_EQ(d.steal(), nullptr);
}

TEST(ChaseLevDequeTest, EmptyAfterInterleavedFillAndDrain) {
    ChaseLevDeque<int> d(2);
    int x = 42;
    for (int round = 0; round < 1000; ++round) {
        d.push(&x);
        EXPECT_EQ(d.pop(), &x);
        EXPECT_EQ(d.pop(), nullptr);
    }
    EXPECT_TRUE(d.empty());
}

// Owner-side growth: pushing far past the initial capacity must preserve
// every element (the grow copy) and leave old arrays readable (retired, not
// freed) — all values drain back out exactly once.
TEST(ChaseLevDequeTest, GrowsPreservingContents) {
    constexpr std::size_t N = 5000;
    ChaseLevDeque<std::size_t> d(2);
    std::vector<std::size_t> values(N);
    for (std::size_t i = 0; i < N; ++i) {
        values[i] = i;
        d.push(&values[i]);
    }
    EXPECT_GE(d.capacity(), N);

    std::vector<bool> seen(N, false);
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t* p = d.pop();
        ASSERT_NE(p, nullptr);
        ASSERT_FALSE(seen[*p]);
        seen[*p] = true;
    }
    EXPECT_EQ(d.pop(), nullptr);
}

// Race-shaped stress: one owner (push/pop, with growth from a tiny initial
// capacity) against 4 thieves. Sum conservation: every element is consumed
// exactly once, by exactly one side — a duplicated or lost element fails the
// per-item counts.
TEST(ChaseLevDequeStressTest, OwnerVersusFourThievesConservesAllItems) {
    constexpr std::size_t N = 100'000;
    constexpr int THIEVES = 4;

    ChaseLevDeque<std::size_t> deque(32); // tiny: force growth under contention
    std::vector<std::size_t> values(N);
    for (std::size_t i = 0; i < N; ++i) values[i] = i;

    std::unique_ptr<std::atomic<int>[]> seen(new std::atomic<int>[N]());
    std::atomic<std::size_t> consumed{0};
    std::atomic<bool> done{false};

    std::vector<std::thread> thieves;
    thieves.reserve(THIEVES);
    for (int t = 0; t < THIEVES; ++t) {
        thieves.emplace_back([&]() {
            while (!done.load(std::memory_order_acquire)) {
                if (std::size_t* p = deque.steal()) {
                    seen[*p].fetch_add(1, std::memory_order_relaxed);
                    consumed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Owner: push everything, popping a few in between to exercise the
    // bottom end (including the single-last-element CAS race), then drain.
    std::size_t pushed = 0;
    while (pushed < N) {
        const std::size_t batch = std::min<std::size_t>(97, N - pushed);
        for (std::size_t i = 0; i < batch; ++i) {
            deque.push(&values[pushed++]);
        }
        for (int i = 0; i < 5; ++i) {
            if (std::size_t* p = deque.pop()) {
                seen[*p].fetch_add(1, std::memory_order_relaxed);
                consumed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    const auto deadline = drain_deadline();
    while (consumed.load(std::memory_order_relaxed) < N) {
        if (std::size_t* p = deque.pop()) {
            seen[*p].fetch_add(1, std::memory_order_relaxed);
            consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::yield(); // thieves are finishing the tail
        }
        if (std::chrono::steady_clock::now() > deadline) {
            ADD_FAILURE() << "drain deadline hit: "
                          << (N - consumed.load(std::memory_order_relaxed))
                          << " items never consumed (lost element)";
            break;
        }
    }
    done.store(true, std::memory_order_release);
    for (auto& t : thieves) t.join();

    EXPECT_EQ(consumed.load(), N);
    for (std::size_t i = 0; i < N; ++i) {
        ASSERT_EQ(seen[i].load(), 1) << "element " << i << " consumed " << seen[i].load() << " times";
    }
    EXPECT_TRUE(deque.empty());
}

namespace {

// Deliberately non-atomic fields: the ONLY thing that publishes them to a
// thief is the release fence in push() paired with steal()'s acquire load of
// `bottom` (Fig. 1's publication edge). No other synchronization touches
// these bytes.
struct Payload {
    std::uint64_t value;
    std::uint64_t checksum; // must always equal payload_mix(value)
};

constexpr std::uint64_t payload_mix(std::uint64_t v) {
    return (v * 0x9E3779B97F4A7C15ull) ^ 0xD1B54A32D192ED03ull;
}

} // namespace

// Torn publication: makes the push()-release-fence <-> steal()-acquire-bottom
// edge load-bearing for real data, not just the slot pointer. The owner
// writes each payload's non-atomic fields immediately before push()ing its
// pointer; thieves dereference and verify checksum == mix(value). Payloads
// are heap-allocated and poisoned up front, so if the release fence in
// push() were deleted, a thief could legally observe the poison (stale
// fields) through the freshly published pointer — a checksum mismatch here,
// and a data race TSan can flag. The owner's pop() path reads the fields
// too (same-thread, trivially ordered) as the control.
TEST(ChaseLevDequeStressTest, NonAtomicPayloadFieldsPublishedByPushFence) {
    constexpr std::size_t N = 150'000;
    constexpr int THIEVES = 3;

    ChaseLevDeque<Payload> deque(64);

    // Heap-allocate every payload before the thieves start, poisoned so the
    // checksum relation does NOT hold until the owner writes real fields.
    std::vector<std::unique_ptr<Payload>> payloads(N);
    for (auto& p : payloads) {
        p = std::make_unique<Payload>();
        p->value = ~0ull;
        p->checksum = 0ull;
    }

    std::atomic<std::size_t> consumed{0};
    std::atomic<std::size_t> torn{0}; // payloads seen with a broken checksum
    std::atomic<bool> done{false};

    std::vector<std::thread> thieves;
    thieves.reserve(THIEVES);
    for (int t = 0; t < THIEVES; ++t) {
        thieves.emplace_back([&]() {
            while (!done.load(std::memory_order_acquire)) {
                if (Payload* p = deque.steal()) {
                    // Non-atomic reads — only the deque's fences order them.
                    if (p->checksum != payload_mix(p->value) || p->value >= N) {
                        torn.fetch_add(1, std::memory_order_relaxed);
                    }
                    consumed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Owner: write the fields INSIDE the loop, right before push() — the
    // fence in push() is the only publication of those writes.
    for (std::size_t i = 0; i < N; ++i) {
        Payload* p = payloads[i].get();
        p->value = i;
        p->checksum = payload_mix(i);
        deque.push(p);
        // Owner path: pop and read every few pushes.
        if ((i % 7) == 0) {
            if (Payload* q = deque.pop()) {
                if (q->checksum != payload_mix(q->value) || q->value >= N) {
                    torn.fetch_add(1, std::memory_order_relaxed);
                }
                consumed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    const auto deadline = drain_deadline();
    while (consumed.load(std::memory_order_relaxed) < N) {
        if (Payload* q = deque.pop()) {
            if (q->checksum != payload_mix(q->value) || q->value >= N) {
                torn.fetch_add(1, std::memory_order_relaxed);
            }
            consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::yield();
        }
        if (std::chrono::steady_clock::now() > deadline) {
            ADD_FAILURE() << "drain deadline hit: "
                          << (N - consumed.load(std::memory_order_relaxed))
                          << " payloads never consumed (lost element)";
            break;
        }
    }
    done.store(true, std::memory_order_release);
    for (auto& t : thieves) t.join();

    EXPECT_EQ(consumed.load(), N);
    EXPECT_EQ(torn.load(), 0u) << "payload fields observed before the push() fence published them";
    EXPECT_TRUE(deque.empty());
}

// Single-last-element duel: pins the one genuinely contended decision in the
// algorithm — the seq_cst CAS on `top` that pop() and steal() race for the
// final element. Each round the owner pushes exactly one item and
// immediately pop()s it while two persistent thieves spin steal()ing;
// exactly one side may win. A double-take (both CAS paths "succeed") drives
// the cumulative claim count past round+1; a lost element (nobody wins)
// stalls the round and trips the deadline with a missing count. Thieves stay
// alive across all rounds so the race window is hit at full temperature.
TEST(ChaseLevDequeStressTest, SingleElementDuelHasExactlyOneConsumerPerRound) {
    constexpr std::size_t ROUNDS = 100'000;
    constexpr int THIEVES = 2;

    ChaseLevDeque<int> deque(4);
    int item = 0;

    std::atomic<std::size_t> claims{0}; // cumulative across rounds — never reset
    std::atomic<bool> done{false};

    std::vector<std::thread> thieves;
    thieves.reserve(THIEVES);
    for (int t = 0; t < THIEVES; ++t) {
        thieves.emplace_back([&]() {
            while (!done.load(std::memory_order_acquire)) {
                if (deque.steal() != nullptr) {
                    claims.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    const auto deadline = drain_deadline();
    bool lost = false;
    for (std::size_t round = 0; round < ROUNDS && !lost; ++round) {
        deque.push(&item);
        if (deque.pop() != nullptr) {
            claims.fetch_add(1, std::memory_order_relaxed);
        }
        // Exactly one consumer must have claimed this round's item before the
        // next round starts. Undershoot forever == lost element (deadline);
        // any overshoot (here or by the final total below) == double-take.
        std::size_t c;
        while ((c = claims.load(std::memory_order_relaxed)) < round + 1) {
            if (std::chrono::steady_clock::now() > deadline) {
                ADD_FAILURE() << "round " << round
                              << ": item consumed by neither owner nor thief (lost element)";
                lost = true;
                break;
            }
        }
        if (!lost && c != round + 1) {
            ADD_FAILURE() << "round " << round << ": " << c - round
                          << " consumers claimed one item (double-take)";
            break;
        }
    }
    done.store(true, std::memory_order_release);
    for (auto& t : thieves) t.join();

    // Final total re-checks the whole run: any late-landing duplicate claim
    // (double-take whose second increment raced past a round check) shows up
    // here as claims > ROUNDS.
    EXPECT_EQ(claims.load(), ROUNDS);
    EXPECT_TRUE(deque.empty());
}

// Torn grow: forces grow_() to run while thieves are actively CAS-ing on
// `top`, from a deliberately tiny capacity. Thieves are throttled while the
// owner over-fills far past the initial capacity (many doublings), then
// released mid-push-storm so the remaining grows race live steal()s. Pins
// two invariants: capacity actually grew (the resize happened under test,
// not vacuously), and every element is consumed exactly once — a torn grow
// (thief reading a half-copied or already-recycled buffer) shows up as a
// duplicate or missing item in the per-element accounting.
TEST(ChaseLevDequeStressTest, GrowUnderContentionPreservesEveryItemOnce) {
    constexpr std::size_t N = 60'000;
    constexpr int THIEVES = 4;

    ChaseLevDeque<std::size_t> deque(2); // smallest legal capacity
    const std::size_t initial_capacity = deque.capacity();

    std::vector<std::size_t> values(N);
    for (std::size_t i = 0; i < N; ++i) values[i] = i;

    std::unique_ptr<std::atomic<int>[]> seen(new std::atomic<int>[N]());
    std::atomic<std::size_t> consumed{0};
    std::atomic<bool> go{false};   // thieves throttle on this
    std::atomic<bool> done{false};

    std::vector<std::thread> thieves;
    thieves.reserve(THIEVES);
    for (int t = 0; t < THIEVES; ++t) {
        thieves.emplace_back([&]() {
            while (!done.load(std::memory_order_acquire)) {
                if (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield(); // throttled: let the owner over-fill
                    continue;
                }
                if (std::size_t* p = deque.steal()) {
                    seen[*p].fetch_add(1, std::memory_order_relaxed);
                    consumed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Phase 1 (thieves throttled): over-fill past the initial capacity many
    // times — every doubling here happens with the queue near-full.
    std::size_t pushed = 0;
    while (pushed < N / 4) {
        deque.push(&values[pushed++]);
    }
    // Phase 2: release the thieves mid-push-storm — the remaining doublings
    // run grow_() concurrently with live steal() CASes on the old buffer.
    go.store(true, std::memory_order_release);
    while (pushed < N) {
        deque.push(&values[pushed++]);
    }

    const auto deadline = drain_deadline();
    while (consumed.load(std::memory_order_relaxed) < N) {
        if (std::size_t* p = deque.pop()) {
            seen[*p].fetch_add(1, std::memory_order_relaxed);
            consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::yield();
        }
        if (std::chrono::steady_clock::now() > deadline) {
            ADD_FAILURE() << "drain deadline hit: "
                          << (N - consumed.load(std::memory_order_relaxed))
                          << " items never consumed (lost element / torn grow)";
            break;
        }
    }
    done.store(true, std::memory_order_release);
    for (auto& t : thieves) t.join();

    EXPECT_GT(deque.capacity(), initial_capacity) << "grow path was never exercised";
    EXPECT_EQ(consumed.load(), N);
    for (std::size_t i = 0; i < N; ++i) {
        ASSERT_EQ(seen[i].load(), 1) << "element " << i << " consumed " << seen[i].load() << " times";
    }
    EXPECT_TRUE(deque.empty());
}
