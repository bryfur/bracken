#include <gtest/gtest.h>

#include "bracken/core/jobs/mpsc_queue.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

using namespace bracken::core::jobs;

namespace {

struct TestNode {
    std::atomic<TestNode*> next{nullptr}; // required by MpscQueue<TestNode>
    int value = 0;
};

} // namespace

TEST(MpscQueueTest, PopOnEmptyReturnsNull) {
    MpscQueue<TestNode> q;
    EXPECT_EQ(q.pop(), nullptr);
    EXPECT_EQ(q.pop(), nullptr); // stable across repeated polls
}

TEST(MpscQueueTest, SingleProducerIsFifo) {
    MpscQueue<TestNode> q;
    constexpr int N = 100;
    std::vector<std::unique_ptr<TestNode>> nodes;
    nodes.reserve(N);
    for (int i = 0; i < N; ++i) {
        nodes.push_back(std::make_unique<TestNode>());
        nodes.back()->value = i;
        q.push(nodes.back().get());
    }
    for (int i = 0; i < N; ++i) {
        TestNode* n = q.pop();
        ASSERT_NE(n, nullptr);
        EXPECT_EQ(n->value, i);
    }
    EXPECT_EQ(q.pop(), nullptr);
}

// Exercises the stub-cycling path (queue repeatedly transitions through
// empty, so the stub is re-pushed and skipped over and over) and node reuse.
TEST(MpscQueueTest, PushPopCyclesReuseNodeThroughStub) {
    MpscQueue<TestNode> q;
    TestNode n;
    for (int i = 0; i < 10'000; ++i) {
        n.value = i;
        q.push(&n);
        TestNode* out = q.pop();
        ASSERT_EQ(out, &n);
        ASSERT_EQ(out->value, i);
        ASSERT_EQ(q.pop(), nullptr);
    }
}

// Race-shaped stress: 4 producers hammer the queue while a single consumer
// drains it. The consumer must tolerate transient nullptr pops (producer
// mid-push) and still receive every node exactly once, in per-producer FIFO
// order.
TEST(MpscQueueStressTest, MultiProducerSingleConsumerConservesAndOrders) {
    constexpr int PRODUCERS = 4;
    constexpr int PER_PRODUCER = 50'000;
    constexpr int TOTAL = PRODUCERS * PER_PRODUCER;

    MpscQueue<TestNode> q;

    // Stable node storage per producer; value encodes (producer, sequence).
    std::vector<std::unique_ptr<TestNode[]>> storage;
    storage.reserve(PRODUCERS);
    for (int p = 0; p < PRODUCERS; ++p) {
        storage.push_back(std::make_unique<TestNode[]>(PER_PRODUCER));
        for (int i = 0; i < PER_PRODUCER; ++i) {
            storage.back()[i].value = p * PER_PRODUCER + i;
        }
    }

    std::vector<std::thread> producers;
    producers.reserve(PRODUCERS);
    for (int p = 0; p < PRODUCERS; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < PER_PRODUCER; ++i) {
                q.push(&storage[p][i]);
            }
        });
    }

    int received = 0;
    int last_seq[PRODUCERS];
    for (int& s : last_seq) s = -1;
    std::vector<int> times_seen(TOTAL, 0);

    while (received < TOTAL) {
        TestNode* n = q.pop();
        if (n == nullptr) {
            std::this_thread::yield(); // empty or producer mid-push; retry
            continue;
        }
        const int producer = n->value / PER_PRODUCER;
        const int seq = n->value % PER_PRODUCER;
        ASSERT_GE(producer, 0);
        ASSERT_LT(producer, PRODUCERS);
        ASSERT_GT(seq, last_seq[producer]) << "per-producer FIFO order violated";
        last_seq[producer] = seq;
        ++times_seen[n->value];
        ++received;
    }
    for (auto& t : producers) t.join();

    EXPECT_EQ(q.pop(), nullptr);
    EXPECT_EQ(received, TOTAL);
    for (int i = 0; i < TOTAL; ++i) {
        ASSERT_EQ(times_seen[i], 1) << "node " << i << " delivered " << times_seen[i] << " times";
    }
}
