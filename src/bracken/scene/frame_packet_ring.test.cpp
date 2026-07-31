#include <gtest/gtest.h>

#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/frame_packet_ring.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

using namespace bracken::scene;

// ---------------------------------------------------------------------------
// Sequential lifecycle
// ---------------------------------------------------------------------------

TEST(PacketRing, ConstructsRetiredPackets) {
    PacketRing ring(2);
    EXPECT_EQ(ring.depth(), 2u);
    EXPECT_EQ(ring.packet(0).state(), FramePacket::State::Retired);
    EXPECT_EQ(ring.packet(1).state(), FramePacket::State::Retired);
    EXPECT_FALSE(ring.has_sealed());
    EXPECT_EQ(ring.try_acquire_sealed(), nullptr);
}

TEST(PacketRing, AcquireForFillAlternatesSlots) {
    PacketRing ring(2);
    FramePacket* seen[4] = {};
    for (int frame = 0; frame < 4; ++frame) {
        FramePacket& p = ring.acquire_for_fill();
        seen[frame] = &p;
        FrameBuilder fb(p);
        fb.seal();
        ring.retire(p); // sequential flow: retire straight from Sealed
    }
    EXPECT_NE(seen[0], seen[1]);
    EXPECT_EQ(seen[0], seen[2]);
    EXPECT_EQ(seen[1], seen[3]);
}

TEST(PacketRing, FullPublishAcquireRetireCycle) {
    PacketRing ring(2);
    for (uint64_t frame = 0; frame < 6; ++frame) {
        FramePacket& p = ring.acquire_for_fill();
        FrameBuilder fb(p);
        fb.draw_model(static_cast<bracken::ModelHandle>(frame + 1), glm::mat4(1.0f));
        fb.set_frame_timing(0.016f, static_cast<double>(frame) * 0.016, frame);
        fb.seal();
        ring.publish(p);
        EXPECT_TRUE(ring.has_sealed());

        FramePacket* c = ring.try_acquire_sealed();
        ASSERT_EQ(c, &p);
        EXPECT_EQ(c->state(), FramePacket::State::Consuming);
        EXPECT_FALSE(ring.has_sealed());
        EXPECT_EQ(ring.try_acquire_sealed(), nullptr); // mailbox drained

        ASSERT_EQ(c->model_commands().size(), 1u);
        EXPECT_EQ(c->model_commands()[0].model_handle, frame + 1);
        EXPECT_EQ(c->frame_index(), frame);

        ring.retire(*c);
        EXPECT_EQ(p.state(), FramePacket::State::Retired);
    }
}

// ---------------------------------------------------------------------------
// Reuse: retire resets cleanly, no cross-frame leakage, memory stabilizes
// ---------------------------------------------------------------------------

TEST(PacketRing, ReuseResetsContentAndSingularState) {
    PacketRing ring(2);

    // Frame 0: heavy content + non-default singular state into slot 0.
    FramePacket& p0 = ring.acquire_for_fill();
    {
        FrameBuilder fb(p0);
        for (int i = 0; i < 500; ++i) {
            fb.draw_model(1, glm::mat4(1.0f));
        }
        auto pal = fb.alloc_bone_palette(8);
        ASSERT_TRUE(pal.valid());
        for (auto& m : pal.matrices) {
            m = glm::mat4(1.0f);
        }
        fb.draw_skinned(2, glm::mat4(1.0f), pal.id);
        fb.ui().add_text("frame zero", 0, 0);
        fb.set_draw_skybox(false);
        fb.set_frame_timing(0.5f, 9.0, 1234);
        fb.seal();
    }
    ring.publish(p0);
    ring.retire(*ring.try_acquire_sealed());

    // Skip slot 1, come back around to slot 0 and verify a minimal frame
    // shows none of frame 0's content or state.
    FramePacket& p1 = ring.acquire_for_fill();
    {
        FrameBuilder fb(p1);
        fb.seal();
    }
    ring.retire(p1);

    FramePacket& p0_again = ring.acquire_for_fill();
    ASSERT_EQ(&p0_again, &p0);
    {
        FrameBuilder fb(p0_again);
        fb.draw_model(9, glm::mat4(2.0f));
        fb.seal();
    }
    ASSERT_EQ(p0_again.model_commands().size(), 1u);
    EXPECT_EQ(p0_again.model_commands()[0].model_handle, 9u);
    EXPECT_TRUE(p0_again.skinned_commands().empty());
    EXPECT_TRUE(p0_again.palettes().empty());
    EXPECT_TRUE(p0_again.ui_commands().empty());
    EXPECT_TRUE(p0_again.should_draw_skybox()); // toggle back to default
    EXPECT_EQ(p0_again.frame_index(), 0u);
    ring.retire(p0_again);
}

TEST(PacketRing, ReuseKeepsArenaCapacityStable) {
    PacketRing ring(2);
    std::size_t caps[8] = {};
    for (int frame = 0; frame < 8; ++frame) {
        FramePacket& p = ring.acquire_for_fill();
        FrameBuilder fb(p);
        for (int i = 0; i < 400; ++i) {
            fb.draw_model(1, glm::mat4(1.0f));
            fb.add_debug_line({0, 0, 0}, {1, 1, 1}, 0xFF00FF00);
        }
        fb.seal();
        caps[frame] = p.bytes_capacity();
        ring.retire(p);
    }
    // Identical per-frame load: after both slots warmed up, capacity must not
    // creep (steady-state frames allocate no new memory).
    EXPECT_EQ(caps[4], caps[6]);
    EXPECT_EQ(caps[5], caps[7]);
}

// ---------------------------------------------------------------------------
// Cross-thread SPSC handoff: single release-publish / acquire-take
// ---------------------------------------------------------------------------

TEST(PacketRing, CrossThreadPublishAcquireHandsOffEveryFrameInOrder) {
    constexpr uint64_t FRAMES = 500;
    PacketRing ring(2);

    std::atomic<int> mismatches{0};
    std::thread consumer([&ring, &mismatches] {
        uint64_t expected = 0;
        while (expected < FRAMES) {
            FramePacket* p = ring.try_acquire_sealed();
            if (p == nullptr) {
                std::this_thread::yield();
                continue;
            }
            if (p->state() != FramePacket::State::Consuming) {
                mismatches.fetch_add(1, std::memory_order_relaxed);
            }
            if (p->frame_index() != expected) {
                mismatches.fetch_add(1, std::memory_order_relaxed);
            }
            if (p->model_commands().size() != 1 ||
                p->model_commands()[0].motion_id != static_cast<uint32_t>(expected)) {
                mismatches.fetch_add(1, std::memory_order_relaxed);
            }
            ring.retire(*p);
            ++expected;
        }
    });

    for (uint64_t frame = 0; frame < FRAMES; ++frame) {
        // Backpressure: wait for the mailbox to drain and the next slot to be
        // retired (depth-2 pipeline discipline).
        while (ring.has_sealed() || ring.peek_fill().state() != FramePacket::State::Retired) {
            std::this_thread::yield();
        }
        FramePacket& p = ring.acquire_for_fill();
        FrameBuilder fb(p);
        fb.draw_model(1, glm::mat4(1.0f), {1, 1, 1, 1}, false, false, static_cast<uint32_t>(frame));
        fb.set_frame_timing(0.016f, static_cast<double>(frame), frame);
        fb.seal();
        ring.publish(p);
    }

    consumer.join();
    EXPECT_EQ(mismatches.load(), 0);
    // Everything delivered; the ring ends quiescent.
    EXPECT_FALSE(ring.has_sealed());
    EXPECT_EQ(ring.packet(0).state(), FramePacket::State::Retired);
    EXPECT_EQ(ring.packet(1).state(), FramePacket::State::Retired);
}

// ---------------------------------------------------------------------------
// Debug contract asserts. Suite name ends in "DeathTest" so gtest schedules
// these before threaded tests.
// ---------------------------------------------------------------------------

#ifndef NDEBUG

TEST(PacketRingDeathTest, PublishUnsealedPacketAsserts) {
    EXPECT_DEATH(
        {
            PacketRing ring(2);
            FramePacket& p = ring.acquire_for_fill(); // still Retired
            ring.publish(p);
        },
        "not Sealed");
}

TEST(PacketRingDeathTest, RetireRetiredPacketAsserts) {
    EXPECT_DEATH(
        {
            PacketRing ring(2);
            ring.retire(ring.packet(0)); // never filled
        },
        "not Consuming/Sealed");
}

TEST(PacketRingDeathTest, AcquireOverrunAsserts) {
    EXPECT_DEATH(
        {
            PacketRing ring(2);
            FrameBuilder fb0(ring.acquire_for_fill()); // slot 0 -> Filling
            FrameBuilder fb1(ring.acquire_for_fill()); // slot 1 -> Filling
            // Wraps to slot 0, which was never sealed/retired: an overrun.
            (void)ring.acquire_for_fill();
        },
        "pipeline overrun");
}

#endif // NDEBUG
