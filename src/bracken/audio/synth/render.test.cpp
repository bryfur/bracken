
#include <gtest/gtest.h>

#include "bracken/audio/synth/render.hpp"

#include <cmath>

using namespace bracken::audio;
using namespace bracken::audio::synth;

namespace {

float peak_of(const AudioClip& c) {
    float p = 0.0f;
    for (const StereoFrame& f : c.frames()) {
        p = std::max(p, std::max(std::fabs(f.left), std::fabs(f.right)));
    }
    return p;
}

} // namespace

TEST(RenderNote, ProducesAClipOfTheRequestedLength) {
    const AudioClip c = render_note(patches::plucked_string(), A4, 0.1f, 0.2f, "test");
    EXPECT_EQ(c.frame_count(),
              static_cast<std::size_t>(frames_from_seconds(0.1f) + frames_from_seconds(0.2f)));
    EXPECT_NEAR(c.duration_seconds(), 0.3f, 1e-3f);
    EXPECT_EQ(c.name(), "test");
}

TEST(RenderNote, IsAudible) {
    EXPECT_GT(peak_of(render_note(patches::plucked_string(), A4, 0.1f, 0.2f, "x")), 0.01f);
}

// A baked clip must be identical every run, or SFX would differ between
// launches and the smoke run would stop being reproducible.
TEST(RenderNote, IsDeterministic) {
    const AudioClip a = render_note(patches::hat(), A4, 0.02f, 0.1f, "a");
    const AudioClip b = render_note(patches::hat(), A4, 0.02f, 0.1f, "b");
    ASSERT_EQ(a.frame_count(), b.frame_count());
    for (std::size_t i = 0; i < a.frame_count(); ++i) {
        ASSERT_EQ(a.frames()[i], b.frames()[i]) << "frame " << i;
    }
}

// The tail exists so the release is captured. A clip cut off mid-release
// clicks on every playback, which is the most common defect in baked SFX.
TEST(RenderNote, TailCapturesTheReleaseSoTheClipEndsQuiet) {
    const AudioClip c = render_note(patches::plucked_string(), A4, 0.05f, 0.6f, "tail");
    ASSERT_GT(c.frame_count(), 512u);

    const auto& f = c.frames();
    float end_peak = 0.0f;
    for (std::size_t i = f.size() - 256; i < f.size(); ++i) {
        end_peak = std::max(end_peak, std::max(std::fabs(f[i].left), std::fabs(f[i].right)));
    }
    EXPECT_LT(end_peak, 0.01f) << "clip must fade out, not stop mid-release";
}

TEST(RenderNote, HigherNoteIsHigherPitched) {
    auto zero_crossings = [](const AudioClip& c) {
        int n = 0;
        const auto& f = c.frames();
        for (std::size_t i = 1; i < f.size(); ++i) {
            if (f[i - 1].left < 0.0f && f[i].left >= 0.0f) {
                ++n;
            }
        }
        return n;
    };
    Instrument i = patches::sine_bass();
    i.filter = dsp::FilterMode::Off; // isolate pitch from the filter sweep
    EXPECT_GT(zero_crossings(render_note(i, A5, 0.2f, 0.05f, "hi")),
              zero_crossings(render_note(i, A3, 0.2f, 0.05f, "lo")));
}

TEST(RenderNote, ZeroTailStillProducesSamples) {
    const AudioClip c = render_note(patches::hat(), A4, 0.05f, 0.0f, "notail");
    EXPECT_EQ(c.frame_count(), static_cast<std::size_t>(frames_from_seconds(0.05f)));
}

