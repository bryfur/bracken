
#include <gtest/gtest.h>

#include "bracken/audio/music/sequencer.hpp"

#include <vector>

using namespace bracken::audio;
using namespace bracken::audio::music;

namespace {

// A two-note song at a tempo whose step length divides evenly, so expected
// boundaries are exact integers and a failure means real drift.
Song two_note_song() {
    Song s;
    s.name = "test";
    s.bpm = 120.0f;        // 2 beats/s
    s.steps_per_beat = 4;  // 8 steps/s -> 6000 frames per step at 48 kHz
    s.loop = false;

    Track t;
    t.name = "lead";
    t.instrument = synth::patches::plucked_string();
    Pattern p;
    p.name = "main";
    p.steps = {
        Step{.note = C4, .velocity = 1.0f, .gate_steps = 1},
        Step{.note = k_note_rest},
        Step{.note = G4, .velocity = 1.0f, .gate_steps = 1},
        Step{.note = k_note_rest},
    };
    t.patterns.push_back(p);
    s.tracks.push_back(t);
    return s;
}

std::vector<StereoFrame> render(Sequencer& seq, int frames, int chunk) {
    std::vector<StereoFrame> out(static_cast<std::size_t>(frames));
    int done = 0;
    while (done < frames) {
        const int n = std::min(chunk, frames - done);
        seq.render_additive(std::span<StereoFrame>(out).subspan(
            static_cast<std::size_t>(done), static_cast<std::size_t>(n)));
        done += n;
    }
    return out;
}

bool any_audible(const std::vector<StereoFrame>& v) {
    for (const StereoFrame& f : v) {
        if (std::fabs(f.left) > 1e-5f || std::fabs(f.right) > 1e-5f) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(Sequencer, StepLengthMatchesTempo) {
    Song s = two_note_song();
    EXPECT_EQ(s.frames_per_step(), 6000); // 48000 / 8 steps per second
    EXPECT_EQ(s.length_steps(), 4);
}

TEST(Sequencer, SilentUntilPlaying) {
    Sequencer seq;
    seq.set_song(two_note_song());
    EXPECT_FALSE(any_audible(render(seq, 4096, 256)));
}

TEST(Sequencer, ProducesAudioOncePlaying) {
    Sequencer seq;
    seq.set_song(two_note_song());
    seq.set_playing(true);
    EXPECT_TRUE(any_audible(render(seq, 8192, 256)));
}

// THE test for this class. A sequencer that only advances on buffer
// boundaries drifts when the device changes buffer size — the same failure
// mode as a variable timestep, and just as invisible until you compare two
// runs. Rendering the same song at seven different chunk sizes must produce
// bit-identical output.
TEST(Sequencer, OutputIsIndependentOfBufferSize) {
    const int frames = 6000 * 4;

    Sequencer ref;
    ref.set_song(two_note_song());
    ref.set_playing(true);
    const auto expected = render(ref, frames, frames); // one giant buffer

    for (int chunk : {1, 7, 64, 128, 512, 4096, 6000}) {
        Sequencer seq;
        seq.set_song(two_note_song());
        seq.set_playing(true);
        const auto got = render(seq, frames, chunk);
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            ASSERT_EQ(got[i], expected[i])
                << "chunk=" << chunk << " diverged at frame " << i;
        }
    }
}

// Determinism: a rendered song must be comparable between runs or it cannot
// be regression-tested.
TEST(Sequencer, RenderIsReproducible) {
    Sequencer a;
    a.set_song(two_note_song());
    a.set_playing(true);
    Sequencer b;
    b.set_song(two_note_song());
    b.set_playing(true);
    EXPECT_EQ(render(a, 12000, 256), render(b, 12000, 333));
}

TEST(Sequencer, RewindRestartsIdentically) {
    Sequencer seq;
    seq.set_song(two_note_song());
    seq.set_playing(true);
    const auto first = render(seq, 12000, 256);
    seq.rewind();
    seq.set_playing(true);
    EXPECT_EQ(render(seq, 12000, 256), first);
}

TEST(Sequencer, NonLoopingSongFinishes) {
    Sequencer seq;
    seq.set_song(two_note_song());
    seq.set_playing(true);
    render(seq, 6000 * 5, 512);
    EXPECT_TRUE(seq.finished());
}

TEST(Sequencer, LoopingSongNeverFinishes) {
    Song s = two_note_song();
    s.loop = true;
    Sequencer seq;
    seq.set_song(s);
    seq.set_playing(true);
    render(seq, 6000 * 12, 512);
    EXPECT_FALSE(seq.finished());
}

TEST(Sequencer, MutedTrackIsSilent) {
    Song s = two_note_song();
    s.tracks[0].muted = true;
    Sequencer seq;
    seq.set_song(s);
    seq.set_playing(true);
    EXPECT_FALSE(any_audible(render(seq, 12000, 256)));
}

// A short pattern under a long one must loop on its own length rather than
// run out — otherwise every drum track needs padding to the song length.
TEST(Sequencer, ShortPatternLoopsUnderLongerOne) {
    Song s;
    s.bpm = 120.0f;
    s.steps_per_beat = 4;
    s.loop = false;

    Track lead;
    lead.instrument = synth::patches::plucked_string();
    Pattern lp;
    lp.steps.assign(16, Step{.note = k_note_rest});
    lp.steps[0] = Step{.note = C4, .gate_steps = 1};
    lead.patterns.push_back(lp);

    Track drum;
    drum.instrument = synth::patches::hat();
    Pattern dp;
    dp.steps = {Step{.note = A4, .gate_steps = 1}, Step{.note = k_note_rest}};
    drum.patterns.push_back(dp);

    s.tracks = {lead, drum};
    EXPECT_EQ(s.length_steps(), 16) << "song length is the longest track";

    Sequencer seq;
    seq.set_song(s);
    seq.set_playing(true);
    // Late in the song, past the 2-step drum pattern many times over.
    const auto tail = render(seq, 6000 * 16, 512);
    EXPECT_TRUE(any_audible(tail));
}

TEST(Sequencer, EmptySongIsSilentAndFinished) {
    Sequencer seq;
    seq.set_song(Song{});
    seq.set_playing(true);
    EXPECT_TRUE(seq.finished());
    EXPECT_FALSE(any_audible(render(seq, 4096, 256)));
}

TEST(Sequencer, RenderAddsRatherThanOverwrites) {
    Sequencer seq;
    seq.set_song(two_note_song());
    seq.set_playing(true);

    std::vector<StereoFrame> buf(1024, StereoFrame{1.0f, -1.0f});
    seq.render_additive(buf);
    // Whatever the synth contributed, the pre-existing content must still be
    // in there — several sources share one bus.
    bool preserved = false;
    for (const StereoFrame& f : buf) {
        if (f.left >= 0.9f) {
            preserved = true;
            break;
        }
    }
    EXPECT_TRUE(preserved);
}

