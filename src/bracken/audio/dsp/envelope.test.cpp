#if BRACKEN_SYNTH

#include <gtest/gtest.h>

#include "bracken/audio/dsp/envelope.hpp"

using namespace bracken::audio;
using namespace bracken::audio::dsp;

namespace {

Envelope::Settings quick() {
    return {.attack_s = 0.01f, .decay_s = 0.01f, .sustain = 0.5f, .release_s = 0.01f};
}

float advance(Envelope& e, int frames) {
    float last = 0.0f;
    for (int i = 0; i < frames; ++i) {
        last = e.next();
    }
    return last;
}

} // namespace

TEST(Envelope, StartsIdleAndSilent) {
    Envelope e;
    e.configure(quick());
    EXPECT_TRUE(e.finished());
    EXPECT_FLOAT_EQ(e.next(), 0.0f);
}

TEST(Envelope, RunsAttackDecayToSustain) {
    Envelope e;
    e.configure(quick());
    e.note_on();

    const int attack = frames_from_seconds(0.01f);
    const int decay = frames_from_seconds(0.01f);

    EXPECT_EQ(e.stage(), Envelope::Stage::Attack);
    advance(e, attack);
    EXPECT_EQ(e.stage(), Envelope::Stage::Decay);
    advance(e, decay);
    EXPECT_EQ(e.stage(), Envelope::Stage::Sustain);
    EXPECT_NEAR(advance(e, 100), 0.5f, 1e-5f);
}

TEST(Envelope, SustainHoldsIndefinitely) {
    Envelope e;
    e.configure(quick());
    e.note_on();
    advance(e, frames_from_seconds(0.02f) + 10);
    EXPECT_NEAR(advance(e, k_sample_rate), 0.5f, 1e-5f) << "sustain must not decay on its own";
    EXPECT_FALSE(e.finished());
}

TEST(Envelope, ReleaseReachesSilenceAndFinishes) {
    Envelope e;
    e.configure(quick());
    e.note_on();
    advance(e, frames_from_seconds(0.02f) + 10);
    e.note_off();
    advance(e, frames_from_seconds(0.01f));
    EXPECT_TRUE(e.finished());
    EXPECT_FLOAT_EQ(e.level(), 0.0f);
}

// The reason release starts from the current level. A note released during
// its attack must ramp DOWN from wherever it got to — jumping up to sustain
// first is an audible click, and staccato notes hit this constantly.
TEST(Envelope, ReleaseDuringAttackDoesNotJumpUp) {
    Envelope e;
    e.configure({.attack_s = 1.0f, .decay_s = 0.01f, .sustain = 1.0f, .release_s = 0.05f});
    e.note_on();

    const float level_at_release = advance(e, frames_from_seconds(0.05f));
    ASSERT_LT(level_at_release, 0.2f) << "should still be early in a 1 s attack";

    e.note_off();
    const float first_release_sample = e.next();
    EXPECT_LE(first_release_sample, level_at_release + 1e-4f)
        << "release must not step the gain upward";
}

TEST(Envelope, ZeroLengthStagesStillRampAndDoNotDivideByZero) {
    Envelope e;
    e.configure({.attack_s = 0.0f, .decay_s = 0.0f, .sustain = 0.5f, .release_s = 0.0f});
    e.note_on();
    for (int i = 0; i < 16; ++i) {
        const float v = e.next();
        EXPECT_FALSE(std::isnan(v));
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
    e.note_off();
    advance(e, 16);
    EXPECT_TRUE(e.finished());
}

TEST(Envelope, HardResetSilencesImmediately) {
    Envelope e;
    e.configure(quick());
    e.note_on();
    advance(e, frames_from_seconds(0.02f));
    e.hard_reset();
    EXPECT_TRUE(e.finished());
    EXPECT_FLOAT_EQ(e.level(), 0.0f);
}

TEST(Envelope, NoteOffOnIdleIsIgnored) {
    Envelope e;
    e.configure(quick());
    e.note_off();
    EXPECT_TRUE(e.finished());
    EXPECT_EQ(e.stage(), Envelope::Stage::Idle);
}

#endif // BRACKEN_SYNTH
