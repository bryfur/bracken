#if BRACKEN_SYNTH

#include <gtest/gtest.h>

#include "bracken/audio/dsp/oscillator.hpp"

#include <cmath>
#include <vector>

using namespace bracken::audio;
using namespace bracken::audio::dsp;

namespace {

std::vector<float> render(Oscillator& o, int frames) {
    std::vector<float> v(static_cast<std::size_t>(frames));
    for (float& s : v) {
        s = o.next();
    }
    return v;
}

int count_rising_zero_crossings(const std::vector<float>& v) {
    int n = 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i - 1] < 0.0f && v[i] >= 0.0f) {
            ++n;
        }
    }
    return n;
}

} // namespace

TEST(Oscillator, SineStaysInRange) {
    Oscillator o;
    o.set_waveform(Waveform::Sine);
    o.set_frequency(440.0f);
    for (float s : render(o, k_sample_rate)) {
        EXPECT_GE(s, -1.001f);
        EXPECT_LE(s, 1.001f);
    }
}

// One second at N Hz must cross zero going upward N times. This is the check
// that catches a phase-increment mistake, which is otherwise inaudible until
// something is a semitone out.
TEST(Oscillator, FrequencyIsAccurate) {
    for (float hz : {110.0f, 440.0f, 1000.0f}) {
        Oscillator o;
        o.set_waveform(Waveform::Sine);
        o.set_frequency(hz);
        const int crossings = count_rising_zero_crossings(render(o, k_sample_rate));
        EXPECT_NEAR(static_cast<float>(crossings), hz, 1.5f) << "at " << hz << " Hz";
    }
}

TEST(Oscillator, ZeroAndNegativeFrequencyAreSilentNotNaN) {
    for (float hz : {0.0f, -100.0f}) {
        Oscillator o;
        o.set_waveform(Waveform::Saw);
        o.set_frequency(hz);
        for (float s : render(o, 256)) {
            EXPECT_FALSE(std::isnan(s));
            EXPECT_FALSE(std::isinf(s));
        }
    }
}

// PolyBLEP's whole purpose. A naive saw at a high fundamental folds energy
// back down as inharmonic partials; the band-limited one should have
// noticeably less total energy above the fundamental. Measured crudely as
// sample-to-sample variation, which aliasing inflates.
TEST(Oscillator, BandLimitedSawIsSmootherThanNaive) {
    Oscillator o;
    o.set_waveform(Waveform::Saw);
    o.set_frequency(4000.0f);
    const auto band_limited = render(o, 4096);

    // Naive reference at the same frequency.
    std::vector<float> naive(4096);
    float phase = 0.0f;
    const float inc = 4000.0f / static_cast<float>(k_sample_rate);
    for (float& s : naive) {
        s = 2.0f * phase - 1.0f;
        phase += inc;
        if (phase >= 1.0f) {
            phase -= 1.0f;
        }
    }

    auto roughness = [](const std::vector<float>& v) {
        double acc = 0.0;
        for (std::size_t i = 1; i < v.size(); ++i) {
            const double d = static_cast<double>(v[i]) - static_cast<double>(v[i - 1]);
            acc += d * d;
        }
        return acc;
    };

    EXPECT_LT(roughness(band_limited), roughness(naive));
}

TEST(Oscillator, SquareIsBipolar) {
    Oscillator o;
    o.set_waveform(Waveform::Square);
    o.set_frequency(220.0f);
    const auto v = render(o, k_sample_rate / 10);
    double mean = 0.0;
    for (float s : v) {
        mean += static_cast<double>(s);
    }
    mean /= static_cast<double>(v.size());
    EXPECT_NEAR(mean, 0.0, 0.05) << "square should have no DC offset";
}

// Determinism: the same reason the effect system's RNG stopped being a
// thread_local mt19937. Noise must be reproducible or a rendered song is not
// comparable between runs.
TEST(Oscillator, NoiseIsDeterministicAcrossInstances) {
    Oscillator a;
    Oscillator b;
    a.set_waveform(Waveform::Noise);
    b.set_waveform(Waveform::Noise);
    EXPECT_EQ(render(a, 512), render(b, 512));
}

TEST(Oscillator, NoiseStaysInRange) {
    Oscillator o;
    o.set_waveform(Waveform::Noise);
    for (float s : render(o, 4096)) {
        EXPECT_GE(s, -1.0f);
        EXPECT_LT(s, 1.0f);
    }
}

TEST(Oscillator, ResetPhaseMakesRepeatedNotesIdentical) {
    Oscillator o;
    o.set_waveform(Waveform::Triangle);
    o.set_frequency(330.0f);
    o.reset_phase();
    const auto first = render(o, 256);
    render(o, 77); // leave it somewhere arbitrary
    o.reset_phase();
    EXPECT_EQ(first, render(o, 256));
}

#endif // BRACKEN_SYNTH
