#include "day_weather.hpp"

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace playground {
namespace {

// Sun path: hour 6 = sunrise (+x horizon), 12 = noon (tilted toward +z),
// 18 = sunset (-x horizon); night is the same circle below the horizon.
constexpr float SUN_TILT_RAD = 0.56f; // ~32 deg southward tilt -> ~58 deg noon sun

// One keyframe of every color apply() drives. All linear HDR (engine
// convention); the sky-gradient colors stay bright hues because the sky
// shader applies its own day/night dimming — only the lighting fields
// (sun, ambient) and fog carry the darkness host-side.
struct SkyKeys {
    glm::vec3 sun, zenith, horizon, ground, amb_sky, amb_ground, fog_h, fog_sun, vol_sun;
};

// Midday — matches the engine defaults / the playground's original look.
const SkyKeys DAY = {
    .sun = {1.00f, 0.95f, 0.90f},
    .zenith = {0.20f, 0.37f, 0.66f},
    .horizon = {0.45f, 0.56f, 0.74f},
    .ground = {0.16f, 0.14f, 0.11f},
    .amb_sky = {0.35f, 0.42f, 0.55f},
    .amb_ground = {0.22f, 0.20f, 0.17f},
    .fog_h = {0.45f, 0.56f, 0.74f},
    .fog_sun = {0.95f, 0.75f, 0.50f},
    .vol_sun = {1.00f, 0.90f, 0.70f},
};

// Sun on the horizon — sunrise/sunset.
const SkyKeys SET = {
    .sun = {0.95f, 0.42f, 0.17f},
    .zenith = {0.11f, 0.17f, 0.36f},
    .horizon = {0.62f, 0.36f, 0.24f},
    .ground = {0.12f, 0.09f, 0.08f},
    .amb_sky = {0.20f, 0.19f, 0.27f},
    .amb_ground = {0.14f, 0.11f, 0.09f},
    .fog_h = {0.52f, 0.36f, 0.28f},
    .fog_sun = {1.10f, 0.52f, 0.22f},
    .vol_sun = {1.05f, 0.52f, 0.24f},
};

// Deep night — ambient is the moonlight (the sky shader draws the moon +
// stars itself once the sun is below the horizon).
const SkyKeys NIGHT = {
    .sun = {0.0f, 0.0f, 0.0f},
    .zenith = {0.10f, 0.15f, 0.32f},  // shader dims this ~30x at night
    .horizon = {0.13f, 0.16f, 0.28f},
    .ground = {0.05f, 0.05f, 0.07f},
    .amb_sky = {0.050f, 0.062f, 0.095f},
    .amb_ground = {0.022f, 0.025f, 0.034f},
    .fog_h = {0.045f, 0.055f, 0.085f},
    .fog_sun = {0.055f, 0.065f, 0.10f},
    .vol_sun = {0.07f, 0.09f, 0.14f},
};

glm::vec3 mixv(const glm::vec3& a, const glm::vec3& b, float t) { return glm::mix(a, b, t); }

SkyKeys mix_keys(const SkyKeys& a, const SkyKeys& b, float t) {
    return {mixv(a.sun, b.sun, t),       mixv(a.zenith, b.zenith, t),
            mixv(a.horizon, b.horizon, t), mixv(a.ground, b.ground, t),
            mixv(a.amb_sky, b.amb_sky, t), mixv(a.amb_ground, b.amb_ground, t),
            mixv(a.fog_h, b.fog_h, t),     mixv(a.fog_sun, b.fog_sun, t),
            mixv(a.vol_sun, b.vol_sun, t)};
}

glm::vec3 desaturate(const glm::vec3& c, float amount) {
    const float luma = glm::dot(c, glm::vec3(0.299f, 0.587f, 0.114f));
    return glm::mix(c, glm::vec3(luma), amount);
}

/** Move value toward target at a fixed rate; returns true if it moved. */
bool approach(float& value, float target, float max_step) {
    const float delta = std::clamp(target - value, -max_step, max_step);
    if (std::abs(delta) < 1e-6f) {
        return false;
    }
    value += delta;
    return true;
}

} // namespace

float DayWeather::rand01() {
    // xorshift32 — private stream so weather rolls never perturb scene RNG.
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return static_cast<float>(rng_ & 0xffffffu) / 16777216.0f;
}

glm::vec3 DayWeather::to_sun() const {
    const float a = (hour - 6.0f) * (glm::pi<float>() / 12.0f);
    return glm::normalize(glm::vec3(std::cos(a), std::sin(a) * std::cos(SUN_TILT_RAD),
                                    std::sin(a) * std::sin(SUN_TILT_RAD)));
}

const char* DayWeather::weather_label() const {
    if (rain_ > 0.05f) {
        return "rain";
    }
    if (overcast_ > 0.55f) {
        return "overcast";
    }
    if (overcast_ > 0.3f) {
        return "cloudy";
    }
    return "clear";
}

void DayWeather::snap() {
    switch (mode) {
        case WeatherMode::Clear:
            target_overcast_ = 0.12f;
            target_rain_ = 0.0f;
            break;
        case WeatherMode::Overcast:
            target_overcast_ = 0.85f;
            target_rain_ = 0.0f;
            break;
        case WeatherMode::Rain:
            target_overcast_ = 0.95f;
            target_rain_ = std::clamp(rain_target, 0.0f, 1.0f);
            break;
        case WeatherMode::Auto:
            break; // keep whatever the last roll picked
    }
    overcast_ = target_overcast_;
    rain_ = target_rain_;
    prev_hour_ = -1.0f; // force the next update() to report dirty
}

void DayWeather::update(float dt) {
    dirty_ = false;
    if (dt <= 0.0f) {
        return;
    }

    if (animate && speed > 0.0f) {
        hour = std::fmod(hour + dt * speed / 3600.0f, 24.0f);
    }
    // Compare-to-last rather than "did I advance": the dev panel writes hour
    // directly at the frame fence, and that edit must re-stage the
    // environment too.
    if (hour != prev_hour_) {
        prev_hour_ = hour;
        dirty_ = true;
    }

    // Weather targets: manual modes pin them; Auto rolls a new state on a
    // real-time cadence so a session sees weather drift by itself.
    switch (mode) {
        case WeatherMode::Clear:
            target_overcast_ = 0.12f;
            target_rain_ = 0.0f;
            break;
        case WeatherMode::Overcast:
            target_overcast_ = 0.85f;
            target_rain_ = 0.0f;
            break;
        case WeatherMode::Rain:
            target_overcast_ = 0.95f;
            target_rain_ = std::clamp(rain_target, 0.0f, 1.0f);
            break;
        case WeatherMode::Auto: {
            auto_timer_ -= dt;
            if (auto_timer_ <= 0.0f) {
                auto_timer_ = 50.0f + rand01() * 70.0f;
                const float roll = rand01();
                if (roll < 0.40f) {
                    target_overcast_ = 0.10f + rand01() * 0.15f;
                    target_rain_ = 0.0f;
                } else if (roll < 0.65f) {
                    target_overcast_ = 0.45f + rand01() * 0.15f;
                    target_rain_ = 0.0f;
                } else if (roll < 0.80f) {
                    target_overcast_ = 0.80f + rand01() * 0.15f;
                    target_rain_ = 0.0f;
                } else {
                    target_overcast_ = 0.95f;
                    target_rain_ = 0.45f + rand01() * 0.55f;
                }
            }
            break;
        }
    }

    // Smooth transitions: full overcast swing over ~12 s, rain over ~9 s.
    dirty_ |= approach(overcast_, target_overcast_, dt / 12.0f);
    dirty_ |= approach(rain_, target_rain_, dt / 9.0f);
}

void DayWeather::apply(bracken::render::EnvironmentState& env) const {
    const glm::vec3 sun = to_sun();
    const float e = sun.y; // elevation term

    // Elevation blend: horizon key at e=0, day by ~25 deg up, night by
    // ~8 deg down (the shader's own duskFade window).
    SkyKeys k = e >= 0.0f ? mix_keys(SET, DAY, glm::smoothstep(0.0f, 0.42f, e))
                          : mix_keys(SET, NIGHT, glm::smoothstep(0.0f, 0.14f, -e));

    // Weather overlay: clouds swallow direct light, gray the sky, pull the
    // fog in. Rain adds haze and thicker volumetrics on top of overcast.
    const float oc = overcast_;
    const float rn = rain_;
    k.sun *= 1.0f - 0.72f * oc;
    k.vol_sun *= 1.0f - 0.65f * oc;
    k.zenith = desaturate(k.zenith, 0.45f * oc) * (1.0f - 0.25f * oc);
    k.horizon = desaturate(k.horizon, 0.45f * oc) * (1.0f - 0.15f * oc);
    k.amb_sky = desaturate(k.amb_sky, 0.35f * oc);
    k.fog_h = desaturate(k.fog_h, 0.40f * oc);
    k.fog_sun = glm::mix(k.fog_sun, k.fog_h, 0.7f * oc);

    env.sun_dir = -sun;
    env.sun_color = k.sun;
    env.sky_zenith = k.zenith;
    env.sky_horizon = k.horizon;
    env.ground_color = k.ground;
    env.ambient_sky = k.amb_sky;
    env.ambient_ground = k.amb_ground;
    env.fog_color_horizon = k.fog_h;
    env.fog_color_sun = k.fog_sun;
    env.fog_volumetric_sun_color = k.vol_sun;

    env.cloud_coverage = glm::mix(0.32f, 0.94f, oc);
    env.cloud_density = glm::mix(1.0f, 1.3f, oc);
    env.cloud_speed = glm::mix(1.0f, 2.4f, std::max(oc, rn));
    env.haze_strength = glm::mix(1.0f, 1.35f, rn);
    env.fog_start = glm::mix(1000.0f, 480.0f, rn);
    env.fog_end = glm::mix(8000.0f, 4200.0f, rn);
    env.fog_volumetric_density = glm::mix(0.0015f, 0.0042f, std::max(rn, 0.55f * oc));
}

} // namespace playground
