#pragma once

// Time-of-day + weather model for the playground.
//
// Pure sim-side logic: update() advances the clock and the weather state
// machine, apply() writes every field it drives into a
// render::EnvironmentState. The engine's procedural sky already responds to
// sun elevation (gradient day-dimming, dusk scattering, starfield + moon
// below the horizon), so this class mainly moves the sun and keys the
// direct/ambient/fog colors to elevation, then overlays the weather
// (overcast darkening + desaturation, rain fog).
//
// Ownership/threading: a plain member of PlaygroundGame, tier sim-stage like
// the rest of the game state. The dev panel edits the public control fields
// at the frame fence (no sim in flight) through the DevPanelView sync.

#include <glm/vec3.hpp>

#include "bracken/contracts/environment.hpp"

namespace playground {

enum class WeatherMode : int { Auto = 0, Clear = 1, Overcast = 2, Rain = 3 };

class DayWeather {
public:
    // ---- Controls (sim-owned; fence-synced with the dev panel) ----
    float hour = 10.6f;   // [0,24); default ~mid-morning, close to the old fixed sun
    float speed = 60.0f;  // game-seconds per real second (60 => a 24-minute day)
    bool animate = true;  // false freezes the sun (weather keeps transitioning)
    WeatherMode mode = WeatherMode::Auto;
    float rain_target = 0.85f; // manual intensity used when mode == Rain

    /** Advance clock + weather. dt is real (sim) seconds. */
    void update(float dt);

    /** Jump the smoothed weather straight to the current mode's targets
     *  (art hooks / screenshots — skips the multi-second transition). */
    void snap();

    /** Write every driven field into env (leaves grading etc. untouched). */
    void apply(bracken::render::EnvironmentState& env) const;

    /** Unit vector TO the sun for the current hour (negate for sun_dir). */
    glm::vec3 to_sun() const;

    float overcast() const { return overcast_; }
    float rain() const { return rain_; }
    const char* weather_label() const;

    /** True when the last update() moved anything apply() depends on. */
    bool dirty() const { return dirty_; }

private:
    float rand01();

    // Smoothed weather state (0..1) and where it is heading.
    float overcast_ = 0.12f;
    float rain_ = 0.0f;
    float target_overcast_ = 0.12f;
    float target_rain_ = 0.0f;
    float auto_timer_ = 40.0f; // real seconds until the next Auto-mode roll
    float prev_hour_ = -1.0f;  // change detector (panel edits included); -1 => first update stages
    bool dirty_ = true;        // first update()/apply() always stages
    uint32_t rng_ = 0x51b7u;   // private stream: keeps the scene RNG untouched
};

} // namespace playground
