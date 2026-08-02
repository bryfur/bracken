#pragma once

#include "bracken/audio/music/song.hpp"

#include <string>


namespace playground {

/**
 * "Hearthsmoke" — the village theme.
 *
 * A slow pastoral loop in D dorian: the mode gives it a folk/medieval colour
 * without the leading tone that would make it sound classical, which suits the
 * village the playground is set in. Four bars, looping, built entirely from
 * the engine's synthesizer — there is no audio asset anywhere in this repo.
 */
[[nodiscard]] bracken::audio::music::Song build_village_theme();

/**
 * Render the theme offline through the real mixer and write it to a WAV.
 *
 * Device-free, so it works headless and on CI. This is the audio equivalent
 * of the smoke run's screenshot: the score is generated rather than authored,
 * so when a patch stops sounding right there is no source asset to compare
 * against — you dump the output and look at it.
 */
[[nodiscard]] bool render_music_to_wav(const std::string& path, float seconds);

} // namespace playground
