#pragma once

#include "bracken/dev/dev_ui.hpp"

// Internal seam between the DevUI core (context/backend/frame lifecycle) and
// the built-in panel implementations. Only dev_ui.cpp calls this; only
// dev_panels.cpp implements it. Both sides are compiled out together when
// BRACKEN_DEV_UI is not defined.

namespace bracken::dev::detail {

/** Draw the built-in "Bracken Dev" window (tabs: Stats, Environment,
 *  Graphics, Camera). Runs inside an active ImGui frame. Sections whose
 *  binding is null are hidden; Graphics additionally requires a non-empty
 *  `apply_settings`. `state` is the owning DevUI's per-instance widget
 *  state (slider values, FPS history). */
void draw_builtin_panels(const DevUI::Bindings& bindings, DevUI::PanelState& state);

} // namespace bracken::dev::detail
