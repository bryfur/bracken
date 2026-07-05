#pragma once

#include "bracken/dev/dev_ui.hpp"

// Internal seam between the DevUI core (context/backend/frame lifecycle) and
// the built-in panel implementations. Only dev_ui.cpp calls this; only
// dev_panels.cpp implements it. Both sides are compiled out together when
// BRACKEN_DEV_UI is not defined.

namespace bracken::dev::detail {

/** Draw the built-in "Bracken Dev" window (tabs: Stats, Environment,
 *  Graphics, Camera). Runs inside an active ImGui frame. Sections whose
 *  binding is null are hidden. */
void draw_builtin_panels(const DevUI::Bindings& bindings);

} // namespace bracken::dev::detail
