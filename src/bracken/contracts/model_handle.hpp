#pragma once

// The model identity contract.
//
// A `ModelHandle` is the only thing the frame contract (`scene/frame_packet.hpp`)
// needs in order to name a mesh: the packet stores handles, the renderer resolves
// them against the `ModelManager` that issued them. Keeping the handle here rather
// than in `model_loader.hpp` is what stops the frame contract — and therefore
// `application.hpp` and `sim_frame.hpp`, which reach it through
// `scene/frame_builder.hpp` — from dragging the whole GPU resource layer
// (`gpu_buffer`, `gpu_texture`, `gpu_types`) and `animation_types.hpp` into every
// host translation unit.
//
// `model_loader.hpp` includes this header, so every existing spelling of
// `bracken::ModelHandle` / `bracken::INVALID_MODEL_HANDLE` keeps working unchanged.

#include <cstdint>

namespace bracken {

// Integer handle for O(1) model lookup (replaces string-based lookups in hot paths).
using ModelHandle = uint32_t;

// The null handle. `ModelManager` never issues it; draws that carry it are
// dropped and counted (`FrameDiagnostics::draws_dropped_unknown_model`).
inline constexpr ModelHandle INVALID_MODEL_HANDLE = 0;

} // namespace bracken
