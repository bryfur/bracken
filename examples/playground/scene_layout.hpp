#pragma once

#include <vector>

namespace playground::layout {

/**
 * One hand-placed prop in the village scene.
 *
 * Coordinates are village metres (plaza centre origin, +x east, +z south —
 * the same space as village_layout.h); the playground converts to world via
 * village_pos(). `model` is a ModelManager name; placements whose model
 * failed to load are skipped, so the list is safe on partial asset sets.
 */
struct Placement {
    const char* model;
    float x = 0.0f;
    float z = 0.0f;
    float rot_deg = 0.0f;
    float scale = 1.0f;
    float sink = 0.0f; // world units sunk into the terrain (hides seams on slopes)
    float y = 0.0f;    // village metres above ground (crate stacking)
};

/** Axis-aligned village-metre rect (kept clear of forest + ground cover). */
struct Rect {
    float x0, z0, x1, z1;
};

/**
 * Areas the ground-cover pass and the forest keep clear: the three roads,
 * both farmsteads (paddock included), and the south rest stop. Authored
 * placements may still dress these — the list only gates the fill layers.
 */
inline constexpr Rect CLEARINGS[] = {
    {-1.0f, -42.0f, 5.0f, -8.0f},  // north road
    {11.0f, -3.0f, 42.0f, 3.0f},   // east road
    {-7.0f, 8.0f, -1.0f, 42.0f},   // south road
    {2.0f, -32.0f, 14.0f, -15.0f}, // north farmstead + wheat paddock
    {21.0f, -11.0f, 34.0f, 12.0f}, // east farmstead + pasture
    {-10.0f, 24.0f, -2.0f, 32.0f}, // south rest stop
};

/**
 * The complete hand-authored dressing for the village scene — every prop,
 * tree, bush, rock, flower clump, mushroom and pebble, laid out area by
 * area (see scene_layout.cpp for the per-area design notes). The only
 * things NOT in this list are the baked village.glb itself and the
 * ground-cover grass-tuft carpet.
 */
std::vector<Placement> build_layout();

} // namespace playground::layout
