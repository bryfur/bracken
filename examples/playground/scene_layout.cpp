#include "scene_layout.hpp"

#include <cmath>
#include <numbers>

// The village dressing, hand-placed area by area. Everything here is in
// village metres (see village_layout.h for the building footprints this is
// composed against):
//
//   plaza x[-12,12] z[-9,9]          streets: north x[0,4] z[-19,-9],
//   tavern x[-8,-2] z[-19,-9]                 east x[12,20] z[-2,2],
//   house_timber x[12,18] z[-6,-2]            south x[-6,-2] z[9,17]
//   cottage x[12,18] z[4,10]         dirt lanes continue the streets to
//   workshop x[-1,5] z[9,17]         z=-34 (north), x=36 (east), z=34 (south)
//   hut x[-12,-8] z[9,13]
//   farm_north x[5,11] z[-29,-25]    farm_east x[26,30] z[-9,-3]
//   house_plaster x[-20,-12] z[-2,2]
//
// Palette note (the nature pack is autumn-stylized): commontrees are green,
// pines dark green, giant pines mid-green — but BIRCHES ARE ORANGE and
// bush_common is BRIGHT RED. Both are used only as deliberate accents
// (birch stands at thirds around the forest ring, single red bushes at
// gates/corners), never as fill — scattered at random they read as noise.

namespace playground::layout {
namespace {

using Layout = std::vector<Placement>;

void add(Layout& out, const Placement& p) {
    out.push_back(p);
}

/**
 * Straight fence line: `count` 2 m modules from (x,z) stepping (dx,dz),
 * alternating the two fence meshes so the posts don't strobe. rot 0 runs the
 * rails along +x, rot 90 along -z (matching the kit's Y-rotation convention).
 * `skip` omits that segment index — a gate gap.
 */
void fence_run(Layout& out, float x, float z, float dx, float dz, int count, float rot, int skip = -1) {
    for (int i = 0; i < count; ++i) {
        if (i == skip) {
            continue;
        }
        add(out, {i % 2 ? "prop_fence_ext" : "prop_fence", x + dx * static_cast<float>(i),
                  z + dz * static_cast<float>(i), rot, 1.0f, 2.0f});
    }
}

/** A brick heap: two offset clusters plus a raised top course read as a pile. */
void brick_pile(Layout& out, float x, float z, float rot) {
    add(out, {"prop_bricks", x, z, rot, 1.35f});
    add(out, {"prop_bricks", x + 0.26f, z + 0.15f, rot + 74.0f, 1.25f});
    add(out, {"prop_bricks", x + 0.12f, z + 0.06f, rot + 35.0f, 1.2f, 0.0f, 0.16f});
}

// ---------------------------------------------------------------------------
// Forest grove templates (local metres around the grove centre). Each stamp
// is a composed vignette: canopy + understory + floor litter, so the forest
// ring reads as designed woodland rather than a tree lawn.
// ---------------------------------------------------------------------------

// Tight dark-pine cluster — the ring's workhorse.
constexpr Placement GROVE_PINES[] = {
    {"nature_pine_1", 0.0f, 0.0f, 0.0f, 1.55f, 8.0f},
    {"nature_pine_2", 2.6f, 1.4f, 70.0f, 1.35f, 8.0f},
    {"nature_pine_3", -2.2f, 2.0f, 160.0f, 1.40f, 8.0f},
    {"nature_pine_2", 1.4f, -2.4f, 250.0f, 1.25f, 8.0f},
    {"nature_pine_1", -1.8f, -1.9f, 320.0f, 1.30f, 8.0f},
    {"nature_fern_2", 0.9f, 2.9f, 40.0f, 0.50f, 4.0f},
    {"nature_mushroom_common", -0.6f, -0.5f, 0.0f, 0.60f, 2.0f},
    {"nature_rock_medium_3", 3.4f, -0.7f, 120.0f, 0.60f, 10.0f},
};

// Broadleaf group with a flowering skirt — the friendly green mass.
constexpr Placement GROVE_OAKS[] = {
    {"nature_commontree_1", 0.0f, 0.0f, 0.0f, 1.45f, 8.0f},
    {"nature_commontree_2", 3.2f, 0.8f, 90.0f, 1.25f, 8.0f},
    {"nature_commontree_3", -2.6f, 1.7f, 200.0f, 1.20f, 8.0f},
    {"nature_bush_common_flowers", 1.5f, 2.8f, 30.0f, 1.05f, 5.0f},
    {"nature_bush_large", -1.3f, -2.3f, 150.0f, 0.90f, 5.0f},
    {"nature_flower_1_group", 2.9f, -1.6f, 0.0f, 0.60f, 3.0f},
    {"nature_mushroom_redcap", -3.4f, -0.6f, 0.0f, 0.55f, 2.0f},
};

// Orange birch stand — the autumn accent, used at thirds around the ring.
constexpr Placement GROVE_BIRCHES[] = {
    {"nature_birch_1", 0.0f, 0.0f, 0.0f, 1.00f, 8.0f},
    {"nature_birch_2", 2.1f, 1.3f, 80.0f, 0.90f, 8.0f},
    {"nature_birch_1", -1.7f, 2.0f, 190.0f, 0.85f, 8.0f},
    {"nature_birch_2", 0.9f, -1.9f, 280.0f, 0.95f, 8.0f},
    {"nature_mushroom_redcap", 0.7f, 1.0f, 0.0f, 0.60f, 2.0f},
    {"nature_fern_2", -2.4f, 0.3f, 60.0f, 0.45f, 4.0f},
    {"nature_clover_1", 1.8f, -0.4f, 0.0f, 0.90f, 3.0f},
};

// Giant-pine landmark with a boulder at its feet.
constexpr Placement GROVE_GIANT[] = {
    {"nature_giantpine_1", 0.0f, 0.0f, 0.0f, 1.05f, 10.0f},
    {"nature_pine_2", 3.9f, 1.6f, 45.0f, 1.30f, 8.0f},
    {"nature_pine_3", -3.3f, 2.3f, 220.0f, 1.35f, 8.0f},
    {"nature_rock_big_1", 2.3f, -2.9f, 300.0f, 0.75f, 22.0f},
    {"nature_fern_2", -1.4f, -2.2f, 120.0f, 0.55f, 4.0f},
    {"nature_fern_2", 1.1f, 2.6f, 200.0f, 0.50f, 4.0f},
    {"nature_mushroom_common", -2.2f, 0.6f, 0.0f, 0.70f, 2.0f},
};

// The broader giant pine over a broadleaf companion.
constexpr Placement GROVE_GIANT_WIDE[] = {
    {"nature_giantpine_2", 0.0f, 0.0f, 0.0f, 1.05f, 12.0f},
    {"nature_commontree_3", 4.2f, -1.1f, 140.0f, 1.30f, 8.0f},
    {"nature_bush_large", -3.2f, 1.6f, 40.0f, 1.00f, 5.0f},
    {"nature_fern_2", 2.6f, 2.1f, 310.0f, 0.50f, 4.0f},
    {"nature_mushroom_common", 1.7f, -2.3f, 0.0f, 0.55f, 2.0f},
};

// Mixed edge grove — one of each canopy layer, for the seams between runs.
constexpr Placement GROVE_MIXED[] = {
    {"nature_commontree_2", 0.0f, 0.0f, 0.0f, 1.35f, 8.0f},
    {"nature_pine_1", 3.1f, -1.3f, 75.0f, 1.45f, 8.0f},
    {"nature_birch_2", -2.9f, -0.9f, 165.0f, 0.90f, 8.0f},
    {"nature_bush_large", 1.3f, 2.5f, 255.0f, 0.95f, 5.0f},
    {"nature_rock_medium_2", -1.6f, 2.9f, 20.0f, 0.70f, 12.0f},
    {"nature_flower_7_group", 3.0f, 1.8f, 0.0f, 0.55f, 3.0f},
};

// ---------------------------------------------------------------------------
// Areas
// ---------------------------------------------------------------------------

/** Plaza: market corner by the north street mouth, planters at the corners. */
void plaza(Layout& out) {
    // Market stall vignette (NE quarter, clear of the spawn + chrome sphere):
    // a parked goods wagon with a crate pile and brick stock beside it.
    add(out, {"prop_wagon", 7.0f, -7.6f, 15.0f});
    add(out, {"prop_crate", 5.7f, -8.3f, 10.0f});
    add(out, {"prop_crate", 6.5f, -8.55f, 55.0f});
    add(out, {"prop_crate", 5.9f, -8.4f, 40.0f, 1.0f, 0.0f, 1.06f}); // stacked
    brick_pile(out, 8.2f, -8.3f, 0.0f);
    // Chrome-sphere corner keeps one lone crate: the mirror ball is the hero.
    add(out, {"prop_crate", 10.6f, -7.6f, 20.0f});
    // Corner planters just outside the border stones (the SE corner sits
    // under the cottage, so its accent moves to the workshop/cottage gap).
    add(out, {"nature_bush_common_flowers", -12.9f, -9.9f, 0.0f, 1.10f, 5.0f});
    add(out, {"nature_flower_1_group", -12.2f, -10.5f, 40.0f, 0.70f, 3.0f});
    add(out, {"nature_flower_3_group", -13.6f, -9.2f, 110.0f, 0.60f, 3.0f});
    add(out, {"nature_bush_common_flowers", 12.9f, -9.9f, 90.0f, 1.05f, 5.0f});
    add(out, {"nature_flower_7_group", 12.3f, -10.4f, 200.0f, 0.65f, 3.0f});
    add(out, {"nature_bush_common_flowers", 8.2f, 9.9f, 180.0f, 1.15f, 5.0f});
    add(out, {"nature_flower_2_group", 7.3f, 10.3f, 20.0f, 0.60f, 3.0f});
    add(out, {"nature_flower_6", 9.2f, 10.2f, 0.0f, 0.95f, 2.0f});
    add(out, {"nature_bush_common_flowers", -13.4f, 9.9f, 270.0f, 1.00f, 5.0f});
    add(out, {"nature_flower_4_group", -13.9f, 9.2f, 300.0f, 0.60f, 3.0f});
}

/** Tavern: door clutter on the plaza, ivy-shade west wall, stores out back. */
void tavern(Layout& out) {
    // Crate pile east of the door (the classic tavern delivery stack).
    add(out, {"prop_crate", -2.7f, -8.1f, 15.0f});
    add(out, {"prop_crate", -2.1f, -8.55f, 40.0f});
    add(out, {"prop_crate", -2.45f, -7.75f, 75.0f});
    add(out, {"prop_crate", -2.4f, -8.1f, 30.0f, 1.0f, 0.0f, 1.06f}); // stacked
    brick_pile(out, -6.8f, -8.3f, 20.0f);
    // West wall (shaded side): a flowering bush, fern, and spare bricks.
    add(out, {"nature_bush_common_flowers", -9.3f, -10.5f, 250.0f, 0.90f, 5.0f});
    add(out, {"nature_fern_2", -9.15f, -12.8f, 130.0f, 0.50f, 4.0f});
    brick_pile(out, -9.4f, -17.0f, 30.0f);
    // Back yard (north, off the road): crate stores + shade mushrooms.
    add(out, {"prop_crate", -6.5f, -20.2f, 4.0f, 1.0f, 1.0f});
    add(out, {"prop_crate", -5.4f, -20.25f, -6.0f, 1.0f, 1.0f});
    add(out, {"prop_crate", -5.95f, -20.2f, 18.0f, 1.0f, 1.0f, 1.06f}); // stacked
    add(out, {"nature_fern_2", -7.9f, -20.3f, 200.0f, 0.55f, 4.0f});
    add(out, {"nature_mushroom_common", -3.1f, -20.0f, 0.0f, 0.80f, 2.0f});
    add(out, {"nature_mushroom_common", -2.6f, -20.4f, 90.0f, 0.60f, 2.0f});
}

/** Workshop: door-front on the plaza, materials yard east, fairy ring SE. */
void workshop(Layout& out) {
    add(out, {"prop_crate", 3.7f, 7.5f, 10.0f});
    add(out, {"prop_crate", 4.35f, 7.85f, 55.0f});
    brick_pile(out, 0.7f, 7.7f, 0.0f);
    // Materials yard along the east wall: brick stock in a row, crate stack,
    // and the work wagon backed up to the yard.
    brick_pile(out, 6.35f, 9.7f, 5.0f);
    brick_pile(out, 6.45f, 10.9f, 20.0f);
    brick_pile(out, 6.3f, 12.1f, -12.0f);
    add(out, {"prop_crate", 6.15f, 13.5f, 12.0f, 1.0f, 1.0f});
    add(out, {"prop_crate", 6.95f, 13.7f, 58.0f, 1.0f, 1.0f});
    add(out, {"prop_crate", 6.55f, 13.6f, 30.0f, 1.0f, 1.0f, 1.06f}); // stacked
    add(out, {"prop_wagon", 8.1f, 10.8f, 272.0f, 1.0f, 1.0f});
    // Behind the workshop: a mossy rock and shade mushrooms.
    add(out, {"nature_rock_medium_2", 2.2f, 18.3f, 40.0f, 0.65f, 10.0f});
    add(out, {"nature_mushroom_common", 0.6f, 18.1f, 0.0f, 0.80f, 2.0f});
    add(out, {"nature_mushroom_common", 1.1f, 18.45f, 120.0f, 0.55f, 2.0f});
    add(out, {"nature_fern_2", 3.4f, 18.2f, 260.0f, 0.50f, 4.0f});
    // Fairy ring in the grass off the yard's SE corner.
    constexpr float RING_X = 7.6f, RING_Z = 16.3f, RING_R = 1.15f;
    for (int i = 0; i < 7; ++i) {
        const float a = static_cast<float>(i) * 51.4f * std::numbers::pi_v<float> / 180.0f;
        add(out, {"nature_mushroom_common", RING_X + std::cos(a) * RING_R, RING_Z + std::sin(a) * RING_R,
                  static_cast<float>(i) * 51.4f, 0.5f + 0.03f * static_cast<float>(i), 2.0f});
    }
}

/** Cottage: vegetable garden in the back pen, flowers along the rear wall. */
void cottage(Layout& out) {
    // Garden pen fencing (west side open toward the cottage's back door).
    add(out, {"prop_fence", 19.5f, 5.0f, 0.0f, 1.0f, 2.0f});
    add(out, {"prop_fence_ext", 21.5f, 5.0f, 0.0f, 1.0f, 2.0f});
    add(out, {"prop_fence_ext", 19.5f, 9.4f, 0.0f, 1.0f, 2.0f});
    add(out, {"prop_fence", 21.5f, 9.4f, 0.0f, 1.0f, 2.0f});
    add(out, {"prop_fence", 22.5f, 6.1f, 90.0f, 1.0f, 2.0f});
    add(out, {"prop_fence_ext", 22.5f, 8.1f, 90.0f, 1.0f, 2.0f});
    // Three planted rows: wheat | leafy greens | flowers.
    for (int i = 0; i < 5; ++i) {
        const float rz = 5.9f + 0.7f * static_cast<float>(i);
        add(out, {"nature_grass_wheat", 20.2f, rz, 25.0f * static_cast<float>(i), 0.65f, 3.0f});
    }
    add(out, {"nature_plant_1", 21.1f, 6.1f, 0.0f, 0.60f, 3.0f});
    add(out, {"nature_plant_3", 21.1f, 6.95f, 80.0f, 0.60f, 3.0f});
    add(out, {"nature_plant_1", 21.1f, 7.8f, 160.0f, 0.55f, 3.0f});
    add(out, {"nature_plant_3", 21.1f, 8.65f, 240.0f, 0.60f, 3.0f});
    add(out, {"nature_flower_2_group", 21.9f, 6.2f, 0.0f, 0.55f, 3.0f});
    add(out, {"nature_flower_4_group", 21.9f, 7.2f, 90.0f, 0.55f, 3.0f});
    add(out, {"nature_flower_3_group", 21.9f, 8.2f, 180.0f, 0.55f, 3.0f});
    // Flowers along the rear (south) wall, bushes bracketing the corners.
    add(out, {"nature_flower_3_group", 13.0f, 11.0f, 10.0f, 0.70f, 3.0f});
    add(out, {"nature_flower_1_group", 15.0f, 11.05f, 100.0f, 0.75f, 3.0f});
    add(out, {"nature_flower_7_group", 17.0f, 11.0f, 190.0f, 0.70f, 3.0f});
    add(out, {"nature_bush_common_flowers", 11.9f, 11.2f, 40.0f, 0.90f, 5.0f});
    add(out, {"nature_bush_long_1", 18.3f, 11.15f, 140.0f, 0.90f, 5.0f});
    // North gap toward the timber house: one deliberate red accent bush.
    add(out, {"nature_bush_common", 17.6f, 3.0f, 60.0f, 0.80f, 5.0f});
    add(out, {"nature_fern_2", 13.6f, 3.05f, 300.0f, 0.50f, 4.0f});
    add(out, {"nature_flower_1_group", 12.6f, 2.9f, 220.0f, 0.55f, 3.0f});
}

/** Hut: the humble corner — mushrooms out back, ferns in the west shade. */
void hut(Layout& out) {
    add(out, {"prop_crate", -9.55f, 8.15f, 20.0f});
    add(out, {"nature_mushroom_redcap", -11.4f, 14.0f, 0.0f, 0.80f, 2.0f});
    add(out, {"nature_mushroom_common", -10.6f, 14.3f, 70.0f, 0.70f, 2.0f});
    add(out, {"nature_mushroom_redcap", -12.1f, 14.5f, 150.0f, 0.55f, 2.0f});
    add(out, {"nature_mushroom_common", -11.8f, 13.85f, 220.0f, 0.50f, 2.0f});
    add(out, {"nature_fern_2", -13.15f, 10.4f, 30.0f, 0.55f, 4.0f});
    add(out, {"nature_fern_2", -13.35f, 11.6f, 140.0f, 0.50f, 4.0f});
    add(out, {"nature_fern_2", -12.95f, 12.6f, 250.0f, 0.45f, 4.0f});
    // East strip between the hut and the south street.
    add(out, {"nature_bush_long_1", -7.1f, 9.2f, 80.0f, 0.75f, 5.0f});
    add(out, {"nature_flower_4_group", -6.95f, 10.6f, 160.0f, 0.60f, 3.0f});
    add(out, {"nature_flower_2_group", -7.05f, 12.0f, 240.0f, 0.55f, 3.0f});
}

/** Plaster house: flower foundation beds, work corner behind, birch frame. */
void plaster_house(Layout& out) {
    // Door-side (plaza edge): crate + bricks against the front wall.
    add(out, {"prop_crate", -11.3f, 0.7f, 15.0f});
    brick_pile(out, -11.3f, -2.5f, 60.0f);
    // North bed.
    add(out, {"nature_flower_1_group", -14.0f, -3.05f, 0.0f, 0.70f, 3.0f});
    add(out, {"nature_flower_3_group", -17.0f, -3.1f, 90.0f, 0.65f, 3.0f});
    add(out, {"nature_bush_common_flowers", -19.9f, -3.05f, 180.0f, 0.95f, 5.0f});
    // South bed.
    add(out, {"nature_bush_common_flowers", -13.8f, 3.1f, 20.0f, 1.00f, 5.0f});
    add(out, {"nature_flower_7_group", -15.6f, 3.15f, 110.0f, 0.65f, 3.0f});
    add(out, {"nature_fern_2", -17.4f, 3.05f, 200.0f, 0.50f, 4.0f});
    // Back-of-house work corner (west).
    add(out, {"prop_crate", -21.3f, -0.9f, 55.0f, 1.0f, 1.0f});
    brick_pile(out, -21.15f, 0.5f, 10.0f);
    add(out, {"nature_rock_medium_2", -23.6f, 2.4f, 200.0f, 0.85f, 12.0f});
    add(out, {"nature_fern_2", -21.7f, 1.7f, 100.0f, 0.55f, 4.0f});
    add(out, {"nature_mushroom_common", -21.0f, -2.0f, 0.0f, 0.70f, 2.0f});
    // Orange birch pair framing the house against the west meadow.
    add(out, {"nature_birch_1", -16.2f, -5.8f, 140.0f, 0.95f, 8.0f});
    add(out, {"nature_birch_2", -13.9f, -4.9f, 15.0f, 0.80f, 8.0f});
}

/** Timber house: door-side stores, big bush bed behind, lane-side fence. */
void timber_house(Layout& out) {
    add(out, {"prop_crate", 11.4f, -4.5f, 25.0f});
    brick_pile(out, 11.45f, -2.5f, 80.0f);
    // North side (behind the house, toward the great oak).
    add(out, {"nature_bush_large", 14.0f, -7.7f, 0.0f, 0.95f, 5.0f});
    add(out, {"nature_bush_common_flowers", 16.6f, -7.5f, 120.0f, 1.00f, 5.0f});
    add(out, {"nature_flower_1_group", 15.3f, -8.0f, 60.0f, 0.70f, 3.0f});
    add(out, {"nature_fern_2", 12.9f, -7.6f, 200.0f, 0.50f, 4.0f});
    // Field fence along the east lane's north shoulder.
    fence_run(out, 13.5f, -2.25f, 2.0f, 0.0f, 4, 0.0f);
    // Parked wagon on the south shoulder, nose toward the village.
    add(out, {"prop_wagon", 17.2f, 3.1f, 262.0f, 1.0f, 1.0f});
    add(out, {"nature_clover_1", 13.2f, 2.8f, 0.0f, 0.90f, 3.0f});
}

/** North road + farmstead: fenced lane, wheat paddock, pine backdrop. */
void north_farm(Layout& out) {
    // Street flanks where the cobbles run (both sides, 4 modules each).
    fence_run(out, -0.25f, -10.5f, 0.0f, -2.0f, 4, 90.0f);
    fence_run(out, 4.25f, -10.5f, 0.0f, -2.0f, 4, 90.0f);
    // Flower verge at the plaza mouth.
    add(out, {"nature_flower_2_group", -0.9f, -9.7f, 30.0f, 0.60f, 3.0f});
    add(out, {"nature_flower_4_group", 5.0f, -9.8f, 140.0f, 0.60f, 3.0f});
    // Wheat paddock north of the farmhouse; gate gap on the farm side.
    fence_run(out, 5.0f, -23.0f, 2.0f, 0.0f, 4, 0.0f, 1); // gate at x=7
    fence_run(out, 5.0f, -17.0f, 2.0f, 0.0f, 4, 0.0f);
    fence_run(out, 4.0f, -22.0f, 0.0f, 2.0f, 3, 90.0f);
    fence_run(out, 12.0f, -22.0f, 0.0f, 2.0f, 3, 90.0f);
    // The crop: five hand-sown wheat rows.
    for (int row = 0; row < 5; ++row) {
        const float rz = -22.0f + static_cast<float>(row);
        for (int i = 0; i < 8; ++i) {
            const float rx = 5.2f + 0.8f * static_cast<float>(i);
            add(out, {"nature_grass_wheat", rx, rz, static_cast<float>((row * 8 + i) * 47 % 360),
                      0.72f, 3.0f});
        }
    }
    add(out, {"nature_clover_1", 4.8f, -17.6f, 0.0f, 0.85f, 3.0f});
    add(out, {"nature_clover_2", 11.3f, -22.3f, 90.0f, 0.80f, 3.0f});
    // Farm door-side stores; hauler's wagon pulled off the lane.
    add(out, {"prop_crate", 5.3f, -24.5f, 30.0f, 1.0f, 1.0f});
    add(out, {"prop_crate", 6.05f, -24.85f, 68.0f, 1.0f, 1.0f});
    brick_pile(out, 4.75f, -24.0f, 15.0f);
    add(out, {"prop_wagon", -0.6f, -23.2f, 8.0f, 1.0f, 1.0f});
    // Red accent bush on the farm's SW corner; pines close the backdrop.
    add(out, {"nature_bush_common", 3.9f, -29.9f, 0.0f, 1.00f, 5.0f});
    add(out, {"nature_giantpine_1", 7.6f, -32.0f, 200.0f, 1.00f, 10.0f});
    add(out, {"nature_pine_2", 10.8f, -31.0f, 80.0f, 1.35f, 8.0f});
    add(out, {"nature_pine_3", 4.6f, -31.4f, 310.0f, 1.30f, 8.0f});
    // Lane-shoulder pebble clusters continuing where the fences end.
    add(out, {"nature_pebble_round_1", 0.3f, -20.5f, 0.0f, 1.20f, 2.0f});
    add(out, {"nature_pebble_round_3", 0.7f, -20.8f, 90.0f, 0.90f, 2.0f});
    add(out, {"nature_pebble_square_1", 3.8f, -22.5f, 45.0f, 1.30f, 2.0f});
    add(out, {"nature_pebble_round_2", 3.5f, -22.1f, 180.0f, 1.00f, 2.0f});
    add(out, {"nature_pebble_round_2", 0.2f, -26.5f, 270.0f, 1.10f, 2.0f});
    add(out, {"nature_pebble_square_2", 3.9f, -29.0f, 20.0f, 1.20f, 2.0f});
    add(out, {"nature_pebble_round_1", 3.6f, -29.4f, 200.0f, 0.80f, 2.0f});
    add(out, {"nature_pebble_round_3", 0.4f, -31.5f, 130.0f, 1.00f, 2.0f});
}

/** East road + farmstead: pasture paddock, door stores, giant-pine canopy. */
void east_farm(Layout& out) {
    // Pasture paddock south of the lane; gate gap on the lane side.
    fence_run(out, 25.0f, 4.0f, 2.0f, 0.0f, 4, 0.0f, 1); // gate at x=27
    fence_run(out, 25.0f, 10.0f, 2.0f, 0.0f, 4, 0.0f);
    fence_run(out, 24.0f, 5.0f, 0.0f, 2.0f, 3, 90.0f);
    fence_run(out, 32.0f, 5.0f, 0.0f, 2.0f, 3, 90.0f);
    // Grazed pasture: clover clumps, short tufts, a few stones.
    add(out, {"nature_clover_1", 25.5f, 5.2f, 0.0f, 1.00f, 3.0f});
    add(out, {"nature_clover_2", 27.2f, 8.4f, 80.0f, 0.90f, 3.0f});
    add(out, {"nature_clover_1", 29.8f, 6.1f, 160.0f, 1.10f, 3.0f});
    add(out, {"nature_clover_2", 30.9f, 8.9f, 240.0f, 0.80f, 3.0f});
    add(out, {"nature_clover_1", 26.3f, 9.0f, 320.0f, 0.85f, 3.0f});
    add(out, {"nature_grass_wispy_short", 28.4f, 5.6f, 40.0f, 0.60f, 3.0f});
    add(out, {"nature_grass_wispy_short", 30.2f, 4.9f, 150.0f, 0.55f, 3.0f});
    add(out, {"nature_pebble_round_1", 27.8f, 7.2f, 0.0f, 1.00f, 2.0f});
    add(out, {"nature_pebble_square_1", 25.2f, 8.8f, 60.0f, 0.90f, 2.0f});
    add(out, {"nature_rock_medium_3", 31.2f, 9.2f, 220.0f, 0.50f, 8.0f});
    // Gate flanks: green flowering bush west, single red accent east.
    add(out, {"nature_bush_common_flowers", 25.9f, 3.4f, 0.0f, 0.90f, 5.0f});
    add(out, {"nature_bush_common", 29.9f, 3.4f, 100.0f, 0.85f, 5.0f});
    // Door-side stores.
    add(out, {"prop_crate", 30.3f, -1.55f, 45.0f, 1.0f, 1.0f});
    brick_pile(out, 25.3f, -1.7f, 20.0f);
    // The giant pine leaning its canopy over the farmhouse.
    add(out, {"nature_giantpine_2", 33.8f, -8.5f, 150.0f, 1.05f, 12.0f});
    // Waymarker where the lane fades into the meadow.
    add(out, {"nature_rock_medium_1", 36.6f, 1.8f, 300.0f, 0.75f, 10.0f});
    add(out, {"nature_pebble_round_2", 36.0f, 1.2f, 0.0f, 1.10f, 2.0f});
    add(out, {"nature_pebble_round_1", 36.9f, 0.9f, 90.0f, 0.80f, 2.0f});
    // Verge flower where the street mouth meets the grass.
    add(out, {"nature_flower_1_group", 12.6f, 2.9f, 0.0f, 0.55f, 3.0f});
}

/** South road + rest stop: fenced lane, then a hauler's campsite. */
void south_road(Layout& out) {
    fence_run(out, -6.25f, 10.5f, 0.0f, 2.0f, 4, 90.0f);
    fence_run(out, -1.75f, 10.5f, 0.0f, 2.0f, 4, 90.0f);
    // Lane-shoulder pebbles past the fences.
    add(out, {"nature_pebble_round_1", -6.6f, 19.5f, 0.0f, 1.20f, 2.0f});
    add(out, {"nature_pebble_round_2", -6.9f, 19.9f, 70.0f, 0.80f, 2.0f});
    add(out, {"nature_pebble_square_1", -1.4f, 21.5f, 140.0f, 1.10f, 2.0f});
    add(out, {"nature_pebble_round_3", -6.8f, 24.0f, 210.0f, 1.00f, 2.0f});
    add(out, {"nature_pebble_round_2", -1.2f, 27.5f, 280.0f, 1.20f, 2.0f});
    add(out, {"nature_pebble_square_2", -0.9f, 27.1f, 350.0f, 0.80f, 2.0f});
    add(out, {"nature_pebble_round_1", -6.5f, 30.0f, 60.0f, 0.90f, 2.0f});
    // A leaning orange birch marks the lane's bend south.
    add(out, {"nature_birch_1", -7.6f, 20.8f, 200.0f, 0.90f, 8.0f});
    add(out, {"nature_bush_large", -1.0f, 19.5f, 130.0f, 0.90f, 5.0f});
    // The campsite: wagon, stone fire ring, rock seats, supplies.
    add(out, {"prop_wagon", -7.3f, 27.2f, 105.0f, 1.0f, 1.0f});
    constexpr float FIRE_X = -4.9f, FIRE_Z = 27.3f, FIRE_R = 0.85f;
    for (int i = 0; i < 7; ++i) {
        const float a = (static_cast<float>(i) * 51.4f + 20.0f) * std::numbers::pi_v<float> / 180.0f;
        add(out, {i % 2 ? "nature_pebble_square_1" : "nature_pebble_square_2",
                  FIRE_X + std::cos(a) * FIRE_R, FIRE_Z + std::sin(a) * FIRE_R,
                  static_cast<float>(i) * 73.0f, 0.95f + 0.08f * static_cast<float>(i % 3), 1.0f});
    }
    add(out, {"nature_rock_medium_1", -3.7f, 27.9f, 40.0f, 0.50f, 6.0f});
    add(out, {"nature_rock_medium_2", -5.6f, 26.1f, 150.0f, 0.45f, 5.0f});
    add(out, {"prop_crate", -6.35f, 28.6f, 15.0f, 1.0f, 1.0f});
    add(out, {"prop_crate", -5.7f, 29.0f, 65.0f, 1.0f, 1.0f});
    brick_pile(out, -6.6f, 29.5f, 0.0f);
    add(out, {"nature_giantpine_1", -8.9f, 25.2f, 20.0f, 1.05f, 10.0f});
    add(out, {"nature_fern_2", -8.1f, 26.0f, 300.0f, 0.50f, 4.0f});
    add(out, {"nature_mushroom_common", -3.0f, 26.2f, 0.0f, 0.60f, 2.0f});
    add(out, {"nature_mushroom_common", -2.7f, 26.6f, 80.0f, 0.45f, 2.0f});
}

/** Meadow: the belt between village and forest — outcrops, drifts, trees. */
void meadow(Layout& out) {
    // The great oak on the plaza's NE approach — the village's landmark tree.
    add(out, {"nature_commontree_1", 8.6f, -11.8f, 25.0f, 1.60f, 8.0f});
    // Rock outcrop A (west meadow, seen from the plaza between the houses).
    add(out, {"nature_rock_big_1", -18.0f, 14.0f, 15.0f, 0.95f, 25.0f});
    add(out, {"nature_rock_medium_3", -16.7f, 14.9f, 170.0f, 0.70f, 12.0f});
    add(out, {"nature_rock_medium_1", -19.0f, 15.3f, 80.0f, 0.50f, 8.0f});
    add(out, {"nature_pebble_round_1", -17.1f, 13.4f, 0.0f, 1.20f, 2.0f});
    add(out, {"nature_pebble_round_3", -18.9f, 13.8f, 90.0f, 0.90f, 2.0f});
    add(out, {"nature_fern_2", -17.4f, 15.7f, 220.0f, 0.55f, 4.0f});
    add(out, {"nature_mushroom_redcap", -18.6f, 13.1f, 0.0f, 0.60f, 2.0f});
    add(out, {"nature_flower_7_group", -16.3f, 14.2f, 130.0f, 0.60f, 3.0f});
    // Outcrop B (NE, past the great oak).
    add(out, {"nature_rock_big_2", 20.5f, -13.5f, 220.0f, 0.85f, 25.0f});
    add(out, {"nature_rock_medium_2", 21.8f, -12.7f, 40.0f, 0.60f, 10.0f});
    add(out, {"nature_bush_common_flowers", 19.4f, -12.6f, 300.0f, 0.95f, 5.0f});
    add(out, {"nature_pebble_round_2", 20.0f, -14.6f, 0.0f, 1.10f, 2.0f});
    add(out, {"nature_pebble_square_2", 21.2f, -14.2f, 70.0f, 0.85f, 2.0f});
    // Outcrop C (SE, off the cottage garden).
    add(out, {"nature_rock_medium_4", 13.5f, 17.5f, 310.0f, 0.95f, 14.0f});
    add(out, {"nature_rock_medium_1", 14.4f, 18.3f, 150.0f, 0.55f, 8.0f});
    add(out, {"nature_flower_2_group", 12.7f, 18.1f, 40.0f, 0.60f, 3.0f});
    add(out, {"nature_pebble_round_1", 14.0f, 16.7f, 180.0f, 1.00f, 2.0f});
    // Monolith D (NW meadow) — a lone standing stone.
    add(out, {"nature_rock_big_1", -24.5f, -9.5f, 265.0f, 1.20f, 35.0f});
    add(out, {"nature_pebble_round_1", -23.4f, -8.9f, 0.0f, 1.30f, 2.0f});
    add(out, {"nature_pebble_round_3", -25.4f, -8.5f, 120.0f, 0.90f, 2.0f});
    // Pine trio on the NE rise.
    add(out, {"nature_pine_1", 16.5f, -16.5f, 0.0f, 1.50f, 8.0f});
    add(out, {"nature_pine_2", 18.7f, -15.6f, 80.0f, 1.30f, 8.0f});
    add(out, {"nature_pine_3", 17.3f, -18.2f, 200.0f, 1.35f, 8.0f});
    add(out, {"nature_fern_2", 17.8f, -16.9f, 140.0f, 0.50f, 4.0f});
    add(out, {"nature_mushroom_common", 16.1f, -15.4f, 0.0f, 0.60f, 2.0f});
    // Broadleaf pair on the west meadow.
    add(out, {"nature_commontree_2", -22.5f, 3.5f, 45.0f, 1.35f, 8.0f});
    add(out, {"nature_commontree_3", -20.6f, 5.4f, 190.0f, 1.15f, 8.0f});
    add(out, {"nature_bush_large", -21.6f, 6.7f, 90.0f, 0.90f, 5.0f});
    // SE shade tree with a red accent bush.
    add(out, {"nature_commontree_1", 18.5f, 20.5f, 120.0f, 1.40f, 8.0f});
    add(out, {"nature_bush_common", 17.2f, 21.3f, 30.0f, 0.85f, 5.0f});
    add(out, {"nature_flower_6", 19.6f, 21.2f, 0.0f, 1.00f, 2.0f});
    // Lone NW birch; lane-side tree east.
    add(out, {"nature_birch_2", -17.8f, -19.5f, 305.0f, 1.00f, 8.0f});
    add(out, {"nature_mushroom_redcap", -16.9f, -18.8f, 60.0f, 0.55f, 2.0f});
    add(out, {"nature_commontree_3", 22.6f, -5.4f, 75.0f, 1.25f, 8.0f});
    // Flower drifts — one per compass quarter, arcing with the terrain.
    add(out, {"nature_flower_1_group", -15.2f, 5.4f, 10.0f, 0.70f, 3.0f}); // west
    add(out, {"nature_flower_3_group", -16.4f, 6.1f, 80.0f, 0.65f, 3.0f});
    add(out, {"nature_flower_7_group", -17.7f, 6.7f, 150.0f, 0.70f, 3.0f});
    add(out, {"nature_flower_2_group", -18.6f, 7.9f, 220.0f, 0.60f, 3.0f});
    add(out, {"nature_flower_4_group", -19.3f, 9.1f, 290.0f, 0.65f, 3.0f});
    add(out, {"nature_flower_6", -16.9f, 7.6f, 0.0f, 1.10f, 2.0f});
    add(out, {"nature_flower_2_group", 6.2f, -10.3f, 40.0f, 0.60f, 3.0f}); // north-east
    add(out, {"nature_flower_1_group", 7.6f, -10.7f, 120.0f, 0.70f, 3.0f});
    add(out, {"nature_flower_4_group", 9.3f, -10.1f, 200.0f, 0.60f, 3.0f});
    add(out, {"nature_flower_7_group", 10.5f, -9.5f, 280.0f, 0.55f, 3.0f});
    add(out, {"nature_flower_6", 8.4f, -9.8f, 0.0f, 1.00f, 2.0f});
    add(out, {"nature_flower_3_group", 19.4f, 12.3f, 60.0f, 0.65f, 3.0f}); // south-east
    add(out, {"nature_flower_1_group", 20.8f, 13.0f, 140.0f, 0.70f, 3.0f});
    add(out, {"nature_flower_7_group", 22.3f, 13.7f, 230.0f, 0.60f, 3.0f});
    add(out, {"nature_flower_6", 21.4f, 14.6f, 0.0f, 1.10f, 2.0f});
    add(out, {"nature_flower_2_group", 23.3f, 14.3f, 310.0f, 0.55f, 3.0f});
    add(out, {"nature_flower_4_group", -11.5f, -12.4f, 20.0f, 0.65f, 3.0f}); // north-west
    add(out, {"nature_flower_2_group", -12.7f, -13.5f, 100.0f, 0.60f, 3.0f});
    add(out, {"nature_flower_1_group", -13.9f, -14.8f, 180.0f, 0.70f, 3.0f});
    add(out, {"nature_flower_6", -12.9f, -12.8f, 0.0f, 1.00f, 2.0f});
    // Old field-boundary fence remnants (a gap where the rails fell).
    add(out, {"prop_fence", -22.0f, -14.0f, 25.0f, 1.0f, 2.0f});
    add(out, {"prop_fence_ext", -20.2f, -14.85f, 25.0f, 1.0f, 2.0f});
    add(out, {"prop_fence", -16.6f, -16.55f, 25.0f, 1.0f, 2.0f});
    add(out, {"prop_fence", 19.0f, 16.0f, 90.0f, 1.0f, 2.0f});
    add(out, {"prop_fence_ext", 19.0f, 18.0f, 90.0f, 1.0f, 2.0f});
    add(out, {"prop_fence", 19.0f, 21.9f, 90.0f, 1.0f, 2.0f});
}

/**
 * River banks (the channel runs north-south along the west map edge,
 * centerline x ~= -31±4 by latitude — see river_center_x in playground.cpp).
 * Reeds hug the east waterline, boulders sit half-submerged in the shallows,
 * and a pebble shoal collects on the inner bend near the village latitude.
 */
void river_banks(Layout& out) {
    // Reed clumps (tall wispy grass) along the east waterline.
    add(out, {"nature_grass_wispy_tall", -27.2f, -16.0f, 10.0f, 1.15f, 2.0f});
    add(out, {"nature_grass_wide_tall", -27.0f, -15.3f, 120.0f, 0.95f, 2.0f});
    add(out, {"nature_grass_wispy_tall", -26.8f, -9.0f, 80.0f, 0.95f, 2.0f});
    add(out, {"nature_grass_wispy_tall", -26.5f, -2.0f, 150.0f, 1.10f, 2.0f});
    add(out, {"nature_grass_wide_tall", -26.3f, -1.2f, 40.0f, 0.90f, 2.0f});
    add(out, {"nature_grass_wispy_tall", -26.7f, 4.0f, 220.0f, 1.00f, 2.0f});
    add(out, {"nature_grass_wispy_tall", -27.3f, 11.0f, 290.0f, 1.20f, 2.0f});
    add(out, {"nature_grass_wide_tall", -27.6f, 11.8f, 200.0f, 0.95f, 2.0f});
    add(out, {"nature_grass_wispy_tall", -28.5f, 18.0f, 60.0f, 1.00f, 2.0f});
    add(out, {"nature_grass_wispy_tall", -29.5f, 24.0f, 130.0f, 1.10f, 2.0f});
    // Half-submerged boulders in the shallows.
    add(out, {"nature_rock_medium_4", -36.0f, 14.5f, 40.0f, 0.75f, 40.0f});
    add(out, {"nature_rock_big_2", -28.3f, -24.0f, 160.0f, 0.55f, 45.0f});
    add(out, {"nature_rock_medium_1", -27.7f, 7.5f, 260.0f, 0.60f, 35.0f});
    // Pebble shoal on the inner bend's sand (east side, village latitude).
    add(out, {"nature_pebble_round_1", -26.4f, -1.2f, 0.0f, 1.30f, 2.0f});
    add(out, {"nature_pebble_round_2", -26.1f, -0.2f, 70.0f, 1.00f, 2.0f});
    add(out, {"nature_pebble_square_1", -26.6f, 0.6f, 140.0f, 1.20f, 2.0f});
    add(out, {"nature_pebble_round_3", -26.2f, 1.4f, 210.0f, 0.90f, 2.0f});
    add(out, {"nature_pebble_round_1", -26.9f, -0.6f, 280.0f, 0.85f, 2.0f});
    // Bank-top accents.
    add(out, {"nature_clover_1", -25.9f, 2.5f, 0.0f, 0.95f, 3.0f});
    add(out, {"nature_flower_2_group", -26.5f, 12.5f, 90.0f, 0.60f, 3.0f});
    add(out, {"nature_clover_2", -27.9f, 21.0f, 180.0f, 0.85f, 3.0f});
    add(out, {"nature_fern_2", -36.6f, 5.6f, 250.0f, 0.55f, 4.0f}); // far bank, under the giant pine
}

/**
 * The forest ring closing the horizon (r 27..42 m): thirty grove stamps
 * every ~12 degrees at alternating radii, with single trees stitching the
 * seams. Road corridors, farmsteads and the rest stop stay open; birch
 * stands (orange) sit at thirds so the autumn accent reads as intentional.
 */
void forest(Layout& out) {
    struct Grove {
        const Placement* tmpl;
        size_t n;
        float x, z, rot;
    };
#define GROVE(T, X, Z, R) {T, std::size(T), X, Z, R}
    constexpr Grove GROVES[] = {
        GROVE(GROVE_GIANT_WIDE, 36.7f, 9.8f, 55.0f),
        GROVE(GROVE_OAKS, 27.4f, 14.6f, 100.0f),
        GROVE(GROVE_PINES, 28.4f, 22.2f, 150.0f),
        GROVE(GROVE_MIXED, 19.3f, 23.0f, 210.0f),
        GROVE(GROVE_BIRCHES, 17.4f, 32.7f, 260.0f),
        GROVE(GROVE_PINES, 8.5f, 29.8f, 310.0f),
        GROVE(GROVE_OAKS, 1.4f, 39.0f, 10.0f),
        GROVE(GROVE_PINES, -7.5f, 35.2f, 65.0f),
        GROVE(GROVE_MIXED, -12.7f, 27.2f, 115.0f),
        GROVE(GROVE_GIANT, -22.2f, 28.4f, 170.0f),
        GROVE(GROVE_OAKS, -23.7f, 19.9f, 225.0f),
        // The west arc hugs the river: groves sit on the near bank crest or
        // across the water on the far bank (never in the channel).
        GROVE(GROVE_PINES, -38.8f, 17.8f, 275.0f), // far bank
        GROVE(GROVE_OAKS, -26.0f, 9.3f, 330.0f),   // near bank
        GROVE(GROVE_GIANT_WIDE, -37.9f, 3.3f, 25.0f), // far bank
        GROVE(GROVE_BIRCHES, -26.0f, -3.5f, 80.0f),   // near bank — riverside birches
        GROVE(GROVE_PINES, -37.2f, -11.8f, 135.0f),   // far bank
        GROVE(GROVE_MIXED, -26.8f, -15.5f, 185.0f),
        GROVE(GROVE_OAKS, -28.2f, -25.4f, 240.0f),
        GROVE(GROVE_PINES, -18.8f, -25.9f, 295.0f),
        GROVE(GROVE_GIANT, -16.1f, -34.4f, 345.0f),
        GROVE(GROVE_MIXED, -7.7f, -31.0f, 40.0f),
        GROVE(GROVE_GIANT_WIDE, -3.5f, -40.0f, 95.0f),
        GROVE(GROVE_PINES, 5.0f, -35.6f, 145.0f),
        GROVE(GROVE_OAKS, 13.3f, -36.7f, 195.0f),
        GROVE(GROVE_BIRCHES, 17.0f, -27.1f, 250.0f),
        GROVE(GROVE_PINES, 26.4f, -27.3f, 300.0f),
        GROVE(GROVE_MIXED, 25.7f, -17.3f, 355.0f),
        GROVE(GROVE_OAKS, 35.2f, -14.2f, 45.0f),
        GROVE(GROVE_PINES, 39.4f, -6.9f, 105.0f),
        GROVE(GROVE_MIXED, 39.8f, 4.2f, 160.0f),
    };
#undef GROVE
    for (const auto& g : GROVES) {
        for (size_t i = 0; i < g.n; ++i) {
            const Placement& p = g.tmpl[i];
            const float c = std::cos(g.rot * std::numbers::pi_v<float> / 180.0f);
            const float s = std::sin(g.rot * std::numbers::pi_v<float> / 180.0f);
            Placement q = p;
            q.x = g.x + c * p.x + s * p.z;
            q.z = g.z + (-s * p.x + c * p.z);
            q.rot_deg = p.rot_deg + g.rot;
            add(out, q);
        }
    }
    // Singles stitching the grove seams (inner ring edge + a few far ones).
    constexpr Placement SINGLES[] = {
        {"nature_commontree_3", 31.5f, 12.8f, 15.0f, 1.20f, 8.0f},
        {"nature_commontree_1", 20.1f, 19.5f, 70.0f, 1.30f, 8.0f},
        {"nature_pine_3", 15.9f, 23.6f, 125.0f, 1.40f, 8.0f},
        {"nature_commontree_2", 10.5f, 26.0f, 180.0f, 1.25f, 8.0f},
        {"nature_pine_2", 4.5f, 28.7f, 235.0f, 1.30f, 8.0f},
        {"nature_commontree_1", -2.9f, 32.9f, 290.0f, 1.35f, 8.0f},
        {"nature_pine_1", -11.7f, 34.0f, 345.0f, 1.45f, 8.0f},
        {"nature_commontree_3", -14.8f, 23.7f, 40.0f, 1.25f, 8.0f},
        {"nature_birch_1", -19.5f, 20.1f, 95.0f, 0.95f, 8.0f},
        {"nature_pine_3", -24.0f, 16.2f, 150.0f, 1.40f, 8.0f},
        {"nature_commontree_1", -26.0f, 10.5f, 205.0f, 1.30f, 8.0f},
        {"nature_pine_2", -25.4f, 5.4f, 260.0f, 1.35f, 8.0f},   // pulled east of the river
        {"nature_commontree_2", -25.0f, -0.8f, 315.0f, 1.30f, 8.0f},
        {"nature_pine_1", -25.2f, -6.9f, 10.0f, 1.40f, 8.0f},
        {"nature_commontree_3", -25.4f, -11.8f, 65.0f, 1.25f, 8.0f},
        {"nature_birch_2", -23.2f, -17.5f, 120.0f, 0.90f, 8.0f},
        {"nature_commontree_1", -18.4f, -21.1f, 175.0f, 1.30f, 8.0f},
        {"nature_pine_2", -14.5f, -25.1f, 230.0f, 1.35f, 8.0f},
        {"nature_commontree_2", -9.3f, -28.5f, 285.0f, 1.30f, 8.0f},
        {"nature_pine_3", -3.7f, -29.8f, 340.0f, 1.45f, 8.0f},
        {"nature_pine_1", 7.7f, -33.1f, 35.0f, 1.45f, 8.0f},
        {"nature_commontree_3", 15.3f, -31.5f, 90.0f, 1.30f, 8.0f},
        {"nature_commontree_1", 17.9f, -22.9f, 145.0f, 1.35f, 8.0f},
        {"nature_pine_2", 22.2f, -18.6f, 200.0f, 1.35f, 8.0f},
        {"nature_commontree_2", 25.6f, -13.6f, 255.0f, 1.30f, 8.0f},
        {"nature_birch_1", 39.8f, -9.9f, 310.0f, 0.95f, 8.0f},
    };
    for (const Placement& p : SINGLES) {
        add(out, p);
    }
    // Outer sentinel band (r 44..49): oversized dark trees behind the grove
    // ring so aerial and long views see forest depth instead of bare slope.
    // The gaps at ~90, ~270 and ~0 degrees keep the road corridors open.
    constexpr Placement OUTER[] = {
        {"nature_pine_1", 46.0f, 9.8f, 0.0f, 1.70f, 12.0f},
        {"nature_giantpine_2", 40.2f, 17.9f, 50.0f, 1.15f, 12.0f},
        {"nature_commontree_1", 39.6f, 28.8f, 100.0f, 1.60f, 12.0f},
        {"nature_pine_2", 29.4f, 32.7f, 150.0f, 1.70f, 12.0f},
        {"nature_giantpine_1", 23.5f, 40.7f, 200.0f, 1.20f, 12.0f},
        {"nature_commontree_2", 13.6f, 41.8f, 250.0f, 1.55f, 12.0f},
        {"nature_pine_3", -12.2f, 45.4f, 300.0f, 1.75f, 12.0f},
        {"nature_pine_1", -20.7f, 38.8f, 350.0f, 1.70f, 12.0f},
        {"nature_giantpine_2", -31.5f, 37.5f, 40.0f, 1.15f, 12.0f},
        {"nature_commontree_1", -34.7f, 27.1f, 90.0f, 1.60f, 12.0f},
        // West side: the sentinel band becomes the far-bank treeline — big
        // silhouettes on the strip between the river and the map edge.
        {"nature_pine_1", -37.6f, 13.0f, 140.0f, 1.60f, 12.0f},
        {"nature_giantpine_1", -37.8f, 4.5f, 190.0f, 1.15f, 12.0f},
        {"nature_pine_3", -37.2f, -3.0f, 240.0f, 1.70f, 12.0f},
        {"nature_pine_2", -37.5f, -13.5f, 290.0f, 1.65f, 12.0f},
        {"nature_giantpine_2", -37.4f, -22.5f, 340.0f, 1.10f, 12.0f},
        {"nature_giantpine_2", -34.7f, -27.1f, 30.0f, 1.15f, 12.0f},
        {"nature_commontree_1", -31.5f, -37.5f, 80.0f, 1.60f, 12.0f},
        {"nature_pine_2", -20.7f, -38.8f, 130.0f, 1.70f, 12.0f},
        {"nature_giantpine_1", -12.9f, -45.2f, 180.0f, 1.20f, 12.0f},
        {"nature_pine_3", 9.1f, -43.0f, 230.0f, 1.75f, 12.0f},
        {"nature_commontree_2", 19.9f, -42.6f, 280.0f, 1.55f, 12.0f},
        {"nature_pine_1", 27.1f, -34.7f, 330.0f, 1.70f, 12.0f},
        {"nature_giantpine_2", 37.5f, -31.5f, 20.0f, 1.15f, 12.0f},
        {"nature_pine_2", 41.5f, -22.1f, 70.0f, 1.70f, 12.0f},
        {"nature_commontree_1", 42.5f, -11.4f, 120.0f, 1.60f, 12.0f},
    };
    for (const Placement& p : OUTER) {
        add(out, p);
    }
}

} // namespace

std::vector<Placement> build_layout() {
    Layout out;
    out.reserve(512);
    plaza(out);
    tavern(out);
    workshop(out);
    cottage(out);
    hut(out);
    plaster_house(out);
    timber_house(out);
    north_farm(out);
    east_farm(out);
    south_road(out);
    meadow(out);
    river_banks(out);
    forest(out);
    return out;
}

} // namespace playground::layout

// ---------------------------------------------------------------------------
// Asset packing (playground --build-pack out.pack)
// ---------------------------------------------------------------------------

#include "bracken/core/asset/asset_pack.hpp"

#include <algorithm>

#include <filesystem>
#include <iostream>

namespace playground {

bool build_asset_pack(const std::string& out_path) {
    namespace fs = std::filesystem;
    const fs::path models = fs::path("assets") / "models";
    if (!fs::exists(models)) {
        std::cerr << "--build-pack: no " << models.string()
                  << " directory here; run from the staged playground output dir\n";
        return false;
    }

    std::vector<std::pair<std::string, std::string>> sources;
    for (const auto& entry : fs::directory_iterator(models)) {
        if (entry.is_regular_file() && entry.path().extension() == ".glb") {
            // Logical name is the path a game would ask for, not where the
            // file happens to sit on this machine.
            sources.emplace_back("models/" + entry.path().filename().string(),
                                 entry.path().string());
        }
    }
    std::sort(sources.begin(), sources.end());

    auto r = bracken::core::asset::AssetPack::build(out_path, sources);
    if (!r) {
        std::cerr << "--build-pack failed: " << bracken::core::asset::to_string(r.error()) << "\n";
        return false;
    }
    std::cout << "packed " << sources.size() << " models into " << out_path << "\n";
    return true;
}

} // namespace playground
