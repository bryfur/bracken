#!/usr/bin/env python3
"""Build the playground's medieval-village assets from the raw asset pack.

Reads the modular building kit from  examples/playground/assets/glTF/  and
writes engine-ready artifacts to      examples/playground/assets/generated/ :

  village.glb        the whole village (buildings, plaza cobbles, vines,
                     stairs) merged into one mesh per canonical material so
                     every texture set uploads to the GPU exactly once
                     (the engine does NOT dedupe textures across meshes)
  prop_*.glb         individual props for instanced scattering at runtime
  nature_*.glb       stylized nature-pack models (assets/nature/glTF/),
                     converted 1:1 with their own materials — trees, bushes,
                     rocks, grass tufts, flowers
  textures/*.png     seamless terrain layer textures (grass/dirt/rock/sand)
                     matched to the pack's palette
  village_layout.h   constexpr layout data (plaza rect, building AABBs, path
                     waypoints, prop/chimney anchors) consumed by playground

The kit's conventions (verified from the source glTF):
  - wall modules are 2 m wide x 3 m tall, spanning x in [-1,1], y in [0,3],
    with the wall plane near z=0 (exterior face toward -Z)
  - Roof_{Flat,Round}Tiles_WxD are complete gabled roofs: gable span W along
    X, ridge length D along Z, with ~0.75 m eaves overhang on every side
  - Roof_Front_* are triangular gable infills for the +-Z ends
  - floors are 2x2 m tiles
Everything is baked at SCALE=100 (metres -> engine units, ~cm) and Y-up.

Usage:
  python3 build_medieval_assets.py            # build everything
  python3 build_medieval_assets.py --preview  # also render preview PNGs
"""

import argparse
import json
import math
import os
import struct
import sys

from PIL import Image, ImageDraw

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.normpath(os.path.join(TOOL_DIR, "..", "assets"))
SRC_DIR = os.path.join(ASSET_DIR, "glTF")
NATURE_SRC = os.path.join(ASSET_DIR, "nature", "glTF")
OUT_DIR = os.path.join(ASSET_DIR, "generated")
MODELS_DIR = os.path.join(OUT_DIR, "models")
TEXCACHE = os.path.join(OUT_DIR, ".texcache")

SCALE = 100.0  # metres -> engine units

# ---------------------------------------------------------------------------
# glTF module reading
# ---------------------------------------------------------------------------

_COMPONENT_FMT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_accessor(gltf, buffers, index):
    acc = gltf["accessors"][index]
    n_comp = _TYPE_COUNT[acc["type"]]
    comp_fmt = _COMPONENT_FMT[acc["componentType"]]
    comp_size = _COMPONENT_SIZE[acc["componentType"]]
    count = acc["count"]
    view = gltf["bufferViews"][acc["bufferView"]]
    data = buffers[view.get("buffer", 0)]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or (n_comp * comp_size)
    out = []
    for i in range(count):
        off = base + i * stride
        vals = struct.unpack_from("<" + comp_fmt * n_comp, data, off)
        out.append(vals if n_comp > 1 else vals[0])
    return out


def node_matrix(node):
    if "matrix" in node:
        m = node["matrix"]  # column-major
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    t = node.get("translation", [0, 0, 0])
    q = node.get("rotation", [0, 0, 0, 1])  # x,y,z,w
    s = node.get("scale", [1, 1, 1])
    x, y, z, w = q
    rot = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m = [[rot[r][c] * s[c] for c in range(3)] + [t[r]] for r in range(3)]
    m.append([0.0, 0.0, 0.0, 1.0])
    return m


def mat_mul(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


_module_cache = {}


def load_module(name, src_dir=None, raw_materials=False):
    """-> list of primitives: dict(material, pos[(x,y,z)], nrm, uv, idx).

    raw_materials keeps the source material names (for packs converted 1:1
    with their own materials) instead of mapping through MATERIALS.
    """
    src_dir = src_dir or SRC_DIR
    cache_key = (src_dir, name)
    if cache_key in _module_cache:
        return _module_cache[cache_key]
    path = os.path.join(src_dir, name + ".gltf")
    with open(path) as f:
        gltf = json.load(f)
    buffers = []
    for buf in gltf.get("buffers", []):
        with open(os.path.join(src_dir, buf["uri"]), "rb") as f:
            buffers.append(f.read())
    prims = []

    def emit_node(node, parent_m):
        m = mat_mul(parent_m, node_matrix(node))
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            for p in mesh["primitives"]:
                attrs = p["attributes"]
                pos = read_accessor(gltf, buffers, attrs["POSITION"])
                nrm = read_accessor(gltf, buffers, attrs["NORMAL"]) if "NORMAL" in attrs else None
                uv = read_accessor(gltf, buffers, attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs else None
                idx = read_accessor(gltf, buffers, p["indices"])
                tp, tn = [], []
                for v in pos:
                    tp.append((
                        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3],
                        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3],
                        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3],
                    ))
                if nrm:
                    for v in nrm:
                        n = (
                            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
                        )
                        l = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
                        tn.append((n[0] / l, n[1] / l, n[2] / l))
                else:
                    tn = [(0.0, 1.0, 0.0)] * len(tp)
                mat_name = gltf["materials"][p["material"]]["name"] if "material" in p else "MI_WoodTrim"
                prims.append({
                    "material": mat_name if raw_materials else canonical_material(mat_name),
                    "pos": tp,
                    "nrm": tn,
                    "uv": uv if uv else [(0.0, 0.0)] * len(tp),
                    "idx": idx,
                })
        for child in node.get("children", []):
            emit_node(gltf["nodes"][child], m)

    ident = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    scene = gltf["scenes"][gltf.get("scene", 0)]
    for ni in scene["nodes"]:
        emit_node(gltf["nodes"][ni], ident)
    _module_cache[cache_key] = prims
    return prims


# ---------------------------------------------------------------------------
# Canonical materials (the pack repeats definitions per file; some are
# duplicates under different names — merging them halves GPU texture copies)
# ---------------------------------------------------------------------------

_MATERIAL_ALIASES = {
    "MI_WoodTrim_Wear": "MI_WoodTrim",  # identical texture set + factors
    "MI_RedBrick": "MI_Brick",          # avoid a texture set for one floor tile
}

# name -> (baseColor, normal, metallicRoughness, flags)
MATERIALS = {
    "MI_WoodTrim": dict(base="T_WoodTrim_BaseColor.png", normal="T_WoodTrim_Normal.png",
                        mr="T_WoodTrim_Roughness.png"),
    "MI_Plaster": dict(base="T_Plaster_BaseColor.png", normal="T_Plaster_Normal.png",
                       mr="T_Plaster_ORM.png", orm=True),
    "MI_Brick": dict(base="T_Brick_BaseColor.png", normal="T_Brick_Normal.png",
                     mr="T_Brick_Roughness.png"),
    "MI_UnevenBrick": dict(base="T_UnevenBrick_BaseColor.png", normal="T_UnevenBrick_Normal.png",
                           mr="T_UnevenBrick_Roughness.png"),
    "MI_RockTrim": dict(base="T_RockTrim_BaseColor.png", normal="T_RockTrim_Normal.png",
                        mr="T_RockTrim_ORM.png", orm=True),
    "MI_FlatTiles": dict(base="T_FlatTiles_BaseColor.png", normal="T_FlatTiles_Normal.png",
                         mr="T_FlatTiles_Roughness.png"),
    "MI_RoundTiles": dict(base="T_RoundTiles_BaseColor.png", normal="T_RoundTiles_Normal.png",
                          mr="T_RoundTiles_Roughness.png"),
    "MI_RoundRocks": dict(base="T_RoundRocks_BaseColor.png", normal="T_RoundRocks_Normal.png",
                          mr="T_RoundRocks_Roughness.png"),
    "MI_MetalOrnaments": dict(base="T_MetalOrnaments_BaseColor.png", mr="T_MetalOrnaments_Roughness.png"),
    "MI_Vine": dict(base="T_VineLeaf_png.png", base_factor=[0.102, 0.349, 0.0, 1.0],
                    alpha_mask=True, rough=0.5),
    # Warm emissive glass: the village's windows glow at dusk (feeds bloom).
    "MI_WindowGlass": dict(base_factor=[0.9, 0.75, 0.5, 0.42], blend=True, rough=0.4,
                           emissive=[1.6, 1.05, 0.45]),
}


def canonical_material(name):
    name = _MATERIAL_ALIASES.get(name, name)
    if name not in MATERIALS:
        sys.exit(f"unknown material '{name}' — add it to MATERIALS")
    return name


# Preview colors (roughly the texture's average) for the offline renderer.
_PREVIEW_COLORS = {
    "MI_WoodTrim": (139, 103, 66), "MI_Plaster": (222, 208, 186), "MI_Brick": (158, 132, 104),
    "MI_UnevenBrick": (168, 148, 122), "MI_RockTrim": (144, 140, 132), "MI_FlatTiles": (156, 96, 70),
    "MI_RoundTiles": (172, 84, 58), "MI_RoundRocks": (150, 144, 132), "MI_MetalOrnaments": (90, 88, 92),
    "MI_Vine": (58, 122, 44), "MI_WindowGlass": (255, 220, 140),
}

# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------


class SceneBuilder:
    """Accumulates transformed kit modules into per-material buckets."""

    def __init__(self):
        self.buckets = {}  # material -> dict(pos, nrm, uv, idx)

    def place(self, module, x, y, z, rot_deg=0.0, mirror_x=False, src_dir=None, raw_materials=False):
        """Place `module` with its origin at (x,y,z) metres, rotated about Y."""
        c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        mx = -1.0 if mirror_x else 1.0
        for prim in load_module(module, src_dir=src_dir, raw_materials=raw_materials):
            b = self.buckets.setdefault(prim["material"],
                                        {"pos": [], "nrm": [], "uv": [], "idx": []})
            basev = len(b["pos"])
            for (px, py, pz) in prim["pos"]:
                px *= mx
                rx = c * px + s * pz
                rz = -s * px + c * pz
                b["pos"].append(((rx + x) * SCALE, (py + y) * SCALE, (rz + z) * SCALE))
            for (nx, ny, nz) in prim["nrm"]:
                nx *= mx
                b["nrm"].append((c * nx + s * nz, ny, -s * nx + c * nz))
            b["uv"].extend(prim["uv"])
            idx = prim["idx"]
            if mirror_x:  # mirroring flips winding
                for i in range(0, len(idx), 3):
                    b["idx"].extend((idx[i] + basev, idx[i + 2] + basev, idx[i + 1] + basev))
            else:
                b["idx"].extend(i + basev for i in idx)


# ---------------------------------------------------------------------------
# Building generator
#
# Local frame: origin = footprint centre at ground level, +Z = front (door
# side), footprint w x d metres on the 2 m grid. Gable span runs along X so
# roofs come from Roof_*Tiles_{w}x{d}; gables face +-Z.
# ---------------------------------------------------------------------------


# Wall window opening -> the matching glass insert set + open shutters.
_WINDOW_INSERTS = {
    "Window_Wide_Round": (["Window_Wide_Round1", "Window_Wide_Round2"], "WindowShutters_Wide_Round_Open"),
    "Window_Wide_Flat": (["Window_Wide_Flat1", "Window_Wide_Flat2"], "WindowShutters_Wide_Flat_Open"),
    "Window_Thin_Round": (["Window_Thin_Round1", "Window_Thin_Round2"], "WindowShutters_Thin_Round_Open"),
    "Window_Thin_Flat": (["Window_Thin_Flat1", "Window_Thin_Flat2"], "WindowShutters_Thin_Flat_Open"),
}


def window_insert_for(wall_name, i):
    """Glass insert (+ sometimes shutters) matching a window wall piece."""
    for key, (inserts, shutters) in _WINDOW_INSERTS.items():
        if key in wall_name:
            out = [inserts[i % len(inserts)]]
            if (i * 7 + 3) % 5 < 3:  # ~60% of windows get open shutters
                out.append(shutters)
            return out
    return []


def wall_ring(sb, recipe, floor, w, d, y):
    """One storey's perimeter walls. Returns nothing; mutates sb."""
    style = recipe["floors"][floor]
    hw, hd = w / 2.0, d / 2.0
    front_cells = w // 2
    side_cells = d // 2

    def pick(kind, i, n, face):
        """kind of wall piece for cell i of n on the given face."""
        if face == "front" and floor == 0 and i == n // 2:
            return None  # door cell, handled separately
        wins = style.get("windows", [])
        if wins and ((i + (0 if face in ("front", "back") else 1)) % 2 == 1):
            return wins[(i // 2) % len(wins)]
        return style["wall"]

    # The kit's decorated exterior face is the module's +Z side (verified in
    # the engine: the -Z side is bare backing). rot 0 keeps +Z facing +Z.
    def put(m, cx, cz, rot, i):
        sb.place(m, cx, y, cz, rot)
        for extra in window_insert_for(m, i):
            sb.place(extra, cx, y, cz, rot)

    for i in range(front_cells):
        cx = -hw + 1.0 + 2.0 * i
        m = pick("front", i, front_cells, "front")
        if m:
            put(m, cx, hd, 0.0, i)
        m = pick("back", i, front_cells, "back")
        if m:
            put(m, -cx, -hd, 180.0, i + 1)
    # Sides (+X outward = rot 90, -X outward = rot 270).
    for i in range(side_cells):
        cz = -hd + 1.0 + 2.0 * i
        m = pick("right", i, side_cells, "right")
        if m:
            put(m, hw, cz, 90.0, i)
        m = pick("left", i, side_cells, "left")
        if m:
            put(m, -hw, -cz, 270.0, i + 1)
    # Corner trims, decorated faces turned out toward their quadrant.
    corner = recipe.get("corner", "Corner_Exterior_Wood")
    for (cx, cz, rot) in ((hw, hd, 0.0), (hw, -hd, 90.0), (-hw, -hd, 180.0), (-hw, hd, 270.0)):
        sb.place(corner, cx, y, cz, rot)


def build_building(sb, recipe, bx, bz, brot):
    """Assemble one building; positions in metres, brot in degrees."""
    w, d = recipe["w"], recipe["d"]
    n_floors = len(recipe["floors"])

    inner = SceneBuilder()
    for f in range(n_floors):
        wall_ring(inner, recipe, f, w, d, 3.0 * f)

    # Door on the front wall's centre cell (ground floor).
    hw, hd = w / 2.0, d / 2.0
    door_cx = -hw + 1.0 + 2.0 * ((w // 2) // 2)
    style0 = recipe["floors"][0]
    inner.place(style0.get("door_wall", "Wall_Plaster_Door_Round"), door_cx, 0.0, hd, 0.0)
    inner.place(recipe.get("door_frame", "DoorFrame_Round_WoodDark"), door_cx, 0.0, hd, 0.0)
    # Door leaf: hinge at local x=0, leaf runs across x in [0, 1.07], so the
    # hinge sits at the opening's left edge (cx - 0.55).
    inner.place(recipe.get("door", "Door_1_Round"), door_cx - 0.55, 0.0, hd, 0.0)

    # Roof slab + gable infills.
    roof_y = 3.0 * n_floors
    roof = recipe.get("roof", f"Roof_FlatTiles_{w}x{d}")
    inner.place(roof, 0.0, roof_y, 0.0)
    gable = recipe.get("gable")
    if gable:
        inner.place(gable, 0.0, roof_y, hd - 0.2, 0.0)
        inner.place(gable, 0.0, roof_y, -(hd - 0.2), 180.0)

    # Chimney on the ridge, off-centre toward the back.
    if recipe.get("chimney"):
        ridge_h = roof_y + recipe.get("ridge_h", 2.4)
        inner.place(recipe["chimney"], 0.0, ridge_h - 0.4, -d * 0.22)

    # Vines on outside walls for the weathered look.
    for (vm, vx, vy, vz, vr) in recipe.get("vines", []):
        inner.place(vm, vx, vy, vz, vr)

    # Re-emit `inner` into the village-space builder (rotate + translate).
    c, s = math.cos(math.radians(brot)), math.sin(math.radians(brot))
    for mat, b in inner.buckets.items():
        out = sb.buckets.setdefault(mat, {"pos": [], "nrm": [], "uv": [], "idx": []})
        basev = len(out["pos"])
        for (px, py, pz) in b["pos"]:
            rx = c * px + s * pz
            rz = -s * px + c * pz
            out["pos"].append((rx + bx * SCALE, py, rz + bz * SCALE))
        for (nx, ny, nz) in b["nrm"]:
            out["nrm"].append((c * nx + s * nz, ny, -s * nx + c * nz))
        out["uv"].extend(b["uv"])
        out["idx"].extend(i + basev for i in b["idx"])


# ---------------------------------------------------------------------------
# The village
# ---------------------------------------------------------------------------

BUILDINGS = [
    # The tavern — big two-storey timber/brick, round-tile roof.
    dict(name="tavern", x=-5.0, z=-14.0, rot=0.0, w=6, d=10,
         floors=[
             dict(wall="Wall_WoodBrick_Straight",
                  windows=["Wall_WoodBrick_Window_Wide_Flat"],
                  door_wall="Wall_WoodBrick_Door_Flat"),
             dict(wall="Wall_Plaster_WoodGrid",
                  windows=["Wall_Plaster_Window_Wide_Round", "Wall_Plaster_Straight"]),
         ],
         corner="Corner_Exterior_Wood",
         roof="Roof_RoundTiles_6x10", gable="Roof_Front_Wood6", ridge_h=4.6,
         door_frame="DoorFrame_Flat_WoodDark", door="Door_2_Flat",
         chimney="Prop_Chimney",
         vines=[("Prop_Vine2", -2.0, 2.3, 5.05, 0.0)]),
    # Timber house east of the plaza.
    dict(name="house_timber", x=15.0, z=-4.0, rot=270.0, w=4, d=6,
         floors=[
             dict(wall="Wall_UnevenBrick_Straight",
                  windows=["Wall_UnevenBrick_Window_Thin_Round"],
                  door_wall="Wall_UnevenBrick_Door_Round"),
             dict(wall="Wall_WoodWear_Straight",
                  windows=["Wall_WoodWear_Window_Wide_Round"]),
         ],
         corner="Corner_Exterior_Wood",
         roof="Roof_FlatTiles_4x6", gable="Roof_Front_Wood4", ridge_h=2.9,
         door_frame="DoorFrame_Round_WoodDark", door="Door_3_Round",
         chimney="Prop_Chimney2"),
    # Brick cottage, single storey, south-east.
    dict(name="cottage", x=15.0, z=7.0, rot=270.0, w=6, d=6,
         floors=[
             dict(wall="Wall_UnevenBrick_Straight",
                  windows=["Wall_UnevenBrick_Window_Wide_Flat"],
                  door_wall="Wall_UnevenBrick_Door_Round"),
         ],
         corner="Corner_Exterior_Brick",
         roof="Roof_RoundTiles_6x6", gable="Roof_Front_Brick6", ridge_h=3.4,
         door_frame="DoorFrame_Round_Brick", door="Door_5_Round",
         chimney="Prop_Chimney",
         vines=[("Prop_Vine1", 0.9, 2.4, 3.05, 0.0), ("Prop_Vine4", -1.6, 2.0, 3.05, 0.0)]),
    # The workshop — long, wood-weathered, big chimney.
    dict(name="workshop", x=2.0, z=13.0, rot=180.0, w=6, d=8,
         floors=[
             dict(wall="Wall_WoodWear_Straight",
                  windows=["Wall_WoodWear_Window_Wide_Flat"],
                  door_wall="Wall_WoodWear_Door_Flat"),
         ],
         corner="Corner_Exterior_Wood",
         roof="Roof_FlatTiles_6x8", gable="Roof_Front_Wood6", ridge_h=3.4,
         door_frame="DoorFrame_Flat_WoodLight", door="Door_4_Flat",
         chimney="Prop_Chimney2"),
    # Small plaster hut, south-west.
    dict(name="hut", x=-10.0, z=11.0, rot=180.0, w=4, d=4,
         floors=[
             dict(wall="Wall_Plaster_Straight",
                  windows=["Wall_Plaster_Window_Thin_Round"]),
         ],
         corner="Corner_Exterior_Wood",
         roof="Roof_RoundTiles_4x4", gable="Roof_Front_Brick4", ridge_h=2.4,
         door_frame="DoorFrame_Round_WoodLight", door="Door_7_Round"),
    # Outlying farmsteads at the road ends — pack-built horizon interest
    # (their footprints get terrain-flattened patches in the playground).
    dict(name="farm_north", x=8.0, z=-27.0, rot=270.0, w=4, d=6,  # faces the north road
         floors=[
             dict(wall="Wall_WoodWear_Straight",
                  windows=["Wall_WoodWear_Window_Wide_Flat"],
                  door_wall="Wall_WoodWear_Door_Flat"),
         ],
         corner="Corner_Exterior_Wood",
         roof="Roof_RoundTiles_4x6", gable="Roof_Front_Wood4", ridge_h=2.9,
         door_frame="DoorFrame_Flat_WoodLight", door="Door_2_Flat",
         chimney="Prop_Chimney2"),
    dict(name="farm_east", x=28.0, z=-6.0, rot=0.0, w=4, d=6,  # faces the east road
         floors=[
             dict(wall="Wall_Plaster_Straight",
                  windows=["Wall_Plaster_Window_Thin_Round"]),
         ],
         corner="Corner_Exterior_Brick",
         roof="Roof_FlatTiles_4x6", gable="Roof_Front_Brick4", ridge_h=2.9,
         door_frame="DoorFrame_Round_WoodDark", door="Door_1_Round",
         chimney="Prop_Chimney",
         vines=[("Prop_Vine1", 0.5, 2.4, 3.05, 0.0)]),
    # Two-storey plaster house, west side.
    dict(name="house_plaster", x=-16.0, z=0.0, rot=90.0, w=4, d=8,
         floors=[
             dict(wall="Wall_Plaster_Straight_Base",
                  windows=["Wall_Plaster_Window_Wide_Flat"]),
             dict(wall="Wall_Plaster_Straight",
                  windows=["Wall_Plaster_Window_Wide_Round", "Wall_Plaster_Straight"]),
         ],
         corner="Corner_Exterior_Wood",
         roof="Roof_FlatTiles_4x8", gable="Roof_Front_Brick4", ridge_h=2.9,
         door_frame="DoorFrame_Flat_WoodDark", door="Door_6_Flat",
         chimney="Prop_Chimney",
         vines=[("Prop_Vine5", 0.9, 2.3, 4.05, 0.0)]),
]

PLAZA = dict(min_x=-12.0, min_z=-9.0, max_x=12.0, max_z=9.0)

# Cobbled streets leaving the plaza (rects of floor tiles).
STREETS = [
    dict(min_x=0.0, min_z=-19.0, max_x=4.0, max_z=-9.0),    # north, past the tavern
    dict(min_x=12.0, min_z=-2.0, max_x=20.0, max_z=2.0),    # east
    dict(min_x=-6.0, min_z=9.0, max_x=-2.0, max_z=17.0),    # south, between hut and workshop
]


def build_plaza(sb):
    """Cobblestone plaza + streets with a stone border."""
    rng = 0x9E3779B9

    def rand01():
        nonlocal rng
        rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF
        return (rng >> 8) / 16777216.0

    def paved_rect(r, accent_center=False):
        x0, z0, x1, z1 = r["min_x"], r["min_z"], r["max_x"], r["max_z"]
        nx, nz = round((x1 - x0) / 2), round((z1 - z0) / 2)
        for ix in range(nx):
            for iz in range(nz):
                cx, cz = x0 + 1.0 + 2.0 * ix, z0 + 1.0 + 2.0 * iz
                # Uniform cobbles with a deliberate brick square at the plaza
                # centre — random mixing reads as noise, not paving.
                tile = "Floor_RoundRocks"
                if accent_center and abs(cx - (x0 + x1) / 2) < 3.0 and abs(cz - (z0 + z1) / 2) < 3.0:
                    tile = "Floor_Brick"
                sb.place(tile, cx, 0.02, cz, 90.0 * int(rand01() * 4))

    paved_rect(PLAZA, accent_center=True)
    for s in STREETS:
        paved_rect(s)

    # Stone border along the plaza's free edges (skip street mouths).
    def street_gap(x, z):
        for s in STREETS:
            if s["min_x"] - 0.1 <= x <= s["max_x"] + 0.1 and s["min_z"] - 0.1 <= z <= s["max_z"] + 0.1:
                return True
        return False

    x0, z0, x1, z1 = PLAZA["min_x"], PLAZA["min_z"], PLAZA["max_x"], PLAZA["max_z"]
    for i in range(round((x1 - x0) / 2)):
        cx = x0 + 1.0 + 2.0 * i
        if not street_gap(cx, z0):
            sb.place("Prop_ExteriorBorder_Straight1", cx, 0.0, z0, 180.0)
        if not street_gap(cx, z1):
            sb.place("Prop_ExteriorBorder_Straight2", cx, 0.0, z1, 0.0)
    for i in range(round((z1 - z0) / 2)):
        cz = z0 + 1.0 + 2.0 * i
        if not street_gap(x0, cz):
            sb.place("Prop_ExteriorBorder_Straight1", x0, 0.0, cz, 270.0)
        if not street_gap(x1, cz):
            sb.place("Prop_ExteriorBorder_Straight2", x1, 0.0, cz, 90.0)


def build_village():
    sb = SceneBuilder()
    build_plaza(sb)
    for b in BUILDINGS:
        build_building(sb, b, b["x"], b["z"], b["rot"])
    return sb


# ---------------------------------------------------------------------------
# GLB writing
# ---------------------------------------------------------------------------


def resize_texture(src_name, size, mode, src_dir=None, alpha_boost=1.0):
    """Returns PNG bytes for the pack texture resized to `size` (cached).

    alpha_boost > 1 rescales the alpha channel upward after the resize:
    downscaling and mipmapping AVERAGE alpha, so thin alpha-tested features
    (leaves, grass blades) lose coverage and dissolve with distance unless
    the stored alpha is biased above the cutoff.
    """
    os.makedirs(TEXCACHE, exist_ok=True)
    tag = "n_" if src_dir else ""
    boost_tag = f"_b{int(alpha_boost * 10)}" if alpha_boost != 1.0 else ""
    cache = os.path.join(TEXCACHE, f"{tag}{os.path.splitext(src_name)[0]}_{size}_{mode}{boost_tag}.png")
    if not os.path.exists(cache):
        img = Image.open(os.path.join(src_dir or SRC_DIR, src_name)).convert(mode)
        if max(img.size) > size:
            img = img.resize((size, size), Image.LANCZOS)
        if alpha_boost != 1.0 and mode == "RGBA":
            r, g, b, a = img.split()
            a = a.point(lambda v: min(255, int(v * alpha_boost)))
            img = Image.merge("RGBA", (r, g, b, a))
        img.save(cache, optimize=True)
    with open(cache, "rb") as f:
        return f.read()


def write_glb(path, buckets, tex_base=1024, tex_normal=512, tex_mr=512, material_specs=None, src_dir=None):
    material_specs = material_specs or MATERIALS
    """Write per-material buckets as a single-mesh, multi-primitive GLB."""
    bin_parts = []
    views = []
    accessors = []
    images = []
    textures = []
    materials = []
    prims = []
    image_cache = {}

    def add_view(data, target=None):
        offset = sum(len(p) for p in bin_parts)
        pad = (4 - offset % 4) % 4
        if pad:
            bin_parts.append(b"\x00" * pad)
            offset += pad
        bin_parts.append(data)
        views.append({k: v for k, v in dict(buffer=0, byteOffset=offset, byteLength=len(data),
                                            target=target).items() if v is not None})
        return len(views) - 1

    def add_accessor(view, comp_type, count, type_, minmax=None):
        acc = dict(bufferView=view, componentType=comp_type, count=count, type=type_)
        if minmax:
            acc["min"], acc["max"] = minmax
        accessors.append(acc)
        return len(accessors) - 1

    def add_texture(src_name, size, mode, alpha_boost=1.0):
        key = (src_name, size, mode, alpha_boost)
        if key not in image_cache:
            png = resize_texture(src_name, size, mode, src_dir=src_dir, alpha_boost=alpha_boost)
            view = add_view(png)
            images.append(dict(mimeType="image/png", bufferView=view,
                               name=os.path.splitext(src_name)[0]))
            textures.append(dict(sampler=0, source=len(images) - 1))
            image_cache[key] = len(textures) - 1
        return image_cache[key]

    for mat_name in sorted(buckets.keys()):
        b = buckets[mat_name]
        spec = material_specs[mat_name]
        pos, nrm, uv, idx = b["pos"], b["nrm"], b["uv"], b["idx"]

        pos_data = b"".join(struct.pack("<3f", *p) for p in pos)
        nrm_data = b"".join(struct.pack("<3f", *n) for n in nrm)
        uv_data = b"".join(struct.pack("<2f", *t) for t in uv)
        idx_data = b"".join(struct.pack("<I", i) for i in idx)
        mins = [min(p[i] for p in pos) for i in range(3)]
        maxs = [max(p[i] for p in pos) for i in range(3)]
        a_pos = add_accessor(add_view(pos_data, 34962), 5126, len(pos), "VEC3", (mins, maxs))
        a_nrm = add_accessor(add_view(nrm_data, 34962), 5126, len(nrm), "VEC3")
        a_uv = add_accessor(add_view(uv_data, 34962), 5126, len(uv), "VEC2")
        a_idx = add_accessor(add_view(idx_data, 34963), 5125, len(idx), "SCALAR")

        pbr = dict(metallicFactor=1.0 if spec.get("metal") else 0.0,
                   roughnessFactor=spec.get("rough", 1.0))
        if spec.get("base_factor"):
            pbr["baseColorFactor"] = spec["base_factor"]
        if spec.get("base"):
            # Alpha-tested foliage keeps full resolution and a boosted alpha
            # channel so coverage survives downscaling + engine mipmaps.
            if spec.get("alpha_mask"):
                pbr["baseColorTexture"] = dict(index=add_texture(spec["base"], max(tex_base, 1024), "RGBA",
                                                                 alpha_boost=1.4))
            else:
                pbr["baseColorTexture"] = dict(index=add_texture(spec["base"], tex_base, "RGBA"))
        if spec.get("mr"):
            pbr["metallicRoughnessTexture"] = dict(index=add_texture(spec["mr"], tex_mr, "RGB"))
        mat = dict(name=mat_name, pbrMetallicRoughness=pbr, doubleSided=True)
        if spec.get("normal"):
            mat["normalTexture"] = dict(index=add_texture(spec["normal"], tex_normal, "RGB"))
        if spec.get("orm"):
            mat["occlusionTexture"] = dict(index=pbr["metallicRoughnessTexture"]["index"])
        if spec.get("emissive"):
            mat["emissiveFactor"] = [1.0, 1.0, 1.0]
            mat["extensions"] = dict(KHR_materials_emissive_strength=dict(emissiveStrength=1.0))
            # Bake the color into emissiveFactor directly (max 1.0 per spec
            # without the extension; the engine applies the strength multiplier).
            e = spec["emissive"]
            strength = max(e)
            mat["emissiveFactor"] = [c / strength for c in e]
            mat["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] = strength
        if spec.get("alpha_mask"):
            mat["alphaMode"] = "MASK"
            # Below the glTF-default 0.5: the engine's box-filtered mips
            # average alpha down, and a lower cutoff keeps distant foliage
            # from dissolving.
            mat["alphaCutoff"] = spec.get("alpha_cutoff", 0.35)
        if spec.get("blend"):
            mat["alphaMode"] = "BLEND"
        materials.append(mat)

        prims.append(dict(attributes=dict(POSITION=a_pos, NORMAL=a_nrm, TEXCOORD_0=a_uv),
                          indices=a_idx, material=len(materials) - 1))

    gltf = dict(
        asset=dict(version="2.0", generator="bracken build_medieval_assets"),
        scene=0,
        scenes=[dict(nodes=[0])],
        nodes=[dict(mesh=0, name=os.path.splitext(os.path.basename(path))[0])],
        meshes=[dict(primitives=prims)],
        materials=materials,
        samplers=[dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)],
        bufferViews=views,
        accessors=accessors,
        extensionsUsed=["KHR_materials_emissive_strength"],
    )
    if images:
        gltf["images"] = images
        gltf["textures"] = textures

    bin_chunk = b"".join(bin_parts)
    bin_chunk += b"\x00" * ((4 - len(bin_chunk) % 4) % 4)
    gltf["buffers"] = [dict(byteLength=len(bin_chunk))]
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)

    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        f.write(json_chunk)
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        f.write(bin_chunk)
    tris = sum(len(b["idx"]) for b in buckets.values()) // 3
    verts = sum(len(b["pos"]) for b in buckets.values())
    print(f"  {os.path.relpath(path, ASSET_DIR):32s} {len(buckets):2d} meshes "
          f"{verts:7d} verts {tris:7d} tris {os.path.getsize(path) / 1e6:6.2f} MB")


# ---------------------------------------------------------------------------
# Props
# ---------------------------------------------------------------------------

PROPS = {
    "prop_crate": ["Prop_Crate"],
    "prop_wagon": ["Prop_Wagon"],
    "prop_fence": ["Prop_WoodenFence_Single"],
    "prop_fence_ext": ["Prop_WoodenFence_Extension1"],
    "prop_bricks": ["Prop_Brick1", "Prop_Brick2", "Prop_Brick3"],
}


def build_props():
    for out_name, modules in PROPS.items():
        sb = SceneBuilder()
        for m in modules:
            sb.place(m, 0.0, 0.0, 0.0)
        write_glb(os.path.join(MODELS_DIR, out_name + ".glb"), sb.buckets,
                  tex_base=512, tex_normal=512, tex_mr=256)


# ---------------------------------------------------------------------------
# Terrain textures — tileable value noise in the pack's palette
# ---------------------------------------------------------------------------


def tileable_noise(size, cell, seed):
    rng = seed

    def rand01():
        nonlocal rng
        rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF
        return (rng >> 8) / 16777216.0

    n = size // cell
    grid = [[rand01() for _ in range(n)] for _ in range(n)]
    img = [[0.0] * size for _ in range(size)]
    for y in range(size):
        gy, fy = divmod(y, cell)
        ty = fy / cell
        ty = ty * ty * (3 - 2 * ty)
        for x in range(size):
            gx, fx = divmod(x, cell)
            tx = fx / cell
            tx = tx * tx * (3 - 2 * tx)
            v00 = grid[gy % n][gx % n]
            v10 = grid[gy % n][(gx + 1) % n]
            v01 = grid[(gy + 1) % n][gx % n]
            v11 = grid[(gy + 1) % n][(gx + 1) % n]
            img[y][x] = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty
    return img


def make_terrain_texture(path, base, dark, size=256, seed=1):
    coarse = tileable_noise(size, size // 8, seed)
    fine = tileable_noise(size, size // 32, seed * 7 + 3)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = 0.65 * coarse[y][x] + 0.35 * fine[y][x]
            px[x, y] = tuple(int(dark[i] + (base[i] - dark[i]) * t) for i in range(3))
    img.save(path, optimize=True)
    print(f"  {os.path.relpath(path, ASSET_DIR)}")


NATURE_PROPS = [
    # trees
    "CommonTree_1", "CommonTree_2", "CommonTree_3", "Birch_1", "Birch_2",
    "Pine_1", "Pine_2", "Pine_3", "GiantPine_1", "GiantPine_2",
    # understory
    "Bush_Common", "Bush_Common_Flowers", "Bush_Large", "Fern_2",
    # rocks
    "Rock_Medium_1", "Rock_Medium_2", "Rock_Medium_3", "Rock_Big_1",
    "Rock_Medium_4", "Rock_Big_2",
    # grass tufts + ground flora
    "Grass_Common_Short", "Grass_Common_Tall", "Grass_Wispy_Short",
    "Grass_Wispy_Tall", "Grass_Wide_Short", "Grass_Wide_Tall", "Grass_Wheat",
    "Clover_1", "Clover_2", "Plant_1", "Plant_2", "Plant_3", "Bush_Long_1",
    "Flower_1_Group", "Flower_2_Group", "Flower_3_Group", "Flower_4_Group",
    "Flower_6", "Flower_7_Group",
    # mushrooms + pebbles
    "Mushroom_Common", "Mushroom_RedCap",
    "Pebble_Round_1", "Pebble_Round_2", "Pebble_Round_3",
    "Pebble_Square_1", "Pebble_Square_2",
]


def build_nature_props():
    """Convert stylized nature-pack models 1:1 into engine .glb props.

    Unlike the building kit these keep their own materials (base color +
    normal + MASK alpha, doubleSided); geometry is baked x100 like everything
    else. Output name: nature_<lowercased model name>.glb.
    """
    for name in NATURE_PROPS:
        src = os.path.join(NATURE_SRC, name + ".gltf")
        with open(src) as f:
            gltf = json.load(f)
        texs = gltf.get("textures", [])
        imgs = gltf.get("images", [])

        def tex_uri(idx):
            if idx is None or idx < 0 or idx >= len(texs):
                return None
            src_i = texs[idx].get("source", -1)
            return imgs[src_i]["uri"] if 0 <= src_i < len(imgs) else None

        materials = {}
        for m in gltf.get("materials", []):
            pbr = m.get("pbrMetallicRoughness", {})
            spec = dict(rough=pbr.get("roughnessFactor", 1.0))
            base = tex_uri(pbr.get("baseColorTexture", {}).get("index"))
            if base:
                spec["base"] = base
            if "baseColorFactor" in pbr:
                spec["base_factor"] = pbr["baseColorFactor"]
            normal = tex_uri(m.get("normalTexture", {}).get("index"))
            if normal:
                spec["normal"] = normal
            if m.get("alphaMode") == "MASK":
                spec["alpha_mask"] = True
            elif m.get("alphaMode") == "BLEND":
                spec["blend"] = True
            materials[m["name"]] = spec

        sb = SceneBuilder()
        sb.place(name, 0.0, 0.0, 0.0, src_dir=NATURE_SRC, raw_materials=True)
        out = os.path.join(MODELS_DIR, f"nature_{name.lower()}.glb")
        write_glb(out, sb.buckets, tex_base=512, tex_normal=512, tex_mr=256,
                  material_specs=materials, src_dir=NATURE_SRC)


def _lerp3(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def make_grass_texture(path, size=512, seed=11):
    """Meadow ground for a texture that tiles EVERY METRE (terrain TEX_SCALE
    0.01): features stay at grass-clump scale (2-6 cm) and moderate contrast,
    or the 1 m repeat reads as a checkerboard. Broad variation comes from the
    shader's dual-scale anti-tiling blend and the vertex tint, not from here.
    """
    fine = tileable_noise(size, size // 40, seed)            # ~2.5 cm blade clumps
    med = tileable_noise(size, size // 16, seed * 5 + 1)     # ~6 cm tufting
    warm = tileable_noise(size, size // 12, seed * 11 + 7)   # small dry specks
    # Domain warp: value noise interpolated on a grid reads as axis-aligned
    # soft squares ("checkerboard") — jittering the lookup coordinates by two
    # more noise fields breaks the grid alignment into organic clumps.
    warp_x = tileable_noise(size, size // 10, seed * 3 + 5)
    warp_y = tileable_noise(size, size // 10, seed * 7 + 9)
    dark = (82, 104, 50)
    mid = (106, 128, 60)
    light = (126, 144, 68)
    warm_c = (138, 136, 58)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            wx = int(x + (warp_x[y][x] - 0.5) * 38.0) % size
            wy = int(y + (warp_y[y][x] - 0.5) * 38.0) % size
            t = 0.55 * fine[wy][wx] + 0.45 * med[wy][wx]
            if t < 0.55:
                c = _lerp3(dark, mid, t / 0.55)
            else:
                c = _lerp3(mid, light, (t - 0.55) / 0.45)
            w = warm[wy][wx]
            if w > 0.78:  # sparse dry blades
                c = _lerp3(c, warm_c, min(1.0, (w - 0.78) / 0.12) * 0.5)
            px[x, y] = tuple(int(v) for v in c)
    img.save(path, optimize=True)
    print(f"  {os.path.relpath(path, ASSET_DIR)}")


def make_dirt_texture(path, size=512, seed=23):
    """Cracked packed earth: jittered-grid voronoi cell edges as crack lines."""
    rng = seed

    def rand01():
        nonlocal rng
        rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF
        return (rng >> 8) / 16777216.0

    cell = size // 9  # crack cell size (px); grid wraps so the texture tiles
    n = size // cell
    seeds = [[(cx * cell + rand01() * cell, cy * cell + rand01() * cell) for cx in range(n)] for cy in range(n)]
    mottle = tileable_noise(size, size // 7, seed * 3 + 5)
    fine = tileable_noise(size, size // 40, seed * 13 + 1)
    base = (164, 134, 94)
    dark = (118, 94, 64)
    crack = (96, 76, 52)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        cy = y // cell
        for x in range(size):
            cx = x // cell
            d1 = d2 = 1e18  # nearest / second-nearest seed distance (torus)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    sx, sy = seeds[(cy + oy) % n][(cx + ox) % n]
                    # unwrapped seed position for correct distances at borders
                    if cx + ox < 0:
                        sx -= size
                    elif cx + ox >= n:
                        sx += size
                    if cy + oy < 0:
                        sy -= size
                    elif cy + oy >= n:
                        sy += size
                    d = (x - sx) ** 2 + (y - sy) ** 2
                    if d < d1:
                        d2 = d1
                        d1 = d
                    elif d < d2:
                        d2 = d
            edge = d2 ** 0.5 - d1 ** 0.5  # ~0 along cell borders
            t = 0.6 * mottle[y][x] + 0.4 * fine[y][x]
            c = _lerp3(dark, base, t)
            if edge < 3.0:  # crack line with soft falloff
                c = _lerp3(crack, c, edge / 3.0)
            px[x, y] = tuple(int(v) for v in c)
    img.save(path, optimize=True)
    print(f"  {os.path.relpath(path, ASSET_DIR)}")


def build_terrain_textures():
    tex_dir = os.path.join(OUT_DIR, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    # Palette tuned to sit under the pack's stylized flora: sunlit clumpy
    # meadow, cracked packed earth for the lanes, grey rock, tan sand.
    make_grass_texture(os.path.join(tex_dir, "grass_seamless.png"))
    make_dirt_texture(os.path.join(tex_dir, "dirt_seamless.png"))
    # NB: all four layers must share dimensions or TerrainRenderer rejects
    # the whole array and falls back to 64x64 procedural noise (a visible
    # checkerboard at the 1 m tiling scale).
    make_terrain_texture(os.path.join(tex_dir, "rock_seamless.png"), (138, 134, 126), (84, 82, 78), size=512, seed=37)
    make_terrain_texture(os.path.join(tex_dir, "sand_seamless.png"), (188, 164, 118), (140, 118, 82), size=512, seed=51)


# ---------------------------------------------------------------------------
# Layout header for the playground
# ---------------------------------------------------------------------------


def building_aabb(b):
    """World-space (metres) footprint AABB after rotation."""
    hw, hd = b["w"] / 2.0 + 0.4, b["d"] / 2.0 + 0.4
    if b["rot"] % 180 != 0:
        hw, hd = hd, hw
    return (b["x"] - hw, b["z"] - hd, b["x"] + hw, b["z"] + hd)


def chimney_top(b):
    """World position (metres) of the chimney mouth, or None."""
    if not b.get("chimney"):
        return None
    n_floors = len(b["floors"])
    lx, lz = 0.0, -b["d"] * 0.22
    c, s = math.cos(math.radians(b["rot"])), math.sin(math.radians(b["rot"]))
    wx = c * lx + s * lz + b["x"]
    wz = -s * lx + c * lz + b["z"]
    return (wx, 3.0 * n_floors + b.get("ridge_h", 2.4) + 2.4, wz)


def write_layout_header():
    lines = [
        "// Generated by tools/build_medieval_assets.py — do not edit.",
        "// Coordinates are in kit metres relative to the village centre;",
        "// the playground multiplies by VILLAGE_SCALE and offsets to world.",
        "#pragma once",
        "",
        "namespace playground::village {",
        "",
        f"inline constexpr float VILLAGE_SCALE = {SCALE}f; // metres -> world units",
        "",
        "struct Rect { float min_x, min_z, max_x, max_z; };",
        "struct Vec3 { float x, y, z; };",
        "",
        f"inline constexpr Rect PLAZA{{{PLAZA['min_x']}f, {PLAZA['min_z']}f, "
        f"{PLAZA['max_x']}f, {PLAZA['max_z']}f}};",
        "",
        "inline constexpr Rect STREETS[] = {",
    ]
    for s in STREETS:
        lines.append(f"    {{{s['min_x']}f, {s['min_z']}f, {s['max_x']}f, {s['max_z']}f}},")
    lines += ["};", "", "inline constexpr Rect BUILDINGS[] = {"]
    for b in BUILDINGS:
        x0, z0, x1, z1 = building_aabb(b)
        lines.append(f"    {{{x0:.2f}f, {z0:.2f}f, {x1:.2f}f, {z1:.2f}f}}, // {b['name']}")
    lines += ["};", "", "inline constexpr Vec3 CHIMNEYS[] = {"]
    for b in BUILDINGS:
        top = chimney_top(b)
        if top:
            lines.append(f"    {{{top[0]:.2f}f, {top[1]:.2f}f, {top[2]:.2f}f}}, // {b['name']}")
    lines += ["};", "",
              "// Door world positions (metres) — orb path anchors.",
              "inline constexpr Vec3 DOORS[] = {"]
    for b in BUILDINGS:
        hw = b["w"] / 2.0
        door_cx = -hw + 1.0 + 2.0 * ((b["w"] // 2) // 2)
        lz = b["d"] / 2.0 + 1.0
        c, s = math.cos(math.radians(b["rot"])), math.sin(math.radians(b["rot"]))
        wx = c * door_cx + s * lz + b["x"]
        wz = -s * door_cx + c * lz + b["z"]
        lines.append(f"    {{{wx:.2f}f, 0.0f, {wz:.2f}f}}, // {b['name']}")
    lines += ["};", "", "} // namespace playground::village", ""]
    path = os.path.join(OUT_DIR, "village_layout.h")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  {os.path.relpath(path, ASSET_DIR)}")


# ---------------------------------------------------------------------------
# Preview renderer (painter's algorithm, orthographic) — sanity-check assembly
# without an engine round-trip.
# ---------------------------------------------------------------------------


def render_preview(buckets, path, size=1400):
    tris = []
    all_pts = []
    for mat, b in buckets.items():
        color = _PREVIEW_COLORS.get(mat, (200, 0, 200))
        pos, idx = b["pos"], b["idx"]
        for i in range(0, len(idx), 3):
            tris.append((color, pos[idx[i]], pos[idx[i + 1]], pos[idx[i + 2]]))
    # Isometric-ish: rotate 40deg about Y, tilt 30deg down.
    ay, ax = math.radians(40), math.radians(-32)
    cy, sy = math.cos(ay), math.sin(ay)
    cx, sx = math.cos(ax), math.sin(ax)
    sun = (0.5, 0.8, 0.3)

    def project(p):
        x = cy * p[0] + sy * p[2]
        z = -sy * p[0] + cy * p[2]
        y = cx * p[1] - sx * z
        depth = sx * p[1] + cx * z
        return x, y, depth

    proj = []
    for color, a, b_, c in tris:
        pa, pb, pc = project(a), project(b_), project(c)
        # Face normal for flat shading (world space).
        u = tuple(b_[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        l = math.sqrt(sum(q * q for q in n)) or 1.0
        lam = abs(sum(n[i] / l * sun[i] for i in range(3)))
        shade = 0.45 + 0.55 * lam
        col = tuple(min(255, int(cq * shade)) for cq in color)
        depth = (pa[2] + pb[2] + pc[2]) / 3.0
        proj.append((depth, col, (pa[0], pa[1]), (pb[0], pb[1]), (pc[0], pc[1])))
        all_pts += [pa, pb, pc]
    if not proj:
        return
    min_x = min(p[0] for p in all_pts); max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts); max_y = max(p[1] for p in all_pts)
    span = max(max_x - min_x, max_y - min_y) or 1.0
    scale = (size - 40) / span
    img = Image.new("RGB", (size, size), (36, 40, 48))
    draw = ImageDraw.Draw(img)
    proj.sort(key=lambda t: t[0])  # far first
    for _, col, a, b_, c in proj:
        pts = [((p[0] - min_x) * scale + 20, size - ((p[1] - min_y) * scale + 20)) for p in (a, b_, c)]
        draw.polygon(pts, fill=col)
    img.save(path)
    print(f"  preview: {os.path.relpath(path, ASSET_DIR)}")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="render preview PNGs")
    ap.add_argument("--only-preview", action="store_true", help="skip glb/texture output")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("assembling village...")
    village = build_village()
    if args.preview or args.only_preview:
        render_preview(village.buckets, os.path.join(OUT_DIR, "preview_village.png"))
        single = SceneBuilder()
        build_building(single, BUILDINGS[0], 0.0, 0.0, 0.0)
        render_preview(single.buckets, os.path.join(OUT_DIR, "preview_tavern.png"), size=1000)
    if args.only_preview:
        return
    print("writing GLBs...")
    write_glb(os.path.join(MODELS_DIR, "village.glb"), village.buckets)
    build_props()
    print("nature props...")
    build_nature_props()
    print("terrain textures...")
    build_terrain_textures()
    write_layout_header()
    print("done.")


if __name__ == "__main__":
    main()
