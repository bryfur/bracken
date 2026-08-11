#!/usr/bin/env python3
"""Pixel-wise comparison of two screenshots.

CLAUDE.md records the same lesson twice: `mean_luma` is an aggregate over 1.44M
pixels and is structurally blind to a change that is large but localized. A
false occlusion cull removes a prop behind a wall; a leaked forward-path change
moves 328 pixels. Both read as a null row. This prints what the mean cannot:
how many pixels moved, by how much, and where.

Usage: tools/pngdiff.py <a.png> <b.png>
"""

from __future__ import annotations

import sys
import zlib
import struct


def read_png(path: str):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat, w, h, bitdepth, color = 8, b"", 0, 0, 0, 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        if ctype == b"IHDR":
            w, h, bitdepth, color = struct.unpack(">IIBB", chunk[:10])
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if bitdepth != 8:
        raise ValueError(f"{path}: only 8-bit supported (got {bitdepth})")
    nch = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw = zlib.decompress(idat)
    stride = w * nch
    out = bytearray(w * h * nch)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        ft = raw[p]
        p += 1
        line = bytearray(raw[p : p + stride])
        p += stride
        if ft == 1:
            for i in range(nch, stride):
                line[i] = (line[i] + line[i - nch]) & 0xFF
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:
            for i in range(stride):
                a = line[i - nch] if i >= nch else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = line[i - nch] if i >= nch else 0
                b = prev[i]
                c = prev[i - nch] if i >= nch else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        out[y * stride : (y + 1) * stride] = line
        prev = line
    return w, h, nch, bytes(out)


def main() -> int:
    wa, ha, ca, a = read_png(sys.argv[1])
    wb, hb, cb, b = read_png(sys.argv[2])
    if (wa, ha) != (wb, hb):
        print(f"size differs: {wa}x{ha} vs {wb}x{hb}")
        return 1

    n = wa * ha
    moved = 0
    total = 0
    worst = 0
    worst_at = (0, 0)
    # Per-eighth-of-frame rows, so a localized change says WHERE.
    bands = [0] * 8
    for i in range(n):
        pa = a[i * ca : i * ca + 3]
        pb = b[i * cb : i * cb + 3]
        d = max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1]), abs(pa[2] - pb[2]))
        if d:
            moved += 1
            total += d
            bands[(i // wa) * 8 // ha] += 1
            if d > worst:
                worst, worst_at = d, (i % wa, i // wa)

    pct = 100.0 * moved / n
    print(f"{wa}x{ha}  pixels moved: {moved} ({pct:.3f}%)")
    if moved:
        print(f"mean |delta| over moved pixels: {total / moved:.3f}/255   max: {worst} at {worst_at}")
        print("rows (top->bottom eighths): " + " ".join(str(x) for x in bands))
    else:
        print("BIT-IDENTICAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
