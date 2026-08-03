#ifndef BRACKEN_DEPTH_LINEAR_HLSLI
#define BRACKEN_DEPTH_LINEAR_HLSLI

// depth_linear.hlsli — THE engine's depth linearization, transliterated.
//
// TRANSLITERATION of contracts/depth_linear.hpp, which is the source of truth
// and carries the full derivation. Change one, change both —
// `contracts/depth_linear.test.cpp` pins the C++ half against the
// inverse-projection reference across the whole depth range.
//
// The short version. glm::perspectiveRH_ZO puts
//     a = projection[2][2] = zf / (zn - zf)
//     b = projection[3][2] = -(zf * zn) / (zf - zn)
// and clip.z / clip.w = b / t - a for a positive view depth t, so
//
//     t = b / (d + a)
//
// exactly, for standard and reversed Z alike. PLUS a, never minus: with the
// engine's swapped near/far both terms are positive, a = near / (far - near) is
// tiny, and getting that sign wrong reads as t / (1 - 2t / far) — 33% high at
// 500 units, 100% high at 1000, and pinned to one constant past far/2. Four
// traced passes carried exactly that typo until 2026-08-02.
//
// Nothing here declares a resource or reads a uniform, so any pass can include
// it (shaders/README.md — declaring an unread resource would change the
// reflected layout; a pure function cannot).

/// Device depth -> positive linear view depth. `a`/`b` are projection[2][2] and
/// projection[3][2] of the SAME (jittered) projection the depth buffer was
/// rendered with. Degenerate a == b == 0 returns 0 rather than a NaN, so a
/// depth-aware upsample whose terms were never filled degrades to plain
/// bilinear instead of poisoning its weight sum.
float brackenLinearViewDepth(float deviceDepth, float a, float b) {
    const float denom = deviceDepth + a;
    return abs(denom) > 1e-9 ? abs(b / denom) : 0.0;
}

#endif // BRACKEN_DEPTH_LINEAR_HLSLI
