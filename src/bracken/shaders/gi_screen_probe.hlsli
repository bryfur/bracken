// gi_screen_probe.hlsli - the screen-probe gather's constants, indexing,
// directions, march step and L1 spherical harmonics, shared by the six passes
// of phase R4b (docs/rt_lighting_plan.md §5.5) and by the forward sampler.
//
// EVERY function here is a transliteration of its counterpart in
// `src/bracken/render/gi/screen_probe_math.hpp`, which is the source of truth
// and is tested with no GPU, no window and no SDL. Same contract as
// gi_probe.hlsli / gi_math.hpp: when one side changes, the other changes in the
// same commit.
//
// Binding-free by construction — no resource is named below — so it can be
// included before or after a shader's binding block.

#ifndef BRACKEN_GI_SCREEN_PROBE_HLSLI
#define BRACKEN_GI_SCREEN_PROBE_HLSLI

// Include-guarded and declares no resource, so taking it here costs a consumer
// nothing (same argument rt_denoise.hlsli makes for rt_sampling.hlsli).
#include "depth_linear.hlsli"

// ---------------------------------------------------------------------------
// The grid — mirrors screen_probe_math.hpp
// ---------------------------------------------------------------------------

#define GI_SP_TILE 16
#define GI_SP_OCT 8
#define GI_SP_RAYS (GI_SP_OCT * GI_SP_OCT)
// float4 quads in a GiScreenProbeSh record: three of SH, one of metadata.
#define GI_SP_SH_QUADS 4

#define GI_SP_FLAG_VALID 1u
#define GI_SP_ANCHOR_SHIFT 8

// Alpha < 0 in the radiance atlas means "no sample resolved for this texel this
// frame" — a miss record the capacity clamp dropped. The filter keeps history
// for such a texel rather than folding a black sample in.
#define GI_SP_UNRESOLVED (-1.0)

#define GI_SP_ANCHOR_ATTEMPTS 9

#define GI_SP_START_MIP 2
#define GI_SP_MAX_STEPS 48
#define GI_SP_THICKNESS_MIN 50.0
#define GI_SP_THICKNESS_REL 0.05
#define GI_SP_WORLD_RAY_CLAMP 600.0

#define GI_SP_TEMPORAL_ALPHA 0.1
#define GI_SP_FIREFLY_MAX 8.0
#define GI_SP_DEPTH_TOLERANCE 0.05
#define GI_SP_NORMAL_POWER 8.0
#define GI_SP_FORWARD_NORMAL_POWER 4.0
#define GI_SP_WEIGHT_FLOOR 0.5

#define GI_SP_PI 3.141592653589793
// THE OCTAHEDRAL MAP IS NOT EQUAL-AREA. Square -> octahedron preserves area,
// but the radial octahedron -> sphere step has Jacobian dOmega = dA / |v|^3
// with v the UN-normalized folded vector, whose length runs from 1 at the six
// octahedron vertices to 1/sqrt(3) at the face centres. Over the 8x8 texels the
// true solid angle spans 0.54x .. 1.46x of 4*pi/64, world-axis texels being the
// over-weighted ones — so a constant weight tilts every probe's L1 lobe toward
// the world axes. The SUM is right, which is why a uniform sphere or hemisphere
// reproduces exactly and hid this. Below: the texel's area in the [-1,1]^2
// square, which `giSpTexelDirectionSolidAngle` divides by |v|^3.
#define GI_SP_TEXEL_AREA ((2.0 / float(GI_SP_OCT)) * (2.0 / float(GI_SP_OCT)))

/// One probe per GI_SP_TILE-square tile; a partial edge tile still gets one.
int2 giSpGrid(int2 screenSize) {
    return int2((screenSize.x + GI_SP_TILE - 1) / GI_SP_TILE,
                (screenSize.y + GI_SP_TILE - 1) / GI_SP_TILE);
}

int giSpIndex(int2 probe, int2 grid) { return probe.y * grid.x + probe.x; }

int2 giSpCoord(int index, int2 grid) { return int2(index % grid.x, index / grid.x); }

// ---------------------------------------------------------------------------
// Anchor placement — the FIXED search order, not a search. See
// screen_probe_math.hpp for why a per-tile depth histogram is the wrong answer.
// ---------------------------------------------------------------------------

int2 giSpAnchorOffset(int attempt) {
    const int H = GI_SP_TILE / 2;
    const int Q = GI_SP_TILE / 4;
    const int L = GI_SP_TILE - 1;
    const int T = GI_SP_TILE - 1 - Q;
    switch (attempt) {
    case 0: return int2(H, H);
    case 1: return int2(Q, Q);
    case 2: return int2(T, Q);
    case 3: return int2(Q, T);
    case 4: return int2(T, T);
    case 5: return int2(0, H);
    case 6: return int2(L, H);
    case 7: return int2(H, 0);
    default: return int2(H, L);
    }
}

uint giSpPackFlags(int2 anchorLocal, bool valid) {
    uint x = uint(anchorLocal.x & 0xF);
    uint y = uint(anchorLocal.y & 0xF);
    return (valid ? GI_SP_FLAG_VALID : 0u) | (x << GI_SP_ANCHOR_SHIFT) |
           (y << (GI_SP_ANCHOR_SHIFT + 4));
}

int2 giSpUnpackAnchor(uint flags) {
    return int2(int((flags >> GI_SP_ANCHOR_SHIFT) & 0xFu),
                int((flags >> (GI_SP_ANCHOR_SHIFT + 4)) & 0xFu));
}

// ---------------------------------------------------------------------------
// Octahedral mapping — the SAME convention as gi_probe.hlsli's giOctDecode and
// prepass_gbuffer.frag.hlsl's octEncode (Cigolle et al. 2014). Spelled out here
// rather than included so this file stays free of the world field's grid
// constants, which none of the screen passes need.
// ---------------------------------------------------------------------------

/// The decode WITHOUT the normalize: a point on |x|+|y|+|z| = 1, whose length is
/// the Jacobian the solid angle above is built from. Split out for that reason
/// alone — every caller that only wants a direction takes `giSpOctDecode`.
float3 giSpOctDecodeRaw(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, float2(-t, -t), float2(t, t));
    return n;
}

float3 giSpOctDecode(float2 e) { return normalize(giSpOctDecodeRaw(e)); }

/// Per-frame sub-texel jitter, R2 sequence in the frame counter alone.
/// Deterministic in `frameIndex` is the requirement: --smoke reproduces bit for
/// bit and the temporal EMA is what integrates the texel's solid angle.
float2 giSpJitter(uint frameIndex) {
    const float R2_A = 0.7548776662466927;
    const float R2_B = 0.5698402909980532;
    float f = float(frameIndex);
    // +0.5 so frame 0 is the texel CENTRE, not its corner: (0,0) lands on the
    // octahedral fold seam, and a whole probe's directions sitting at their
    // texels' corners is a one-sided set whose L1 projection is biased.
    return float2(frac(0.5 + R2_A * f), frac(0.5 + R2_B * f));
}

/// World-space direction of octahedral texel `texel`. FULL SPHERE, in world
/// space: one probe serves a whole tile and a tile's pixels do not share a
/// normal. Only the hemisphere above the anchor is actually traced.
float3 giSpTexelDirection(int texel, uint frameIndex) {
    int tx = texel % GI_SP_OCT;
    int ty = texel / GI_SP_OCT;
    float2 oct = (float2(float(tx), float(ty)) + giSpJitter(frameIndex)) / float(GI_SP_OCT);
    return giSpOctDecode(oct);
}

/// The same direction, plus the SOLID ANGLE that texel subtends this frame —
/// the quadrature weight its radiance must carry into an integral over the
/// sphere. Evaluated at the JITTERED sample rather than the texel centre, which
/// is the unbiased choice: E[dA/|v(x_f)|^3] over the jitter sequence is the
/// texel's true solid angle, and the temporal EMA is what takes that
/// expectation. CPU mirror: `screen_probe_texel_solid_angle`.
float3 giSpTexelDirectionSolidAngle(int texel, uint frameIndex, out float solidAngle) {
    int tx = texel % GI_SP_OCT;
    int ty = texel / GI_SP_OCT;
    float2 oct = (float2(float(tx), float(ty)) + giSpJitter(frameIndex)) / float(GI_SP_OCT);
    float3 v = giSpOctDecodeRaw(oct);
    float len = length(v);
    solidAngle = GI_SP_TEXEL_AREA / (len * len * len);
    return v / len;
}

// ---------------------------------------------------------------------------
// Atlas addressing
// ---------------------------------------------------------------------------

int2 giSpRadianceTexel(int2 probe, int texel) {
    return int2(probe.x * GI_SP_OCT + (texel % GI_SP_OCT), probe.y * GI_SP_OCT + (texel / GI_SP_OCT));
}


// ---------------------------------------------------------------------------
// The hierarchical-Z march
// ---------------------------------------------------------------------------

/// The level-`level` cell grid, in cells per unit UV: `screenSize / 2^level`,
/// NOT `hzbMipSize(level)`.
///
/// The pyramid's footprints are NOT uniform — render/hzb.hpp extends the LAST
/// destination texel of each axis to the end of the source so an odd dimension's
/// trailing row is not dropped, so a 72-row image's mip 4 has footprints of
/// 16, 16, 16 and 24 rows. Addressing it with floor(uv * 4) asks the wrong texel
/// above uv 0.667, and the march then walks through whatever that texel was
/// reporting. CPU mirror: `screen_trace_cell_grid`, and the test that found it.
float2 giSpCellGrid(int2 screenSize, int level) {
    return float2(screenSize) / float(1 << level);
}

/// The pyramid texel a UV addresses at `level`. Clamped for the LOOKUP only —
/// the step function keeps the unclamped grid, so a ray in the oversized last
/// cell leaves the screen rather than stalling against a boundary that is not
/// there.
int2 giSpCell(float2 uv, float2 cells, int2 mipSize) {
    return clamp(int2(floor(uv * cells)), int2(0, 0), max(mipSize - 1, int2(0, 0)));
}

/// Distance along a screen-space ray to the next cell boundary of a
/// `cellCount`-resolution grid. `p` is UV, `d` is the ray's WHOLE UV extent, so
/// the result is in the ray's own [0,1] parameterization.
///
/// CPU mirror: `screen_trace_cell_step`, checked against a brute-force scan.
/// An axis with d == 0 never crosses and yields infinity; both zero yields
/// infinity, which the caller must treat as "cannot advance".
float giSpCellStep(float2 p, float2 d, float2 cellCount) {
    float2 cell = floor(p * cellCount);
    float2 plane = cell + float2(d.x >= 0.0 ? 1.0 : 0.0, d.y >= 0.0 ? 1.0 : 0.0);
    float2 target = plane / cellCount;
    const float INF = 3.402823466e+38;
    float tx = abs(d.x) > 1e-12 ? (target.x - p.x) / d.x : INF;
    float ty = abs(d.y) > 1e-12 ? (target.y - p.y) / d.y : INF;
    return min(max(tx, 0.0), max(ty, 0.0));
}

/// Assumed thickness of a screen surface in linear view depth: a floor plus a
/// fraction of the distance. See screen_probe_math.hpp for the tunnelling /
/// shadow-volume trade this sits between.
float giSpThickness(float linearDepth) {
    return max(GI_SP_THICKNESS_MIN, abs(linearDepth) * GI_SP_THICKNESS_REL);
}

/// Reversed-Z device depth -> positive linear view depth, from the same
/// projection[2][2] / projection[3][2] pair the other traced passes carry.
float giSpLinearDepth(float deviceDepth, float a, float b) {
    // depth_linear.hlsli is the formula's one home. The only difference is the
    // degenerate case: a screen trace wants a very large distance there, not 0,
    // so a caller that forgot to test for sky gets an obviously wrong hit
    // instead of one at the eye. CPU mirror: screen_trace_linear_depth.
    float denom = deviceDepth + a;
    return abs(denom) < 1e-9 ? 1e9 : brackenLinearViewDepth(deviceDepth, a, b);
}

// ---------------------------------------------------------------------------
// L1 spherical harmonics
// ---------------------------------------------------------------------------

/// (Y00, Y1-1, Y10, Y11) at `n`, in the order the SH atlas stores them.
float4 giSpShBasis(float3 n) {
    const float Y0 = 0.28209479177387814;
    const float Y1 = 0.4886025119029199;
    return float4(Y0, Y1 * n.y, Y1 * n.z, Y1 * n.x);
}

/**
 * Cosine-weighted MEAN INCOMING RADIANCE (irradiance / pi) for a surface facing
 * `n`. `shR/shG/shB` are the four coefficients of each colour channel.
 *
 * THE UNIT IS LOAD-BEARING: the same convention gi_sample.hlsli returns and the
 * IBL irradiance cube stores, so this substitutes for the world field's answer
 * with no scale factor anywhere.
 *
 * Ramamoorthi's convolution A_0 = pi, A_1 = 2*pi/3, divided through by pi.
 * Clamped at zero — an L1 fit to a one-sided distribution goes negative
 * opposite the lobe, and a negative irradiance is a hole, not a dimmer answer.
 */
float3 giSpShEval(float4 shR, float4 shG, float4 shB, float3 n) {
    float4 y = giSpShBasis(n);
    float4 w = float4(y.x, y.y * (2.0 / 3.0), y.z * (2.0 / 3.0), y.w * (2.0 / 3.0));
    return max(float3(dot(shR, w), dot(shG, w), dot(shB, w)), float3(0, 0, 0));
}

// ---------------------------------------------------------------------------
// Bilateral weight, shared by the spatial filter and the forward gather
// ---------------------------------------------------------------------------

float giSpBilateral(float camDistA, float camDistB, float3 normalA, float3 normalB,
                    float normalPower) {
    float scale = max(GI_SP_DEPTH_TOLERANCE * max(camDistA, 1.0), 1.0);
    float depthTerm = exp(-abs(camDistA - camDistB) / scale);
    float cosine = max(0.0, dot(normalA, normalB));
    return depthTerm * pow(cosine, normalPower);
}

// ---------------------------------------------------------------------------
// The probe record, mirroring gpu::GiScreenProbeHeader (32 B) and
// gpu::GiScreenProbeMiss (16 B). Declared here so all six passes agree.
// ---------------------------------------------------------------------------

struct GiScreenProbe {
    float3 worldPosition;
    uint flags;
    float3 normal;
    float cameraDistance;
};

/// The published per-probe L1 SH record, mirroring gpu::GiScreenProbeSh (64 B).
/// A STORAGE BUFFER rather than a texture: SDL3 GPU hard-caps a shader stage at
/// 16 samplers and the scene shaders are already there, while they have
/// storage-buffer headroom. See the CPU struct for the full argument.
struct GiScreenProbeSh {
    float4 shR;
    float4 shG;
    float4 shB;
    /// xyz = anchor normal, w = anchor camera distance (0 = no probe here).
    float4 meta;
};

struct GiScreenProbeMissRecord {
    uint probeIndex;
    uint texelIndex;
    float resumeDistance;
    uint _pad0;
};

/// Byte offset of miss record `i` in the miss buffer. Record 0 starts at 16
/// because the first 16 bytes are the header — `[0] = miss count`, written by
/// the march pass's one InterlockedAdd per group and read by the args pass and
/// by stage B. Keeping the count IN the record buffer is what holds stage B at
/// eight read-only storage buffers (SDL3 GPU's ceiling) instead of nine.
#define GI_SP_MISS_HEADER_BYTES 16
uint giSpMissRecordOffset(uint index) { return GI_SP_MISS_HEADER_BYTES + index * 16u; }

#endif // BRACKEN_GI_SCREEN_PROBE_HLSLI
