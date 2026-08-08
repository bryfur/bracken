#ifndef BRACKEN_RT_SAMPLING_HLSLI
#define BRACKEN_RT_SAMPLING_HLSLI

// This file used to carry a byte-identical private copy of
// interleavedGradientNoise, which was fine until R3 gave the reflection pass
// both this include AND shadow_common.hlsli (which includes lighting.hlsli for
// the same function) and the two collided. One definition.
//
// rtSample2D no longer calls it — its per-pixel offset has to be 2D and a
// scalar hash cannot be — but the include stays: sixteen shaders include this
// header and several reach lighting.hlsli through it.
#include "lighting.hlsli"

// Sampling and packing shared by the ray-traced shadow/reflection passes and
// their denoisers.
//
// THE SEQUENCE IS THE WHOLE DESIGN. One ray per pixel per frame only looks
// clean after a denoiser if consecutive frames sample DIFFERENT points and
// neighbouring pixels sample different points from each other. Get either
// wrong and the temporal filter averages a set of samples that were already
// correlated, so the variance never drops and the result is a blur with the
// original noise still in it.
//
// So: a per-pixel spatial offset (a hash, decorrelating neighbours) advanced
// per frame by the R2 low-discrepancy sequence (decorrelating time). R2 rather
// than Halton because it is two multiply-adds and a frac, and its 2D
// distribution beats Halton's low bases at the ~16-32 sample counts a
// temporal filter actually reaches.

/// Fractional part of the plastic-constant sequence — R2, Roberts 2018. The
/// two constants are 1/phi2 and 1/phi2^2 where phi2 is the plastic number.
float2 r2Sequence(uint index) {
    return frac(float2(0.7548776662466927, 0.5698402909980532) * float(index));
}

/// Seed a per-pixel, per-frame random STREAM.
///
/// rtSample2D answers "the one sample this pixel takes this frame", and is the
/// right tool when there is exactly one — its whole value is that consecutive
/// frames land at low-discrepancy offsets. A reservoir needs a DOZEN decisions
/// per pixel (which candidate, whether to accept it, ...) and those are
/// independent Bernoulli/uniform draws, not a set being stratified; feeding them
/// from one low-discrepancy sequence would correlate the candidate choice with
/// the acceptance test. So: a stream, seeded from the same (pixel, frame) pair
/// so it is exactly as replayable — which is what the deterministic smoke gate
/// and the A-SVGF gradient work of §5.7 both require.
uint rtSeed(uint2 pixel, uint frameIndex) {
    uint h = pixel.x * 73856093u ^ pixel.y * 19349663u ^ frameIndex * 83492791u;
    // One round of the "lowbias32" finalizer — enough to decorrelate the three
    // multiplied lanes above, which is all the seed has to do.
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

/// The per-pixel, per-frame 2D sample point.
///
/// A per-pixel Cranley-Patterson rotation of the R2 sequence: the pixel's whole
/// sequence is translated on the unit torus, which is measure-preserving and
/// leaves R2's low-discrepancy property intact across that pixel's frames.
///
/// THE OFFSET MUST BE 2D, and it used to be one scalar broadcast into both
/// lanes. That is a legal rotation per pixel — the temporal half of this file's
/// header contract was always satisfied — but it put every pixel on screen onto
/// the SAME diagonal line {(t, t + c) mod 1} at any fixed frame, because the
/// two lanes differed by the frame's constant `r2.y - r2.x` and nothing else.
/// So the frame's spatial sample set was one-dimensional: neighbouring pixels
/// picked points perfectly correlated along one axis, which is exactly the
/// failure the header's second sentence warns about, and the spatial filters
/// (both a-trous DI passes, the shadow filter) average over those neighbours.
///
/// Still a pure function of (pixel, frameIndex) — that is what the A-SVGF
/// gradient replay of §5.7 and the deterministic smoke gate both require, and
/// it is why the offset is hashed from the pixel alone rather than carried in a
/// history texture.
float2 rtSample2D(uint2 pixel, uint frameIndex) {
    const uint h = rtSeed(pixel, 0u);
    const float2 offset = float2(h & 0xffffu, h >> 16u) * (1.0 / 65536.0);
    return frac(r2Sequence(frameIndex) + offset);
}

/// Next uniform float in [0, 1) from a stream seeded by rtSeed. PCG-XSH-RR's
/// output function over an LCG state: two multiplies and three shifts, and it
/// passes the only test that matters here — neighbouring seeds produce
/// uncorrelated streams.
float rtNextFloat(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w) * (1.0 / 4294967296.0);
}

/// Orthonormal basis around `n`. Duff et al. 2017 — branchless, and stable as
/// n crosses the z = 0 plane, which the naive cross-with-up construction is
/// not (it degenerates and makes the sampled cone spin).
void buildBasis(float3 n, out float3 t, out float3 b) {
    const float s = n.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (s + n.z);
    const float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

/// Uniformly sample a direction inside a cone of half-angle acos(cosThetaMax)
/// around `dir`. This is what turns a hard shadow into a soft one: the sun is
/// a disk about half a degree across, not a point, so the correct shadow ray
/// goes to a random point ON that disk.
float3 sampleCone(float3 dir, float cosThetaMax, float2 xi) {
    const float cosTheta = lerp(cosThetaMax, 1.0, xi.x);
    const float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    const float phi = 6.283185307179586 * xi.y;
    float3 t, b;
    buildBasis(dir, t, b);
    return normalize(t * (cos(phi) * sinTheta) + b * (sin(phi) * sinTheta) + dir * cosTheta);
}

/// GGX visible-normal-distribution sample (Heitz 2018), in tangent space with
/// the surface normal on +z. Sampling the VISIBLE normals rather than the full
/// distribution is what keeps reflection variance low at grazing angles, where
/// naive NDF sampling produces rays that go straight into the surface and have
/// to be thrown away.
float3 sampleGGXVndf(float3 viewTangent, float alpha, float2 xi) {
    // Stretch the view vector so the problem becomes sampling a hemisphere.
    const float3 vh = normalize(float3(alpha * viewTangent.x, alpha * viewTangent.y, viewTangent.z));

    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 t1 =
        lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    const float3 t2 = cross(vh, t1);

    const float r = sqrt(xi.x);
    const float phi = 6.283185307179586 * xi.y;
    const float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    const float s = 0.5 * (1.0 + vh.z);
    p2 = lerp(sqrt(saturate(1.0 - p1 * p1)), p2, s);

    const float3 nh = p1 * t1 + p2 * t2 + sqrt(saturate(1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(float3(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

// ---------------------------------------------------------------------------
// Packed visibility tiles.
//
// The traced shadow signal is one BIT per pixel, packed 8x4 into a uint. At
// half res that is 22 KB for a 1600x900 frame instead of 360 KB as R8, and —
// the reason it is worth the packing at all — the tile classifier reads the
// visibility of 32 pixels in ONE fetch, so deciding "this whole tile is lit,
// skip every filter tap in it" costs almost nothing.
// ---------------------------------------------------------------------------

static const uint2 RT_TILE_SIZE = uint2(8, 4);

/// Tile grid dimensions for a traced signal of `size` pixels.
uint2 rtTileCount(uint2 size) {
    return (size + RT_TILE_SIZE - 1) / RT_TILE_SIZE;
}

/// Bit index within a tile for a pixel-in-tile coordinate.
uint rtTileBit(uint2 pixelInTile) {
    return pixelInTile.y * RT_TILE_SIZE.x + pixelInTile.x;
}

/// Unpack one pixel's visibility from its tile word.
float rtUnpackVisibility(uint tileBits, uint2 pixelInTile) {
    return (tileBits & (1u << rtTileBit(pixelInTile))) != 0u ? 1.0 : 0.0;
}

#endif // BRACKEN_RT_SAMPLING_HLSLI
