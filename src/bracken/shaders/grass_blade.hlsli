/**
 * Shared grass-blade placement/animation math.
 *
 * Included by grass.vert.hlsl (forward) and prepass_grass.vert.hlsl (depth
 * prepass + velocity). Both includers must declare the SAME GrassUniforms
 * cbuffer, heightmap/splatmap textures+samplers, and ChunkOrigins buffer
 * BEFORE including this file — the function bodies reference those globals.
 * Sharing one source guarantees both passes compute bit-identical blade
 * world positions, which the forward pass's EQUAL depth test relies on.
 */

uint murmurHash(uint seed) {
    seed ^= seed >> 16;
    seed *= 0x85ebca6bu;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35u;
    seed ^= seed >> 16;
    return seed;
}

uint murmurHash2(uint a, uint b) {
    return murmurHash(a ^ (b * 0x9e3779b9u));
}

float hashFloat(int2 p) {
    uint h = murmurHash2(asuint(p.x), asuint(p.y));
    return float(h & 0x00FFFFFFu) / float(0x01000000u);
}

float2 hashFloat2(int2 p) {
    uint h1 = murmurHash2(asuint(p.x), asuint(p.y));
    uint h2 = murmurHash(h1);
    return float2(
        float(h1 & 0x00FFFFFFu) / float(0x01000000u),
        float(h2 & 0x00FFFFFFu) / float(0x01000000u)
    );
}

float sampleTerrainHeight(float2 worldXZ) {
    float2 hmUV = float2(
        (worldXZ.x - heightmapOriginX) / heightmapWorldSize,
        (worldXZ.y - heightmapOriginZ) / heightmapWorldSize
    );
    hmUV = saturate(hmUV);
    float raw = heightmapTexture.SampleLevel(heightmapSampler, hmUV, 0).r;
    return raw * (heightmapMaxHeight - heightmapMinHeight) + heightmapMinHeight;
}

struct BladeResult {
    float3 worldPos;  // final animated vertex position in world space
    float3 bladeFwd;  // blade-frame forward (shading normal input)
    float heightT;    // vertex height fraction along the blade [0, 1]
    bool alive;       // false: fully collapsed — emit a clipped degenerate
};

// Derives the blade vertex world position from (chunk, instance, vertex).
// `inputPos` is the blade-mesh vertex position (x/z = width axes, y = [0,1]
// height). Dead blades early-out before any texture fetch.
BladeResult computeBlade(float3 inputPos, uint instanceID) {
    BladeResult result;
    result.worldPos = float3(0.0, 0.0, 0.0);
    result.bladeFwd = float3(0.0, 0.0, 1.0);
    result.heightT = 0.0;
    result.alive = false;

    // Decode: which chunk + which blade in the chunk.
    uint chunk_idx = instanceID / bladesPerChunkSq;
    uint blade_idx = instanceID % bladesPerChunkSq;

    float2 chunk_origin = ChunkOrigins[chunk_idx].xz;

    uint bx = blade_idx % bladesPerChunkSide;
    uint bz = blade_idx / bladesPerChunkSide;

    // Blade cell within chunk (regular grid).
    float2 cellPos = chunk_origin + float2((float)bx, (float)bz) * bladeSpacing + 0.5 * bladeSpacing;

    // Deterministic per-blade seed from world-space cell coordinate.
    int2 seed = int2(floor(cellPos / max(bladeSpacing, 0.001)));

    // Natural jitter within cell (visual variety, not density tricks).
    float2 jitter = (hashFloat2(seed) - 0.5) * bladeSpacing * 1.0;
    float2 worldXZ = cellPos + jitter;

    // Distance from camera for LOD.
    float2 toCell = worldXZ - cameraPos.xz;
    float dist = length(toCell);

    // === SMOOTH LOD FADE ===
    // Blades smoothly shrink over the last 30% of view distance. No discrete
    // bands, no pop-in.
    float fade_start = grassViewDistance * 0.70;
    float fade_t = saturate((dist - fade_start) / max(grassViewDistance - fade_start, 1.0));
    float lod_scale = 1.0 - fade_t;

    // Clump density via continuous attenuation (not threshold). Sparser clumps
    // look natural without pop-in.
    int2 clumpSeed = int2(floor(cellPos / (bladeSpacing * 6.0)));
    float clump = hashFloat(clumpSeed + int2(42, 17));
    lod_scale *= 0.6 + 0.4 * clump;

    // Early-collapse before any texture fetch: beyond view distance the blade
    // contributes nothing, so skip splat + heightmap reads entirely.
    if (lod_scale <= 0.001) {
        return result;
    }

    // === SPLAT-AWARE PLACEMENT ===
    // Grass only grows where the terrain splatmap's grass layer (R) dominates;
    // dirt/rock/sand fade blades to nothing.
    if (splatEnabled > 0.5) {
        float2 splatUV = saturate((worldXZ - splatParams.xy) / splatParams.zw);
        float grassWeight = splatmapTexture.SampleLevel(splatmapSampler, splatUV, 0).r;
        lod_scale *= smoothstep(0.15, 0.5, grassWeight);
        if (lod_scale <= 0.001) {
            return result;
        }
    }

    // Single heightmap sample, only for live blades.
    float terrainY = sampleTerrainHeight(worldXZ);
    float3 bladeUp = float3(0.0, 1.0, 0.0);

    // Per-instance blade properties.
    float h1 = hashFloat(seed);
    float h2 = hashFloat(seed + int2(1, 0));
    float h3 = hashFloat(seed + int2(3, 3));

    float bladeHeight = (2.5 + h2 * 4.0 + h3 * h3 * 4.0) * lod_scale;
    // ~1.3x wider than the old billboard blades so hash-yawed blades stay
    // readable edge-on.
    float bladeWidth  = (0.6 + hashFloat(seed + int2(0, 1)) * 0.5) * 1.3;

    // === STABLE ORIENTATION ===
    // Per-blade hash yaw over the full 2π. World-anchored: the meadow no longer
    // rotates/re-lights as the camera orbits.
    float yaw = h1 * 6.2831853;
    float cosY = cos(yaw);
    float sinY = sin(yaw);
    float3 bladeRight = float3(cosY, 0.0, sinY);
    float3 bladeFwd = float3(-sinY, 0.0, cosY); // cross(bladeRight, bladeUp)

    // Transform vertex into blade frame.
    float3 localPos = bladeRight * (inputPos.x * bladeWidth)
                    + bladeUp    * (inputPos.y * bladeHeight)
                    + bladeFwd   * (inputPos.z * bladeWidth);

    // === BLADE CURVE ===
    float heightT = inputPos.y;
    float curveMagnitude = 0.15 * bladeHeight;
    // Length-guarded normalize: the hash can land exactly on (0.5, 0.5),
    // and normalize(0) would NaN the whole blade.
    float2 curveDirRaw = hashFloat2(seed + int2(5, 7)) - 0.5;
    float2 curveDir = curveDirRaw / max(length(curveDirRaw), 1e-4);
    float curveAmount = heightT * heightT * curveMagnitude;
    float3 curveOffset = float3(curveDir.x, 0.0, curveDir.y) * curveAmount;

    // === LAYERED WIND ===
    float windFactor = heightT;  // wind affects tip more than base
    float p1 = time * 2.0 + worldXZ.x * 0.10 + worldXZ.y * 0.10;
    float p2 = time * 3.7 + worldXZ.x * 0.07 - worldXZ.y * 0.13;
    float windWave = sin(p1) * 0.7 + sin(p2) * 0.3;
    float2 windOffset = windDirection * windStrength * windFactor * windWave;

    float3 worldPos = float3(worldXZ.x, terrainY, worldXZ.y) + localPos + curveOffset;
    worldPos.x += windOffset.x;
    worldPos.z += windOffset.y;

    result.worldPos = worldPos;
    result.bladeFwd = bladeFwd;
    result.heightT = heightT;
    result.alive = true;
    return result;
}
