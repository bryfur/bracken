/**
 * Grass Vertex Shader - SDL3 GPU API (chunked, splat-aware)
 *
 * Design:
 *  - CPU frustum-culls grass chunks and uploads visible chunk origins
 *    into a storage buffer. Each chunk contains BLADES_PER_CHUNK^2 blades.
 *  - Vertex shader derives per-blade position from (chunk_origin, blade_idx)
 *    and applies a SMOOTH LOD fade over the last 30% of view distance.
 *  - LOD fade and the splatmap grass mask are evaluated BEFORE the heightmap
 *    fetch; fully collapsed blades early-out with a degenerate position and
 *    skip the texture reads entirely.
 *  - Blade orientation is a stable per-blade hash yaw (full 2π) — no
 *    camera-facing billboarding, so the meadow doesn't re-orient/re-light as
 *    the camera orbits. Blades are widened ~1.3x to stay readable edge-on.
 */

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
    [[vk::location(3)]] float4 color : COLOR0;
    uint instanceID : SV_InstanceID;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float2 texCoord : TEXCOORD1;
    float3 worldPos : TEXCOORD2;
    float4 color : TEXCOORD3;
    float viewDepth : TEXCOORD4;
};

// Per-frame uniforms
[[vk::binding(0, 1)]]
cbuffer GrassUniforms {
    float4x4 viewProjection;
    float3 cameraPos;          // world camera position (xz used)
    float time;
    float splatWorldSize;      // world size for splatmap UV (origin at 0,0)
    float splatEnabled;        // 1 when a splatmap is bound at sampler slot 1
    float _pad0;
    float grassViewDistance;
    float2 windDirection;
    float windStrength;
    float chunkSize;           // world units per chunk side
    float bladeSpacing;        // world units between blades within a chunk
    uint bladesPerChunkSide;   // sqrt of blades per chunk
    uint bladesPerChunkSq;     // bladesPerChunkSide * bladesPerChunkSide
    float _pad1;
    float heightmapOriginX;
    float heightmapOriginZ;
    float heightmapWorldSize;
    float heightmapMinHeight;
    float heightmapMaxHeight;
    float3 cameraForward;      // world-space camera forward (view depth projection)
};

// Heightmap texture + sampler (vertex stage set 0 = samplers first).
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D heightmapTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState heightmapSampler;

// Terrain splatmap (R = grass weight), CLAMP sampler. Bound to the heightmap
// texture as a placeholder when splatEnabled == 0.
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D splatmapTexture;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState splatmapSampler;

// Per-visible-chunk origins (xz = world origin). Filled by CPU each frame.
// Within vertex set 0, storage buffers come after samplers (2 samplers -> binding 2).
[[vk::binding(2, 0)]]
StructuredBuffer<float4> ChunkOrigins;

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

// Fully collapsed blade: emit a clipped degenerate without touching textures.
VSOutput deadBlade() {
    VSOutput output;
    output.position = float4(0.0, 0.0, 2.0, 1.0); // z/w > 1 -> clipped
    output.normal = float3(0.0, 1.0, 0.0);
    output.texCoord = float2(0.0, 0.0);
    output.worldPos = float3(0.0, 0.0, 0.0);
    output.color = float4(0.0, 0.0, 0.0, 0.0);
    output.viewDepth = 0.0;
    return output;
}

VSOutput VSMain(VSInput input) {
    VSOutput output;

    // Decode: which chunk + which blade in the chunk.
    uint chunk_idx = input.instanceID / bladesPerChunkSq;
    uint blade_idx = input.instanceID % bladesPerChunkSq;

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
        return deadBlade();
    }

    // === SPLAT-AWARE PLACEMENT ===
    // Grass only grows where the terrain splatmap's grass layer (R) dominates;
    // dirt/rock/sand fade blades to nothing.
    if (splatEnabled > 0.5) {
        float2 splatUV = saturate(worldXZ / splatWorldSize);
        float grassWeight = splatmapTexture.SampleLevel(splatmapSampler, splatUV, 0).r;
        lod_scale *= smoothstep(0.15, 0.5, grassWeight);
        if (lod_scale <= 0.001) {
            return deadBlade();
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
    float3 localPos = bladeRight * (input.position.x * bladeWidth)
                    + bladeUp    * (input.position.y * bladeHeight)
                    + bladeFwd   * (input.position.z * bladeWidth);

    // === BLADE CURVE ===
    float heightT = input.position.y;
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

    // Base AO: darken blade base, lighten tip.
    float ao = lerp(0.45, 1.0, heightT);
    float4 vertColor = input.color;
    vertColor.rgb *= ao;

    output.worldPos = worldPos;
    // Shading normal leans heavily toward up so lighting reads as a grounded
    // meadow surface instead of per-blade facets flipping with yaw.
    output.normal = normalize(lerp(bladeFwd, float3(0.0, 1.0, 0.0), 0.7));
    output.texCoord = input.texCoord;
    output.color = vertColor;
    // View-space depth (-viewPos.z), matching the other mesh shaders — radial
    // distance would skew cluster Z-slice + cascade selection at screen edges.
    output.viewDepth = dot(worldPos - cameraPos, cameraForward);
    output.position = mul(viewProjection, float4(worldPos, 1.0));

    return output;
}
