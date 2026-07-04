/**
 * "GTAO" Fragment Shader (ao_mode == 2)
 *
 * Naming note: despite the settings name, this is NOT GTAO (Jimenez et al.) —
 * there is no per-direction horizon search or arc integration. It accumulates
 * cosine-weighted, distance-attenuated point samples along rotated
 * screen-space directions, i.e. an HBAO-style directional variant. The mode
 * name is kept for settings-file compatibility.
 *
 * Reads the depth buffer, reconstructs view-space positions, estimates normals
 * from screen-space derivatives, and marches the precomputed directions.
 * Outputs a single-channel AO value (1.0 = no occlusion, 0.0 = full occlusion).
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

#define KERNEL_SIZE 24

[[vk::binding(0, 3)]]
cbuffer GTAOUniforms : register(b0, space3) {
    float4x4 projection;
    float4x4 invProjection;
    float2 screenSize;
    float2 invScreenSize;
    float radius;
    float bias;
    int numDirections;
    int numSteps;
    float radiusScale;
    float maxDistance; // view-space far cutoff for AO (fades over the last 10%)
    // Scalars (not a float array) so the kernel lands at byte 176, matching
    // the CPU struct: HLSL would pad float[2] to one register per element.
    float _padding0;
    float _padding1;
    // xy = evenly spaced unit-circle direction (first numDirections entries used)
    float4 sampleKernel[KERNEL_SIZE];
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float> depthTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState depthSampler : register(s0, space2);

static const float PI = 3.14159265359;

// Reconstruct view-space position from UV + depth
float3 viewPosFromDepth(float2 uv, float depth) {
    float4 clipPos = float4(uv * 2.0 - 1.0, depth, 1.0);
    clipPos.y = -clipPos.y; // Vulkan Y flip
    float4 viewPos = mul(invProjection, clipPos);
    return viewPos.xyz / viewPos.w;
}

// Simple hash for pseudo-random rotation per pixel
float interleavedGradientNoise(float2 pos) {
    return frac(52.9829189 * frac(dot(pos, float2(0.06711056, 0.00583715))));
}

float PSMain(PSInput input) : SV_Target {
    float depth = depthTexture.SampleLevel(depthSampler, input.texCoord, 0);

    // Sky never gets AO: reversed-Z cleared/skybox depth is exactly 0.0 (same
    // test as volumetric_fog.frag). The explicit maxDistance cutoff below is
    // the real far-field fade; this only skips true sky pixels.
    if (depth <= 0.0) {
        return 1.0;
    }

    float3 viewPos = viewPosFromDepth(input.texCoord, depth);

    // Explicit far-field cutoff in linear view distance: past maxDistance the
    // AO footprint is subpixel at half resolution anyway. Fade over the last
    // 10% so the boundary doesn't pop.
    float distanceFade = saturate((maxDistance - (-viewPos.z)) / max(0.1 * maxDistance, 1e-3));
    if (distanceFade <= 0.0) {
        return 1.0;
    }

    // Reconstruct normal from screen-space derivatives of view position
    float3 dpdx = ddx(viewPos);
    float3 dpdy = ddy(viewPos);
    float3 viewNormal = normalize(cross(dpdy, dpdx));

    // Per-pixel rotation angle for noise reduction: one sincos per pixel,
    // base directions come precomputed from the CPU kernel.
    float rotationAngle = interleavedGradientNoise(input.position.xy) * 2.0 * PI;
    float ca = cos(rotationAngle);
    float sa = sin(rotationAngle);

    float occlusion = 0.0;
    float sampleCount = 0.0;
    float effectiveRadius = radius * radiusScale;

    // Screen-space radius scaled by projection
    float screenRadius = effectiveRadius * projection[0][0] * 0.5 / max(-viewPos.z, 0.1);
    // Clamp to avoid too large or too small radii
    screenRadius = clamp(screenRadius, 2.0 * invScreenSize.x, 0.2);

    int dirCount = min(numDirections, KERNEL_SIZE);
    for (int dir = 0; dir < dirCount; ++dir) {
        float2 base = sampleKernel[dir].xy;
        float2 direction = float2(base.x * ca - base.y * sa, base.x * sa + base.y * ca);

        for (int step = 1; step <= numSteps; ++step) {
            float t = float(step) / float(numSteps);
            float2 sampleUV = input.texCoord + direction * screenRadius * t;

            // Bounds check
            if (any(sampleUV < 0.0) || any(sampleUV > 1.0)) {
                continue;
            }

            float sampleDepth = depthTexture.SampleLevel(depthSampler, sampleUV, 0);
            float3 samplePos = viewPosFromDepth(sampleUV, sampleDepth);

            float3 sampleVec = samplePos - viewPos;
            float sampleDist = length(sampleVec);

            if (sampleDist < 0.001) {
                continue;
            }

            float3 sampleDir = sampleVec / sampleDist;

            // Cosine-weighted contribution
            float cosAngle = max(0.0, dot(viewNormal, sampleDir) - bias);

            // Distance falloff
            float falloff = 1.0 - smoothstep(0.0, effectiveRadius, sampleDist);

            occlusion += cosAngle * falloff;
            sampleCount += 1.0;
        }
    }

    if (sampleCount > 0.0) {
        occlusion /= sampleCount;
    }

    float ao = saturate(1.0 - occlusion);
    return lerp(1.0, ao, distanceFade);
}
