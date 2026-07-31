// water_sim.comp.hlsl - 2D wave-equation water simulation (one substep).
//
// State texture: RG16F, R = height at step n, G = height at step n-1.
// Ping-pong: reads stateIn (steps n / n-1), writes stateOut (steps n+1 / n).
// The field advects downstream (semi-Lagrangian back-trace along the flow)
// so ripples drift with the river; a border band damps waves to nothing
// before they can reflect off the simulation window's edges.
//
// Dispatched once per fixed substep by render::WaterRenderer.

static const uint MAX_IMPULSES = 8;

[[vk::binding(0, 2)]]
cbuffer WaterSimUniforms : register(b0, space2) {
    uint resolution;      // square state texture size (texels)
    float waveC2Dt2;      // (c * dt / dx)^2 — wave speed term, pre-squared
    float damping;        // per-substep amplitude retention (e.g. 0.996)
    float advectTexels;   // texels to back-trace along -flowDir per substep
    float2 flowDir;       // normalized flow direction in texel space
    uint impulseCount;    // live entries in impulses[]
    float borderTexels;   // edge damp band width (texels)
    // xy = center (texels), z = radius (texels), w = height amplitude added
    float4 impulses[MAX_IMPULSES];
};

[[vk::binding(0, 0)]] Texture2D<float2> stateIn : register(t0, space0);
[[vk::binding(0, 1)]] RWTexture2D<float2> stateOut : register(u0, space1);

// Bilinear read of the previous state at a fractional texel position
// (needed by the advection back-trace; Load() has no filtering).
float2 sampleState(float2 p) {
    p = clamp(p, 0.0, (float)(resolution - 1));
    int2 p0 = int2(p);
    int2 p1 = min(p0 + 1, int(resolution - 1));
    float2 f = frac(p);
    float2 s00 = stateIn[int2(p0.x, p0.y)];
    float2 s10 = stateIn[int2(p1.x, p0.y)];
    float2 s01 = stateIn[int2(p0.x, p1.y)];
    float2 s11 = stateIn[int2(p1.x, p1.y)];
    return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= resolution || id.y >= resolution) {
        return;
    }

    // Back-trace along the flow so the whole field drifts downstream.
    float2 src = float2(id.xy) - flowDir * advectTexels;

    float2 state = sampleState(src);
    float h = state.x;
    float hOld = state.y;

    // 5-point Laplacian around the advected source position.
    float lap = sampleState(src + float2(1, 0)).x + sampleState(src - float2(1, 0)).x +
                sampleState(src + float2(0, 1)).x + sampleState(src - float2(0, 1)).x - 4.0 * h;

    // Verlet step of the wave equation, with amplitude damping.
    float hNew = (2.0 * h - hOld + waveC2Dt2 * lap) * damping;

    // Gaussian impulse splats (ripple sources injected by the host).
    for (uint i = 0; i < impulseCount; ++i) {
        float2 d = float2(id.xy) - impulses[i].xy;
        float r = max(impulses[i].z, 1.0);
        hNew += impulses[i].w * exp(-dot(d, d) / (r * r));
    }

    // Fade to calm inside the border band: the window edge absorbs waves
    // instead of reflecting them back into the scene.
    float edge = min(min((float)id.x, (float)id.y),
                     min((float)(resolution - 1 - id.x), (float)(resolution - 1 - id.y)));
    hNew *= smoothstep(0.0, borderTexels, edge);

    stateOut[id.xy] = float2(hNew, h);
}
