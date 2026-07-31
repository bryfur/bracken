/**
 * IBL Split-Sum BRDF LUT Fragment Shader
 *
 * Karis split-sum second sum: environment BRDF scale/bias (A, B) such that
 * specular = prefiltered * (F0 * A + B). Rendered ONCE at init into an RG16F
 * 2D LUT indexed by (NdotV, perceptual roughness) — u = NdotV, v = roughness.
 * Smith-Schlick geometry with the IBL k = alpha/2 remapping (alpha = r*r).
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

static const float PI_IBL = 3.14159265358979323846;
static const uint LUT_SAMPLE_COUNT = 1024u;

float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley(uint i, uint n) {
    return float2(float(i) / float(n), radicalInverseVdC(i));
}

float3 importanceSampleGGX(float2 xi, float3 N, float rough) {
    float a = rough * rough;
    float phi = 2.0 * PI_IBL * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 H = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    float3 upAxis = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(upAxis, N));
    float3 B = cross(N, T);
    return normalize(T * H.x + B * H.y + N * H.z);
}

// Smith-Schlick geometry, IBL remap: k = alpha / 2 (Karis SIGGRAPH 2013).
float geometrySmithIBL(float NdotV, float NdotL, float rough) {
    float a = rough * rough;
    float k = a / 2.0;
    float gV = NdotV / (NdotV * (1.0 - k) + k);
    float gL = NdotL / (NdotL * (1.0 - k) + k);
    return gV * gL;
}

float4 PSMain(PSInput input) : SV_Target {
    float NdotV = max(input.texCoord.x, 1e-3);
    float rough = input.texCoord.y;

    float3 V = float3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    float3 N = float3(0.0, 0.0, 1.0);

    float A = 0.0;
    float B = 0.0;
    [loop]
    for (uint i = 0u; i < LUT_SAMPLE_COUNT; ++i) {
        float2 xi = hammersley(i, LUT_SAMPLE_COUNT);
        float3 H = importanceSampleGGX(xi, N, rough);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = saturate(L.z);
        float NdotH = saturate(H.z);
        float VdotH = saturate(dot(V, H));
        if (NdotL > 0.0) {
            float G = geometrySmithIBL(NdotV, NdotL, rough);
            float GVis = (G * VdotH) / max(NdotH * NdotV, 1e-6);
            float Fc = pow(1.0 - VdotH, 5.0);
            A += (1.0 - Fc) * GVis;
            B += Fc * GVis;
        }
    }

    return float4(A / float(LUT_SAMPLE_COUNT), B / float(LUT_SAMPLE_COUNT), 0.0, 1.0);
}
