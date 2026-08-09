/**
 * bent_cone.hlsli — the four channels of the screen-space AO texture, in one
 * place, so the pass that writes them and the five shaders that read them
 * cannot drift apart.
 *
 * Since R5a the GTAO chain (gtao.frag.hlsl, SSILVB visibility bitmask) no
 * longer produces a scalar. Every texture in the chain is RGBA:
 *
 *   R  ambient occlusion, 1 = unoccluded  (what it always was)
 *   GB the BENT NORMAL — the mean unoccluded direction — in WORLD space,
 *      octahedral-encoded to [0,1]^2 with the engine's existing mapping
 *      (`octEncode` in prepass_gbuffer.frag.hlsl, Cigolle et al. 2014; the CPU
 *      mirror is `render::gi::oct_encode`). World space and not view space
 *      because every consumer shades in world space and none of them has the
 *      view matrix, and because a world-space direction survives the temporal
 *      pass's reprojection across camera motion unchanged.
 *   A  the CONE APERTURE, stored as sin^2(alpha) where alpha is the cone's
 *      half-angle: the fraction of the hemisphere's PROJECTED solid angle the
 *      cone covers, which is exactly the factor the diffuse GI term wants and
 *      is the quantity that averages linearly under a filter. Recover the
 *      half-angle with alpha = asin(sqrt(A)).
 *
 * PRECISION. RGBA8_UNORM at the raw and full-resolution ends (RGBA16F through
 * the temporal history, where accumulation would otherwise stagnate against
 * the 8-bit grid). 8:8 octahedral is ~1 degree of worst-case angular error —
 * far under the angular scale a diffuse cone resolves — and 8 bits of aperture
 * is a 0.4% step in a factor that multiplies ambient only.
 *
 * THE ESCAPE VALUE. A >= BENT_CONE_OPEN is "this pixel carries no usable
 * directional information", so GB may be anything and the shading normal is
 * used instead. That is not a special case bolted on — it is what makes a 1x1
 * WHITE dummy texture (what every scene shader binds when AO is off, and what
 * the full-res target is cleared to before the first frame) decode to exactly
 * the pre-R5a behaviour: AO 1, gate 1, bent normal = shading normal. No uniform
 * flag is needed anywhere.
 *
 * TWO DIFFERENT THINGS LAND ON IT, which is why the gate ramps rather than
 * branches (see BENT_CONE_TRUST): a genuinely open pixel, whose cone IS the
 * full hemisphere and whose mean direction is the shading normal by definition;
 * and a lobe too SPLIT to fit a cone at all (two open sides under an eave),
 * which the resultant->cone inversion also clamps to aperture 1 while its AO
 * says 0.5. The first wants an ungated GI term, the second wants the scalar.
 */

#ifndef BRACKEN_BENT_CONE_HLSLI
#define BRACKEN_BENT_CONE_HLSLI

/// Aperture at or above which the pass is declaring "no directional data".
/// 0.998 rather than 1.0 because the value survives an 8-bit round trip and a
/// bilinear filter; a genuinely open pixel lands here anyway, and its bent
/// normal IS its shading normal, so the fallback is exact rather than a guess.
#define BENT_CONE_OPEN 0.998

/// Aperture below which the cone fit is trusted outright. Between this and
/// BENT_CONE_OPEN the GI gate RAMPS to the scalar, because the resultant->cone
/// inversion in gtao.frag.hlsl clamps at the open hemisphere for ANY lobe too
/// split to fit a cone (resultant <= 2/3 -> aperture 1; render/gtao.hpp
/// gtao_cone_cos_from_resultant documents the clamp). An aperture near 1
/// therefore means "very wide OR unfittable", and only the second wants the
/// scalar gate: an overhead occluder with open sides — an eave, a cart
/// underside, a ceiling — has aperture 1 and AO 0.5. Without the ramp the gate
/// steps from ~0.996 straight to the scalar AO across ONE RGBA8 code, which
/// crawls along the isoline as the estimate dithers over it.
/// 0.95 is alpha >= 77 degrees — within the 3-slice estimator's own noise of
/// the clamp — and 0.95..0.998 is 12 codes, enough for a ramp.
#define BENT_CONE_TRUST 0.95

/// Unit vector -> [0,1]^2. Verbatim `octEncode` from prepass_gbuffer.frag.hlsl.
float2 bentOctEncode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, float2(1.0, 1.0), float2(-1.0, -1.0));
    }
    return e * 0.5 + 0.5;
}

/// [0,1]^2 -> unit vector. Verbatim `octDecode` from rt_trace.hlsli.
float3 bentOctDecode(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, float2(-t, -t), float2(t, t));
    return normalize(n);
}

/// Normalise a filtered direction sum, falling back when the taps cancelled.
/// Every filter in the AO chain averages DECODED directions and re-encodes, so
/// each of them needs this: octahedral coordinates are discontinuous across the
/// fold seam and averaging them there points at neither input.
float3 bentNormalize(float3 v, float3 fallback) {
    float l2 = dot(v, v);
    return (l2 > 1e-8) ? (v * rsqrt(l2)) : fallback;
}

/// The AO texel, decoded.
struct BentCone {
    float ao;          ///< scalar occlusion, 1 = unoccluded
    float3 direction;  ///< world-space bent normal (== the shading normal when !valid)
    float aperture;    ///< sin^2(alpha) of the cone half-angle
    bool valid;        ///< false = fully open / no data; `direction` is the shading normal
};

/// Decode an AO texel. `shadingNormal` is the world-space normal the caller
/// shades with — it is what `direction` falls back to for an open cone.
BentCone decodeBentCone(float4 texel, float3 shadingNormal) {
    BentCone c;
    c.ao = texel.r;
    c.aperture = texel.a;
    c.valid = texel.a < BENT_CONE_OPEN;
    c.direction = c.valid ? bentOctDecode(texel.gb) : shadingNormal;
    return c;
}

/**
 * The gate the DIFFUSE GI term is multiplied by, in place of the scalar screen
 * AO (docs/rt_lighting_plan.md §5.6's double-count rule: bent-cone gating
 * REPLACES the scalar-AO multiply for that term, it never joins it).
 *
 * `materialAo` is the baked (ORM / occlusionStrength) occlusion, which is a
 * DIFFERENT measurement at a different scale and still min-combines exactly as
 * it does with screen AO today. `scalarGate` is what the caller would have used
 * without a cone — returned unchanged when the pixel carries no cone, which is
 * what keeps the AO-off and pre-R5a paths bit-identical.
 *
 * The curve is not a taste knob: irradiance sampled along the bent normal is
 * the mean radiance over the hemisphere about that direction, and the cosine-
 * weighted solid angle of a cone of half-angle alpha about its own axis is
 * pi*sin^2(alpha). Dividing by the pi the irradiance convention already
 * carries leaves sin^2(alpha) — the stored aperture itself.
 *
 * The BENT_CONE_TRUST..BENT_CONE_OPEN ramp is what keeps the two branches from
 * meeting at a cliff: the aperture saturates at 1 both for a genuinely open
 * pixel and for a lobe too split to fit a cone at all, and only the second
 * wants the scalar. smoothstep reaches exactly 1 at BENT_CONE_OPEN, so the
 * valid branch meets the escape branch there and the escape path — the 1x1
 * white dummy, sky, far field, ao_strength == 0 — is untouched.
 */
float bentConeGiGate(BentCone c, float materialAo, float scalarGate) {
    return c.valid ? lerp(min(materialAo, c.aperture), scalarGate,
                          smoothstep(BENT_CONE_TRUST, BENT_CONE_OPEN, c.aperture))
                   : scalarGate;
}

#endif // BRACKEN_BENT_CONE_HLSLI
