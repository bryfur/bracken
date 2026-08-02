// reflection_gate.hlsli - the ONE definition of "how much of this pixel's
// specular comes from a traced reflection instead of the IBL probe".
//
// Three shaders have to agree on this number or the image is wrong in one of
// two visible ways:
//
//   rt_reflection.comp.hlsl  writes it into the traced target's alpha, and
//                            composite.frag.hlsl adds `rtSpec.rgb * rtSpec.a`
//   model.frag.hlsl          scales the IBL specular lobe by (1 - w)
//   model_instanced.frag.hlsl  likewise
//
// The invariant is `tracedWeight + iblWeight == 1` at every pixel. Disagree
// and a surface either loses its specular entirely (dark, because neither side
// supplied it) or gets it twice (blown out, which is what made the chrome
// sphere clip). Sharing the function -- rather than repeating the expression
// -- is what makes that structural instead of a convention.
//
// SMOOTHSTEP, NOT A CUTOFF. A binary `roughness <= maxRoughness` test puts a
// hard seam through any surface whose roughness crosses the threshold: on one
// side a full mirror, on the other the probe, with nothing in between. It also
// turns any small disagreement between the prepass G-buffer's roughness and
// the forward pass's into a full-strength flip, where a fade turns it into a
// proportionally small blend-ratio error. Production reflection systems fade;
// this is the cheapest form of it.
//
// `fadeStart` is where the handover begins (weight 1) and `maxRoughness` where
// it completes (weight 0). Both are pushed from
// render::RtReflectionSettings, so the two sides read the same numbers rather
// than each deriving them.

#ifndef REFLECTION_GATE_HLSLI
#define REFLECTION_GATE_HLSLI

float rtReflectionWeight(float roughness, float fadeStart, float maxRoughness) {
    // maxRoughness == 0 is the engine's "reflections are off" encoding: the
    // forward shaders must then keep their full IBL lobe.
    if (maxRoughness <= 0.0) {
        return 0.0;
    }
    // Guard the degenerate fadeStart >= maxRoughness (a host could configure
    // it): smoothstep with edge0 >= edge1 is undefined in HLSL.
    float lo = min(fadeStart, maxRoughness - 1e-4);
    return 1.0 - smoothstep(lo, maxRoughness, roughness);
}

#endif // REFLECTION_GATE_HLSLI
