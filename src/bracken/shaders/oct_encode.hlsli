#ifndef BRACKEN_OCT_ENCODE_HLSLI
#define BRACKEN_OCT_ENCODE_HLSLI

// Octahedral normal ENCODING, the writer half of the G-buffer's normal channel.
//
// One definition, for the same reason rt_sampling.hlsli gives for
// interleavedGradientNoise: this used to be a byte-identical private copy in
// prepass_gbuffer.frag.hlsl and another in prepass_gbuffer_terrain.frag.hlsl,
// and when the MASK variant needed it too the choice was a third copy or this
// file. An encoder that disagrees with its decoder by a sign produces a normal
// that is wrong only in one octant, which is indistinguishable from a denoiser
// artefact in every consumer that reads it.
//
// The DECODERS live elsewhere and are deliberately not merged in here: rt_trace.hlsli,
// gi_probe.hlsli and gi_screen_probe.hlsli each carry `octDecode` next to the
// buffer they decode, and bent_cone.hlsli's `bentOctEncode` is a separate
// signal with its own range contract. This header is only for shaders that
// WRITE the prepass G-buffer.

// Cigolle et al. 2014. Unit vector -> [0,1]^2.
float2 octEncode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, float2(1.0, 1.0), float2(-1.0, -1.0));
    }
    return e * 0.5 + 0.5;
}

#endif // BRACKEN_OCT_ENCODE_HLSLI
