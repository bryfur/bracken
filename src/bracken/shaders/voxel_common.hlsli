#ifndef BRACKEN_VOXEL_COMMON_HLSLI
#define BRACKEN_VOXEL_COMMON_HLSLI

// voxel_common.hlsli - the voxel radiance clipmap's addressing, cascade
// selection and cone-step math, shared by every voxel pass.
//
// EVERY function here is a transliteration of its counterpart in
// `src/bracken/render/gi/voxel_math.hpp`, which is the source of truth and is
// tested with no GPU, no window and no SDL. When one side changes, the other
// changes in the same commit — a clipmap whose CPU and GPU disagree about which
// texel holds which point in the world produces lighting that is wrong
// everywhere and looks like a shading bug for a week.
//
// Binding-free by construction: no resource is named below, so this can be
// included before or after a shader's binding block.

// ---------------------------------------------------------------------------
// Dimensions — mirrors voxel_math.hpp
// ---------------------------------------------------------------------------

#define VOXEL_RESOLUTION 96
#define VOXEL_CASCADE_COUNT 3
#define VOXEL_BASE_SIZE 25.0
#define VOXEL_CASCADE_MARGIN 2.0

#define VOXEL_CONE_HIT_ALPHA 0.5
#define VOXEL_CONE_TERMINATE_ALPHA 0.98

float voxelSize(int cascade) { return VOXEL_BASE_SIZE * float(1 << cascade); }

float voxelCascadeExtent(int cascade) { return voxelSize(cascade) * float(VOXEL_RESOLUTION); }

float voxelCascadeHalfExtent(int cascade) { return voxelCascadeExtent(cascade) * 0.5; }

// ---------------------------------------------------------------------------
// Torus addressing
// ---------------------------------------------------------------------------

/// Euclidean modulo. HLSL `%` truncates toward zero like C++, and every wrap
/// here crosses the origin — the naive form folds -1 onto 1 and tears the
/// volume along the world axes.
int voxelWrapAxis(int v, int n) {
    int m = v % n;
    return m < 0 ? m + n : m;
}

int3 voxelWrap(int3 absolute) {
    return int3(voxelWrapAxis(absolute.x, VOXEL_RESOLUTION),
                voxelWrapAxis(absolute.y, VOXEL_RESOLUTION),
                voxelWrapAxis(absolute.z, VOXEL_RESOLUTION));
}

/// World position -> that cascade's absolute lattice coordinate.
int3 voxelCoordAt(int cascade, float3 world) {
    return int3(floor(world / voxelSize(cascade)));
}

/// World position of a voxel's CENTRE. The half-voxel offset is what makes a
/// trilinear tap at this position return exactly this voxel.
float3 voxelCenter(int cascade, int3 absolute) {
    return (float3(absolute) + 0.5) * voxelSize(cascade);
}

/**
 * Is this absolute lattice coordinate inside the cascade's window?
 *
 * Torus addressing wraps ANY coordinate to a valid texel, so this is the only
 * thing standing between a sample outside the window and a confident answer
 * read from the opposite side of the world. EVERY read must test it.
 */
bool voxelInWindow(int3 absolute, int3 base) {
    int3 local = absolute - base;
    return all(local >= 0) && all(local < VOXEL_RESOLUTION);
}

/**
 * World position -> normalized volume texture coordinate.
 *
 * Because storage is a torus this is simply the world position in units of a
 * whole window: the wrap a REPEAT sampler performs is exactly the wrap the
 * addressing wants, which is why the volumes are created with a repeating
 * sampler rather than a clamped one.
 */
float3 voxelSampleUvw(int cascade, float3 world) {
    return world / (voxelSize(cascade) * float(VOXEL_RESOLUTION));
}

// ---------------------------------------------------------------------------
// Cascade selection
// ---------------------------------------------------------------------------

/**
 * The finest cascade whose window contains `world` with VOXEL_CASCADE_MARGIN
 * voxels to spare, or -1 when even the coarsest does not.
 *
 * The margin is not politeness: a trilinear tap reaches half a voxel past the
 * sample and a mip tap reaches further, so a sample accepted at the window edge
 * would filter against texels belonging to the opposite side of the torus.
 */
int voxelSelectCascade(float3 world, float3 camera) {
    [unroll]
    for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
        float half = voxelCascadeHalfExtent(c) - VOXEL_CASCADE_MARGIN * voxelSize(c);
        float3 d = abs(world - camera);
        if (all(d <= half)) {
            return c;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Cone tracing
// ---------------------------------------------------------------------------

/**
 * Diameter of a cone of half-angle `tanHalfAngle` at distance `t`, floored at
 * one cascade-0 voxel.
 *
 * This is what turns a cone trace into a MIP SELECTION: the right mip is the
 * one whose voxels are about as wide as the cone is there, so the single
 * filtered tap taken at that level stands in for the whole cone cross-section.
 * That substitution is the entire technique.
 */
float voxelConeDiameter(float t, float tanHalfAngle) {
    return max(2.0 * t * tanHalfAngle, VOXEL_BASE_SIZE);
}

/// Fractional on purpose — the sampler interpolates between mips, so a cone
/// that widens smoothly gets a smoothly widening filter instead of stepping.
float voxelConeMip(float diameter, int cascade) {
    return max(log2(diameter / voxelSize(cascade)), 0.0);
}

/**
 * Step length along a cone at distance `t`.
 *
 * Proportional to the cone's own width, which is what bounds the sample count:
 * near the apex the cone is thin and steps are short, far away it is wide and
 * steps are long, so the step count grows with the LOGARITHM of distance. The
 * 0.5 is a half-diameter step, the rate at which front-to-back compositing does
 * not under-count a surface it passes through. The floor is what makes the loop
 * finite at t = 0.
 */
float voxelConeStep(float t, float tanHalfAngle) {
    return max(0.5 * voxelConeDiameter(t, tanHalfAngle), VOXEL_BASE_SIZE * 0.5);
}

/**
 * Front-to-back alpha compositing, one cone step. `accum` is (rgb premultiplied
 * radiance, a occlusion).
 *
 * Standard over-operator: a sample contributes only what the samples in front
 * of it left unblocked, which is what stops a cone that hit a wall from
 * reporting the room behind it.
 */
float4 voxelConeAccumulate(float4 accum, float4 sample) {
    float transmittance = 1.0 - accum.a;
    return float4(accum.rgb + transmittance * sample.a * sample.rgb,
                  accum.a + transmittance * sample.a);
}

#endif // BRACKEN_VOXEL_COMMON_HLSLI
