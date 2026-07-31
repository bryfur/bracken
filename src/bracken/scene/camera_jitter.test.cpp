#include "bracken/scene/camera_jitter.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

using namespace bracken::scene;

// =============================================================================
// Halton sequence values
// =============================================================================

TEST(CameraJitter, HaltonBase2KnownValues) {
    // Radical inverse base 2: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, 1/16
    EXPECT_FLOAT_EQ(halton(1, 2), 0.5f);
    EXPECT_FLOAT_EQ(halton(2, 2), 0.25f);
    EXPECT_FLOAT_EQ(halton(3, 2), 0.75f);
    EXPECT_FLOAT_EQ(halton(4, 2), 0.125f);
    EXPECT_FLOAT_EQ(halton(5, 2), 0.625f);
    EXPECT_FLOAT_EQ(halton(6, 2), 0.375f);
    EXPECT_FLOAT_EQ(halton(7, 2), 0.875f);
    EXPECT_FLOAT_EQ(halton(8, 2), 0.0625f);
}

TEST(CameraJitter, HaltonBase3KnownValues) {
    // Radical inverse base 3: 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9, 8/9
    EXPECT_FLOAT_EQ(halton(1, 3), 1.0f / 3.0f);
    EXPECT_FLOAT_EQ(halton(2, 3), 2.0f / 3.0f);
    EXPECT_FLOAT_EQ(halton(3, 3), 1.0f / 9.0f);
    EXPECT_FLOAT_EQ(halton(4, 3), 4.0f / 9.0f);
    EXPECT_FLOAT_EQ(halton(5, 3), 7.0f / 9.0f);
    EXPECT_FLOAT_EQ(halton(6, 3), 2.0f / 9.0f);
    EXPECT_FLOAT_EQ(halton(7, 3), 5.0f / 9.0f);
    EXPECT_FLOAT_EQ(halton(8, 3), 8.0f / 9.0f);
}

TEST(CameraJitter, JitterSequenceCenteredAndBounded) {
    for (uint64_t frame = 0; frame < 64; ++frame) {
        const glm::vec2 j = halton_jitter(frame);
        EXPECT_GE(j.x, -0.5f) << "frame " << frame;
        EXPECT_LT(j.x, 0.5f) << "frame " << frame;
        EXPECT_GE(j.y, -0.5f) << "frame " << frame;
        EXPECT_LT(j.y, 0.5f) << "frame " << frame;
    }
    // First sample of the (2,3) sequence: (0.5 - 0.5, 1/3 - 0.5) = (0, -1/6)
    EXPECT_FLOAT_EQ(halton_jitter(0).x, 0.0f);
    EXPECT_FLOAT_EQ(halton_jitter(0).y, 1.0f / 3.0f - 0.5f);
}

TEST(CameraJitter, JitterSequenceRepeats) {
    for (uint64_t frame = 0; frame < JITTER_SEQUENCE_LENGTH; ++frame) {
        EXPECT_EQ(halton_jitter(frame), halton_jitter(frame + JITTER_SEQUENCE_LENGTH));
    }
}

TEST(CameraJitter, JitterSamplesAreDistinctWithinPeriod) {
    for (uint64_t a = 0; a < JITTER_SEQUENCE_LENGTH; ++a) {
        for (uint64_t b = a + 1; b < JITTER_SEQUENCE_LENGTH; ++b) {
            EXPECT_NE(halton_jitter(a), halton_jitter(b)) << "frames " << a << " and " << b;
        }
    }
}

// =============================================================================
// Projection jitter
// =============================================================================

namespace {

// Engine-convention reversed-Z RH_ZO projection (near/far swapped).
glm::mat4 reversed_z_projection() {
    return glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 15000.0f, 5.0f);
}

glm::vec2 project_to_ndc(const glm::mat4& proj, const glm::vec3& view_pos) {
    const glm::vec4 clip = proj * glm::vec4(view_pos, 1.0f);
    return glm::vec2(clip) / clip.w;
}

} // namespace

TEST(CameraJitter, JitteredProjectionShiftsNdcBySubPixelOffset) {
    const glm::mat4 proj = reversed_z_projection();
    const float w = 1600.0f;
    const float h = 900.0f;
    const glm::vec2 jitter(0.25f, -0.375f); // pixels

    const glm::mat4 jittered = jitter_projection(proj, jitter, w, h);

    // The NDC displacement must be exactly 2*jitter/viewport at ANY depth.
    for (const glm::vec3& p : {glm::vec3(3.0f, -2.0f, -20.0f), glm::vec3(-40.0f, 8.0f, -500.0f),
                               glm::vec3(0.0f, 0.0f, -9000.0f)}) {
        const glm::vec2 base = project_to_ndc(proj, p);
        const glm::vec2 moved = project_to_ndc(jittered, p);
        EXPECT_NEAR(moved.x - base.x, 2.0f * jitter.x / w, 1e-6f);
        EXPECT_NEAR(moved.y - base.y, 2.0f * jitter.y / h, 1e-6f);
    }
}

TEST(CameraJitter, JitterDoesNotTouchDepthOrW) {
    const glm::mat4 proj = reversed_z_projection();
    const glm::mat4 jittered = jitter_projection(proj, {0.5f, 0.5f}, 1600.0f, 900.0f);

    const glm::vec4 p(3.0f, 4.0f, -100.0f, 1.0f);
    const glm::vec4 base = proj * p;
    const glm::vec4 moved = jittered * p;
    EXPECT_FLOAT_EQ(base.z, moved.z);
    EXPECT_FLOAT_EQ(base.w, moved.w);
}

TEST(CameraJitter, ZeroJitterIsIdentity) {
    const glm::mat4 proj = reversed_z_projection();
    EXPECT_EQ(jitter_projection(proj, {0.0f, 0.0f}, 1600.0f, 900.0f), proj);
}

TEST(CameraJitter, DegenerateViewportIsPassthrough) {
    const glm::mat4 proj = reversed_z_projection();
    EXPECT_EQ(jitter_projection(proj, {0.5f, 0.5f}, 0.0f, 900.0f), proj);
}
