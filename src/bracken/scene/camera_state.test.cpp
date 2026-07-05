#include "bracken/scene/camera_state.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

using namespace bracken::scene;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(CameraState, DefaultConstructionZeroInitializes) {
    CameraState cam{};
    // All matrices should be zero-initialized (aggregate init)
    EXPECT_EQ(cam.view, glm::mat4(0.0f));
    EXPECT_EQ(cam.projection, glm::mat4(0.0f));
    EXPECT_EQ(cam.view_projection, glm::mat4(0.0f));
    EXPECT_EQ(cam.position, glm::vec3(0.0f));
}

// ---------------------------------------------------------------------------
// view_projection == projection * view
// ---------------------------------------------------------------------------

TEST(CameraState, ViewProjectionEqualsProjectionTimesView) {
    CameraState cam{};
    cam.view = glm::lookAt(glm::vec3(0.0f, 10.0f, 20.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    cam.projection = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 1000.0f);
    cam.view_projection = cam.projection * cam.view;

    glm::mat4 expected = cam.projection * cam.view;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            EXPECT_FLOAT_EQ(cam.view_projection[col][row], expected[col][row])
                << "mismatch at [" << col << "][" << row << "]";
        }
    }
}

// ---------------------------------------------------------------------------
// Position extraction from view matrix matches stored position
// ---------------------------------------------------------------------------

TEST(CameraState, PositionMatchesViewMatrixInverse) {
    glm::vec3 eye(5.0f, 3.0f, -8.0f);
    glm::vec3 target(0.0f, 0.0f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    CameraState cam{};
    cam.view = glm::lookAt(eye, target, up);
    cam.projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 100.0f);
    cam.view_projection = cam.projection * cam.view;
    cam.position = eye;

    // Extract position from the inverse of the view matrix
    glm::mat4 inv_view = glm::inverse(cam.view);
    glm::vec3 extracted_pos(inv_view[3]);

    EXPECT_NEAR(cam.position.x, extracted_pos.x, 1e-4f);
    EXPECT_NEAR(cam.position.y, extracted_pos.y, 1e-4f);
    EXPECT_NEAR(cam.position.z, extracted_pos.z, 1e-4f);
}

// ---------------------------------------------------------------------------
// Identity matrices produce identity view_projection
// ---------------------------------------------------------------------------

TEST(CameraState, IdentityMatricesProduceIdentityViewProjection) {
    CameraState cam{};
    cam.view = glm::mat4(1.0f);
    cam.projection = glm::mat4(1.0f);
    cam.view_projection = cam.projection * cam.view;

    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float expected = (col == row) ? 1.0f : 0.0f;
            EXPECT_FLOAT_EQ(cam.view_projection[col][row], expected);
        }
    }
}

// ---------------------------------------------------------------------------
// View matrix rotation part is orthonormal (proper rotation, no scale/skew)
// ---------------------------------------------------------------------------

TEST(CameraState, ViewMatrixRotationIsOrthonormal) {
    CameraState cam{};
    cam.view = glm::lookAt(glm::vec3(3.0f, 7.0f, -2.0f), glm::vec3(1.0f, 0.5f, 4.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat3 rot(cam.view);
    glm::mat3 product = rot * glm::transpose(rot);
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            float expected = (col == row) ? 1.0f : 0.0f;
            EXPECT_NEAR(product[col][row], expected, 1e-5f) << "mismatch at [" << col << "][" << row << "]";
        }
    }
    // Determinant +1: a proper rotation, not a reflection.
    EXPECT_NEAR(glm::determinant(rot), 1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Projection is right-handed with reversed zero-to-one depth
// (GLM_FORCE_DEPTH_ZERO_TO_ONE + the engine's reversed-Z convention: the
// near/far arguments are swapped in the glm::perspective call, mapping the
// near plane to depth 1 and the far plane to depth 0).
// ---------------------------------------------------------------------------

TEST(CameraState, ProjectionIsRightHandedReversedZeroToOneDepth) {
    constexpr float near_plane = 0.1f;
    constexpr float far_plane = 100.0f;
    CameraState cam{};
    // Engine reversed-Z idiom: swap near/far in the RH_ZO perspective call.
    cam.projection = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, far_plane, near_plane);

    // Right-handed: the camera looks down -Z, so view-space points in front of
    // the camera (negative z) must project to positive clip w.
    glm::vec4 clip_near = cam.projection * glm::vec4(0.0f, 0.0f, -near_plane, 1.0f);
    glm::vec4 clip_far = cam.projection * glm::vec4(0.0f, 0.0f, -far_plane, 1.0f);
    EXPECT_GT(clip_near.w, 0.0f);
    EXPECT_GT(clip_far.w, 0.0f);

    // Reversed zero-to-one depth: near plane maps to NDC z == 1, far to 0.
    EXPECT_NEAR(clip_near.z / clip_near.w, 1.0f, 1e-5f);
    EXPECT_NEAR(clip_far.z / clip_far.w, 0.0f, 1e-6f);

    // A point between the planes stays strictly inside (0, 1).
    glm::vec4 clip_mid = cam.projection * glm::vec4(0.0f, 0.0f, -10.0f, 1.0f);
    float ndc_mid = clip_mid.z / clip_mid.w;
    EXPECT_GT(ndc_mid, 0.0f);
    EXPECT_LT(ndc_mid, 1.0f);

    // A point behind the camera (positive view-space z) lands at negative w.
    glm::vec4 clip_behind = cam.projection * glm::vec4(0.0f, 0.0f, near_plane, 1.0f);
    EXPECT_LT(clip_behind.w, 0.0f);
}

// ---------------------------------------------------------------------------
// Reversed-Z depth precision: with the engine camera range (near 5, far
// 15000), two distant surfaces one unit apart must still land on distinct
// float32 device depths, with the nearer one at GREATER depth (the reversed-Z
// ordering the scene pipelines test against).
// ---------------------------------------------------------------------------

TEST(CameraState, ReversedZKeepsDistantDepthsDistinct) {
    // The engine camera projection (systems/camera_system.cpp): reversed-Z
    // via swapped near/far.
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 15000.0f, 5.0f);

    auto device_depth = [&](float view_z) {
        glm::vec4 clip = proj * glm::vec4(0.0f, 0.0f, view_z, 1.0f);
        return clip.z / clip.w;
    };

    const float d_near = device_depth(-10000.0f);
    const float d_far = device_depth(-10001.0f);

    // Both are valid device depths...
    EXPECT_GT(d_near, 0.0f);
    EXPECT_LT(d_near, 1.0f);
    EXPECT_GT(d_far, 0.0f);
    // ...and strictly ordered: the nearer surface has the GREATER device
    // depth, and the two do not collapse onto the same float32 value (under
    // standard-Z at this range both round to the same value near 1.0 — the
    // exact artifact reversed-Z exists to fix).
    EXPECT_GT(d_near, d_far);

    // Recovering the near plane from the reversed-Z matrix uses
    // n = m32 / (m22 + 1) (scene_renderer.cpp); pin that formula here.
    EXPECT_NEAR(proj[3][2] / (proj[2][2] + 1.0f), 5.0f, 1e-3f);
}

// ---------------------------------------------------------------------------
// view * inverse(view) == identity
// ---------------------------------------------------------------------------

TEST(CameraState, ViewTimesInverseViewIsIdentity) {
    CameraState cam{};
    cam.view = glm::lookAt(glm::vec3(-4.0f, 12.0f, 9.0f), glm::vec3(2.0f, 1.0f, -3.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 product = cam.view * glm::inverse(cam.view);
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float expected = (col == row) ? 1.0f : 0.0f;
            EXPECT_NEAR(product[col][row], expected, 1e-5f) << "mismatch at [" << col << "][" << row << "]";
        }
    }
}

// ---------------------------------------------------------------------------
// Fov changes are reflected in the projection matrix
// ---------------------------------------------------------------------------

TEST(CameraState, FovChangeIsReflectedInProjection) {
    constexpr float aspect = 16.0f / 9.0f;
    CameraState cam{};
    cam.projection = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 1000.0f);

    // Perspective scale: [1][1] = 1 / tan(fov/2), [0][0] = [1][1] / aspect.
    const float focal = 1.0f / std::tan(glm::radians(60.0f) * 0.5f);
    EXPECT_NEAR(cam.projection[1][1], focal, 1e-5f);
    EXPECT_NEAR(cam.projection[0][0], focal / aspect, 1e-5f);

    // A wider fov shrinks both scale factors.
    glm::mat4 wider = glm::perspective(glm::radians(90.0f), aspect, 0.1f, 1000.0f);
    EXPECT_LT(wider[1][1], cam.projection[1][1]);
    EXPECT_LT(wider[0][0], cam.projection[0][0]);
}

// ---------------------------------------------------------------------------
// Aspect changes are reflected in the projection matrix
// ---------------------------------------------------------------------------

TEST(CameraState, AspectChangeIsReflectedInProjection) {
    const float fov = glm::radians(70.0f);
    constexpr float wide_aspect = 21.0f / 9.0f;
    glm::mat4 wide = glm::perspective(fov, wide_aspect, 0.1f, 500.0f);
    glm::mat4 square = glm::perspective(fov, 1.0f, 0.1f, 500.0f);

    // Vertical scale depends only on fov; horizontal scale shrinks as aspect widens.
    EXPECT_FLOAT_EQ(wide[1][1], square[1][1]);
    EXPECT_LT(wide[0][0], square[0][0]);
    EXPECT_NEAR(wide[0][0], wide[1][1] / wide_aspect, 1e-5f);
    EXPECT_FLOAT_EQ(square[0][0], square[1][1]);
}

// ---------------------------------------------------------------------------
// Camera position is recoverable from the inverse view for arbitrary eyes
// ---------------------------------------------------------------------------

TEST(CameraState, PositionRecoverableFromInverseViewForVariousEyes) {
    const glm::vec3 eyes[] = {
        {10.0f, 2.0f, 0.0f},
        {-3.5f, 8.0f, 12.25f},
        {0.0f, 100.0f, -50.0f},
    };
    for (const glm::vec3& eye : eyes) {
        CameraState cam{};
        cam.view = glm::lookAt(eye, glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        cam.position = eye;

        glm::vec3 recovered(glm::inverse(cam.view)[3]);
        EXPECT_NEAR(recovered.x, cam.position.x, 1e-3f) << "eye (" << eye.x << ", " << eye.y << ", " << eye.z << ")";
        EXPECT_NEAR(recovered.y, cam.position.y, 1e-3f) << "eye (" << eye.x << ", " << eye.y << ", " << eye.z << ")";
        EXPECT_NEAR(recovered.z, cam.position.z, 1e-3f) << "eye (" << eye.x << ", " << eye.y << ", " << eye.z << ")";
    }
}
