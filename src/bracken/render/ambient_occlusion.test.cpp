#include "bracken/render/ambient_occlusion.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace bracken::render;
using bracken::gpu::GTAOUniforms;

namespace {

constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float TOLERANCE = 1e-5f;

} // namespace

TEST(GtaoDirectionKernel, EvenlySpacedUnitCircleDirections) {
    const auto kernel = detail::gtao_direction_kernel(6);

    for (int i = 0; i < GTAOUniforms::KERNEL_SIZE; ++i) {
        const float angle = (static_cast<float>(i % 6) / 6.0f) * TWO_PI;
        EXPECT_NEAR(kernel[i].x, std::cos(angle), TOLERANCE) << "entry " << i;
        EXPECT_NEAR(kernel[i].y, std::sin(angle), TOLERANCE) << "entry " << i;
        EXPECT_FLOAT_EQ(kernel[i].z, 0.0f);
        EXPECT_FLOAT_EQ(kernel[i].w, 0.0f);
    }
}

// Pins the L6 fix: the kernel used to be a function-local static that latched
// the first call's num_directions forever, so changing the direction count at
// runtime silently kept the old spacing. Each call must honor its argument.
TEST(GtaoDirectionKernel, RegeneratesWhenDirectionCountChanges) {
    const auto six = detail::gtao_direction_kernel(6);
    const auto four = detail::gtao_direction_kernel(4);

    // 6 directions: entry 1 at 60 degrees.
    EXPECT_NEAR(six[1].x, 0.5f, TOLERANCE);
    EXPECT_NEAR(six[1].y, std::sqrt(3.0f) / 2.0f, TOLERANCE);

    // 4 directions requested AFTER 6: entry 1 must now be at 90 degrees,
    // not the latched 60-degree spacing of the first call.
    EXPECT_NEAR(four[1].x, 0.0f, TOLERANCE);
    EXPECT_NEAR(four[1].y, 1.0f, TOLERANCE);

    // And asking for 6 again returns the 6-direction spacing.
    const auto six_again = detail::gtao_direction_kernel(6);
    EXPECT_NEAR(six_again[1].x, six[1].x, TOLERANCE);
    EXPECT_NEAR(six_again[1].y, six[1].y, TOLERANCE);
}

TEST(GtaoDirectionKernel, ClampsNonPositiveCountToOne) {
    for (const int dirs : {0, -3}) {
        const auto kernel = detail::gtao_direction_kernel(dirs);
        for (int i = 0; i < GTAOUniforms::KERNEL_SIZE; ++i) {
            EXPECT_NEAR(kernel[i].x, 1.0f, TOLERANCE) << "dirs " << dirs << " entry " << i;
            EXPECT_NEAR(kernel[i].y, 0.0f, TOLERANCE) << "dirs " << dirs << " entry " << i;
        }
    }
}
