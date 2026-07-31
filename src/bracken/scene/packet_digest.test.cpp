#include <gtest/gtest.h>

#include "bracken/contracts/camera_state.hpp"
#include "bracken/contracts/light.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/packet_digest.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace bracken::scene;

namespace {

// Builds a small but representative frame: model draws, a light, a camera and
// some UI. Every test below starts from this and perturbs exactly one thing,
// so a failure names the field the digest stopped noticing.
void fill_reference_frame(FrameBuilder& fb) {
    fb.draw_model(1, glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
    fb.draw_model(2, glm::scale(glm::mat4(1.0f), glm::vec3(2.0f)));

    bracken::render::lighting::PointLight light{};
    light.position = {10.0f, 5.0f, -2.0f};
    light.radius = 25.0f;
    light.color = {1.0f, 0.5f, 0.25f};
    light.intensity = 4.0f;
    fb.add_point_light(light);

    CameraState cam{};
    cam.view = glm::lookAt(glm::vec3(0, 5, 10), glm::vec3(0), glm::vec3(0, 1, 0));
    cam.projection = glm::mat4(1.0f);
    cam.view_projection = cam.view;
    cam.position = {0, 5, 10};
    fb.set_camera(cam);

    fb.ui().add_text("score 100", 10.0f, 20.0f, 1.0f, 0xFFFFFFFF);
}

uint64_t digest_of(void (*build)(FrameBuilder&)) {
    FramePacket p;
    FrameBuilder fb(p);
    build(fb);
    fb.seal();
    return packet_digest(p);
}

} // namespace

TEST(PacketDigest, IdenticalFramesDigestIdentically) {
    EXPECT_EQ(digest_of(fill_reference_frame), digest_of(fill_reference_frame));
}

TEST(PacketDigest, EmptyFrameIsStableAndNonZero) {
    FramePacket a;
    FrameBuilder fa(a);
    fa.seal();
    FramePacket b;
    FrameBuilder fb(b);
    fb.seal();
    EXPECT_EQ(packet_digest(a), packet_digest(b));
    EXPECT_NE(packet_digest(a), 0u);
}

// Each of the following pins one field the gate must not go blind to. If a
// refactor drops a field from the digest, exactly one of these fails.

TEST(PacketDigest, DetectsChangedTransform) {
    const uint64_t base = digest_of(fill_reference_frame);
    const uint64_t moved = digest_of([](FrameBuilder& fb) {
        fill_reference_frame(fb);
        fb.draw_model(3, glm::translate(glm::mat4(1.0f), glm::vec3(0.001f, 0.0f, 0.0f)));
    });
    EXPECT_NE(base, moved);
}

TEST(PacketDigest, DetectsChangedModelHandle) {
    FramePacket a;
    FrameBuilder fa(a);
    fa.draw_model(1, glm::mat4(1.0f));
    fa.seal();

    FramePacket b;
    FrameBuilder fb(b);
    fb.draw_model(2, glm::mat4(1.0f)); // same transform, different asset
    fb.seal();

    EXPECT_NE(packet_digest(a), packet_digest(b));
}

TEST(PacketDigest, DetectsDroppedDraw) {
    FramePacket a;
    FrameBuilder fa(a);
    fa.draw_model(1, glm::mat4(1.0f));
    fa.draw_model(2, glm::mat4(1.0f));
    fa.seal();

    FramePacket b;
    FrameBuilder fb(b);
    fb.draw_model(1, glm::mat4(1.0f));
    fb.seal();

    EXPECT_NE(packet_digest(a), packet_digest(b));
}

TEST(PacketDigest, DetectsChangedLightIntensity) {
    FramePacket a;
    FrameBuilder fa(a);
    bracken::render::lighting::PointLight la{};
    la.intensity = 1.0f;
    fa.add_point_light(la);
    fa.seal();

    FramePacket b;
    FrameBuilder fb(b);
    bracken::render::lighting::PointLight lb{};
    lb.intensity = 1.5f;
    fb.add_point_light(lb);
    fb.seal();

    EXPECT_NE(packet_digest(a), packet_digest(b));
}

TEST(PacketDigest, DetectsChangedUiText) {
    FramePacket a;
    FrameBuilder fa(a);
    fa.ui().add_text("score 100", 0.0f, 0.0f, 1.0f, 0xFFFFFFFF);
    fa.seal();

    FramePacket b;
    FrameBuilder fb(b);
    fb.ui().add_text("score 101", 0.0f, 0.0f, 1.0f, 0xFFFFFFFF);
    fb.seal();

    // The playground's live FPS readout was exactly this: a wall-clock-derived
    // string in the packet that made the smoke run irreproducible.
    EXPECT_NE(packet_digest(a), packet_digest(b));
}

TEST(PacketDigest, DetectsChangedCameraPosition) {
    FramePacket a;
    FrameBuilder fa(a);
    CameraState ca{};
    ca.position = {0, 0, 0};
    fa.set_camera(ca);
    fa.seal();

    FramePacket b;
    FrameBuilder fb(b);
    CameraState cb{};
    cb.position = {0, 0, 0.5f};
    fb.set_camera(cb);
    fb.seal();

    EXPECT_NE(packet_digest(a), packet_digest(b));
}

// Floats are hashed by bit pattern, so signed zero is a real difference. This
// pins that choice: a value-based hash would silently merge these.
TEST(PacketDigest, DistinguishesSignedZero) {
    // Set the element directly: glm::translate would compute the translation
    // column as a sum, and 0*-0.0 + ... normalizes back to +0.0.
    glm::mat4 pos_zero(1.0f);
    pos_zero[3][0] = 0.0f;

    glm::mat4 neg_zero(1.0f);
    neg_zero[3][0] = -0.0f;

    ASSERT_EQ(pos_zero[3][0], neg_zero[3][0]); // equal as values...

    FramePacket a;
    FrameBuilder fa(a);
    fa.draw_model(1, pos_zero);
    fa.seal();

    FramePacket b;
    FrameBuilder fb(b);
    fb.draw_model(1, neg_zero);
    fb.seal();

    EXPECT_NE(packet_digest(a), packet_digest(b)); // ...distinct as bit patterns
}
