// The umbrella, used the way a game would use it.
//
// This TU includes `bracken/bracken.hpp` and NOTHING ELSE from the engine —
// no second bracken include, deliberately, and the reviewer should keep it
// that way. That is the whole assertion: if `bracken.hpp` were missing a
// header the reference consumer needs, or if one of the 24 headers stopped
// compiling standalone in a TU with no prior includes, this file would fail
// to compile. `public_headers.test.cpp` checks the same property by scanning;
// this one checks it by *being* the consumer, which is the version that
// catches a missing declaration rather than a missing file.
//
// The runtime half is secondary. It exists so this is a test rather than a
// compile-only stub with no failure mode, and it stays on the GPU-free part of
// the surface — a headless Application subclass describing a frame through a
// FrameSink — so ctest can actually run it with no window and no device.

#include "bracken/bracken.hpp"

#include <gtest/gtest.h>
#include <type_traits>

namespace {

// A game, near enough: subclass Application, describe a frame, stop.
class UmbrellaGame final : public bracken::Application {
public:
    int sim_frames = 0;

protected:
    bool on_init() override { return true; }

    void on_simulate(bracken::SimFrame& sim) override {
        bracken::scene::FrameSink& frame = sim.frame();

        // The environment contract.
        bracken::render::EnvironmentState env;
        env.sun_dir = glm::normalize(glm::vec3(0.3f, -1.0f, 0.2f));
        env.sun_color = glm::vec3(1.0f, 0.95f, 0.85f);
        frame.set_environment(env);

        // A light, described with the host descriptor rather than the GPU POD.
        bracken::render::lighting::PointLightDesc lamp;
        lamp.position = glm::vec3(0.0f, 2.0f, 0.0f);
        lamp.color = glm::vec3(1.0f, 0.8f, 0.4f);
        lamp.radius = 8.0f;
        lamp.intensity = 4.0f;
        frame.add_point_light(lamp);

        ++sim_frames;
        if (sim_frames >= 2) {
            quit();
        }
    }
};

} // namespace

// Every type a host names should be spellable after the umbrella alone. These
// are compile-time assertions dressed as a test body; the static_asserts are
// the point.
TEST(BrackenUmbrella, PublicTypesAreReachable) {
    static_assert(std::is_base_of_v<bracken::Application, UmbrellaGame>);
    static_assert(std::is_abstract_v<bracken::Application>);

    // Frame description.
    static_assert(sizeof(bracken::scene::FrameSink) > 0);
    static_assert(sizeof(bracken::render::EnvironmentState) > 0);
    static_assert(sizeof(bracken::render::lighting::PointLightDesc) > 0);
    static_assert(sizeof(bracken::render::lighting::SpotLightDesc) > 0);

    // Assets, renderer, camera.
    static_assert(sizeof(bracken::Model) > 0);
    static_assert(sizeof(bracken::ModelManager) > 0);
    static_assert(sizeof(bracken::Heightmap) > 0);
    static_assert(sizeof(bracken::scene::SceneRenderer) > 0);
    static_assert(sizeof(bracken::render::TerrainRenderer) > 0);
    static_assert(sizeof(bracken::render::GrassRenderer) > 0);
    static_assert(sizeof(bracken::render::WaterRenderer) > 0);
    static_assert(sizeof(bracken::render::IBL) > 0);
    static_assert(sizeof(bracken::systems::CameraController) > 0);

    // Animation, effects, physics, procedural, services.
    static_assert(bracken::animation::MAX_BONES == 128);
    static_assert(sizeof(bracken::animation::AnimationPlayer) > 0);
    static_assert(sizeof(bracken::animation::AnimationStateMachine) > 0);
    static_assert(sizeof(bracken::EffectDefinition) > 0);
    static_assert(sizeof(bracken::physics::PhysicsWorld) > 0);
    static_assert(sizeof(bracken::procedural::TreeGenerator) > 0);
    static_assert(sizeof(bracken::procedural::RockGenerator) > 0);
    static_assert(sizeof(bracken::core::Logger) > 0);
    static_assert(sizeof(bracken::core::jobs::JobSystem) > 0);
    static_assert(sizeof(bracken::dev::DevUI) > 0);

    SUCCEED();
}

// The umbrella is enough to write and run a host. This is application.test.cpp's
// headless pattern (no init_engine, no window, no renderer): the pipeline
// coordinator needs only the job-system singleton.
TEST(BrackenUmbrella, HostRunsHeadlessFromTheUmbrellaAlone) {
    if (!bracken::core::jobs::JobSystem::instance().is_initialized()) {
        bracken::core::jobs::JobSystem::instance().init();
    }

    UmbrellaGame game;
    game.run();

    // quit() lands in sim frame 2; the pipeline (depth 2) may kick one more
    // before observing the flag — application.test.cpp documents the same
    // window.
    EXPECT_GE(game.sim_frames, 2);
    EXPECT_LE(game.sim_frames, 3);
}
