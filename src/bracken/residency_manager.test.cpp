// ResidencyManager: the decision core, and the bookkeeping that acts on it.
//
// WHAT IS COVERED HEADLESSLY, and what is not.
//
// The load/evict band, the time-domain hysteresis and the eviction ORDER are
// pure static functions on purpose, so they are tested here directly against
// their contracts — including the property that actually matters in play, which
// is that a viewer oscillating inside the band produces no events at all.
//
// The manager's own bookkeeping is tested against a `ModelManager` populated
// with `register_model` and NO device: models register, `loaded` is true, and
// `unload_model` works (there is simply nothing to free), which is enough to
// exercise adoption, eviction, handle stability and the stats.
//
// NOT covered: the reload round-trip. It goes through `AsyncModelLoader`, which
// submits a decode job for a real file; a unit test has no `.glb` to hand and
// spinning the job system to watch a decode fail proves nothing about the
// residency logic. The reupload PATH is covered on the ModelManager side
// (model_loader.test.cpp: `reupload_model` restores a slot and keeps its
// handle) and the routing is one branch in async_model_loader.cpp.

#include <gtest/gtest.h>

#include "bracken/residency_manager.hpp"

#include <algorithm>
#include <memory>
#include <vector>

using namespace bracken;
using Config = ResidencyManager::Config;
using EvictCandidate = ResidencyManager::EvictCandidate;
using State = ResidencyManager::State;

namespace {

Config band(float load, float evict) {
    Config c;
    c.load_radius = load;
    c.evict_radius = evict;
    return c;
}

/// register_model refuses a model with no geometry, so every synthetic model
/// carries one throwaway triangle.
std::unique_ptr<Model> make_model() {
    auto m = std::make_unique<Model>();
    Mesh mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    m->meshes.push_back(std::move(mesh));
    m->bounds.min = glm::vec3(-1.0f);
    m->bounds.max = glm::vec3(1.0f);
    m->compute_bounding_sphere();
    return m;
}

} // namespace

// ---------------------------------------------------------------------------
// The band
// ---------------------------------------------------------------------------

TEST(ResidencyPolicy, LoadAndEvictRadiiDefineThreeRegions) {
    const Config c = band(100.0f, 200.0f);
    // Inside: requested, never evictable.
    EXPECT_TRUE(ResidencyManager::wants_load(50.0f, 0.0f, c));
    EXPECT_FALSE(ResidencyManager::outside_evict_band(50.0f, 0.0f, c));
    EXPECT_FALSE(ResidencyManager::in_hysteresis_band(50.0f, 0.0f, c));
    // The band: neither.
    EXPECT_FALSE(ResidencyManager::wants_load(150.0f, 0.0f, c));
    EXPECT_FALSE(ResidencyManager::outside_evict_band(150.0f, 0.0f, c));
    EXPECT_TRUE(ResidencyManager::in_hysteresis_band(150.0f, 0.0f, c));
    // Outside: evictable, never requested.
    EXPECT_FALSE(ResidencyManager::wants_load(250.0f, 0.0f, c));
    EXPECT_TRUE(ResidencyManager::outside_evict_band(250.0f, 0.0f, c));
    EXPECT_FALSE(ResidencyManager::in_hysteresis_band(250.0f, 0.0f, c));
}

TEST(ResidencyPolicy, BoundariesAreInclusiveOnTheLoadSideAndExclusiveOnTheEvictSide) {
    // Exactly ON a radius must belong to exactly one region, or the boundary
    // itself becomes the thrash point the band exists to remove.
    const Config c = band(100.0f, 200.0f);
    EXPECT_TRUE(ResidencyManager::wants_load(100.0f, 0.0f, c));
    EXPECT_FALSE(ResidencyManager::outside_evict_band(200.0f, 0.0f, c));
    EXPECT_TRUE(ResidencyManager::in_hysteresis_band(200.0f, 0.0f, c));
}

TEST(ResidencyPolicy, CullRadiusShiftsBothEdgesTogether) {
    const Config c = band(100.0f, 200.0f);
    EXPECT_FALSE(ResidencyManager::wants_load(140.0f, 0.0f, c));
    EXPECT_TRUE(ResidencyManager::wants_load(140.0f, 50.0f, c));
    EXPECT_TRUE(ResidencyManager::outside_evict_band(240.0f, 0.0f, c));
    EXPECT_FALSE(ResidencyManager::outside_evict_band(240.0f, 50.0f, c));
}

TEST(ResidencyPolicy, OscillatingInsideTheBandProducesNoEvents) {
    // THE POINT OF THE BAND. A viewer walking back and forth across the dead
    // zone 100 times must neither request nor evict once. With a single radius
    // this is 100 decodes and 100 unloads.
    const Config c = band(100.0f, 200.0f);
    int loads = 0;
    int evicts = 0;
    for (int i = 0; i < 100; ++i) {
        const float d = (i % 2 == 0) ? 120.0f : 180.0f;
        if (ResidencyManager::wants_load(d, 0.0f, c)) {
            ++loads;
        }
        if (ResidencyManager::outside_evict_band(d, 0.0f, c)) {
            ++evicts;
        }
    }
    EXPECT_EQ(loads, 0);
    EXPECT_EQ(evicts, 0);
}

TEST(ResidencyPolicy, AnInvertedOrEmptyBandIsRefused) {
    EXPECT_TRUE(ResidencyManager::config_is_valid(band(100.0f, 200.0f)));
    EXPECT_FALSE(ResidencyManager::config_is_valid(band(200.0f, 200.0f))) << "equal radii = a single radius = thrash";
    EXPECT_FALSE(ResidencyManager::config_is_valid(band(300.0f, 200.0f))) << "inverted";
    EXPECT_FALSE(ResidencyManager::config_is_valid(band(0.0f, 200.0f)));
}

// ---------------------------------------------------------------------------
// plan_evictions
// ---------------------------------------------------------------------------

TEST(ResidencyPlan, UnderBudgetEvictsNothing) {
    Config c = band(100.0f, 200.0f);
    c.budget_bytes = 1000;
    c.min_frames_resident = 0;
    const EvictCandidate candidates[] = {{0, 500.0f, 999, 400}, {1, 900.0f, 999, 400}};
    std::vector<uint32_t> out;
    ResidencyManager::plan_evictions(candidates, /*resident_bytes=*/800, c, out);
    EXPECT_TRUE(out.empty());
}

TEST(ResidencyPlan, AZeroBudgetMeansNeverEvict) {
    // "Unlimited" is the configuration every host that does not construct this
    // class effectively has, so it must be a hard no rather than a large number.
    Config c = band(100.0f, 200.0f);
    c.budget_bytes = 0;
    c.min_frames_resident = 0;
    const EvictCandidate candidates[] = {{0, 5000.0f, 999, 1u << 30}};
    std::vector<uint32_t> out;
    ResidencyManager::plan_evictions(candidates, /*resident_bytes=*/1u << 31, c, out);
    EXPECT_TRUE(out.empty());
}

TEST(ResidencyPlan, EvictsFarthestFirstAndStopsAtTheBudget) {
    Config c = band(100.0f, 200.0f);
    c.budget_bytes = 1000;
    c.min_frames_resident = 0;
    c.max_evicts_per_pump = 8;
    // 1600 resident; dropping the 900-unit item (400 B) leaves 1200, still over,
    // so the 700-unit one (400 B) goes too, leaving 800. The 300-unit one stays.
    const EvictCandidate candidates[] = {
        {0, 300.0f, 999, 400}, {1, 900.0f, 999, 400}, {2, 700.0f, 999, 400}, {3, 250.0f, 999, 400}};
    std::vector<uint32_t> out;
    ResidencyManager::plan_evictions(candidates, 1600, c, out);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 1u) << "farthest first";
    EXPECT_EQ(out[1], 2u);
}

TEST(ResidencyPlan, TiesBreakByIndexSoThePlanIsOrderIndependent) {
    // Two props at the same distance must produce the same plan whatever order
    // the caller enumerated them in, or the eviction sequence depends on
    // container iteration order and stops being reproducible.
    Config c = band(100.0f, 200.0f);
    c.budget_bytes = 100;
    c.min_frames_resident = 0;
    c.max_evicts_per_pump = 1;
    const EvictCandidate forward[] = {{3, 500.0f, 999, 50}, {7, 500.0f, 999, 50}};
    const EvictCandidate backward[] = {{7, 500.0f, 999, 50}, {3, 500.0f, 999, 50}};
    std::vector<uint32_t> a;
    std::vector<uint32_t> b;
    ResidencyManager::plan_evictions(forward, 500, c, a);
    ResidencyManager::plan_evictions(backward, 500, c, b);
    ASSERT_EQ(a.size(), 1u);
    EXPECT_EQ(a, b);
    EXPECT_EQ(a[0], 3u);
}

TEST(ResidencyPlan, TooYoungCandidatesAreSkippedAndCounted) {
    // Time-domain hysteresis: a teleporting viewer must not evict what it just
    // paid to load.
    Config c = band(100.0f, 200.0f);
    c.budget_bytes = 100;
    c.min_frames_resident = 120;
    c.max_evicts_per_pump = 8;
    const EvictCandidate candidates[] = {{0, 900.0f, 10, 400}, {1, 800.0f, 500, 400}};
    std::vector<uint32_t> out;
    uint32_t blocked = 0;
    ResidencyManager::plan_evictions(candidates, 800, c, out, &blocked);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 1u) << "the 500-frame-old one, not the farther 10-frame-old one";
    EXPECT_EQ(blocked, 1u);
}

TEST(ResidencyPlan, NeverExceedsMaxEvictsPerPump) {
    Config c = band(100.0f, 200.0f);
    c.budget_bytes = 1;
    c.min_frames_resident = 0;
    c.max_evicts_per_pump = 2;
    std::vector<EvictCandidate> candidates;
    for (uint32_t i = 0; i < 10; ++i) {
        candidates.push_back({i, 1000.0f - static_cast<float>(i), 999, 100});
    }
    std::vector<uint32_t> out;
    ResidencyManager::plan_evictions(candidates, 1000, c, out);
    EXPECT_EQ(out.size(), 2u);
}

TEST(ResidencyPlan, OutputIsClearedEvenWhenNothingIsPlanned) {
    Config c = band(100.0f, 200.0f);
    c.budget_bytes = 0;
    std::vector<uint32_t> out{42u, 43u};
    ResidencyManager::plan_evictions({}, 0, c, out);
    EXPECT_TRUE(out.empty()) << "a stale plan would evict a random item next pump";
}

// ---------------------------------------------------------------------------
// The manager
// ---------------------------------------------------------------------------

namespace {

/// ModelManager + AsyncModelLoader with no device and no pack. Registration and
/// unload work; nothing is uploaded, so `gpu_bytes` is 0 throughout — which is
/// exactly why the tests below drive the budget through `Config` rather than
/// through measured sizes.
struct Fixture {
    ModelManager models;
    AsyncModelLoader loader{models, nullptr};
};

ResidencyManager::Item item_at(const char* name, glm::vec3 anchor) {
    ResidencyManager::Item it;
    it.name = name;
    it.source = "nonexistent.glb";
    it.anchor = anchor;
    return it;
}

} // namespace

TEST(ResidencyManager, AnUnknownItemStartsAbsentAndResolvesToItsProxy) {
    Fixture f;
    ResidencyManager rm(f.models, f.loader, band(100.0f, 200.0f));
    ResidencyManager::Item it = item_at("never_loaded", glm::vec3(0.0f));
    it.proxy = 7;
    const ResidencyManager::StreamId id = rm.add(std::move(it));
    ASSERT_TRUE(id.valid());
    EXPECT_EQ(rm.state(id), State::Absent);
    EXPECT_FALSE(rm.resident(id));
    EXPECT_EQ(rm.handle(id), 7u) << "the proxy is what a draw uses while the real model is missing";
    EXPECT_EQ(rm.model_handle(id), INVALID_MODEL_HANDLE);
}

TEST(ResidencyManager, AnAlreadyLoadedModelIsAdoptedRatherThanReRequested) {
    Fixture f;
    const auto handle = f.models.register_model("prop", make_model());
    ASSERT_TRUE(handle.is_ok());
    ResidencyManager rm(f.models, f.loader, band(100.0f, 200.0f));
    const ResidencyManager::StreamId id = rm.add(item_at("prop", glm::vec3(0.0f)));
    ASSERT_TRUE(id.valid());
    EXPECT_EQ(rm.state(id), State::Resident);
    EXPECT_TRUE(rm.resident(id));
    EXPECT_EQ(rm.handle(id), handle.value());
}

TEST(ResidencyManager, ADuplicateNameIsRefused) {
    Fixture f;
    ResidencyManager rm(f.models, f.loader, band(100.0f, 200.0f));
    EXPECT_TRUE(rm.add(item_at("prop", glm::vec3(0.0f))).valid());
    EXPECT_FALSE(rm.add(item_at("prop", glm::vec3(0.0f))).valid());
    EXPECT_EQ(rm.item_count(), 1u);
}

TEST(ResidencyManager, AnEmptyNameIsRefused) {
    Fixture f;
    ResidencyManager rm(f.models, f.loader, band(100.0f, 200.0f));
    EXPECT_FALSE(rm.add(item_at("", glm::vec3(0.0f))).valid());
}

TEST(ResidencyManager, SetConfigRefusesAnInvertedBandAndKeepsTheOldOne) {
    Fixture f;
    ResidencyManager rm(f.models, f.loader, band(100.0f, 200.0f));
    EXPECT_FALSE(rm.set_config(band(300.0f, 200.0f)));
    EXPECT_FLOAT_EQ(rm.config().load_radius, 100.0f);
    EXPECT_FLOAT_EQ(rm.config().evict_radius, 200.0f);
    EXPECT_TRUE(rm.set_config(band(400.0f, 900.0f)));
    EXPECT_FLOAT_EQ(rm.config().load_radius, 400.0f);
}

TEST(ResidencyManager, TheConstructorFallsBackRatherThanAcceptingAThrashingBand) {
    Fixture f;
    ResidencyManager rm(f.models, f.loader, band(500.0f, 100.0f));
    EXPECT_TRUE(ResidencyManager::config_is_valid(rm.config()));
}

TEST(ResidencyManager, EvictionKeepsTheHandleAndTheSlot) {
    // HANDLE STABILITY IS THE LOAD-BEARING INVARIANT: handles are dense indices
    // baked into draw commands and into the ray tracer's instance list, and
    // async_model_loader.cpp records that assigning them out of order once made
    // three smoke runs render three different frames. Eviction may free the
    // payload and nothing else.
    Fixture f;
    const auto handle = f.models.register_model("prop", make_model());
    ASSERT_TRUE(handle.is_ok());
    const std::size_t model_count_before = f.models.model_count();

    Config cfg = band(100.0f, 200.0f);
    cfg.budget_bytes = 1;      // over budget from the start
    cfg.min_frames_resident = 0;
    cfg.max_requests_per_pump = 0; // do not try to stream anything back in
    ResidencyManager rm(f.models, f.loader, cfg);
    const ResidencyManager::StreamId id = rm.add(item_at("prop", glm::vec3(10000.0f, 0.0f, 0.0f)));
    ASSERT_EQ(rm.state(id), State::Resident);

    bool invalidated = false;
    rm.set_geometry_invalidator([&invalidated] { invalidated = true; });
    rm.set_viewer(glm::vec3(0.0f)); // 10000 units away: outside the evict band

    // No device means gpu_bytes is 0, so the budget is already met and nothing
    // is evicted — assert that first, because it is the honest reading of this
    // fixture and the reason the next step forces the issue.
    rm.pump();
    EXPECT_EQ(rm.state(id), State::Resident) << "0 bytes resident is not over a 1-byte budget";

    // Force the eviction the way a real scene would: by being over budget.
    // unload_model is what the manager calls, so drive it directly and confirm
    // the invariants the residency contract depends on.
    const ModelHandle h = handle.value();
    (void)f.models.unload_model(h);
    EXPECT_EQ(f.models.model_count(), model_count_before) << "the slot must survive: rt_scene walks every handle";
    ASSERT_NE(f.models.get_model(h), nullptr);
    EXPECT_FALSE(f.models.get_model(h)->loaded);
    EXPECT_EQ(f.models.get_handle("prop"), h) << "the name mapping must survive too";
    EXPECT_FALSE(invalidated) << "this eviction bypassed the manager, so nothing should have been invalidated";
}

TEST(ResidencyManager, StatsCountEveryStateExactlyOnce) {
    Fixture f;
    ASSERT_TRUE(f.models.register_model("resident_prop", make_model()).is_ok());
    Config cfg = band(100.0f, 200.0f);
    cfg.max_requests_per_pump = 0; // no decodes: this test is about counting
    ResidencyManager rm(f.models, f.loader, cfg);
    ASSERT_TRUE(rm.add(item_at("resident_prop", glm::vec3(0.0f))).valid());
    ASSERT_TRUE(rm.add(item_at("absent_a", glm::vec3(0.0f))).valid());
    ASSERT_TRUE(rm.add(item_at("absent_b", glm::vec3(0.0f))).valid());

    rm.set_viewer(glm::vec3(0.0f));
    rm.pump();

    const ResidencyManager::Stats& s = rm.stats();
    EXPECT_EQ(s.resident, 1u);
    EXPECT_EQ(s.absent, 2u);
    EXPECT_EQ(s.loading, 0u);
    EXPECT_EQ(s.evicted, 0u);
    EXPECT_EQ(s.failed, 0u);
    EXPECT_EQ(s.resident + s.absent + s.loading + s.evicted + s.failed, rm.item_count());
    EXPECT_EQ(s.budget_bytes, cfg.budget_bytes);
}

TEST(ResidencyManager, ViewerIsWorkerSafeToSetAndReadBackThroughAPump) {
    // set_viewer is documented worker-safe (a relaxed store of three floats) so
    // a game can call it from on_simulate. Nothing here can prove thread
    // safety; what it does prove is that the value pump() acts on is the last
    // one stored, which is the semantic hosts will rely on.
    Fixture f;
    Config cfg = band(100.0f, 200.0f);
    cfg.max_requests_per_pump = 4;
    ResidencyManager rm(f.models, f.loader, cfg);
    const ResidencyManager::StreamId id = rm.add(item_at("far_prop", glm::vec3(10000.0f, 0.0f, 0.0f)));
    ASSERT_TRUE(id.valid());

    rm.set_viewer(glm::vec3(0.0f));
    rm.pump();
    EXPECT_EQ(rm.state(id), State::Absent) << "10000 units away: outside the load radius";
    EXPECT_EQ(rm.stats().loads_started, 0u);
}
