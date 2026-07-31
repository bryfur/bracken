#include <gtest/gtest.h>

#include "bracken/core/asset/asset_registry.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using bracken::core::asset::AssetRegistry;

namespace {

struct StubAsset {
    int value = 0;
    bool reload_from(const fs::path& p) {
        std::ifstream in(p);
        if (!in) {
            return false;
        }
        in >> value;
        return true;
    }
};

fs::path tmp_path(const std::string& name) {
    auto p = fs::temp_directory_path() / ("mmo_assetreg_" + std::to_string(::getpid()) + "_" + name);
    return p;
}

void write_int(const fs::path& p, int v) {
    std::ofstream o(p, std::ios::trunc);
    o << v;
}

} // namespace

TEST(AssetRegistry, LoadAndGet) {
    auto path = tmp_path("a");
    write_int(path, 42);

    AssetRegistry<StubAsset> reg;
    auto h = reg.load("a", path);
    ASSERT_TRUE(h.is_valid());

    auto* a = reg.get(h);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->value, 42);

    fs::remove(path);
}

TEST(AssetRegistry, ReloadUpdatesValue) {
    auto path = tmp_path("b");
    write_int(path, 1);

    AssetRegistry<StubAsset> reg;
    auto h = reg.load("b", path);
    ASSERT_TRUE(h.is_valid());
    EXPECT_EQ(reg.get(h)->value, 1);

    write_int(path, 7);
    EXPECT_TRUE(reg.reload(h));
    EXPECT_EQ(reg.get(h)->value, 7);

    fs::remove(path);
}

TEST(AssetRegistry, FindByName) {
    auto path = tmp_path("c");
    write_int(path, 9);

    AssetRegistry<StubAsset> reg;
    auto h = reg.load("named", path);
    ASSERT_TRUE(h.is_valid());

    auto h2 = reg.find("named");
    EXPECT_EQ(h, h2);
    EXPECT_EQ(reg.find("missing").is_valid(), false);

    fs::remove(path);
}

TEST(AssetRegistry, UnloadInvalidatesHandle) {
    auto path = tmp_path("d");
    write_int(path, 5);

    AssetRegistry<StubAsset> reg;
    auto h = reg.load("d", path);
    ASSERT_TRUE(h.is_valid());

    reg.unload(h);
    EXPECT_EQ(reg.get(h), nullptr);

    fs::remove(path);
}

TEST(AssetRegistry, CustomLoadFn) {
    auto path = tmp_path("e");
    write_int(path, 0);

    AssetRegistry<StubAsset> reg;
    reg.set_load_fn([](const fs::path&, StubAsset& out) {
        out.value = 1234;
        return true;
    });
    auto h = reg.load("e", path);
    ASSERT_TRUE(h.is_valid());
    EXPECT_EQ(reg.get(h)->value, 1234);

    fs::remove(path);
}

TEST(AssetRegistry, FailedLoadReturnsInvalid) {
    AssetRegistry<StubAsset> reg;
    auto h = reg.load("ghost", tmp_path("does_not_exist"));
    EXPECT_FALSE(h.is_valid());
}

// Duplicate-name load must replace (unload) the previous entry instead of
// orphaning it: the old handle dies, the name maps to the new entry.
TEST(AssetRegistry, DuplicateNameLoadReplacesPreviousEntry) {
    auto path_a = tmp_path("dup_a");
    auto path_b = tmp_path("dup_b");
    write_int(path_a, 1);
    write_int(path_b, 2);

    AssetRegistry<StubAsset> reg;
    auto a = reg.load("hero", path_a);
    ASSERT_TRUE(a.is_valid());
    auto b = reg.load("hero", path_b);
    ASSERT_TRUE(b.is_valid());

    // Old entry was unloaded, not orphaned.
    EXPECT_EQ(reg.get(a), nullptr);
    // Name resolves to the new entry.
    EXPECT_EQ(reg.find("hero"), b);
    ASSERT_NE(reg.get(b), nullptr);
    EXPECT_EQ(reg.get(b)->value, 2);

    fs::remove(path_a);
    fs::remove(path_b);
}

// The exact corrupting sequence from the audit: load "hero" a, load "hero" b,
// unload(a) — find("hero") must still return b (unloading a stale handle must
// not erase the newer entry's name mapping).
TEST(AssetRegistry, UnloadStaleHandleKeepsNewerNameMapping) {
    auto path_a = tmp_path("stale_a");
    auto path_b = tmp_path("stale_b");
    write_int(path_a, 10);
    write_int(path_b, 20);

    AssetRegistry<StubAsset> reg;
    auto a = reg.load("hero", path_a);
    ASSERT_TRUE(a.is_valid());
    auto b = reg.load("hero", path_b);
    ASSERT_TRUE(b.is_valid());

    reg.unload(a); // stale handle: must be a no-op for the name mapping

    EXPECT_EQ(reg.find("hero"), b);
    ASSERT_NE(reg.get(b), nullptr);
    EXPECT_EQ(reg.get(b)->value, 20);

    fs::remove(path_a);
    fs::remove(path_b);
}

// A failed duplicate-name load must leave the existing entry fully intact.
TEST(AssetRegistry, FailedDuplicateNameLoadKeepsOldEntry) {
    auto path_a = tmp_path("keep_a");
    write_int(path_a, 5);

    AssetRegistry<StubAsset> reg;
    auto a = reg.load("hero", path_a);
    ASSERT_TRUE(a.is_valid());

    auto bad = reg.load("hero", tmp_path("keep_does_not_exist"));
    EXPECT_FALSE(bad.is_valid());

    EXPECT_EQ(reg.find("hero"), a);
    ASSERT_NE(reg.get(a), nullptr);
    EXPECT_EQ(reg.get(a)->value, 5);

    fs::remove(path_a);
}
