#include <gtest/gtest.h>

#include "bracken/core/asset/asset_pack.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace bracken::core::asset;

namespace {

std::filesystem::path temp_dir() {
    return std::filesystem::temp_directory_path() / "bracken_pack_test";
}

struct Fixture {
    std::filesystem::path dir = temp_dir();
    Fixture() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
    }
    ~Fixture() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    std::string write(const std::string& leaf, const std::string& contents) const {
        const std::string p = (dir / leaf).string();
        std::ofstream out(p, std::ios::binary);
        out << contents;
        return p;
    }
    [[nodiscard]] std::string pack_path() const { return (dir / "test.pack").string(); }
};

std::string as_string(const std::vector<std::byte>& b) {
    return std::string(reinterpret_cast<const char*>(b.data()), b.size());
}

} // namespace

TEST(AssetPack, RoundTripsEntries) {
    Fixture f;
    const auto a = f.write("a.bin", "hello");
    const auto b = f.write("b.bin", "a somewhat longer payload than the first one");

    std::vector<std::pair<std::string, std::string>> sources{{"models/a.glb", a},
                                                             {"textures/b.png", b}};
    ASSERT_TRUE(AssetPack::build(f.pack_path(), sources));

    auto opened = AssetPack::open(f.pack_path());
    ASSERT_TRUE(opened) << to_string(opened.error());
    const AssetPack& pack = opened.value();

    EXPECT_EQ(pack.entries().size(), 2u);
    EXPECT_TRUE(pack.contains("models/a.glb"));
    EXPECT_TRUE(pack.contains("textures/b.png"));
    EXPECT_FALSE(pack.contains("nope"));

    auto ra = pack.read("models/a.glb");
    ASSERT_TRUE(ra);
    EXPECT_EQ(as_string(ra.value()), "hello");

    auto rb = pack.read("textures/b.png");
    ASSERT_TRUE(rb);
    EXPECT_EQ(as_string(rb.value()), "a somewhat longer payload than the first one");
}

// Entries are sorted at build time so open() can binary-search a 50k-entry TOC
// instead of scanning it.
TEST(AssetPack, EntriesAreSortedByName) {
    Fixture f;
    std::vector<std::pair<std::string, std::string>> sources{
        {"zeta", f.write("z.bin", "z")},
        {"alpha", f.write("a.bin", "a")},
        {"mid", f.write("m.bin", "m")}};
    ASSERT_TRUE(AssetPack::build(f.pack_path(), sources));

    auto pack = AssetPack::open(f.pack_path());
    ASSERT_TRUE(pack);
    const auto entries = pack.value().entries();
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].name, "alpha");
    EXPECT_EQ(entries[1].name, "mid");
    EXPECT_EQ(entries[2].name, "zeta");
}

TEST(AssetPack, MissingEntryIsReported) {
    Fixture f;
    std::vector<std::pair<std::string, std::string>> sources{{"only", f.write("o.bin", "x")}};
    ASSERT_TRUE(AssetPack::build(f.pack_path(), sources));

    auto pack = AssetPack::open(f.pack_path());
    ASSERT_TRUE(pack);
    auto r = pack.value().read("absent");
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), PackError::EntryNotFound);
}

TEST(AssetPack, EmptyPackIsValid) {
    Fixture f;
    ASSERT_TRUE(AssetPack::build(f.pack_path(), {}));
    auto pack = AssetPack::open(f.pack_path());
    ASSERT_TRUE(pack);
    EXPECT_TRUE(pack.value().entries().empty());
}

TEST(AssetPack, EmptyFileEntryRoundTrips) {
    Fixture f;
    std::vector<std::pair<std::string, std::string>> sources{{"zero", f.write("z.bin", "")}};
    ASSERT_TRUE(AssetPack::build(f.pack_path(), sources));
    auto pack = AssetPack::open(f.pack_path());
    ASSERT_TRUE(pack);
    auto r = pack.value().read("zero");
    ASSERT_TRUE(r);
    EXPECT_TRUE(r.value().empty());
}

TEST(AssetPack, MissingPackIsReported) {
    auto r = AssetPack::open((temp_dir() / "does_not_exist.pack").string());
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), PackError::FileNotFound);
}

TEST(AssetPack, NonPackFileIsRejected) {
    Fixture f;
    const std::string p = f.write("bogus.pack", "this is not a pack, just some text bytes here");
    auto r = AssetPack::open(p);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), PackError::NotAPack);
}

TEST(AssetPack, FutureVersionIsRefused) {
    Fixture f;
    std::vector<std::pair<std::string, std::string>> sources{{"x", f.write("x.bin", "x")}};
    ASSERT_TRUE(AssetPack::build(f.pack_path(), sources));
    {
        std::fstream io(f.pack_path(), std::ios::binary | std::ios::in | std::ios::out);
        io.seekp(8, std::ios::beg);
        const uint32_t future = k_pack_format_version + 1;
        io.write(reinterpret_cast<const char*>(&future), sizeof(future));
    }
    auto r = AssetPack::open(f.pack_path());
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), PackError::FutureVersion);
}

// A TOC pointing outside the file must be caught at open, not turned into an
// out-of-range read later.
TEST(AssetPack, TruncatedPackIsRejected) {
    Fixture f;
    std::vector<std::pair<std::string, std::string>> sources{
        {"a", f.write("a.bin", std::string(4096, 'a'))},
        {"b", f.write("b.bin", std::string(4096, 'b'))}};
    ASSERT_TRUE(AssetPack::build(f.pack_path(), sources));

    const auto size = std::filesystem::file_size(f.pack_path());
    std::filesystem::resize_file(f.pack_path(), size / 2);

    auto r = AssetPack::open(f.pack_path());
    EXPECT_FALSE(r) << "a truncated pack must not open";
}

TEST(AssetPack, BuildRejectsAMissingSource) {
    Fixture f;
    std::vector<std::pair<std::string, std::string>> sources{
        {"present", f.write("p.bin", "x")},
        {"absent", (f.dir / "not_there.bin").string()}};
    auto r = AssetPack::build(f.pack_path(), sources);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), PackError::FileNotFound);
    EXPECT_FALSE(std::filesystem::exists(f.pack_path()))
        << "a failed build must not leave a partial pack that later looks openable";
    EXPECT_FALSE(std::filesystem::exists(f.pack_path() + ".tmp"));
}

// The reason read() is worker-safe: the async loader pulls entries from many
// threads at once.
TEST(AssetPack, ConcurrentReadsAreSafeAndCorrect) {
    Fixture f;
    std::vector<std::pair<std::string, std::string>> sources;
    for (int i = 0; i < 16; ++i) {
        const std::string name = "entry" + std::to_string(i);
        sources.emplace_back(name, f.write(name + ".bin", std::string(1024, char('a' + i))));
    }
    ASSERT_TRUE(AssetPack::build(f.pack_path(), sources));

    auto opened = AssetPack::open(f.pack_path());
    ASSERT_TRUE(opened);
    const AssetPack& pack = opened.value();

    std::vector<std::thread> threads;
    std::atomic<int> mismatches{0};
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&pack, &mismatches] {
            for (int rep = 0; rep < 20; ++rep) {
                for (int i = 0; i < 16; ++i) {
                    auto r = pack.read("entry" + std::to_string(i));
                    if (!r || r.value().size() != 1024 ||
                        as_string(r.value()) != std::string(1024, char('a' + i))) {
                        mismatches.fetch_add(1);
                    }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(mismatches.load(), 0);
}
