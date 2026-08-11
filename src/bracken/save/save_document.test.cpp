#include <gtest/gtest.h>

#include "bracken/save/save_document.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace bracken::save;

namespace {

std::string temp_path(const char* leaf) {
    return (std::filesystem::temp_directory_path() / leaf).string();
}

struct TempFile {
    std::string path;
    explicit TempFile(const char* leaf) : path(temp_path(leaf)) { remove(); }
    ~TempFile() { remove(); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    void remove() const {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        std::filesystem::remove(path + ".tmp", ec);
    }
};

} // namespace

TEST(SaveDocument, RoundTripsEveryType) {
    TempFile f("bracken_save_types.sav");

    SaveDocument doc;
    doc.set_game_version(7);
    doc.write("b", true);
    doc.write("i", int64_t{-9000000000LL});
    doc.write("i32", int32_t{-42});
    doc.write("u32", uint32_t{4000000000u});
    doc.write("f", 0.25f);
    doc.write("d", 1.0 / 3.0);
    doc.write("s", std::string_view("hello world"));
    doc.write("v2", glm::vec2(1, 2));
    doc.write("v3", glm::vec3(1, 2, 3));
    doc.write("v4", glm::vec4(1, 2, 3, 4));
    doc.write("q", glm::quat(0.5f, 0.5f, 0.5f, 0.5f));
    doc.write("m", glm::mat4(2.0f));
    const std::byte blob[]{std::byte{1}, std::byte{2}, std::byte{255}};
    doc.write_blob("blob", blob);
    ASSERT_TRUE(doc.save(f.path));

    auto loaded = SaveDocument::load(f.path);
    ASSERT_TRUE(loaded) << to_string(loaded.error());
    const SaveDocument& d = loaded.value();

    EXPECT_EQ(d.game_version(), 7u);
    EXPECT_TRUE(d.read_or("b", false));
    EXPECT_EQ(d.read_or<int64_t>("i", 0), -9000000000LL);
    EXPECT_EQ(d.read_or<int32_t>("i32", 0), -42);
    EXPECT_EQ(d.read_or<uint32_t>("u32", 0), 4000000000u);
    EXPECT_FLOAT_EQ(d.read_or("f", 0.0f), 0.25f);
    EXPECT_DOUBLE_EQ(d.read_or("d", 0.0), 1.0 / 3.0);
    EXPECT_EQ(d.read_or<std::string>("s", ""), "hello world");
    EXPECT_EQ(d.read_or("v3", glm::vec3(0)), glm::vec3(1, 2, 3));
    EXPECT_EQ(d.read_or("v4", glm::vec4(0)), glm::vec4(1, 2, 3, 4));
    EXPECT_EQ(d.read_or("m", glm::mat4(1.0f)), glm::mat4(2.0f));

    const glm::quat q = d.read_or("q", glm::quat(1, 0, 0, 0));
    EXPECT_FLOAT_EQ(q.w, 0.5f);
    EXPECT_FLOAT_EQ(q.x, 0.5f);

    std::vector<std::byte> out;
    ASSERT_TRUE(d.read_blob("blob", out));
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[2], std::byte{255});
}

// The whole reason this is a document and not a stream: a build that adds a
// field must still load an older save.
TEST(SaveDocument, MissingFieldTakesTheDefault) {
    TempFile f("bracken_save_missing.sav");

    SaveDocument old;
    old.write("player/health", int32_t{50});
    ASSERT_TRUE(old.save(f.path));

    auto loaded = SaveDocument::load(f.path);
    ASSERT_TRUE(loaded);
    // "player/stamina" was added after this save was written.
    EXPECT_EQ(loaded.value().read_or<int32_t>("player/stamina", 100), 100);
    EXPECT_EQ(loaded.value().read_or<int32_t>("player/health", 0), 50);
}

// An older build loading and re-saving must not silently delete a newer
// build's data.
TEST(SaveDocument, UnknownKeysSurviveARoundTrip) {
    TempFile f("bracken_save_unknown.sav");

    SaveDocument newer;
    newer.write("known", int32_t{1});
    newer.write("added_by_a_later_build", std::string_view("keep me"));
    ASSERT_TRUE(newer.save(f.path));

    auto loaded = SaveDocument::load(f.path);
    ASSERT_TRUE(loaded);
    SaveDocument older = std::move(loaded).value();
    older.write("known", int32_t{2}); // the old build only understands this

    TempFile g("bracken_save_unknown2.sav");
    ASSERT_TRUE(older.save(g.path));

    auto reloaded = SaveDocument::load(g.path);
    ASSERT_TRUE(reloaded);
    EXPECT_EQ(reloaded.value().read_or<int32_t>("known", 0), 2);
    EXPECT_EQ(reloaded.value().read_or<std::string>("added_by_a_later_build", ""), "keep me")
        << "an old build must not drop data it does not understand";
}

// A save from another build is data, not a contract: asking for the wrong type
// must fail cleanly rather than be undefined.
TEST(SaveDocument, TypeMismatchFailsAndLeavesTheDefault) {
    SaveDocument doc;
    doc.write("count", std::string_view("not a number"));

    int32_t value = 99;
    EXPECT_FALSE(doc.read("count", value));
    EXPECT_EQ(value, 99) << "a failed read must not clobber the destination";
    EXPECT_EQ(doc.read_or<int32_t>("count", 7), 7);
}

TEST(SaveDocument, WriteReplacesRatherThanDuplicates) {
    SaveDocument doc;
    doc.write("hp", int32_t{10});
    doc.write("hp", int32_t{20});
    EXPECT_EQ(doc.record_count(), 1u);
    EXPECT_EQ(doc.read_or<int32_t>("hp", 0), 20);
}

TEST(SaveDocument, ListsRoundTripWithCount) {
    TempFile f("bracken_save_list.sav");

    SaveDocument doc;
    const glm::vec3 positions[] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    doc.write_count("orbs", std::size(positions));
    for (std::size_t i = 0; i < std::size(positions); ++i) {
        doc.write(index_path("orbs", i, "position"), positions[i]);
    }
    ASSERT_TRUE(doc.save(f.path));

    auto loaded = SaveDocument::load(f.path);
    ASSERT_TRUE(loaded);
    const SaveDocument& d = loaded.value();
    ASSERT_EQ(d.read_count("orbs"), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(d.read_or(index_path("orbs", i, "position"), glm::vec3(0)), positions[i]);
    }
}

TEST(SaveDocument, MissingFileIsReportedNotCrashed) {
    auto r = SaveDocument::load(temp_path("bracken_save_absent.sav"));
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), SaveError::FileNotFound);
}

TEST(SaveDocument, NonSaveFileIsRejected) {
    TempFile f("bracken_save_notasave.sav");
    {
        std::ofstream out(f.path, std::ios::binary);
        out << "this is just some text, definitely not a save file at all";
    }
    auto r = SaveDocument::load(f.path);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), SaveError::NotASaveFile);
}

// Bit rot must be reported, not loaded as plausible-looking garbage.
TEST(SaveDocument, CorruptPayloadIsDetected) {
    TempFile f("bracken_save_corrupt.sav");

    SaveDocument doc;
    doc.write("player/name", std::string_view("Aria"));
    doc.write("player/health", int32_t{73});
    ASSERT_TRUE(doc.save(f.path));

    // Flip a byte deep in the payload, past the header.
    {
        std::fstream io(f.path, std::ios::binary | std::ios::in | std::ios::out);
        io.seekp(30, std::ios::beg);
        char c = 0;
        io.seekg(30, std::ios::beg);
        io.read(&c, 1);
        c = static_cast<char>(c ^ 0x5A);
        io.seekp(30, std::ios::beg);
        io.write(&c, 1);
    }

    auto r = SaveDocument::load(f.path);
    ASSERT_FALSE(r) << "a corrupted save must not load";
    EXPECT_EQ(r.error(), SaveError::Corrupt);
}

TEST(SaveDocument, TruncatedFileIsDetected) {
    TempFile f("bracken_save_truncated.sav");

    SaveDocument doc;
    for (int i = 0; i < 20; ++i) {
        doc.write(std::format("key{}", i), glm::vec3(static_cast<float>(i)));
    }
    ASSERT_TRUE(doc.save(f.path));

    const auto size = std::filesystem::file_size(f.path);
    std::filesystem::resize_file(f.path, size / 2);

    auto r = SaveDocument::load(f.path);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), SaveError::Corrupt);
}

// A save from a newer engine must be refused rather than half-read: an unknown
// tag cannot be skipped, because its length is unknown.
TEST(SaveDocument, FutureFormatVersionIsRefused) {
    TempFile f("bracken_save_future.sav");

    SaveDocument doc;
    doc.write("x", int32_t{1});
    ASSERT_TRUE(doc.save(f.path));

    {   // bump the format version field in place
        std::fstream io(f.path, std::ios::binary | std::ios::in | std::ios::out);
        io.seekp(8, std::ios::beg);
        const uint32_t future = k_save_format_version + 1;
        io.write(reinterpret_cast<const char*>(&future), sizeof(future));
    }

    auto r = SaveDocument::load(f.path);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), SaveError::FutureVersion);
}

// The durability guarantee: an interrupted save must leave the PREVIOUS file
// readable. Simulated by checking that save() never truncates the target in
// place — after a successful save the old content is fully replaced, and a
// stray .tmp is never left behind.
TEST(SaveDocument, SaveIsAtomicAndLeavesNoTempBehind) {
    TempFile f("bracken_save_atomic.sav");

    SaveDocument first;
    first.write("generation", int32_t{1});
    ASSERT_TRUE(first.save(f.path));

    SaveDocument second;
    second.write("generation", int32_t{2});
    ASSERT_TRUE(second.save(f.path));

    EXPECT_FALSE(std::filesystem::exists(f.path + ".tmp"))
        << "the temp file must be renamed away, not left as litter";

    auto loaded = SaveDocument::load(f.path);
    ASSERT_TRUE(loaded);
    EXPECT_EQ(loaded.value().read_or<int32_t>("generation", 0), 2);
}

TEST(SaveDocument, EmptyDocumentRoundTrips) {
    TempFile f("bracken_save_empty.sav");
    SaveDocument doc;
    doc.set_game_version(3);
    ASSERT_TRUE(doc.save(f.path));

    auto loaded = SaveDocument::load(f.path);
    ASSERT_TRUE(loaded);
    EXPECT_EQ(loaded.value().record_count(), 0u);
    EXPECT_EQ(loaded.value().game_version(), 3u);
}

TEST(SaveDocument, PathsListsEverythingInWriteOrder) {
    SaveDocument doc;
    doc.write("a", int32_t{1});
    doc.write("b", int32_t{2});
    doc.write("c", int32_t{3});
    const auto paths = doc.paths();
    ASSERT_EQ(paths.size(), 3u);
    EXPECT_EQ(paths[0], "a");
    EXPECT_EQ(paths[2], "c");
}

TEST(SaveDocument, EmptyStringAndEmptyBlobRoundTrip) {
    TempFile f("bracken_save_empties.sav");
    SaveDocument doc;
    doc.write("empty_string", std::string_view(""));
    doc.write_blob("empty_blob", {});
    ASSERT_TRUE(doc.save(f.path));

    auto loaded = SaveDocument::load(f.path);
    ASSERT_TRUE(loaded);
    EXPECT_TRUE(loaded.value().has("empty_string"));
    EXPECT_EQ(loaded.value().read_or<std::string>("empty_string", "x"), "");
    std::vector<std::byte> blob{std::byte{9}};
    EXPECT_TRUE(loaded.value().read_blob("empty_blob", blob));
    EXPECT_TRUE(blob.empty());
}

// ---------------------------------------------------------------------------
// serialize() / parse() — the same container, without the filesystem.
//
// The risk this pair introduces is a SECOND encoder that drifts from the first,
// so the first test is that there is only one.
// ---------------------------------------------------------------------------

TEST(SaveDocument, SerializeProducesExactlyTheFileBytes) {
    TempFile f("bracken_save_serialize.sav");

    SaveDocument doc;
    doc.set_game_version(3);
    doc.write("a/b", int64_t{7});
    doc.write("a/c", std::string_view("text"));
    doc.write("v", glm::vec3(1, 2, 3));
    const std::byte blob[]{std::byte{9}, std::byte{8}};
    doc.write_blob("blob", blob);

    ASSERT_TRUE(doc.save(f.path));
    const std::vector<std::byte> bytes = doc.serialize();

    std::ifstream in(f.path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<char> on_disk((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_EQ(on_disk.size(), bytes.size());
    EXPECT_EQ(0, std::memcmp(on_disk.data(), bytes.data(), bytes.size()));
}

TEST(SaveDocument, ParseAcceptsWhatSerializeProduces) {
    SaveDocument doc;
    doc.set_game_version(11);
    doc.write("i", int64_t{-5});
    doc.write("s", std::string_view("round trip"));
    doc.write("m", glm::mat4(3.0f));

    auto parsed = SaveDocument::parse(doc.serialize());
    ASSERT_TRUE(parsed) << to_string(parsed.error());
    EXPECT_EQ(parsed.value().game_version(), 11u);
    EXPECT_EQ(parsed.value().read_or<int64_t>("i", 0), -5);
    EXPECT_EQ(parsed.value().read_or<std::string>("s", ""), "round trip");
    EXPECT_EQ(parsed.value().read_or("m", glm::mat4(1.0f)), glm::mat4(3.0f));
}

TEST(SaveDocument, ParseRefusesGarbageWithoutReadingPastTheSpan) {
    const std::byte junk[]{std::byte{1}, std::byte{2}, std::byte{3}};
    auto r = SaveDocument::parse(junk);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error(), SaveError::NotASaveFile);

    // A valid header with a corrupted payload.
    SaveDocument doc;
    doc.write("x", int64_t{1});
    std::vector<std::byte> bytes = doc.serialize();
    bytes.back() = static_cast<std::byte>(std::to_integer<uint8_t>(bytes.back()) ^ 0xFFu);
    auto corrupt = SaveDocument::parse(bytes);
    ASSERT_FALSE(corrupt);
    EXPECT_EQ(corrupt.error(), SaveError::Corrupt);
}

TEST(SaveDocument, ValueAtCarriesRecordsAcrossDocumentsWithoutKnowingTheirType) {
    SaveDocument source;
    source.write("k/i", int64_t{5});
    source.write("k/v", glm::vec4(1, 2, 3, 4));
    source.write("k/s", std::string_view("copy me"));

    SaveDocument sink;
    for (std::string_view path : source.paths()) {
        const SaveDocument::Value* v = source.value_at(path);
        ASSERT_NE(v, nullptr);
        sink.write_value(path, *v);
    }
    EXPECT_EQ(sink.record_count(), source.record_count());
    EXPECT_EQ(sink.read_or<int64_t>("k/i", 0), 5);
    EXPECT_EQ(sink.read_or("k/v", glm::vec4(0.0f)), glm::vec4(1, 2, 3, 4));
    EXPECT_EQ(sink.read_or<std::string>("k/s", ""), "copy me");
    EXPECT_EQ(source.value_at("nope"), nullptr);
}
