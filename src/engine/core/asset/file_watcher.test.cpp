#include <gtest/gtest.h>

#include "engine/core/asset/file_watcher.hpp"

#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using mmo::engine::core::asset::FileWatcher;

namespace {

fs::path make_temp_dir(const std::string& prefix) {
    static std::atomic<uint32_t> counter{0};
    std::random_device rd;
    auto pid = static_cast<uint32_t>(getpid());
    auto id = counter.fetch_add(1) ^ rd();
    auto p = fs::temp_directory_path() / ("mmo_" + prefix + "_" + std::to_string(pid) + "_" + std::to_string(id));
    fs::create_directories(p);
    return p;
}

void write_file(const fs::path& p, const std::string& contents) {
    std::ofstream o(p, std::ios::trunc | std::ios::binary);
    o << contents;
}

// Per-test scratch directory under /tmp/claude-fw-test-<pid>, removed on scope
// exit so early ASSERT bail-outs still clean up.
struct ScopedTempDir {
    explicit ScopedTempDir(const std::string& name)
        : root(fs::path("/tmp") / ("claude-fw-test-" + std::to_string(getpid()))), path(root / name) {
        fs::create_directories(path);
    }
    ~ScopedTempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::remove(root, ec); // best-effort; only succeeds once the root is empty
    }
    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;

    fs::path root;
    fs::path path;
};

// Stamp `p` with an exact mtime `seconds_ago` in the past (`nanos` for extra
// uniqueness). Being well past the watcher's 100ms settle window, the change
// is eligible for pickup on the very next worker pass — no mtime-granularity
// or settle sleeps needed.
bool set_mtime(const fs::path& p, long seconds_ago, long nanos) {
    timespec times[2];
    clock_gettime(CLOCK_REALTIME, &times[0]); // atime = now
    times[1].tv_sec = times[0].tv_sec - seconds_ago;
    times[1].tv_nsec = nanos;
    return utimensat(AT_FDCWD, p.c_str(), times, 0) == 0;
}

// Repeatedly re-stamp `p` with a fresh, already-settled mtime and drain the
// main-thread queue until `predicate` holds. Bounded, and immune to the
// first-scan priming race: whichever pass primes the cache, the next pass
// sees a different mtime and fires.
bool bump_until(FileWatcher& w, const fs::path& p, const std::function<bool()>& predicate, int budget = 100) {
    for (int i = 0; i < budget; ++i) {
        if (!set_mtime(p, 7200 - i, (i + 1) * 1000L)) {
            return false;
        }
        w.poll_main_thread();
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(20ms);
    }
    return false;
}

// Drain callbacks for up to `budget` iterations of 50ms each, until `predicate`
// becomes true. Returns true on success.
bool drain_until(FileWatcher& w, std::function<bool()> predicate, int budget = 80) {
    for (int i = 0; i < budget; ++i) {
        w.poll_main_thread();
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(50ms);
    }
    return false;
}

class FileWatcherFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Use 50ms poll interval for snappy tests.
        ASSERT_TRUE(watcher.init(50ms));
    }
    void TearDown() override { watcher.shutdown(); }
    FileWatcher watcher;
};

} // namespace

TEST_F(FileWatcherFixture, FileChangeFiresCallback) {
    auto dir = make_temp_dir("fw_file");
    auto file = dir / "a.txt";
    write_file(file, "hello");
    // Prime mtime cache by yielding past the settle window.
    std::this_thread::sleep_for(150ms);

    std::atomic<int> hits{0};
    auto h = watcher.watch_file(file, [&](const fs::path&) { hits.fetch_add(1); });
    ASSERT_NE(h, FileWatcher::k_invalid_handle);

    // Allow the first scan to prime the cache.
    std::this_thread::sleep_for(150ms);
    watcher.poll_main_thread();
    EXPECT_EQ(hits.load(), 0);

    write_file(file, "modified");
    EXPECT_TRUE(drain_until(watcher, [&] { return hits.load() >= 1; }));

    fs::remove_all(dir);
}

TEST_F(FileWatcherFixture, DirectoryWatchHonoursExtensionFilter) {
    auto dir = make_temp_dir("fw_dir");
    write_file(dir / "init.txt", "x");
    write_file(dir / "init.bin", "y");
    std::this_thread::sleep_for(150ms);

    std::atomic<int> txt_hits{0};
    std::atomic<int> any_hits{0};
    auto h = watcher.watch_directory(dir, ".txt", [&](const fs::path& p) {
        any_hits.fetch_add(1);
        if (p.extension() == ".txt") {
            txt_hits.fetch_add(1);
        }
    });
    ASSERT_NE(h, FileWatcher::k_invalid_handle);

    std::this_thread::sleep_for(150ms);
    watcher.poll_main_thread();
    EXPECT_EQ(any_hits.load(), 0);

    write_file(dir / "init.txt", "x2");
    write_file(dir / "init.bin", "y2");

    EXPECT_TRUE(drain_until(watcher, [&] { return txt_hits.load() >= 1; }));
    // The bin file must never produce a callback.
    EXPECT_EQ(any_hits.load(), txt_hits.load());

    fs::remove_all(dir);
}

TEST_F(FileWatcherFixture, UnwatchStopsCallbacks) {
    auto dir = make_temp_dir("fw_unwatch");
    auto file = dir / "f.txt";
    write_file(file, "0");
    std::this_thread::sleep_for(150ms);

    std::atomic<int> hits{0};
    auto h = watcher.watch_file(file, [&](const fs::path&) { hits.fetch_add(1); });
    std::this_thread::sleep_for(150ms);
    watcher.poll_main_thread();

    watcher.unwatch(h);

    write_file(file, "1");
    std::this_thread::sleep_for(400ms);
    watcher.poll_main_thread();
    EXPECT_EQ(hits.load(), 0);

    fs::remove_all(dir);
}

TEST_F(FileWatcherFixture, MultipleWatchersOnSameFileBothFire) {
    auto dir = make_temp_dir("fw_multi");
    auto file = dir / "shared.txt";
    write_file(file, "v0");
    std::this_thread::sleep_for(150ms);

    std::atomic<int> a{0}, b{0};
    watcher.watch_file(file, [&](const fs::path&) { a.fetch_add(1); });
    watcher.watch_file(file, [&](const fs::path&) { b.fetch_add(1); });

    std::this_thread::sleep_for(150ms);
    watcher.poll_main_thread();

    write_file(file, "v1");
    EXPECT_TRUE(drain_until(watcher, [&] { return a.load() >= 1 && b.load() >= 1; }));

    fs::remove_all(dir);
}

TEST_F(FileWatcherFixture, MultipleWatchedFilesReportOnlyTheChangedOne) {
    ScopedTempDir tmp("multi_files");
    auto file_a = tmp.path / "a.txt";
    auto file_b = tmp.path / "b.txt";
    auto file_c = tmp.path / "c.txt";
    for (const auto& f : {file_a, file_b, file_c}) {
        write_file(f, "v0");
        ASSERT_TRUE(set_mtime(f, 7210, 0));
    }

    std::atomic<int> a_hits{0}, b_hits{0}, c_hits{0};
    std::vector<fs::path> b_paths; // only touched on this thread, via poll_main_thread
    ASSERT_NE(watcher.watch_file(file_a, [&](const fs::path&) { a_hits.fetch_add(1); }),
              FileWatcher::k_invalid_handle);
    ASSERT_NE(watcher.watch_file(file_b,
                                 [&](const fs::path& p) {
                                     b_hits.fetch_add(1);
                                     b_paths.push_back(p);
                                 }),
              FileWatcher::k_invalid_handle);
    ASSERT_NE(watcher.watch_file(file_c, [&](const fs::path&) { c_hits.fetch_add(1); }),
              FileWatcher::k_invalid_handle);

    // Touch only b; only b's watch may fire.
    ASSERT_TRUE(bump_until(watcher, file_b, [&] { return b_hits.load() >= 1; }));
    EXPECT_EQ(a_hits.load(), 0);
    EXPECT_EQ(c_hits.load(), 0);
    for (const auto& p : b_paths) {
        EXPECT_EQ(p, file_b);
    }
}

TEST_F(FileWatcherFixture, MtimeChangeAloneIsDetected) {
    ScopedTempDir tmp("mtime_only");
    auto file = tmp.path / "same_bytes.txt";
    write_file(file, "contents never rewritten");
    ASSERT_TRUE(set_mtime(file, 7210, 0));

    std::atomic<int> hits{0};
    ASSERT_NE(watcher.watch_file(file, [&](const fs::path&) { hits.fetch_add(1); }), FileWatcher::k_invalid_handle);

    // bump_until only calls utimensat — the file content is never touched.
    EXPECT_TRUE(bump_until(watcher, file, [&] { return hits.load() >= 1; }));
}

TEST_F(FileWatcherFixture, UnwatchedHandleStaysSilentWhileSiblingFires) {
    ScopedTempDir tmp("unwatch_sibling");
    auto kept = tmp.path / "kept.txt";
    auto dropped = tmp.path / "dropped.txt";
    write_file(kept, "k");
    write_file(dropped, "d");
    ASSERT_TRUE(set_mtime(kept, 7210, 0));
    ASSERT_TRUE(set_mtime(dropped, 7210, 0));

    std::atomic<int> kept_hits{0}, dropped_hits{0};
    auto h_dropped = watcher.watch_file(dropped, [&](const fs::path&) { dropped_hits.fetch_add(1); });
    auto h_kept = watcher.watch_file(kept, [&](const fs::path&) { kept_hits.fetch_add(1); });
    ASSERT_NE(h_dropped, FileWatcher::k_invalid_handle);
    ASSERT_NE(h_kept, FileWatcher::k_invalid_handle);

    watcher.unwatch(h_dropped);

    // Touch both files every iteration; the kept watch is the barrier proving
    // scans ran after the unwatch. Two fires => two distinct worker passes.
    bool ok = false;
    for (int i = 0; i < 100 && !ok; ++i) {
        ASSERT_TRUE(set_mtime(kept, 7200 - i, (i + 1) * 1000L));
        ASSERT_TRUE(set_mtime(dropped, 7200 - i, (i + 1) * 1000L));
        watcher.poll_main_thread();
        ok = kept_hits.load() >= 2;
        if (!ok) {
            std::this_thread::sleep_for(20ms);
        }
    }
    EXPECT_TRUE(ok);
    EXPECT_EQ(dropped_hits.load(), 0);
}

TEST_F(FileWatcherFixture, CallbackReceivesExactWatchedPath) {
    ScopedTempDir tmp("exact_path");
    auto file = tmp.path / "payload.txt";
    write_file(file, "x");
    ASSERT_TRUE(set_mtime(file, 7210, 0));

    std::vector<fs::path> received; // only touched on this thread, via poll_main_thread
    ASSERT_NE(watcher.watch_file(file, [&](const fs::path& p) { received.push_back(p); }),
              FileWatcher::k_invalid_handle);

    ASSERT_TRUE(bump_until(watcher, file, [&] { return !received.empty(); }));
    EXPECT_EQ(received.front(), file);
    EXPECT_EQ(received.front().string(), file.string());
}

TEST_F(FileWatcherFixture, MissingPathsAreWatchedGracefully) {
    ScopedTempDir tmp("missing");
    auto missing_file = tmp.path / "no_such_dir" / "ghost.txt";
    auto missing_dir = tmp.path / "not_a_dir";
    auto sentinel = tmp.path / "sentinel.txt";
    write_file(sentinel, "s");
    ASSERT_TRUE(set_mtime(sentinel, 7210, 0));

    std::atomic<int> ghost_hits{0};
    auto h_file = watcher.watch_file(missing_file, [&](const fs::path&) { ghost_hits.fetch_add(1); });
    auto h_dir = watcher.watch_directory(missing_dir, ".txt", [&](const fs::path&) { ghost_hits.fetch_add(1); });
    ASSERT_NE(h_file, FileWatcher::k_invalid_handle);
    ASSERT_NE(h_dir, FileWatcher::k_invalid_handle);

    std::atomic<int> sentinel_hits{0};
    ASSERT_NE(watcher.watch_file(sentinel, [&](const fs::path&) { sentinel_hits.fetch_add(1); }),
              FileWatcher::k_invalid_handle);

    // Two sentinel fires prove at least two full worker passes have scanned
    // the missing paths without crashing or emitting events for them.
    ASSERT_TRUE(bump_until(watcher, sentinel, [&] { return sentinel_hits.load() >= 2; }));
    EXPECT_EQ(ghost_hits.load(), 0);

    watcher.unwatch(h_file);
    watcher.unwatch(h_dir);
}

TEST_F(FileWatcherFixture, FileCreatedAfterWatchIsReported) {
    ScopedTempDir tmp("created_later");
    auto future_file = tmp.path / "appears_later.txt";
    auto sentinel = tmp.path / "sentinel.txt";
    write_file(sentinel, "s");
    ASSERT_TRUE(set_mtime(sentinel, 7210, 0));

    std::vector<fs::path> received; // only touched on this thread, via poll_main_thread
    ASSERT_NE(watcher.watch_file(future_file, [&](const fs::path& p) { received.push_back(p); }),
              FileWatcher::k_invalid_handle);
    std::atomic<int> sentinel_hits{0};
    ASSERT_NE(watcher.watch_file(sentinel, [&](const fs::path&) { sentinel_hits.fetch_add(1); }),
              FileWatcher::k_invalid_handle);

    // Barrier: once the sentinel fires, a full pass has scanned the missing
    // file's watch, so its appearance below cannot be swallowed as first-scan.
    ASSERT_TRUE(bump_until(watcher, sentinel, [&] { return sentinel_hits.load() >= 1; }));
    ASSERT_TRUE(received.empty());

    write_file(future_file, "now I exist");
    ASSERT_TRUE(set_mtime(future_file, 7000, 0)); // already settled

    EXPECT_TRUE(drain_until(watcher, [&] { return !received.empty(); }));
    ASSERT_FALSE(received.empty());
    EXPECT_EQ(received.front(), future_file);
}

TEST_F(FileWatcherFixture, NullCallbackReturnsInvalidHandle) {
    EXPECT_EQ(watcher.watch_file("/tmp/never_read.txt", {}), FileWatcher::k_invalid_handle);
    EXPECT_EQ(watcher.watch_directory("/tmp", ".txt", {}), FileWatcher::k_invalid_handle);
}

TEST(FileWatcherDestruction, DestroyWhileWatchingJoinsPromptly) {
    ScopedTempDir tmp("dtor");
    auto file = tmp.path / "watched.txt";
    write_file(file, "x");
    ASSERT_TRUE(set_mtime(file, 7210, 0));

    auto start = std::chrono::steady_clock::now();
    {
        FileWatcher local;
        ASSERT_TRUE(local.init(10ms));
        ASSERT_NE(local.watch_file(file, [](const fs::path&) {}), FileWatcher::k_invalid_handle);
        ASSERT_NE(local.watch_directory(tmp.path, "", [](const fs::path&) {}), FileWatcher::k_invalid_handle);
        // No shutdown(): the destructor must stop and join the worker itself.
    }
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(elapsed_ms, 2000);
}

TEST(FileWatcherLifecycle, WatchBeforeInitReturnsInvalidHandle) {
    FileWatcher w;
    EXPECT_FALSE(w.is_initialized());
    EXPECT_EQ(w.watch_file("/tmp/never_read.txt", [](const fs::path&) {}), FileWatcher::k_invalid_handle);
    EXPECT_EQ(w.watch_directory("/tmp", ".txt", [](const fs::path&) {}), FileWatcher::k_invalid_handle);
    // Both are safe no-ops on an uninitialized watcher.
    w.unwatch(FileWatcher::k_invalid_handle);
    w.poll_main_thread();
}

TEST(FileWatcherLifecycle, InitIsExclusiveAndShutdownIdempotent) {
    FileWatcher w;
    ASSERT_TRUE(w.init(10ms));
    EXPECT_FALSE(w.init(10ms)); // second init rejected
    EXPECT_TRUE(w.is_initialized());
    w.shutdown();
    EXPECT_FALSE(w.is_initialized());
    w.shutdown();              // second shutdown is a no-op
    EXPECT_TRUE(w.init(10ms)); // re-init after shutdown works
    w.shutdown();
}
