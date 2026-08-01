#pragma once

#include "bracken/core/result.hpp"

#include <cstdint>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bracken::core::asset {

enum class PackError : uint8_t {
    FileNotFound,
    NotAPack,       ///< magic mismatch
    Corrupt,        ///< truncated, bad TOC, or content hash mismatch
    FutureVersion,  ///< written by a newer engine than this build understands
    WriteFailed,
    EntryNotFound,
};

[[nodiscard]] const char* to_string(PackError e) noexcept;

inline constexpr uint32_t k_pack_format_version = 1;

/**
 * A read-only bundle of asset files in one container.
 *
 * WHY. A game that ships 50 000 loose files spends a meaningful fraction of its
 * load time in the filesystem rather than in its own code, and cannot be
 * distributed as one artifact. A pack turns N opens into one, makes reads
 * sequential, and gives patching and integrity checking somewhere to live.
 *
 * NOT read wholly into memory. The table of contents is, but entry bytes are
 * read on demand — a pack is allowed to be larger than RAM, which is the whole
 * point of having one for a large world. Reads are serialized by an internal
 * mutex so several worker threads can pull entries concurrently; that is
 * deliberate and cheap, because decoding an asset costs far more than reading
 * its bytes, so the lock is never the bottleneck.
 *
 * Tier: open() is main-thread. read()/contains()/entries() are worker-safe.
 */
class AssetPack {
public:
    struct Entry {
        std::string name; ///< logical path, e.g. "models/tree.glb"
        uint64_t offset = 0;
        uint64_t size = 0;
    };

    AssetPack() = default;
    AssetPack(AssetPack&&) noexcept;
    AssetPack& operator=(AssetPack&&) noexcept;
    AssetPack(const AssetPack&) = delete;
    AssetPack& operator=(const AssetPack&) = delete;

    [[nodiscard]] static Result<AssetPack, PackError> open(const std::string& path);

    [[nodiscard]] bool is_open() const noexcept { return stream_ != nullptr; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] std::span<const Entry> entries() const noexcept { return entries_; }
    [[nodiscard]] bool contains(std::string_view name) const;

    /** Read one entry's bytes. Worker-safe. */
    [[nodiscard]] Result<std::vector<std::byte>, PackError> read(std::string_view name) const;

    /**
     * Build a pack from (name, file path) pairs.
     *
     * Writes to a temp file and renames, like SaveDocument — an interrupted
     * pack build must not leave a half-written file that later looks openable.
     */
    [[nodiscard]] static Result<void, PackError>
    build(const std::string& out_path,
          std::span<const std::pair<std::string, std::string>> name_and_source);

private:
    [[nodiscard]] const Entry* find(std::string_view name) const;

    std::string path_;
    std::vector<Entry> entries_;
    // Mutable + mutex: read() is logically const (a pack is immutable) but has
    // to seek a shared stream, and callers are worker threads.
    mutable std::unique_ptr<std::ifstream> stream_;
    mutable std::mutex read_mutex_;
};

} // namespace bracken::core::asset
