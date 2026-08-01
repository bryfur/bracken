#include "bracken/core/asset/asset_pack.hpp"

#include "bracken/core/logger.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>

namespace bracken::core::asset {
namespace {

constexpr char k_magic[8] = {'B', 'R', 'K', 'P', 'A', 'C', 'K', '\0'};

/** Blobs start on this boundary so a future mmap-backed reader can hand out
 *  suitably aligned pointers without copying. */
constexpr uint64_t k_blob_align = 16;

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

uint64_t hash_bytes(uint64_t h, std::span<const std::byte> bytes) {
    for (std::byte b : bytes) {
        h ^= static_cast<uint64_t>(std::to_integer<uint8_t>(b));
        h *= 0x100000001b3ull;
    }
    return h;
}

template <typename T> void put_pod(std::vector<std::byte>& out, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* p = reinterpret_cast<const std::byte*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

} // namespace

const char* to_string(PackError e) noexcept {
    switch (e) {
    case PackError::FileNotFound: return "pack file not found";
    case PackError::NotAPack: return "not a bracken asset pack";
    case PackError::Corrupt: return "pack is corrupt or truncated";
    case PackError::FutureVersion: return "pack written by a newer engine";
    case PackError::WriteFailed: return "could not write the pack";
    case PackError::EntryNotFound: return "entry not present in the pack";
    }
    return "unknown";
}

AssetPack::AssetPack(AssetPack&& o) noexcept
    : path_(std::move(o.path_)), entries_(std::move(o.entries_)), stream_(std::move(o.stream_)) {}

AssetPack& AssetPack::operator=(AssetPack&& o) noexcept {
    if (this != &o) {
        path_ = std::move(o.path_);
        entries_ = std::move(o.entries_);
        stream_ = std::move(o.stream_);
    }
    return *this;
}

const AssetPack::Entry* AssetPack::find(std::string_view name) const {
    // Entries are sorted by name at build time, so this is a binary search
    // rather than a scan — a pack with 50k entries is the case that matters.
    auto it = std::lower_bound(entries_.begin(), entries_.end(), name,
                               [](const Entry& e, std::string_view n) { return e.name < n; });
    return (it != entries_.end() && it->name == name) ? &*it : nullptr;
}

bool AssetPack::contains(std::string_view name) const { return find(name) != nullptr; }

Result<AssetPack, PackError> AssetPack::open(const std::string& path) {
    auto stream = std::make_unique<std::ifstream>(path, std::ios::binary);
    if (!*stream) {
        return Err(PackError::FileNotFound);
    }

    char magic[sizeof(k_magic)]{};
    uint32_t version = 0;
    uint32_t entry_count = 0;
    uint64_t toc_offset = 0;
    uint64_t content_hash = 0;

    stream->read(magic, sizeof(magic));
    stream->read(reinterpret_cast<char*>(&version), sizeof(version));
    stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));
    stream->read(reinterpret_cast<char*>(&toc_offset), sizeof(toc_offset));
    stream->read(reinterpret_cast<char*>(&content_hash), sizeof(content_hash));
    if (!*stream || std::memcmp(magic, k_magic, sizeof(magic)) != 0) {
        BRACKEN_LOG_ERROR("asset", "AssetPack::open('{}'): not a bracken pack", path);
        return Err(PackError::NotAPack);
    }
    if (version > k_pack_format_version) {
        BRACKEN_LOG_ERROR("asset", "AssetPack::open('{}'): version {} is newer than this build ({})",
                          path, version, k_pack_format_version);
        return Err(PackError::FutureVersion);
    }

    const auto file_size = static_cast<uint64_t>(std::filesystem::file_size(path));
    if (toc_offset > file_size) {
        return Err(PackError::Corrupt);
    }

    stream->seekg(static_cast<std::streamoff>(toc_offset), std::ios::beg);
    AssetPack pack;
    pack.entries_.reserve(entry_count);
    for (uint32_t i = 0; i < entry_count; ++i) {
        uint16_t name_len = 0;
        stream->read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        stream->read(name.data(), name_len);
        Entry e;
        e.name = std::move(name);
        stream->read(reinterpret_cast<char*>(&e.offset), sizeof(e.offset));
        stream->read(reinterpret_cast<char*>(&e.size), sizeof(e.size));
        if (!*stream || e.offset + e.size > file_size) {
            BRACKEN_LOG_ERROR("asset", "AssetPack::open('{}'): TOC entry {} is out of range", path,
                              i);
            return Err(PackError::Corrupt);
        }
        pack.entries_.push_back(std::move(e));
    }

    pack.path_ = path;
    pack.stream_ = std::move(stream);
    BRACKEN_LOG_INFO("asset", "Opened pack {} ({} entries)", path, pack.entries_.size());
    return pack;
}

Result<std::vector<std::byte>, PackError> AssetPack::read(std::string_view name) const {
    const Entry* e = find(name);
    if (e == nullptr) {
        return Err(PackError::EntryNotFound);
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(e->size));
    {
        // One stream shared by every worker: seek and read are a single
        // critical section. Decoding an asset costs far more than reading it,
        // so this is never where the time goes.
        std::scoped_lock lock(read_mutex_);
        stream_->clear();
        stream_->seekg(static_cast<std::streamoff>(e->offset), std::ios::beg);
        stream_->read(reinterpret_cast<char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
        if (!*stream_) {
            return Err(PackError::Corrupt);
        }
    }
    return bytes;
}

Result<void, PackError>
AssetPack::build(const std::string& out_path,
                 std::span<const std::pair<std::string, std::string>> name_and_source) {
    // Sorted so open() can binary-search the TOC.
    std::vector<std::pair<std::string, std::string>> sorted(name_and_source.begin(),
                                                            name_and_source.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    const std::string tmp = out_path + ".tmp";

    // Every failure path below must take the temp file with it. A guard rather
    // than a remove() at each return: a stray .pack.tmp is not just litter, it
    // is a file that looks like a build artifact and will be shipped by
    // whatever globs the output directory.
    struct TempGuard {
        std::string path;
        bool armed = true;
        ~TempGuard() {
            if (armed) {
                std::error_code ec;
                std::filesystem::remove(path, ec);
            }
        }
    } guard{tmp};

    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
        BRACKEN_LOG_ERROR("asset", "AssetPack::build('{}'): cannot open temp file", out_path);
        return Err(PackError::WriteFailed);
    }

    // Header is written twice: once as a placeholder so blob offsets are real,
    // then again at the end once the TOC offset is known.
    constexpr uint64_t k_header_bytes = sizeof(k_magic) + 4 + 4 + 8 + 8;
    std::vector<std::byte> zeros(k_header_bytes, std::byte{0});
    out.write(reinterpret_cast<const char*>(zeros.data()),
              static_cast<std::streamsize>(zeros.size()));

    std::vector<Entry> entries;
    entries.reserve(sorted.size());
    uint64_t cursor = k_header_bytes;
    uint64_t content_hash = 0xcbf29ce484222325ull;

    for (const auto& [name, source] : sorted) {
        std::ifstream in(source, std::ios::binary);
        if (!in) {
            BRACKEN_LOG_ERROR("asset", "AssetPack::build: cannot read source '{}'", source);
            return Err(PackError::FileNotFound);
        }
        std::vector<char> data((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());

        const uint64_t aligned = align_up(cursor, k_blob_align);
        for (uint64_t pad = cursor; pad < aligned; ++pad) {
            out.put('\0');
        }
        cursor = aligned;

        Entry e;
        e.name = name;
        e.offset = cursor;
        e.size = data.size();
        content_hash = hash_bytes(
            content_hash, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(data.data()), data.size()));

        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        cursor += data.size();
        entries.push_back(std::move(e));
    }

    const uint64_t toc_offset = cursor;
    std::vector<std::byte> toc;
    for (const Entry& e : entries) {
        put_pod(toc, static_cast<uint16_t>(e.name.size()));
        const auto* p = reinterpret_cast<const std::byte*>(e.name.data());
        toc.insert(toc.end(), p, p + e.name.size());
        put_pod(toc, e.offset);
        put_pod(toc, e.size);
    }
    out.write(reinterpret_cast<const char*>(toc.data()),
              static_cast<std::streamsize>(toc.size()));

    out.seekp(0, std::ios::beg);
    const auto entry_count = static_cast<uint32_t>(entries.size());
    out.write(k_magic, sizeof(k_magic));
    out.write(reinterpret_cast<const char*>(&k_pack_format_version), 4);
    out.write(reinterpret_cast<const char*>(&entry_count), 4);
    out.write(reinterpret_cast<const char*>(&toc_offset), 8);
    out.write(reinterpret_cast<const char*>(&content_hash), 8);
    out.flush();

    if (!out) {
        return Err(PackError::WriteFailed);
    }
    out.close();

    std::error_code ec;
    std::filesystem::rename(tmp, out_path, ec);
    if (ec) {
        BRACKEN_LOG_ERROR("asset", "AssetPack::build('{}'): rename failed: {}", out_path,
                          ec.message());
        return Err(PackError::WriteFailed);
    }
    guard.armed = false; // the rename consumed it

    BRACKEN_LOG_INFO("asset", "Built pack {} ({} entries, {} bytes)", out_path, entries.size(),
                     toc_offset);
    return {};
}

} // namespace bracken::core::asset
