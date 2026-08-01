#include "bracken/save/save_document.hpp"

#include "bracken/core/logger.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>

namespace bracken::save {
namespace {

constexpr char k_magic[8] = {'B', 'R', 'K', 'S', 'A', 'V', 'E', '\0'};

/** Type tags are part of the on-disk format — append only, never renumber. */
enum class Tag : uint8_t {
    Bool = 0,
    Int64 = 1,
    Double = 2,
    String = 3,
    Blob = 4,
    Vec2 = 5,
    Vec3 = 6,
    Vec4 = 7,
    Quat = 8,
    Mat4 = 9,
};

// FNV-1a over the payload. Not cryptographic and not meant to be: it detects a
// truncated or bit-rotted file, which is what actually happens to saves.
uint64_t checksum(std::span<const std::byte> bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (std::byte b : bytes) {
        h ^= static_cast<uint64_t>(std::to_integer<uint8_t>(b));
        h *= 0x100000001b3ull;
    }
    return h;
}

void put(std::vector<std::byte>& out, const void* data, std::size_t n) {
    const auto* p = static_cast<const std::byte*>(data);
    out.insert(out.end(), p, p + n);
}

template <typename T> void put_pod(std::vector<std::byte>& out, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    put(out, &v, sizeof(T));
}

/** Sequential reader that can fail but never reads out of bounds. Every get
 *  checks first, so a truncated file produces `ok == false` rather than a
 *  crash — a corrupt save must be a message, not a segfault. */
struct Cursor {
    std::span<const std::byte> data;
    std::size_t pos = 0;
    bool ok = true;

    bool take(void* dst, std::size_t n) {
        if (!ok || pos + n > data.size()) {
            ok = false;
            return false;
        }
        std::memcpy(dst, data.data() + pos, n);
        pos += n;
        return true;
    }

    template <typename T> bool take_pod(T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        return take(&v, sizeof(T));
    }

    bool take_bytes(std::vector<std::byte>& v, std::size_t n) {
        if (!ok || pos + n > data.size()) {
            ok = false;
            return false;
        }
        v.assign(data.begin() + static_cast<std::ptrdiff_t>(pos),
                 data.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
        return true;
    }
};

} // namespace

const char* to_string(SaveError e) noexcept {
    switch (e) {
    case SaveError::FileNotFound: return "file not found";
    case SaveError::NotASaveFile: return "not a bracken save file";
    case SaveError::Corrupt: return "save file is corrupt or truncated";
    case SaveError::FutureVersion: return "save written by a newer engine format";
    case SaveError::WriteFailed: return "could not write the save file";
    }
    return "unknown";
}

std::string index_path(std::string_view prefix, std::size_t index, std::string_view leaf) {
    return std::format("{}/{}/{}", prefix, index, leaf);
}

// ---- record access --------------------------------------------------------

SaveDocument::Record* SaveDocument::find(std::string_view path) {
    auto it = std::find_if(records_.begin(), records_.end(),
                           [&](const Record& r) { return r.path == path; });
    return it == records_.end() ? nullptr : &*it;
}

const SaveDocument::Record* SaveDocument::find(std::string_view path) const {
    auto it = std::find_if(records_.begin(), records_.end(),
                           [&](const Record& r) { return r.path == path; });
    return it == records_.end() ? nullptr : &*it;
}

void SaveDocument::set(std::string_view path, Value v) {
    if (Record* existing = find(path)) {
        existing->value = std::move(v);
        return;
    }
    records_.push_back(Record{std::string(path), std::move(v)});
}

// ---- writing --------------------------------------------------------------

void SaveDocument::write(std::string_view p, bool v) { set(p, v); }
void SaveDocument::write(std::string_view p, int64_t v) { set(p, v); }
void SaveDocument::write(std::string_view p, double v) { set(p, v); }
void SaveDocument::write(std::string_view p, std::string_view v) { set(p, std::string(v)); }
void SaveDocument::write(std::string_view p, const glm::vec2& v) { set(p, v); }
void SaveDocument::write(std::string_view p, const glm::vec3& v) { set(p, v); }
void SaveDocument::write(std::string_view p, const glm::vec4& v) { set(p, v); }
void SaveDocument::write(std::string_view p, const glm::quat& v) { set(p, v); }
void SaveDocument::write(std::string_view p, const glm::mat4& v) { set(p, v); }

void SaveDocument::write_blob(std::string_view p, std::span<const std::byte> bytes) {
    set(p, std::vector<std::byte>(bytes.begin(), bytes.end()));
}

void SaveDocument::write_count(std::string_view path, std::size_t count) {
    write(std::format("{}/count", path), static_cast<int64_t>(count));
}

// ---- reading --------------------------------------------------------------

bool SaveDocument::has(std::string_view path) const { return find(path) != nullptr; }

namespace {
template <typename T> bool read_as(const SaveDocument::Value* v, T& out) {
    if (v == nullptr) {
        return false;
    }
    if (const T* typed = std::get_if<T>(v)) {
        out = *typed;
        return true;
    }
    return false;
}
} // namespace

bool SaveDocument::read(std::string_view p, bool& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}

bool SaveDocument::read(std::string_view p, int64_t& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}

bool SaveDocument::read(std::string_view p, int32_t& out) const {
    int64_t wide = 0;
    if (!read(p, wide)) {
        return false;
    }
    out = static_cast<int32_t>(wide);
    return true;
}

bool SaveDocument::read(std::string_view p, uint32_t& out) const {
    int64_t wide = 0;
    if (!read(p, wide)) {
        return false;
    }
    out = static_cast<uint32_t>(wide);
    return true;
}

bool SaveDocument::read(std::string_view p, double& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}

bool SaveDocument::read(std::string_view p, float& out) const {
    double wide = 0.0;
    if (!read(p, wide)) {
        return false;
    }
    out = static_cast<float>(wide);
    return true;
}

bool SaveDocument::read(std::string_view p, std::string& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}

bool SaveDocument::read(std::string_view p, glm::vec2& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}
bool SaveDocument::read(std::string_view p, glm::vec3& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}
bool SaveDocument::read(std::string_view p, glm::vec4& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}
bool SaveDocument::read(std::string_view p, glm::quat& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}
bool SaveDocument::read(std::string_view p, glm::mat4& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}

bool SaveDocument::read_blob(std::string_view p, std::vector<std::byte>& out) const {
    const Record* r = find(p);
    return read_as(r ? &r->value : nullptr, out);
}

std::size_t SaveDocument::read_count(std::string_view path) const {
    const int64_t n = read_or<int64_t>(std::format("{}/count", path), 0);
    return n > 0 ? static_cast<std::size_t>(n) : 0;
}

std::vector<std::string_view> SaveDocument::paths() const {
    std::vector<std::string_view> out;
    out.reserve(records_.size());
    for (const Record& r : records_) {
        out.emplace_back(r.path);
    }
    return out;
}

// ---- files ----------------------------------------------------------------

core::Result<void, SaveError> SaveDocument::save(const std::string& path) const {
    std::vector<std::byte> payload;

    for (const Record& r : records_) {
        const auto key_len = static_cast<uint16_t>(r.path.size());
        put_pod(payload, key_len);
        put(payload, r.path.data(), r.path.size());

        std::visit(
            [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, bool>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Bool));
                    put_pod(payload, static_cast<uint8_t>(v ? 1 : 0));
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Int64));
                    put_pod(payload, v);
                } else if constexpr (std::is_same_v<T, double>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Double));
                    put_pod(payload, v);
                } else if constexpr (std::is_same_v<T, std::string>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::String));
                    put_pod(payload, static_cast<uint32_t>(v.size()));
                    put(payload, v.data(), v.size());
                } else if constexpr (std::is_same_v<T, std::vector<std::byte>>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Blob));
                    put_pod(payload, static_cast<uint32_t>(v.size()));
                    put(payload, v.data(), v.size());
                } else if constexpr (std::is_same_v<T, glm::vec2>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Vec2));
                    put(payload, &v.x, sizeof(float) * 2);
                } else if constexpr (std::is_same_v<T, glm::vec3>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Vec3));
                    put(payload, &v.x, sizeof(float) * 3);
                } else if constexpr (std::is_same_v<T, glm::vec4>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Vec4));
                    put(payload, &v.x, sizeof(float) * 4);
                } else if constexpr (std::is_same_v<T, glm::quat>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Quat));
                    const float q[4] = {v.x, v.y, v.z, v.w};
                    put(payload, q, sizeof(q));
                } else if constexpr (std::is_same_v<T, glm::mat4>) {
                    put_pod(payload, static_cast<uint8_t>(Tag::Mat4));
                    put(payload, &v[0][0], sizeof(float) * 16);
                }
            },
            r.value);
    }

    const uint64_t sum = checksum(payload);

    // Write to a sibling temp file, then rename. rename() is atomic within a
    // filesystem, so a crash or power loss leaves the OLD save intact rather
    // than a half-written one. Writing straight to the target is how games
    // ship "my save is gone" bugs.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            BRACKEN_LOG_ERROR("save", "save('{}'): cannot open temp file '{}'", path, tmp);
            return core::Err(SaveError::WriteFailed);
        }
        const uint32_t payload_bytes = static_cast<uint32_t>(payload.size());
        out.write(k_magic, sizeof(k_magic));
        out.write(reinterpret_cast<const char*>(&format_version_), sizeof(format_version_));
        out.write(reinterpret_cast<const char*>(&game_version_), sizeof(game_version_));
        out.write(reinterpret_cast<const char*>(&payload_bytes), sizeof(payload_bytes));
        out.write(reinterpret_cast<const char*>(&sum), sizeof(sum));
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
        out.flush();
        if (!out) {
            BRACKEN_LOG_ERROR("save", "save('{}'): write failed", path);
            std::error_code ignore;
            std::filesystem::remove(tmp, ignore);
            return core::Err(SaveError::WriteFailed);
        }
    } // close before rename

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        BRACKEN_LOG_ERROR("save", "save('{}'): rename from temp failed: {}", path, ec.message());
        std::error_code ignore;
        std::filesystem::remove(tmp, ignore);
        return core::Err(SaveError::WriteFailed);
    }

    BRACKEN_LOG_INFO("save", "Wrote {} ({} records, {} bytes)", path, records_.size(),
                     payload.size());
    return {};
}

core::Result<SaveDocument, SaveError> SaveDocument::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return core::Err(SaveError::FileNotFound);
    }

    std::vector<char> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    constexpr std::size_t k_header = sizeof(k_magic) + 4 + 4 + 4 + 8;
    if (file.size() < k_header || std::memcmp(file.data(), k_magic, sizeof(k_magic)) != 0) {
        BRACKEN_LOG_ERROR("save", "load('{}'): not a bracken save file", path);
        return core::Err(SaveError::NotASaveFile);
    }

    std::size_t off = sizeof(k_magic);
    uint32_t format_version = 0;
    uint32_t game_version = 0;
    uint32_t payload_bytes = 0;
    uint64_t stored_sum = 0;
    std::memcpy(&format_version, file.data() + off, 4); off += 4;
    std::memcpy(&game_version, file.data() + off, 4); off += 4;
    std::memcpy(&payload_bytes, file.data() + off, 4); off += 4;
    std::memcpy(&stored_sum, file.data() + off, 8); off += 8;

    if (format_version > k_save_format_version) {
        // Refuse rather than guess: a newer container may have tags this build
        // cannot skip correctly, and half-reading a save is worse than not.
        BRACKEN_LOG_ERROR("save", "load('{}'): format version {} is newer than this build ({})",
                          path, format_version, k_save_format_version);
        return core::Err(SaveError::FutureVersion);
    }
    if (file.size() - off < payload_bytes) {
        BRACKEN_LOG_ERROR("save", "load('{}'): truncated (want {} payload bytes, have {})", path,
                          payload_bytes, file.size() - off);
        return core::Err(SaveError::Corrupt);
    }

    std::span<const std::byte> payload(reinterpret_cast<const std::byte*>(file.data() + off),
                                       payload_bytes);
    if (checksum(payload) != stored_sum) {
        BRACKEN_LOG_ERROR("save", "load('{}'): checksum mismatch; file is corrupt", path);
        return core::Err(SaveError::Corrupt);
    }

    SaveDocument doc;
    doc.format_version_ = format_version;
    doc.game_version_ = game_version;

    Cursor c{payload};
    while (c.ok && c.pos < payload.size()) {
        uint16_t key_len = 0;
        if (!c.take_pod(key_len)) {
            break;
        }
        std::vector<std::byte> key_bytes;
        if (!c.take_bytes(key_bytes, key_len)) {
            break;
        }
        std::string key(reinterpret_cast<const char*>(key_bytes.data()), key_bytes.size());

        uint8_t tag = 0;
        if (!c.take_pod(tag)) {
            break;
        }

        Value value;
        switch (static_cast<Tag>(tag)) {
        case Tag::Bool: {
            uint8_t b = 0;
            c.take_pod(b);
            value = (b != 0);
            break;
        }
        case Tag::Int64: {
            int64_t v = 0;
            c.take_pod(v);
            value = v;
            break;
        }
        case Tag::Double: {
            double v = 0.0;
            c.take_pod(v);
            value = v;
            break;
        }
        case Tag::String: {
            uint32_t n = 0;
            c.take_pod(n);
            std::vector<std::byte> bytes;
            c.take_bytes(bytes, n);
            value = std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            break;
        }
        case Tag::Blob: {
            uint32_t n = 0;
            c.take_pod(n);
            std::vector<std::byte> bytes;
            c.take_bytes(bytes, n);
            value = std::move(bytes);
            break;
        }
        case Tag::Vec2: {
            glm::vec2 v{};
            c.take(&v.x, sizeof(float) * 2);
            value = v;
            break;
        }
        case Tag::Vec3: {
            glm::vec3 v{};
            c.take(&v.x, sizeof(float) * 3);
            value = v;
            break;
        }
        case Tag::Vec4: {
            glm::vec4 v{};
            c.take(&v.x, sizeof(float) * 4);
            value = v;
            break;
        }
        case Tag::Quat: {
            float q[4] = {0, 0, 0, 1};
            c.take(q, sizeof(q));
            value = glm::quat(q[3], q[0], q[1], q[2]); // glm::quat is (w,x,y,z)
            break;
        }
        case Tag::Mat4: {
            glm::mat4 v{1.0f};
            c.take(&v[0][0], sizeof(float) * 16);
            value = v;
            break;
        }
        default:
            // An unknown tag means this build cannot compute the record's
            // length, so it cannot skip to the next one either. Stop rather
            // than resynchronize on garbage.
            BRACKEN_LOG_WARN("save", "load('{}'): unknown value tag {} at '{}'; stopping", path,
                             tag, key);
            c.ok = false;
            break;
        }

        if (c.ok) {
            doc.records_.push_back(Record{std::move(key), std::move(value)});
        }
    }

    if (!c.ok && c.pos < payload.size()) {
        BRACKEN_LOG_WARN("save", "load('{}'): stopped early after {} records", path,
                         doc.records_.size());
    }

    BRACKEN_LOG_INFO("save", "Loaded {} ({} records, game version {})", path,
                     doc.records_.size(), doc.game_version_);
    return doc;
}

} // namespace bracken::save
