#pragma once

#include "bracken/core/result.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace bracken::save {

enum class SaveError : uint8_t {
    FileNotFound,
    NotASaveFile,   ///< magic mismatch — wrong file entirely
    Corrupt,        ///< checksum failed or the payload is truncated
    FutureVersion,  ///< written by a newer engine format than this build knows
    WriteFailed,
};

[[nodiscard]] const char* to_string(SaveError e) noexcept;

/** Bump only when the on-disk container changes shape. Host game data has its
 *  own independent version (SaveDocument::game_version). */
inline constexpr uint32_t k_save_format_version = 1;

/**
 * A saved game: typed values addressed by path.
 *
 * WHY A DOCUMENT AND NOT A STREAM. A stream format makes writing easy and
 * every subsequent schema change hard: fields must be read back in exactly the
 * order they were written, so inserting one in the middle invalidates every
 * existing save. Here each value carries its path, so a build can add fields,
 * remove them, or reorder its own code freely — a reader asks for what it
 * wants and takes a default when it is absent. That is the difference between
 * a save system you can ship a patch against and one you cannot.
 *
 * UNKNOWN KEYS SURVIVE A ROUND TRIP. Loading keeps every record, including
 * ones this build does not understand, so an older build that loads and
 * re-saves does not silently delete a newer build's data.
 *
 * PATHS are plain strings, by convention '/'-separated:
 *
 *     doc.write("player/position", pos);
 *     doc.write("player/health", 87);
 *     doc.write("world/seed", seed);
 *
 * There is no enforced hierarchy — a path is just a key. The convention exists
 * so a dumped save reads like an outline instead of a pile.
 *
 * DURABILITY. save() writes to a temporary file and renames it over the target,
 * so an interrupted save leaves the previous one intact. A save file that is
 * half-written because the console lost power is one of the worst bugs a game
 * can ship, and it is entirely preventable here rather than in each game.
 * Every payload carries a checksum, so a corrupt file is reported rather than
 * loaded as garbage.
 *
 * Tier: owned. Build and query it on whatever thread owns it; it touches no
 * engine globals. save()/load() do blocking file IO — keep them off the frame.
 */
class SaveDocument {
public:
    using Value = std::variant<bool, int64_t, double, std::string, std::vector<std::byte>,
                               glm::vec2, glm::vec3, glm::vec4, glm::quat, glm::mat4>;

    // ---- Identity --------------------------------------------------------

    /** Host-defined content version, for the game's own migrations. The engine
     *  never interprets it; it just round-trips it and hands it back so a
     *  reader can decide how to interpret what follows. */
    void set_game_version(uint32_t v) noexcept { game_version_ = v; }
    [[nodiscard]] uint32_t game_version() const noexcept { return game_version_; }

    /** The engine container version the file was written with. */
    [[nodiscard]] uint32_t format_version() const noexcept { return format_version_; }

    // ---- Writing ---------------------------------------------------------

    void write(std::string_view path, bool v);
    void write(std::string_view path, int32_t v) { write(path, static_cast<int64_t>(v)); }
    void write(std::string_view path, int64_t v);
    void write(std::string_view path, uint32_t v) { write(path, static_cast<int64_t>(v)); }
    void write(std::string_view path, float v) { write(path, static_cast<double>(v)); }
    void write(std::string_view path, double v);
    void write(std::string_view path, std::string_view v);
    void write(std::string_view path, const glm::vec2& v);
    void write(std::string_view path, const glm::vec3& v);
    void write(std::string_view path, const glm::vec4& v);
    void write(std::string_view path, const glm::quat& v);
    void write(std::string_view path, const glm::mat4& v);
    void write_blob(std::string_view path, std::span<const std::byte> bytes);

    /** Record how many elements a list has. Pair with indexed paths:
     *
     *      doc.write_count("orbs", orbs.size());
     *      for (i) doc.write(save::index_path("orbs", i, "position"), orbs[i].pos);
     *
     * Storing the count explicitly means a reader never has to probe for the
     * first missing index, which would stop early on a sparse list. */
    void write_count(std::string_view path, std::size_t count);

    // ---- Reading ---------------------------------------------------------

    [[nodiscard]] bool has(std::string_view path) const;

    /** Read into `out`; leaves it untouched and returns false when the path is
     *  absent OR holds a different type. A save from another build is data,
     *  not a contract — a type mismatch must not be undefined behavior. */
    [[nodiscard]] bool read(std::string_view path, bool& out) const;
    [[nodiscard]] bool read(std::string_view path, int32_t& out) const;
    [[nodiscard]] bool read(std::string_view path, int64_t& out) const;
    [[nodiscard]] bool read(std::string_view path, uint32_t& out) const;
    [[nodiscard]] bool read(std::string_view path, float& out) const;
    [[nodiscard]] bool read(std::string_view path, double& out) const;
    [[nodiscard]] bool read(std::string_view path, std::string& out) const;
    [[nodiscard]] bool read(std::string_view path, glm::vec2& out) const;
    [[nodiscard]] bool read(std::string_view path, glm::vec3& out) const;
    [[nodiscard]] bool read(std::string_view path, glm::vec4& out) const;
    [[nodiscard]] bool read(std::string_view path, glm::quat& out) const;
    [[nodiscard]] bool read(std::string_view path, glm::mat4& out) const;
    [[nodiscard]] bool read_blob(std::string_view path, std::vector<std::byte>& out) const;

    /** Read, or return `fallback`. The common case: a field added after this
     *  save was written should not need a `has()` first. */
    template <typename T>
    [[nodiscard]] T read_or(std::string_view path, T fallback) const {
        T value = fallback;
        return read(path, value) ? value : fallback;
    }

    [[nodiscard]] std::size_t read_count(std::string_view path) const;

    // ---- Files -----------------------------------------------------------

    /** Serialize and durably replace `path`: written to `path.tmp` first, then
     *  renamed, so an interrupted save leaves the previous file intact. */
    [[nodiscard]] core::Result<void, SaveError> save(const std::string& path) const;

    [[nodiscard]] static core::Result<SaveDocument, SaveError> load(const std::string& path);

    /** Number of stored records, including ones this build does not read. */
    [[nodiscard]] std::size_t record_count() const noexcept { return records_.size(); }

    /** Every path present, in write order. For dumping a save to a log. */
    [[nodiscard]] std::vector<std::string_view> paths() const;

private:
    struct Record {
        std::string path;
        Value value;
    };

    Record* find(std::string_view path);
    [[nodiscard]] const Record* find(std::string_view path) const;
    void set(std::string_view path, Value v);

    std::vector<Record> records_;
    uint32_t game_version_ = 0;
    uint32_t format_version_ = k_save_format_version;
};

/** Build `prefix/index/leaf` — the convention SaveDocument's list support
 *  expects. Provided so every host does not hand-roll the same std::format. */
[[nodiscard]] std::string index_path(std::string_view prefix, std::size_t index,
                                     std::string_view leaf);

} // namespace bracken::save
