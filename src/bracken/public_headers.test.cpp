// The public/internal boundary, as a build check instead of a doc paragraph.
//
// Four facts used to be maintained by hand in prose — `CLAUDE.md` ("Jolt types
// must stay out of public bracken headers"), `README.md`, `docs/architecture.md`
// and `docs/engine_design_notes.md` all state some version of it, and nothing
// in the build could tell you when it stopped being true. It held only because
// Jolt is linked PRIVATE, which is a *link*-time property being used to police
// an *include*-time invariant: a consumer that happens to have Jolt on its own
// include path would have compiled `physics/jolt_job_system.hpp` right out of
// the public tree without a word.
//
// Phase 11 moves that header to `physics/internal/` and states the rule here:
//
//   (a) no header outside an `internal/` directory may transitively reach
//       Jolt, ImGui, tinygltf or stb_;
//   (b) every header `bracken.hpp` names exists, is `bracken/`-rooted, and is
//       itself public;
//   (c) every bracken header the reference consumer (examples/playground)
//       includes is reachable from `bracken.hpp` — i.e. the umbrella really is
//       "the supported surface" and not a stale list;
//   (d) `application.hpp` pulls in no `gpu/` header. This is Phase 6b's win
//       (before the `ModelHandle` hoist, `application.hpp -> sim_frame.hpp ->
//       frame_builder.hpp -> frame_packet.hpp -> model_loader.hpp` exported
//       `gpu_buffer.hpp`, `gpu_texture.hpp` and `gpu_types.hpp` to every host
//       TU, to name one type), and it is the one that silently regresses.
//
// Honest note about (a): it passes on day one, on the same commit that moves
// the offending header. Its value is preventing the regression, not finding a
// bug. (d) is the same shape.
//
// `docs/refactor_plan.md` §"Phase 11" asked for a broader version of (d): "no
// header reachable from bracken.hpp pulls gpu_buffer.hpp or gpu_texture.hpp
// unless it is itself a gpu/ header". That is false against the code, and
// should be. `TerrainRenderer`, `GrassRenderer`, `WaterRenderer`,
// `AutoExposure`, `IBL` and `SceneRenderer` all hold GPU resources by value and
// have to name their types; a host that configures a sub-renderer is compiling
// the GPU layer by definition. The property actually worth guarding is the one
// about the HOST ENTRY POINT — a game's every TU includes `application.hpp` —
// which is what (d) checks.
//
// "Compiles standalone" is deliberately NOT checked here: a runtime scan cannot
// prove it. `src/bracken/CMakeLists.txt` generates one TU per umbrella header
// into the `bracken_public_headers` object library, reading the list out of
// `bracken.hpp` itself, so the compiler proves it at build time. This test's
// job is the include GRAPH; that target's job is compilability.
//
// GTEST_SKIP when BRACKEN_REPO_ROOT is unavailable — same mechanism as
// core/module_layering.test.cpp and core/logger_discipline.test.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// Third-party include prefixes that must never appear in the public tree.
// Matched against the raw include target (angle or quoted) as a prefix.
const char* const FORBIDDEN[] = {
    "Jolt/",      // rigid-body backend: linked PRIVATE, pimpl'd behind physics_world.hpp
    "imgui",      // dev overlay: dev_ui.hpp forward-declares, dev_ui.cpp includes
    "tiny_gltf",  // model loading: model_loader.cpp only
    "stb_",       // tinygltf's image codecs, same
};

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// A path (relative to src/bracken) is internal when any of its directory
// components is literally "internal".
bool is_internal(const std::string& rel) {
    return starts_with(rel, "internal/") || rel.find("/internal/") != std::string::npos;
}

bool is_test_or_bench(const std::string& rel) {
    return ends_with(rel, ".test.cpp") || ends_with(rel, ".bench.cpp");
}

// The include target on `line` — quoted or angled — with a flag for which.
// Empty target means the line is not an include.
struct Include {
    std::string target;
    bool quoted = false;
};

// Anchored at the start of the line on purpose. A loose `find('#')` also
// matches a `#include` quoted inside a comment — bracken.hpp's preamble
// contains exactly that, showing the reader how to spell the include — and
// would then report the umbrella as including itself. The CMake side that
// generates the standalone-compile TUs anchors the same way; the two parsers
// must agree on the header list or one of them is silently checking a
// different set.
Include parse_include(const std::string& line) {
    const std::size_t hash = line.find_first_not_of(" \t");
    if (hash == std::string::npos || line[hash] != '#') {
        return {};
    }
    std::size_t i = hash + 1;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        ++i;
    }
    if (line.compare(i, 7, "include") != 0) {
        return {};
    }
    i += 7;
    const std::size_t quote = line.find('"', i);
    const std::size_t angle = line.find('<', i);
    if (quote != std::string::npos && (angle == std::string::npos || quote < angle)) {
        const std::size_t close = line.find('"', quote + 1);
        if (close == std::string::npos) {
            return {};
        }
        return {line.substr(quote + 1, close - quote - 1), true};
    }
    if (angle != std::string::npos) {
        const std::size_t close = line.find('>', angle + 1);
        if (close == std::string::npos) {
            return {};
        }
        return {line.substr(angle + 1, close - angle - 1), false};
    }
    return {};
}

std::string normalize(const std::string& path) {
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i <= path.size()) {
        const std::size_t j = path.find('/', i);
        const std::string seg = path.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (seg == ".." && !parts.empty()) {
            parts.pop_back();
        } else if (!seg.empty() && seg != ".") {
            parts.push_back(seg);
        }
        if (j == std::string::npos) {
            break;
        }
        i = j + 1;
    }
    std::string out;
    for (const std::string& p : parts) {
        if (!out.empty()) {
            out += '/';
        }
        out += p;
    }
    return out;
}

struct SourceFile {
    std::string rel; // relative to src/bracken, POSIX
    std::vector<Include> includes;
};

fs::path repo_root() {
#ifdef BRACKEN_REPO_ROOT
    const fs::path root(BRACKEN_REPO_ROOT);
    if (fs::is_directory(root / "src" / "bracken")) {
        return root;
    }
#endif
    return {};
}

std::vector<Include> includes_of(const fs::path& file) {
    std::vector<Include> out;
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
        if (Include inc = parse_include(line); !inc.target.empty()) {
            out.push_back(std::move(inc));
        }
    }
    return out;
}

std::map<std::string, SourceFile> scan(const fs::path& src_root) {
    std::map<std::string, SourceFile> files;
    for (const auto& entry : fs::recursive_directory_iterator(src_root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string ext = entry.path().extension().string();
        if (ext != ".hpp" && ext != ".cpp") {
            continue;
        }
        SourceFile f;
        f.rel = fs::relative(entry.path(), src_root).generic_string();
        f.includes = includes_of(entry.path());
        files.emplace(f.rel, std::move(f));
    }
    return files;
}

// Resolve a quoted include to a path relative to src/bracken, or empty when it
// does not name an in-tree header.
std::string resolve(const fs::path& src_root, const std::string& includer_rel, const std::string& target) {
    std::string candidate;
    if (starts_with(target, "bracken/")) {
        candidate = target.substr(std::string("bracken/").size());
    } else {
        const std::size_t slash = includer_rel.rfind('/');
        const std::string dir = slash == std::string::npos ? std::string() : includer_rel.substr(0, slash + 1);
        candidate = normalize(dir + target);
    }
    return fs::exists(src_root / candidate) ? candidate : std::string();
}

// Transitive closure of in-tree headers reachable from `seeds` (inclusive).
std::set<std::string> closure(const fs::path& src_root, const std::map<std::string, SourceFile>& files,
                              const std::vector<std::string>& seeds) {
    std::set<std::string> seen;
    std::vector<std::string> stack = seeds;
    while (!stack.empty()) {
        const std::string node = stack.back();
        stack.pop_back();
        if (!seen.insert(node).second) {
            continue;
        }
        const auto it = files.find(node);
        if (it == files.end()) {
            continue;
        }
        for (const Include& inc : it->second.includes) {
            if (!inc.quoted) {
                continue;
            }
            if (std::string target = resolve(src_root, node, inc.target); !target.empty()) {
                stack.push_back(std::move(target));
            }
        }
    }
    return seen;
}

// The `bracken/<path>` targets listed in the umbrella, in file order.
std::vector<std::string> umbrella_headers(const fs::path& src_root) {
    std::vector<std::string> out;
    for (const Include& inc : includes_of(src_root / "bracken.hpp")) {
        if (inc.quoted && starts_with(inc.target, "bracken/")) {
            out.push_back(inc.target.substr(std::string("bracken/").size()));
        }
    }
    return out;
}

const char* forbidden_hit(const std::string& target) {
    for (const char* prefix : FORBIDDEN) {
        if (starts_with(target, prefix)) {
            return prefix;
        }
    }
    return nullptr;
}

} // namespace

// (a) The invariant CLAUDE.md, README.md, architecture.md and the design notes
//     all assert in prose. A header that reaches a vendored dependency drags
//     that dependency's include path into every consumer that names it —
//     which is exactly what `target_link_libraries(bracken PRIVATE Jolt)`
//     was silently relied upon to prevent.
TEST(PublicHeaders, NoThirdPartyLeakageOutsideInternalDirectories) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const fs::path src = root / "src" / "bracken";
    const std::map<std::string, SourceFile> files = scan(src);

    std::vector<std::string> violations;
    for (const auto& [rel, file] : files) {
        if (!ends_with(rel, ".hpp") || is_internal(rel)) {
            continue;
        }
        // Reachability, not just the direct line: a public header that includes
        // a public header that includes Jolt is just as broken.
        for (const std::string& reached : closure(src, files, {rel})) {
            if (is_internal(reached)) {
                violations.push_back(rel + " reaches INTERNAL header bracken/" + reached);
                continue;
            }
            const auto it = files.find(reached);
            if (it == files.end()) {
                continue;
            }
            for (const Include& inc : it->second.includes) {
                if (const char* hit = forbidden_hit(inc.target)) {
                    violations.push_back(rel + " reaches bracken/" + reached + ", which includes " + inc.target +
                                         " (forbidden prefix \"" + hit + "\")");
                }
            }
        }
    }
    std::sort(violations.begin(), violations.end());
    violations.erase(std::unique(violations.begin(), violations.end()), violations.end());

    std::string report;
    for (const std::string& v : violations) {
        report += "\n  " + v;
    }
    EXPECT_TRUE(violations.empty())
        << "a public bracken header must not reach a vendored dependency. Move the header under "
           "<module>/internal/ (see physics/internal/jolt_job_system.hpp) or pimpl the type away:"
        << report;
}

// (b) The umbrella is only worth having if it is real: every line resolves, is
//     spelled the way the include root demands, and names a PUBLIC header.
TEST(PublicHeaders, UmbrellaNamesRealPublicHeaders) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const fs::path src = root / "src" / "bracken";

    const std::vector<std::string> listed = umbrella_headers(src);
    EXPECT_GE(listed.size(), 20u) << "bracken.hpp parsed as " << listed.size()
                                  << " headers — the umbrella or this parser is broken";

    std::vector<std::string> problems;
    std::set<std::string> seen;
    for (const std::string& h : listed) {
        if (!fs::exists(src / h)) {
            problems.push_back("bracken/" + h + " does not exist");
        }
        if (is_internal(h)) {
            problems.push_back("bracken/" + h + " is an internal header and cannot be part of the public surface");
        }
        if (!seen.insert(h).second) {
            problems.push_back("bracken/" + h + " is listed twice");
        }
    }
    // Every include in the umbrella must be bracken/-rooted quoted form.
    for (const Include& inc : includes_of(src / "bracken.hpp")) {
        if (!inc.quoted || !starts_with(inc.target, "bracken/")) {
            problems.push_back("#include " + inc.target + " — the umbrella takes only quoted bracken/-rooted paths");
        }
    }

    std::string report;
    for (const std::string& p : problems) {
        report += "\n  " + p;
    }
    EXPECT_TRUE(problems.empty()) << "src/bracken/bracken.hpp:" << report;
}

// (c) "The supported surface" is a claim about the reference consumer. If the
//     playground reaches for a header the umbrella cannot supply, then either
//     the umbrella is stale or that header should not have been needed — and
//     both of those are worth being told about at ctest time rather than
//     discovering when a game copies the example.
TEST(PublicHeaders, UmbrellaCoversWhatTheReferenceConsumerIncludes) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const fs::path src = root / "src" / "bracken";
    const fs::path playground = root / "examples" / "playground";
    if (!fs::is_directory(playground)) {
        GTEST_SKIP() << "examples/playground unavailable";
    }

    const std::map<std::string, SourceFile> files = scan(src);
    const std::set<std::string> supported = closure(src, files, umbrella_headers(src));

    std::vector<std::string> uncovered;
    for (const auto& entry : fs::recursive_directory_iterator(playground)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string ext = entry.path().extension().string();
        if (ext != ".hpp" && ext != ".cpp") {
            continue;
        }
        for (const Include& inc : includes_of(entry.path())) {
            if (!inc.quoted || !starts_with(inc.target, "bracken/")) {
                continue;
            }
            const std::string rel = inc.target.substr(std::string("bracken/").size());
            if (!supported.contains(rel)) {
                uncovered.push_back(fs::relative(entry.path(), root).generic_string() + ": bracken/" + rel);
            }
        }
    }
    std::sort(uncovered.begin(), uncovered.end());
    uncovered.erase(std::unique(uncovered.begin(), uncovered.end()), uncovered.end());

    std::string report;
    for (const std::string& u : uncovered) {
        report += "\n  " + u;
    }
    EXPECT_TRUE(uncovered.empty()) << "examples/playground includes bracken headers the umbrella does not reach. "
                                      "Either add them to src/bracken/bracken.hpp (and say what they are for) or "
                                      "stop needing them:"
                                   << report;
}

// (d) Phase 6b, still true. A host TU that includes application.hpp must not
//     be compiling the GPU layer — that chain existed only so frame_packet.hpp
//     could name `ModelHandle`, and hoisting it into contracts/ cut it.
TEST(PublicHeaders, ApplicationHeaderDoesNotExportTheGpuLayer) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const fs::path src = root / "src" / "bracken";
    const std::map<std::string, SourceFile> files = scan(src);

    std::vector<std::string> leaked;
    for (const std::string& reached : closure(src, files, {"application.hpp"})) {
        if (starts_with(reached, "gpu/")) {
            leaked.push_back("bracken/" + reached);
        }
    }

    std::string report;
    for (const std::string& l : leaked) {
        report += "\n  " + l;
    }
    EXPECT_TRUE(leaked.empty()) << "application.hpp must not drag the GPU layer into every host TU "
                                   "(docs/architecture.md \"Module layering\"):"
                                << report;
}

// A scan that finds nothing proves nothing — pin the corpus, same as
// module_layering.test.cpp's ScanActuallySeesTheTree.
TEST(PublicHeaders, ScanActuallySeesTheTree) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const fs::path src = root / "src" / "bracken";
    const std::map<std::string, SourceFile> files = scan(src);

    EXPECT_GT(files.size(), 200u) << "expected the whole engine tree, got " << files.size() << " files";

    // The internal tree is non-empty and the forbidden-prefix matcher works —
    // otherwise (a) would pass by matching nothing at all.
    std::size_t internal_headers = 0;
    std::size_t forbidden_sites = 0;
    for (const auto& [rel, file] : files) {
        if (ends_with(rel, ".hpp") && is_internal(rel)) {
            ++internal_headers;
        }
        if (is_test_or_bench(rel)) {
            continue;
        }
        for (const Include& inc : file.includes) {
            if (forbidden_hit(inc.target) != nullptr) {
                ++forbidden_sites;
            }
        }
    }
    EXPECT_GE(internal_headers, 1u) << "no <module>/internal/ header found — the convention has been lost";
    EXPECT_GE(forbidden_sites, 5u) << "the forbidden-prefix matcher found almost nothing; it is probably broken "
                                      "(the engine does include Jolt, ImGui and tinygltf — from .cpp files)";

    EXPECT_TRUE(is_internal("physics/internal/jolt_job_system.hpp"));
    EXPECT_FALSE(is_internal("physics/physics_world.hpp"));
    EXPECT_FALSE(is_internal("render/internals.hpp")); // substring, not a component
}
