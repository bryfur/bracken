// Module layering: the engine's header graph is a DAG, and it stays one.
//
// Until Phase 6b the production module graph had a six-module strongly-
// connected component — {@root, dev, render, render/lighting, scene, systems} —
// with 21 elementary cycles of length <= 4. There were zero *file*-level include
// cycles, which is why it compiled and why the coupling was invisible: nothing
// in the build ever complained. This test makes the layering a checkable fact
// instead of an implied one.
//
// It is a source scan for the same reason logger_discipline.test.cpp is: the
// property being checked is about `#include` lines, and the compiler cannot
// report "this edge points the wrong way" — only "this cycle does not compile",
// which is the failure we already never had.
//
// Three assertions:
//   (a) the module graph over headers has no cycle          -- what 6b made true
//   (b) contracts/ includes only core/, contracts/, glm, <std>  -- leaf-layer rule
//   (c) core/, memory/ and gpu/ never include upward         -- baseline guard
//
// GTEST_SKIP when BRACKEN_REPO_ROOT is unavailable (an installed/packaged build
// has no source tree to scan), same mechanism as the logger-discipline test.
//
// ---------------------------------------------------------------------------
// What counts as a "module"
// ---------------------------------------------------------------------------
// A node is the directory a file sits in, relative to src/bracken, with
// `core/jobs`, `core/asset` and `render/lighting` kept distinct from their
// parents — they are separately-documented modules with their own contracts.
//
// Files that sit *directly* in src/bracken/ are each their own node. That is
// deliberate and it is the accurate attribution, not a loophole: there is no
// root module. `application.hpp` (top of the stack) and `model_loader.hpp` (an
// asset-layer leaf that render/ and scene/ reach up for by path) are unrelated
// files that happen to share a directory, and bucketing them together would
// manufacture a cycle out of `sim_frame.hpp -> scene/frame_builder.hpp` plus
// `scene/scene_renderer.hpp -> model_loader.hpp` where no header actually
// depends on another header cyclically. Singleton nodes are strictly more
// precise: every genuine cycle a `@root` bucket would catch — e.g. if
// `model_loader.hpp` ever grew an include of `scene/` — is still caught here,
// because the cycle then runs through the individual file.
//
// docs/architecture.md "Module layering" states the intended order in prose;
// this test is what keeps it true.

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

// Sub-directories that are modules in their own right rather than part of their
// parent. Longest-prefix match wins.
const char* const SUBMODULES[] = {
    "core/asset",
    "core/jobs",
    "render/lighting",
};

// `rel` is a path relative to src/bracken, POSIX-separated.
std::string module_of(const std::string& rel) {
    const std::size_t slash = rel.rfind('/');
    if (slash == std::string::npos) {
        return rel; // loose root file: its own node — see the header comment
    }
    const std::string dir = rel.substr(0, slash);
    for (const char* sub : SUBMODULES) {
        if (dir == sub) {
            return dir;
        }
    }
    const std::size_t first = dir.find('/');
    return first == std::string::npos ? dir : dir.substr(0, first);
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_test_or_bench(const std::string& rel) {
    return ends_with(rel, ".test.cpp") || ends_with(rel, ".bench.cpp");
}

// The quoted-include target of `line`, or empty. Only quoted includes can name
// an in-tree header; angle includes are always external.
std::string quoted_include(const std::string& line) {
    const std::size_t hash = line.find('#');
    if (hash == std::string::npos) {
        return {};
    }
    std::size_t i = hash + 1;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        ++i;
    }
    if (line.compare(i, 7, "include") != 0) {
        return {};
    }
    const std::size_t open = line.find('"', i + 7);
    if (open == std::string::npos) {
        return {};
    }
    const std::size_t close = line.find('"', open + 1);
    if (close == std::string::npos) {
        return {};
    }
    return line.substr(open + 1, close - open - 1);
}

// The angle-include target of `line`, or empty.
std::string angle_include(const std::string& line) {
    const std::size_t hash = line.find('#');
    if (hash == std::string::npos) {
        return {};
    }
    std::size_t i = hash + 1;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        ++i;
    }
    if (line.compare(i, 7, "include") != 0) {
        return {};
    }
    const std::size_t open = line.find('<', i + 7);
    if (open == std::string::npos) {
        return {};
    }
    const std::size_t close = line.find('>', open + 1);
    if (close == std::string::npos) {
        return {};
    }
    return line.substr(open + 1, close - open - 1);
}

// Normalize a/b/../c -> a/c. Inputs here never escape the tree.
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
    std::string rel;                   // relative to src/bracken, POSIX
    std::vector<std::string> quoted;   // raw quoted include targets
    std::vector<std::string> angled;   // raw angle include targets
};

// Resolve a quoted include target to a path relative to src/bracken, or empty
// if it does not name an in-tree header (SDL3/..., third-party, etc).
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

std::vector<SourceFile> scan(const fs::path& src_root) {
    std::vector<SourceFile> files;
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
        std::ifstream in(entry.path());
        std::string line;
        while (std::getline(in, line)) {
            if (std::string q = quoted_include(line); !q.empty()) {
                f.quoted.push_back(std::move(q));
            } else if (std::string a = angle_include(line); !a.empty()) {
                f.angled.push_back(std::move(a));
            }
        }
        files.push_back(std::move(f));
    }
    std::sort(files.begin(), files.end(), [](const SourceFile& a, const SourceFile& b) { return a.rel < b.rel; });
    return files;
}

// Locates src/bracken, or an empty path when the source tree is unavailable.
fs::path source_root() {
#ifdef BRACKEN_REPO_ROOT
    const fs::path src = fs::path(BRACKEN_REPO_ROOT) / "src" / "bracken";
    if (fs::is_directory(src)) {
        return src;
    }
#endif
    return {};
}

} // namespace

// (a) The whole point of Phase 6b. A cycle here means two modules cannot be
//     read, tested or reasoned about independently, and that a header change in
//     either rebuilds both.
TEST(ModuleLayering, HeaderGraphIsAcyclic) {
    const fs::path src = source_root();
    if (src.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const std::vector<SourceFile> files = scan(src);

    // module -> module -> one witness edge ("a.hpp -> b.hpp"), first in sorted
    // order so the failure message is deterministic.
    std::map<std::string, std::map<std::string, std::string>> edges;
    for (const SourceFile& f : files) {
        if (!ends_with(f.rel, ".hpp")) {
            continue; // (a) is about compile coupling: headers only
        }
        const std::string from = module_of(f.rel);
        for (const std::string& q : f.quoted) {
            const std::string target = resolve(src, f.rel, q);
            if (target.empty()) {
                continue;
            }
            const std::string to = module_of(target);
            if (to == from) {
                continue;
            }
            edges[from].emplace(to, f.rel + " -> " + target);
        }
    }

    // Iterative depth-first cycle detection; reports the first cycle found with
    // the include line responsible for each hop.
    enum class Mark { White, Grey, Black };
    std::map<std::string, Mark> mark;
    for (const auto& [from, tos] : edges) {
        mark.emplace(from, Mark::White);
        for (const auto& [to, _] : tos) {
            mark.emplace(to, Mark::White);
        }
    }

    std::vector<std::string> path;
    std::string report;
    auto edge_witness = [&](const std::string& a, const std::string& b) {
        auto it = edges.find(a);
        if (it == edges.end()) {
            return std::string("?");
        }
        auto jt = it->second.find(b);
        return jt == it->second.end() ? std::string("?") : jt->second;
    };

    // Explicit stack: (node, index of next successor to visit).
    for (const auto& [start, _] : mark) {
        if (mark[start] != Mark::White || !report.empty()) {
            continue;
        }
        std::vector<std::pair<std::string, std::size_t>> stack{{start, 0}};
        mark[start] = Mark::Grey;
        path.push_back(start);
        while (!stack.empty() && report.empty()) {
            auto& [node, next] = stack.back();
            std::vector<std::string> succs;
            if (auto it = edges.find(node); it != edges.end()) {
                for (const auto& [to, __] : it->second) {
                    succs.push_back(to);
                }
            }
            if (next < succs.size()) {
                const std::string to = succs[next];
                ++next;
                if (mark[to] == Mark::Grey) {
                    // Found a cycle: everything from `to` onward in `path`.
                    const auto begin = std::find(path.begin(), path.end(), to);
                    report = "\n  module cycle:";
                    for (auto it = begin; it != path.end(); ++it) {
                        const std::string& a = *it;
                        const std::string& b = (it + 1 == path.end()) ? to : *(it + 1);
                        report += "\n    " + a + " -> " + b + "   (" + edge_witness(a, b) + ")";
                    }
                } else if (mark[to] == Mark::White) {
                    mark[to] = Mark::Grey;
                    path.push_back(to);
                    stack.emplace_back(to, 0);
                }
            } else {
                mark[node] = Mark::Black;
                path.pop_back();
                stack.pop_back();
            }
        }
    }

    EXPECT_TRUE(report.empty()) << "the engine header graph must be a DAG (docs/architecture.md "
                                  "\"Module layering\"):"
                               << report;
}

// (b) The leaf-layer rule, stated in src/bracken/contracts/README.md. If a
//     contract header ever reaches sideways or upward it stops being a leaf and
//     the DAG argument collapses — every module includes this layer.
TEST(ModuleLayering, ContractsAreALeafLayer) {
    const fs::path src = source_root();
    if (src.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }

    std::vector<std::string> violations;
    for (const SourceFile& f : scan(src)) {
        if (!starts_with(f.rel, "contracts/") || is_test_or_bench(f.rel)) {
            continue;
        }
        for (const std::string& q : f.quoted) {
            const std::string target = resolve(src, f.rel, q);
            if (target.empty()) {
                violations.push_back(f.rel + ": #include \"" + q + "\" — contracts/ may not depend on anything "
                                                                   "outside the engine tree");
                continue;
            }
            if (!starts_with(target, "core/") && !starts_with(target, "contracts/")) {
                violations.push_back(f.rel + ": #include \"bracken/" + target +
                                     "\" — contracts/ may include only bracken/core/*, bracken/contracts/*, "
                                     "glm and the standard library");
            }
        }
    }

    std::string report;
    for (const std::string& v : violations) {
        report += "\n  " + v;
    }
    EXPECT_TRUE(violations.empty()) << "src/bracken/contracts/README.md: the leaf-layer rule:" << report;
}

// (c) The baseline this engine has always had and must not lose: core/ is a
//     perfect sink (fan-out 0), and neither memory/ nor gpu/ knows anything
//     about the renderer, the scene, or the host. Tests and benchmarks are
//     exempt — cross-layer contract tests (gpu/uniform_layout.test.cpp,
//     core/contracts.test.cpp) exist precisely to check the layers agree.
TEST(ModuleLayering, LowerLayersNeverIncludeUpward) {
    const fs::path src = source_root();
    if (src.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }

    const std::set<std::string> lower = {"core", "core/asset", "core/jobs", "memory", "gpu"};
    const std::set<std::string> upper = {"render", "render/lighting", "scene",       "systems",
                                         "effects", "dev",            "procedural", "physics",
                                         "animation", "render_graph"};

    std::vector<std::string> violations;
    for (const SourceFile& f : scan(src)) {
        if (is_test_or_bench(f.rel) || !lower.contains(module_of(f.rel))) {
            continue;
        }
        for (const std::string& q : f.quoted) {
            const std::string target = resolve(src, f.rel, q);
            if (target.empty()) {
                continue;
            }
            const std::string to = module_of(target);
            const bool root_header = target.find('/') == std::string::npos;
            if (upper.contains(to) || root_header) {
                violations.push_back(f.rel + ": #include \"bracken/" + target + "\"");
            }
        }
    }

    std::string report;
    for (const std::string& v : violations) {
        report += "\n  " + v;
    }
    EXPECT_TRUE(violations.empty()) << "core/, memory/ and gpu/ are below everything else and must not "
                                       "include render/, scene/, systems/, effects/, dev/ or a root header:"
                                    << report;
}

// A scan that finds nothing proves nothing. Pin the corpus so a broken path,
// a renamed directory or a busted include parser fails loudly instead of
// silently passing all three checks above.
TEST(ModuleLayering, ScanActuallySeesTheTree) {
    const fs::path src = source_root();
    if (src.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const std::vector<SourceFile> files = scan(src);
    EXPECT_GT(files.size(), 200u) << "expected the whole engine tree, got " << files.size() << " files";

    std::size_t contracts = 0;
    std::size_t in_tree_edges = 0;
    for (const SourceFile& f : files) {
        if (starts_with(f.rel, "contracts/")) {
            ++contracts;
        }
        for (const std::string& q : f.quoted) {
            if (!resolve(src, f.rel, q).empty()) {
                ++in_tree_edges;
            }
        }
    }
    EXPECT_GE(contracts, 8u) << "contracts/ should hold the leaf layer";
    EXPECT_GT(in_tree_edges, 300u) << "include parsing produced implausibly few in-tree edges";

    // The attribution rule itself, spelled out by example.
    EXPECT_EQ(module_of("core/logger.hpp"), "core");
    EXPECT_EQ(module_of("core/jobs/job_system.hpp"), "core/jobs");
    EXPECT_EQ(module_of("render/lighting/light.hpp"), "render/lighting");
    EXPECT_EQ(module_of("render/bloom.hpp"), "render");
    EXPECT_EQ(module_of("model_loader.hpp"), "model_loader.hpp");
}
