// The docs cite the code. This checks that the citations still resolve.
//
// `docs/architecture.md` opens with "Everything in this file is verified
// against the code it cites" — a promise that, until now, nothing could keep.
// Across ~20k lines of documentation there are several hundred backticked
// citations of files and identifiers, and the only way a stale one surfaced
// was a reader noticing.
//
// WHAT THIS CHECKS, AND WHY NOT LINES
// -----------------------------------
// The obvious design — assert every `path:NNN` citation still points at the
// right line — is the wrong one, and it is worth saying why in the file that
// declines to do it. It would institutionalize line numbers at exactly the
// moment they should be abandoned: this refactor moved ~3,700 lines of code
// between translation units, and a line-keyed suite would have turned every
// one of those moves into a doc chore enforced by a red test. Worse, it makes
// the *citation format* load-bearing, so the cheapest way to keep the suite
// green becomes "stop citing".
//
// So the checks here are anchored, not line-numbered:
//
//   (a) every cited PATH exists;
//   (b) every cited IDENTIFIER appears somewhere in the file cited next to it
//       (grep, not line — a moved function still passes, a renamed one does not);
//   (c) every module directory under src/bracken has a row in architecture.md's
//       module map — a new module cannot appear undocumented;
//   (d) every engine .hpp includes in-tree headers `bracken/`-rooted, never by
//       relative path (the include-root convention CLAUDE.md states).
//
// Line numbers get an ADVISORY drift report on stdout: a `path:NNN` past the
// end of its file is definitely stale and worth printing, but it does not fail
// the suite. Printing is the right severity for a fact the reader can check in
// two seconds and the test cannot check at all.
//
// WHAT IS OUT OF SCOPE
// -------------------
// `docs/refactor_plan.md` and `docs/audits/` are excluded by design. They are
// historical records: the plan's whole method is quoting the code *as it was
// before* each phase, with the line numbers it had then. Making them track the
// tree would destroy the evidence they exist to preserve.
//
// GTEST_SKIP when BRACKEN_REPO_ROOT is unavailable — same mechanism as
// core/module_layering.test.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// Documentation that describes the CURRENT tree. See "out of scope" above for
// what is deliberately missing.
const char* const DOCS[] = {
    "CLAUDE.md",
    "README.md",
    "docs/architecture.md",
    "docs/technical_spec.md",
    "docs/user_guide.md",
    "docs/effects_system.md",
    "docs/engine_design_notes.md",
    "docs/multithreading_plan.md",
    "docs/skinned_mesh_format.md",
    "src/bracken/shaders/README.md",
    "src/bracken/contracts/README.md",
};

// Prefixes a citation may be resolved against, in order.
const char* const ROOTS[] = {
    "", "src/", "src/bracken/", "src/bracken/shaders/", "docs/", "examples/", "examples/playground/",
};

// Citations that name a file in a VENDORED dependency (fetched into build/_deps,
// absent from this repo) or a build artifact. Matched as a substring of the
// citation. Each one is a real, checkable fact about a third-party file — it is
// simply not a fact about a file we ship.
const char* const EXTERNAL[] = {
    "imgui", "Jolt/", "tiny_gltf", "stb_", "json.hpp", "SDL_gpu.h", "SDL3/", "benchmark/", "gtest/",
};
// NOTE: "assets/" used to be exempted here, because the raw packs and their
// outputs were untracked. Since 2026-08-01 examples/playground/assets/ is
// tracked in git and holds only files the playground references by name, so
// asset citations are checkable like any other and are no longer excused.

// Files the docs name as an EXAMPLE of something the reader would create. They
// are supposed not to exist; that is the whole point of the passage. Keep this
// list tiny — every entry is a hole in the check.
const char* const ILLUSTRATIVE[] = {
    "vignette.frag.hlsl", // user_guide.md "adding a post-process pass" walkthrough
    "myshader.hlsl",      // user_guide.md "misnamed file" gotcha
};

// Directories under src/bracken that are not modules and need no module-map row.
bool is_non_module_dir(const std::string& dir) {
    // `<module>/internal/` is by construction not a module — it is the private
    // half of its parent (see src/bracken/bracken.hpp, public_headers.test.cpp).
    return dir == "internal" || dir.rfind("internal/", 0) == 0 || dir.find("/internal") != std::string::npos;
}

bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

bool contains(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> read_lines(const fs::path& p) {
    std::vector<std::string> lines;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(std::move(line));
    }
    return lines;
}

// A backticked code span, with the offsets of its delimiters so neighbouring
// spans can be related to each other precisely (see the pairing rule below).
struct Span {
    std::string text;
    std::size_t begin = 0; // index of the opening backtick
    std::size_t end = 0;   // one past the closing backtick
};

// The code spans on one line, in order. Handles ``double`` fences by treating a
// run of backticks as one delimiter.
std::vector<Span> code_spans(const std::string& line) {
    std::vector<Span> spans;
    std::size_t i = 0;
    while (i < line.size()) {
        if (line[i] != '`') {
            ++i;
            continue;
        }
        std::size_t ticks = 0;
        while (i + ticks < line.size() && line[i + ticks] == '`') {
            ++ticks;
        }
        const std::string fence(ticks, '`');
        const std::size_t open = i + ticks;
        const std::size_t close = line.find(fence, open);
        if (close == std::string::npos) {
            break;
        }
        spans.push_back({line.substr(open, close - open), i, close + ticks});
        i = close + ticks;
    }
    return spans;
}

const char* const SOURCE_EXTS[] = {".hpp", ".cpp", ".hlsl", ".hlsli", ".md", ".txt", ".json", ".py", ".cmake"};

bool has_source_ext(const std::string& s) {
    for (const char* ext : SOURCE_EXTS) {
        const std::string e(ext);
        if (s.size() > e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0) {
            return true;
        }
    }
    return false;
}

struct Citation {
    std::string path;  // the file part, as written
    std::string lines; // whatever followed ':' — may be "12", "12-30", "12–30", or empty
};

// A span is a path citation when the part before the first ':' ends in a source
// extension and the whole thing is a plain path.
//
// Rejected on sight, because they name a SET of files rather than one:
// globs (`*.test.cpp`), brace expansions (`terrain.{vert,frag}.hlsl`),
// placeholders (`{name}.vert.hlsl`, `bracken/<module>/<file>.hpp`), and bare
// extensions (`.comp.hlsl`). Checking those would mean reimplementing shell
// globbing to verify a claim the reader already reads as a pattern.
bool parse_citation(const std::string& span, Citation& out) {
    if (span.empty() || span[0] == '.') {
        return false;
    }
    if (span.find_first_of(" \t*{}<>,") != std::string::npos) {
        return false;
    }
    const std::size_t colon = span.find(':');
    const std::string path = colon == std::string::npos ? span : span.substr(0, colon);
    if (!has_source_ext(path)) {
        return false;
    }
    out.path = path;
    out.lines = colon == std::string::npos ? std::string() : span.substr(colon + 1);
    return true;
}

bool is_external(const std::string& path) {
    for (const char* e : EXTERNAL) {
        if (contains(path, e)) {
            return true;
        }
    }
    for (const char* i : ILLUSTRATIVE) {
        if (path == i || (path.size() > std::string(i).size() &&
                          path.compare(path.size() - std::string(i).size(), std::string(i).size(), i) == 0)) {
            return true;
        }
    }
    return false;
}

// Directories the index must never enter. PRUNED, not filtered: the build tree
// plus .git holds hundreds of thousands of files, and merely skipping them
// after the iterator has already descended turns this scan from milliseconds
// into tens of seconds — worse when something is writing into one of them.
//
// `.claude/worktrees` is pruned for a DIFFERENT and stronger reason: it holds
// git worktrees — whole second checkouts of this same repo. Descending into one
// puts a second `bvh.hpp`, a second `pbr.hlsli` and a second everything else
// into an index keyed by BASENAME, and `resolve_citation` then answers with
// whichever copy the directory walk happened to reach first. That makes every
// citation check in this file a coin flip on a mid-edit copy of the tree: the
// suite fails, and it fails naming a path no one is working on. Only that one
// subdirectory is pruned, not `.claude` itself, because `.claude/RULES.md` is a
// tracked file the docs legitimately cite.
//
// Takes the path relative to the repo root rather than a bare basename, because
// "worktrees" alone is too generic a name to prune repo-wide.
bool is_pruned_dir(const fs::path& relative) {
    const std::string name = relative.filename().string();
    if (starts_with(name, "build") || name == ".git" || name == "_deps" || name == ".cache") {
        return true;
    }
    return relative.generic_string() == ".claude/worktrees";
}

// An index of every tracked repo file by basename, for citations written as a
// bare filename (`scene_renderer.cpp`) rather than a full path.
std::map<std::string, std::vector<std::string>> basename_index(const fs::path& root) {
    std::map<std::string, std::vector<std::string>> index;
    fs::recursive_directory_iterator it(root);
    const fs::recursive_directory_iterator end;
    for (; it != end; ++it) {
        const fs::directory_entry& entry = *it;
        if (entry.is_directory()) {
            if (is_pruned_dir(fs::relative(entry.path(), root))) {
                it.disable_recursion_pending();
            }
            continue;
        }
        if (entry.is_regular_file()) {
            index[entry.path().filename().string()].push_back(
                fs::relative(entry.path(), root).generic_string());
        }
    }
    return index;
}

// Resolve a citation to a repo-relative path, or empty.
//
// Handles the docs' `unit.hpp/.cpp` idiom (one unit, two files) by falling back
// to the part before the trailing `/.`: `render/bloom.hpp/.cpp` resolves as
// `render/bloom.hpp`. Checking one half of a pair is enough — a unit does not
// half-move.
std::string resolve_one(const fs::path& root, const std::map<std::string, std::vector<std::string>>& index,
                        const std::string& cited) {
    if (cited.find('/') == std::string::npos) {
        const auto it = index.find(cited);
        return it == index.end() ? std::string() : it->second.front();
    }
    for (const char* prefix : ROOTS) {
        const std::string candidate = std::string(prefix) + cited;
        if (fs::is_regular_file(root / candidate)) {
            return candidate;
        }
    }
    return {};
}

std::string resolve(const fs::path& root, const std::map<std::string, std::vector<std::string>>& index,
                    const std::string& cited) {
    if (std::string direct = resolve_one(root, index, cited); !direct.empty()) {
        return direct;
    }
    if (const std::size_t pair = cited.rfind("/."); pair != std::string::npos && pair > 0) {
        return resolve_one(root, index, cited.substr(0, pair));
    }
    return {};
}

// Does this span look like a code identifier a reader would expect to find in
// the neighbouring file? Deliberately conservative: prose in backticks is
// common, and a false positive here is a permanently-red suite.
bool is_identifier_span(const std::string& s) {
    if (s.empty() || s.size() > 80) {
        return false;
    }
    std::string body = s;
    if (body.size() > 2 && body.compare(body.size() - 2, 2, "()") == 0) {
        body.resize(body.size() - 2);
    }
    bool has_upper = false;
    bool has_scope = false;
    for (std::size_t i = 0; i < body.size(); ++i) {
        const char c = body[i];
        if (c == ':') {
            if (i + 1 >= body.size() || body[i + 1] != ':') {
                return false;
            }
            has_scope = true;
            ++i;
            continue;
        }
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        if (!ok) {
            return false;
        }
        if (c >= 'A' && c <= 'Z') {
            has_upper = true;
        }
    }
    if (body.empty() || (body[0] >= '0' && body[0] <= '9')) {
        return false;
    }
    // Require some evidence of code-ness: a scope operator, an underscore, or a
    // capital that is not just a sentence-initial word.
    return has_scope || contains(body, "_") || (has_upper && body.find_first_of("abcdefghijklmnopqrstuvwxyz") !=
                                                                  std::string::npos &&
                                                (body[0] >= 'A' && body[0] <= 'Z'));
}

// The last component of a qualified name: `FrameBuilder::seal` -> `seal`.
std::string last_component(const std::string& id) {
    const std::size_t pos = id.rfind("::");
    return pos == std::string::npos ? id : id.substr(pos + 2);
}

// THE PAIRING RULE, and it is deliberately narrow: an identifier is "cited in"
// a file only when the doc writes exactly
//
//     `Identifier` (`path/to/file.hpp`)
//
// — the parenthesised-source idiom these docs use everywhere. Anything looser
// (nearest code span, same sentence, same line) pairs `SceneRenderer` with
// `light_cluster.hpp` in a sentence saying the cluster grid is DRIVEN BY the
// scene renderer, and a check with false positives is a check that gets
// deleted. The cost of the narrow rule is coverage; the benefit is that every
// failure it reports is real.
bool is_parenthesised_source(const std::string& line, const Span& id, const Span& path) {
    if (path.begin < id.end || path.end >= line.size() || line[path.end] != ')') {
        return false;
    }
    const std::string gap = line.substr(id.end, path.begin - id.end);
    return gap == " (" || gap == "(";
}

fs::path repo_root() {
#ifdef BRACKEN_REPO_ROOT
    const fs::path root(BRACKEN_REPO_ROOT);
    if (fs::is_directory(root / "src" / "bracken")) {
        return root;
    }
#endif
    return {};
}

} // namespace

// (a) A citation that names a file which no longer exists is the loudest
//     possible form of doc rot, and the cheapest to detect.
TEST(DocsCoherence, EveryCitedPathExists) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const auto index = basename_index(root);

    std::vector<std::string> missing;
    std::size_t checked = 0;
    for (const char* doc : DOCS) {
        const fs::path path = root / doc;
        if (!fs::is_regular_file(path)) {
            missing.push_back(std::string(doc) + " — the documentation map itself is stale");
            continue;
        }
        const std::vector<std::string> lines = read_lines(path);
        for (std::size_t n = 0; n < lines.size(); ++n) {
            for (const Span& span : code_spans(lines[n])) {
                Citation cite;
                if (!parse_citation(span.text, cite) || is_external(cite.path)) {
                    continue;
                }
                ++checked;
                if (resolve(root, index, cite.path).empty()) {
                    missing.push_back(std::string(doc) + ":" + std::to_string(n + 1) + " cites `" + cite.path +
                                      "`, which does not exist");
                }
            }
        }
    }
    EXPECT_GT(checked, 300u) << "only " << checked << " path citations parsed — the scanner is probably broken";

    std::string report;
    for (const std::string& m : missing) {
        report += "\n  " + m;
    }
    EXPECT_TRUE(missing.empty()) << "documentation cites files that do not exist:" << report;
}

// (b) The identifier half. A doc that says `Foo::bar` (`foo.hpp`) is making a
//     claim about that file's contents; a rename breaks it and a move does not.
//     That is exactly the sensitivity we want, and it is why this is a grep and
//     not a line check.
TEST(DocsCoherence, CitedIdentifiersAppearInTheCitedFile) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const auto index = basename_index(root);

    std::map<std::string, std::string> file_cache;
    auto text_of = [&](const std::string& rel) -> const std::string& {
        auto it = file_cache.find(rel);
        if (it == file_cache.end()) {
            it = file_cache.emplace(rel, read_file(root / rel)).first;
        }
        return it->second;
    };

    std::vector<std::string> broken;
    std::size_t checked = 0;
    for (const char* doc : DOCS) {
        const fs::path path = root / doc;
        if (!fs::is_regular_file(path)) {
            continue;
        }
        const std::vector<std::string> lines = read_lines(path);
        for (std::size_t n = 0; n < lines.size(); ++n) {
            const std::vector<Span> spans = code_spans(lines[n]);
            for (std::size_t s = 1; s < spans.size(); ++s) {
                Citation cite;
                if (!parse_citation(spans[s].text, cite) || is_external(cite.path)) {
                    continue;
                }
                const Span& id = spans[s - 1];
                Citation ignored;
                if (parse_citation(id.text, ignored) || !is_identifier_span(id.text)) {
                    continue;
                }
                if (!is_parenthesised_source(lines[n], id, spans[s])) {
                    continue;
                }
                const std::string rel = resolve(root, index, cite.path);
                if (rel.empty()) {
                    continue; // reported by EveryCitedPathExists
                }
                ++checked;
                const std::string& text = text_of(rel);
                std::string body = id.text;
                if (body.size() > 2 && body.compare(body.size() - 2, 2, "()") == 0) {
                    body.resize(body.size() - 2);
                }
                if (!contains(text, body) && !contains(text, last_component(body))) {
                    broken.push_back(std::string(doc) + ":" + std::to_string(n + 1) + " says `" + id.text +
                                     "` is in `" + cite.path + "` (" + rel + ") — it is not there");
                }
            }
        }
    }
    EXPECT_GT(checked, 50u) << "only " << checked << " identifier/path pairs found — the pairing heuristic is "
                               "probably broken";

    std::string report;
    for (const std::string& b : broken) {
        report += "\n  " + b;
    }
    EXPECT_TRUE(broken.empty()) << "documentation names identifiers that are not in the file it cites "
                                   "(renamed? moved to another file? then move the citation too):"
                                << report;
}

// (c) The module map is the first thing a reader meets and the easiest thing to
//     forget. A new directory of headers with no row is an undocumented module.
TEST(DocsCoherence, EveryModuleDirectoryHasAnArchitectureRow) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const std::string arch = read_file(root / "docs" / "architecture.md");
    ASSERT_FALSE(arch.empty()) << "docs/architecture.md is missing or empty";

    const fs::path src = root / "src" / "bracken";
    std::set<std::string> dirs;
    for (const auto& entry : fs::recursive_directory_iterator(src)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".hpp") {
            continue;
        }
        const std::string rel = fs::relative(entry.path(), src).generic_string();
        const std::size_t slash = rel.rfind('/');
        if (slash == std::string::npos) {
            continue; // root files: covered by the `src/bracken/` (root) row
        }
        const std::string dir = rel.substr(0, slash);
        if (!is_non_module_dir(dir)) {
            dirs.insert(dir);
        }
    }
    EXPECT_GE(dirs.size(), 12u) << "only " << dirs.size() << " module directories found — scan is broken";

    std::vector<std::string> undocumented;
    for (const std::string& dir : dirs) {
        // A row is a markdown table cell holding the directory as a code span.
        if (!contains(arch, "| `" + dir + "/`")) {
            undocumented.push_back(dir + "/");
        }
    }

    std::string report;
    for (const std::string& u : undocumented) {
        report += "\n  " + u;
    }
    EXPECT_TRUE(undocumented.empty()) << "docs/architecture.md \"Module map\" has no row for:" << report
                                      << "\n(add `| `<dir>/` | <namespace> | <contents> |`)";
}

// (d) The include-root convention, stated in CLAUDE.md ("engine headers are
//     included with a `bracken/`-rooted path") and never checked. A relative
//     include compiles fine and quietly makes the header un-relocatable.
TEST(DocsCoherence, EngineHeadersUseBrackenRootedIncludes) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const fs::path src = root / "src" / "bracken";

    std::vector<std::string> violations;
    std::size_t scanned = 0;
    for (const auto& entry : fs::recursive_directory_iterator(src)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".hpp") {
            continue;
        }
        ++scanned;
        const std::string rel = fs::relative(entry.path(), src).generic_string();
        std::ifstream in(entry.path());
        std::string line;
        while (std::getline(in, line)) {
            const std::size_t hash = line.find('#');
            if (hash == std::string::npos) {
                continue;
            }
            std::size_t i = hash + 1;
            while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
                ++i;
            }
            if (line.compare(i, 7, "include") != 0) {
                continue;
            }
            const std::size_t open = line.find('"', i + 7);
            if (open == std::string::npos) {
                continue;
            }
            const std::size_t close = line.find('"', open + 1);
            if (close == std::string::npos) {
                continue;
            }
            const std::string target = line.substr(open + 1, close - open - 1);
            if (starts_with(target, "bracken/")) {
                continue;
            }
            // A quoted include that resolves relative to this header is an
            // in-tree header spelled the wrong way. Anything else is
            // third-party and quoted by that library's own convention.
            const fs::path sibling = entry.path().parent_path() / target;
            if (fs::exists(sibling)) {
                violations.push_back(rel + ": #include \"" + target + "\" — spell it \"bracken/...\"");
            }
        }
    }
    EXPECT_GT(scanned, 100u) << "only " << scanned << " headers scanned";

    std::string report;
    for (const std::string& v : violations) {
        report += "\n  " + v;
    }
    EXPECT_TRUE(violations.empty()) << "CLAUDE.md: engine headers are included with a `bracken/`-rooted path:"
                                    << report;
}

// ADVISORY. Not a failure: see the header comment on why line-keyed doc tests
// are a trap. A `path:NNN` past the end of its file is unambiguously stale
// though, so print it — the reader can fix it in seconds and the suite never
// goes red over a code move.
TEST(DocsCoherence, LineCitationDriftIsReportedNotEnforced) {
    const fs::path root = repo_root();
    if (root.empty()) {
        GTEST_SKIP() << "source tree unavailable (BRACKEN_REPO_ROOT unset or not a directory)";
    }
    const auto index = basename_index(root);

    std::map<std::string, std::size_t> length_cache;
    auto length_of = [&](const std::string& rel) {
        auto it = length_cache.find(rel);
        if (it == length_cache.end()) {
            it = length_cache.emplace(rel, read_lines(root / rel).size()).first;
        }
        return it->second;
    };

    std::vector<std::string> drift;
    std::size_t line_citations = 0;
    for (const char* doc : DOCS) {
        const fs::path path = root / doc;
        if (!fs::is_regular_file(path)) {
            continue;
        }
        const std::vector<std::string> lines = read_lines(path);
        for (std::size_t n = 0; n < lines.size(); ++n) {
            for (const Span& span : code_spans(lines[n])) {
                Citation cite;
                if (!parse_citation(span.text, cite) || cite.lines.empty() || is_external(cite.path)) {
                    continue;
                }
                if (cite.lines[0] < '0' || cite.lines[0] > '9') {
                    continue;
                }
                const std::string rel = resolve(root, index, cite.path);
                if (rel.empty()) {
                    continue;
                }
                ++line_citations;
                const std::size_t cited_line = static_cast<std::size_t>(std::stoul(cite.lines));
                if (cited_line > length_of(rel)) {
                    drift.push_back(std::string(doc) + ":" + std::to_string(n + 1) + "  `" + span.text + "`  (" + rel +
                                    " has " + std::to_string(length_of(rel)) + " lines)");
                }
            }
        }
    }

    // GTEST_LOG_, not the standard streams: core/logger_discipline.test.cpp
    // bans those across the whole tree, tests included.
    GTEST_LOG_(INFO) << "[docs] " << line_citations << " line-keyed citations; " << drift.size()
                     << " point past the end of their file.";
    for (const std::string& d : drift) {
        GTEST_LOG_(INFO) << "[docs]   " << d;
    }
    GTEST_LOG_(INFO) << "[docs] Advisory only — prefer citing an identifier over a line number.";
    SUCCEED();
}
