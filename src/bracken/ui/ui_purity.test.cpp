// UI-module purity: `bracken::ui` runs in the SIM STAGE and must therefore be
// unable to reach the GPU, SDL's GPU surface, or the renderer at all.
//
// WHY THIS IS A SOURCE SCAN AND NOT AN ASSERTION. The tier macros
// (BRACKEN_ASSERT_MAIN_THREAD, BRACKEN_ASSERT_NOT_SIM_STAGE) are debug-only
// runtime checks on the CALLEE, and they only fire if the offending call is
// actually executed on a bad thread by a test that happens to exercise it.
// "The UI module cannot touch the GPU" is a stronger and simpler claim: there
// is no include path from `src/bracken/ui/` to anything GPU-shaped, so the
// call cannot be written. A scan proves that for every line of the module at
// once, including lines nobody has a test for yet.
//
// It also pins the OTHER direction, which is the one that breaks the header
// DAG rather than the threading: `render/` must never include `ui/`. scene/
// already includes render/, and ui/ includes scene/, so a render -> ui edge
// closes render -> ui -> scene -> render and
// core/module_layering.test.cpp:HeaderGraphIsAcyclic fails somewhere that does
// not name the file that caused it.
//
// SCOPE, stated honestly: this scans DIRECT include lines, not the transitive
// closure the way public_headers.test.cpp does. That is enough for the mistake
// it exists to catch (somebody reaches for a renderer type from a widget), and
// the transitive half is already covered — the module's only engine dependency
// is `scene/frame_builder.hpp`, whose own includes are contracts/, core/,
// memory/ and `scene/frame_packet.hpp`, none of which reach gpu/ or render/.
// If ui/ ever grows a second engine dependency, widen this to a closure walk.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

/// Include targets no file under src/bracken/ui/ may name. Matched as a
/// substring of the whole `#include` line, which is enough: the spellings are
/// distinctive and the alternative (parsing the target) buys nothing here.
///
/// NOTE what is deliberately ABSENT. `bracken/input_handler.hpp` is allowed,
/// and it includes SDL3/SDL_events.h and SDL3/SDL_scancode.h — `ui_input.hpp`
/// maps an `InputSnapshot`, which is a plain-data type whose key edges are
/// indexed by SDL scancode. Naming a scancode is not touching a device; every
/// entry below is a resource or a command stream that only the main thread may
/// own.
struct Forbidden {
    const char* needle;
    const char* why;
};

const Forbidden FORBIDDEN[] = {
    {"bracken/gpu/", "the GPU layer is main-thread tier; the sim stage may not name it"},
    {"bracken/render/", "render/ is main-thread tier, AND ui/ -> render/ -> ... -> ui/ would cycle"},
    {"bracken/scene/scene_renderer", "SceneRenderer::render_frame is main-thread; emit through the lane instead"},
    {"SDL3/SDL_gpu.h", "an SDL GPU handle may not appear in a sim-stage type"},
    {"SDL3/SDL_render.h", "SDL's 2D renderer is main-thread tier"},
    {"SDL3_ttf/", "text is measured through contracts/font_metrics.hpp, never through live SDL_ttf"},
};

std::string to_posix(const fs::path& p) {
    return p.generic_string();
}

/// The `#include` target on a line, or empty when the line is not an include.
/// Anchored at the first non-blank character so a commented-out include is
/// ignored — the same rule public_headers.test.cpp uses.
std::string include_line(const std::string& line) {
    std::size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        ++i;
    }
    if (line.compare(i, 1, "#") != 0) {
        return {};
    }
    ++i;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        ++i;
    }
    if (line.compare(i, 7, "include") != 0) {
        return {};
    }
    return line.substr(i + 7);
}

std::vector<fs::path> sources_under(const fs::path& dir) {
    std::vector<fs::path> out;
    if (!fs::is_directory(dir)) {
        return out;
    }
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string ext = entry.path().extension().string();
        if (ext == ".cpp" || ext == ".hpp") {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

TEST(UiPurity, TheModuleCannotReachTheGpuOrTheRenderer) {
#ifndef BRACKEN_REPO_ROOT
    GTEST_SKIP() << "BRACKEN_REPO_ROOT not defined; cannot locate the source tree";
#else
    const fs::path root(BRACKEN_REPO_ROOT);
    const fs::path ui = root / "src" / "bracken" / "ui";
    if (!fs::is_directory(ui)) {
        GTEST_SKIP() << "source tree not present at " << ui.string();
    }

    const std::vector<fs::path> files = sources_under(ui);
    // A corpus guard: if the glob ever stops finding the module, an empty scan
    // would pass silently and this test would be worth nothing.
    ASSERT_GE(files.size(), 8u) << "expected the ui/ module to have more files than this";

    std::vector<std::string> violations;
    for (const fs::path& p : files) {
        const std::string rel = to_posix(fs::relative(p, root));
        if (rel.find("ui_purity.test.cpp") != std::string::npos) {
            continue; // this file names every forbidden spelling on purpose
        }
        std::ifstream in(p);
        ASSERT_TRUE(in) << "cannot read " << rel;
        std::string line;
        int lineno = 0;
        while (std::getline(in, line)) {
            ++lineno;
            const std::string target = include_line(line);
            if (target.empty()) {
                continue;
            }
            for (const Forbidden& f : FORBIDDEN) {
                if (target.find(f.needle) != std::string::npos) {
                    violations.push_back(rel + ":" + std::to_string(lineno) + ": includes " + f.needle + " — " +
                                         f.why);
                }
            }
        }
    }

    std::string report;
    for (const std::string& v : violations) {
        report += "\n  " + v;
    }
    EXPECT_TRUE(violations.empty()) << "bracken::ui is sim-stage code and must stay device-free:" << report;
#endif
}

TEST(UiPurity, RenderNeverIncludesUi) {
#ifndef BRACKEN_REPO_ROOT
    GTEST_SKIP() << "BRACKEN_REPO_ROOT not defined; cannot locate the source tree";
#else
    const fs::path root(BRACKEN_REPO_ROOT);
    const fs::path render = root / "src" / "bracken" / "render";
    if (!fs::is_directory(render)) {
        GTEST_SKIP() << "source tree not present at " << render.string();
    }

    const std::vector<fs::path> files = sources_under(render);
    ASSERT_GE(files.size(), 20u) << "expected render/ to have more files than this";

    std::vector<std::string> violations;
    for (const fs::path& p : files) {
        std::ifstream in(p);
        ASSERT_TRUE(in) << "cannot read " << p.string();
        std::string line;
        int lineno = 0;
        while (std::getline(in, line)) {
            ++lineno;
            const std::string target = include_line(line);
            if (target.find("bracken/ui/") != std::string::npos) {
                violations.push_back(to_posix(fs::relative(p, root)) + ":" + std::to_string(lineno));
            }
        }
    }

    std::string report;
    for (const std::string& v : violations) {
        report += "\n  " + v;
    }
    EXPECT_TRUE(violations.empty())
        << "render/ must not include ui/ — that closes render -> ui -> scene -> render:" << report;
#endif
}

// The metrics mirror is the thing that let the UI leave the main thread at all,
// so pin that it too stays leaf-layer clean. contracts/ has its own general
// rule in core/module_layering.test.cpp; this is the specific one that matters
// here: font_metrics must not reach back into the renderer that produces it.
TEST(UiPurity, FontMetricsDoesNotReachTheTextRenderer) {
#ifndef BRACKEN_REPO_ROOT
    GTEST_SKIP() << "BRACKEN_REPO_ROOT not defined; cannot locate the source tree";
#else
    const fs::path root(BRACKEN_REPO_ROOT);
    const fs::path dir = root / "src" / "bracken" / "contracts";
    if (!fs::is_directory(dir)) {
        GTEST_SKIP() << "source tree not present at " << dir.string();
    }
    for (const char* name : {"font_metrics.hpp", "font_metrics.cpp", "ui_texture.hpp"}) {
        const fs::path p = dir / name;
        ASSERT_TRUE(fs::is_regular_file(p)) << p.string() << " is missing";
        std::ifstream in(p);
        ASSERT_TRUE(in);
        std::string line;
        while (std::getline(in, line)) {
            const std::string target = include_line(line);
            if (target.empty()) {
                continue;
            }
            EXPECT_EQ(target.find("bracken/render/"), std::string::npos)
                << name << " must not include render/: it is the LEAF-layer type render/ produces";
            EXPECT_EQ(target.find("SDL3"), std::string::npos) << name << " must not include SDL";
        }
    }
#endif
}
