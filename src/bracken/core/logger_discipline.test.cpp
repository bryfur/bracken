// Log-channel discipline: the engine has exactly one log channel.
//
// Every diagnostic an engine subsystem emits must reach `core::Logger`, because
// that is the only thing a host can install a `LogSink` on. Writing to
// `SDL_Log`, `std::cerr` or `std::cout` instead produces output the host cannot
// capture, filter, or route — which is precisely the failure that made GPU,
// pipeline, shader and renderer boot errors invisible to every consumer.
//
// This test walks the engine source tree and fails on any reintroduction.
// It is a source scan rather than a link-time check on purpose: `SDL_Log` is a
// variadic C function in a dependency we cannot poison, and the point is to
// catch the call site, not the symbol.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

// The only files allowed to talk to a raw output channel:
//   core/logger.cpp  — the sink implementation; it *is* the bridge to SDL.
//   core/assert.hpp  — abort path; must still print when the logger is gone or
//                      is itself the thing that failed.
//   this file        — it names the forbidden spellings.
const char* const ALLOWED[] = {
    "src/bracken/core/logger.cpp",
    "src/bracken/core/assert.hpp",
    "src/bracken/core/logger_discipline.test.cpp",
};

// Spellings that bypass core::Logger. Kept as fragments so this file's own
// mention of them does not have to be special-cased anywhere else.
struct Banned {
    const char* needle;
    const char* fix;
};
const Banned BANNED[] = {
    {"SDL_Log(", "use BRACKEN_LOG_{ERROR,WARN,INFO,DEBUG}(LOG_TAG, ...)"},
    {"std::cerr", "use BRACKEN_LOG_ERROR(LOG_TAG, ...)"},
    {"std::cout", "use BRACKEN_LOG_INFO(LOG_TAG, ...)"},
};

bool is_allowed(const std::string& rel) {
    for (const char* a : ALLOWED) {
        if (rel == a) {
            return true;
        }
    }
    return false;
}

std::string to_posix(const fs::path& p) {
    std::string s = p.generic_string();
    return s;
}

} // namespace

TEST(LoggerDiscipline, EngineSourcesUseOnlyCoreLogger) {
#ifndef BRACKEN_REPO_ROOT
    GTEST_SKIP() << "BRACKEN_REPO_ROOT not defined; cannot locate the source tree";
#else
    const fs::path root(BRACKEN_REPO_ROOT);
    const fs::path src = root / "src" / "bracken";
    if (!fs::is_directory(src)) {
        GTEST_SKIP() << "source tree not present at " << src.string();
    }

    std::vector<std::string> violations;
    for (const auto& entry : fs::recursive_directory_iterator(src)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const fs::path& p = entry.path();
        const std::string ext = p.extension().string();
        if (ext != ".cpp" && ext != ".hpp") {
            continue;
        }
        const std::string rel = to_posix(fs::relative(p, root));
        if (is_allowed(rel)) {
            continue;
        }

        std::ifstream in(p);
        ASSERT_TRUE(in) << "cannot read " << rel;
        std::string line;
        int lineno = 0;
        while (std::getline(in, line)) {
            ++lineno;
            for (const Banned& b : BANNED) {
                if (line.find(b.needle) != std::string_view::npos) {
                    violations.push_back(rel + ":" + std::to_string(lineno) + ": " + b.needle + " — " + b.fix);
                }
            }
        }
    }

    std::string report;
    for (const std::string& v : violations) {
        report += "\n  " + v;
    }
    EXPECT_TRUE(violations.empty()) << "engine sources must log through core::Logger only:" << report;
#endif
}
