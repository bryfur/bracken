#include "playground.hpp"

#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <SDL3_image/SDL_image.h>

namespace {

// The smoke run saves a screenshot mid-flight; reject a missing file or a
// degenerate (black / solid-color) image so a silently broken renderer that
// still hits 60fps can't pass the integration test.
bool verify_smoke_screenshot(const char* path) {
    SDL_Surface* loaded = IMG_Load(path);
    if (!loaded) {
        std::cerr << "[smoke] screenshot FAIL: unreadable '" << path << "': " << SDL_GetError() << "\n";
        return false;
    }
    SDL_Surface* rgb = SDL_ConvertSurface(loaded, SDL_PIXELFORMAT_RGB24);
    SDL_DestroySurface(loaded);
    if (!rgb) {
        std::cerr << "[smoke] screenshot FAIL: convert: " << SDL_GetError() << "\n";
        return false;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    const auto* pixels = static_cast<const uint8_t*>(rgb->pixels);
    for (int y = 0; y < rgb->h; ++y) {
        const uint8_t* row = pixels + static_cast<size_t>(y) * rgb->pitch;
        for (int x = 0; x < rgb->w; ++x) {
            const uint8_t* p = row + static_cast<size_t>(x) * 3;
            double luma = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            sum += luma;
            sum_sq += luma * luma;
        }
    }
    const double n = static_cast<double>(rgb->w) * rgb->h;
    const double mean = sum / n;
    const double stddev = std::sqrt(std::max(0.0, sum_sq / n - mean * mean));
    const bool ok = mean > 2.0 && mean < 250.0 && stddev > 5.0;
    std::cout << std::format("[smoke] screenshot {} path={} {}x{} mean_luma={:.1f} stddev={:.1f}\n",
                             ok ? "OK" : "FAIL", path, rgb->w, rgb->h, mean, stddev);
    SDL_DestroySurface(rgb);
    return ok;
}

} // namespace

// Usage: playground [--smoke [frames]] [--engine-camera=on|off]
//   --smoke: self-drive for N frames (default 240) and exit — the integration-
//            test mode. Prints "[smoke] OK ..." and "[smoke] screenshot OK ..."
//            on success; exit code 1 if the run or the screenshot check fails,
//            2 on bad arguments.
//   --engine-camera=off: drive the camera rig from the demo instead of letting
//            the engine drive it (Application::set_camera_driven_by_engine) —
//            coverage for the escape hatch a game with a custom rig uses.
int main(int argc, char** argv) {
    int smoke_frames = -1;
    bool engine_camera = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--engine-camera=off") == 0) {
            engine_camera = false;
        } else if (std::strcmp(argv[i], "--engine-camera=on") == 0) {
            engine_camera = true;
        } else if (std::strcmp(argv[i], "--smoke") == 0) {
            smoke_frames = 240;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                const char* arg = argv[++i];
                const char* arg_end = arg + std::strlen(arg);
                int frames = 0;
                auto [parse_end, ec] = std::from_chars(arg, arg_end, frames);
                if (ec != std::errc{} || parse_end != arg_end || frames <= 0) {
                    std::cerr << "invalid --smoke frame count (expected a positive integer): "
                              << arg << "\n";
                    return 2;
                }
                smoke_frames = frames;
            }
        } else {
            std::cerr << "unknown argument: " << argv[i] << "\n";
            return 2;
        }
    }

    playground::PlaygroundGame game(smoke_frames, engine_camera);
    if (!game.init_engine()) {
        return 1;
    }
    if (!game.start()) {
        game.shutdown_engine();
        return 1;
    }
    game.run();
    game.stop();
    game.shutdown_engine();

    if (smoke_frames > 0 && !verify_smoke_screenshot(playground::PlaygroundGame::SMOKE_SCREENSHOT_PATH)) {
        return 1;
    }
    return 0;
}
