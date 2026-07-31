#include "bracken/sim_frame.hpp"

#include "bracken/application.hpp"

#include <utility>

namespace bracken {

// MainFrame passthroughs live out-of-line so sim_frame.hpp does not include
// application.hpp (Application includes sim_frame.hpp for the hook types).

float MainFrame::fps() const {
    return app_->fps();
}

int MainFrame::screen_width() const {
    return app_->screen_width();
}

int MainFrame::screen_height() const {
    return app_->screen_height();
}

void MainFrame::request_screenshot(std::string path) {
    app_->request_screenshot(std::move(path));
}

} // namespace bracken
