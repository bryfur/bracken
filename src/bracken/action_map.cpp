#include "bracken/action_map.hpp"

#include "bracken/core/logger.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>

namespace bracken {
namespace {

const char* source_token(BindingSource s) {
    switch (s) {
    case BindingSource::Key: return "key";
    case BindingSource::MouseButton: return "mouse";
    case BindingSource::ControllerButton: return "pad";
    }
    return "key";
}

bool source_from_token(std::string_view t, BindingSource& out) {
    if (t == "key") {
        out = BindingSource::Key;
        return true;
    }
    if (t == "mouse") {
        out = BindingSource::MouseButton;
        return true;
    }
    if (t == "pad") {
        out = BindingSource::ControllerButton;
        return true;
    }
    return false;
}

} // namespace

ActionId ActionMap::add(std::string_view name) {
    if (const ActionId existing = find(name); existing.valid()) {
        return existing;
    }
    if (actions_.size() >= ActionId::k_invalid) {
        BRACKEN_LOG_ERROR("input", "ActionMap::add('{}'): too many actions", name);
        return {};
    }
    actions_.push_back(Action{std::string(name), {}});
    held_.push_back(0);
    prev_held_.push_back(0);
    return ActionId{static_cast<uint16_t>(actions_.size() - 1)};
}

ActionId ActionMap::find(std::string_view name) const {
    for (std::size_t i = 0; i < actions_.size(); ++i) {
        if (actions_[i].name == name) {
            return ActionId{static_cast<uint16_t>(i)};
        }
    }
    return {};
}

std::string_view ActionMap::name_of(ActionId id) const {
    return id.value < actions_.size() ? std::string_view(actions_[id.value].name)
                                      : std::string_view{};
}

void ActionMap::bind(ActionId id, Binding binding) {
    if (id.value >= actions_.size()) {
        return;
    }
    auto& list = actions_[id.value].bindings;
    if (std::find(list.begin(), list.end(), binding) == list.end()) {
        list.push_back(binding);
    }
}

void ActionMap::unbind(ActionId id, Binding binding) {
    if (id.value >= actions_.size()) {
        return;
    }
    auto& list = actions_[id.value].bindings;
    list.erase(std::remove(list.begin(), list.end(), binding), list.end());
}

void ActionMap::clear_bindings(ActionId id) {
    if (id.value < actions_.size()) {
        actions_[id.value].bindings.clear();
    }
}

std::span<const Binding> ActionMap::bindings(ActionId id) const {
    if (id.value >= actions_.size()) {
        return {};
    }
    return actions_[id.value].bindings;
}

void ActionMap::rebind_exclusive(ActionId id, Binding binding) {
    if (id.value >= actions_.size()) {
        return;
    }
    // Take it away from everyone else first: two actions sharing one key is
    // the defect a rebinding screen is most likely to produce.
    for (std::size_t i = 0; i < actions_.size(); ++i) {
        if (i != id.value) {
            unbind(ActionId{static_cast<uint16_t>(i)}, binding);
        }
    }
    clear_bindings(id);
    bind(id, binding);
}

bool ActionMap::evaluate(const Binding& b, const InputSnapshot& input) const {
    switch (b.source) {
    case BindingSource::Key:
        return input.key_held(static_cast<SDL_Scancode>(b.code));
    case BindingSource::MouseButton:
        // The snapshot only carries a held flag for the left button; the right
        // button is edge-only, so a "held" query on it reports the edge. That
        // is a limitation of InputSnapshot, not of the binding, and it is
        // better to report the edge than to always answer false.
        return static_cast<MouseButton>(b.code) == MouseButton::Left ? input.mouse_left_held
                                                                     : input.mouse_right_pressed;
    case BindingSource::ControllerButton:
        if (!input.has_controller) {
            return false;
        }
        return static_cast<ControllerButton>(b.code) == ControllerButton::Primary
                   ? input.controller_primary
                   : input.controller_secondary;
    }
    return false;
}

void ActionMap::update(const InputSnapshot& input) {
    prev_held_ = held_;
    for (std::size_t i = 0; i < actions_.size(); ++i) {
        bool down = false;
        for (const Binding& b : actions_[i].bindings) {
            if (evaluate(b, input)) {
                down = true;
                break; // any binding triggers the action
            }
        }
        held_[i] = down ? 1u : 0u;
    }
}

bool ActionMap::held(ActionId id) const {
    return id.value < held_.size() && held_[id.value] != 0;
}

bool ActionMap::pressed(ActionId id) const {
    return id.value < held_.size() && held_[id.value] != 0 && prev_held_[id.value] == 0;
}

bool ActionMap::released(ActionId id) const {
    return id.value < held_.size() && held_[id.value] == 0 && prev_held_[id.value] != 0;
}

float ActionMap::axis(ActionId negative, ActionId positive) const {
    const float n = held(negative) ? 1.0f : 0.0f;
    const float p = held(positive) ? 1.0f : 0.0f;
    return p - n;
}

glm::vec2 ActionMap::vector(ActionId left, ActionId right, ActionId down, ActionId up) const {
    glm::vec2 v{axis(left, right), axis(down, up)};
    const float len = std::sqrt(v.x * v.x + v.y * v.y);
    // Normalize only when past unit length: holding one direction must stay
    // exactly 1, and diagonals must not be 1.41x faster.
    if (len > 1.0f) {
        v /= len;
    }
    return v;
}

bool ActionMap::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        BRACKEN_LOG_ERROR("input", "ActionMap::save('{}'): could not open for writing", path);
        return false;
    }
    out << "# bracken action bindings\n";
    out << "# <action>=<source>:<code>[,<source>:<code>...]\n";
    for (const Action& a : actions_) {
        out << a.name << '=';
        for (std::size_t i = 0; i < a.bindings.size(); ++i) {
            if (i > 0) {
                out << ',';
            }
            out << source_token(a.bindings[i].source) << ':' << a.bindings[i].code;
        }
        out << '\n';
    }
    return static_cast<bool>(out);
}

bool ActionMap::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return false; // absent file is not an error: defaults stand
    }

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            BRACKEN_LOG_WARN("input", "{}:{}: expected <action>=<bindings>", path, line_no);
            continue;
        }

        const std::string name = line.substr(0, eq);
        const ActionId id = find(name);
        if (!id.valid()) {
            // An action this build no longer has. Skip it and keep going —
            // one stale line must not discard the rest of a player's setup.
            BRACKEN_LOG_WARN("input", "{}:{}: unknown action '{}'; ignored", path, line_no, name);
            continue;
        }

        std::vector<Binding> parsed;
        std::istringstream fields(line.substr(eq + 1));
        std::string field;
        bool ok = true;
        while (std::getline(fields, field, ',')) {
            if (field.empty()) {
                continue;
            }
            const auto colon = field.find(':');
            if (colon == std::string::npos) {
                ok = false;
                break;
            }
            BindingSource source{};
            if (!source_from_token(field.substr(0, colon), source)) {
                ok = false;
                break;
            }
            const std::string_view digits(field.data() + colon + 1, field.size() - colon - 1);
            int code = 0;
            const auto [ptr, ec] =
                std::from_chars(digits.data(), digits.data() + digits.size(), code);
            if (ec != std::errc{} || ptr != digits.data() + digits.size() || code < 0) {
                ok = false;
                break;
            }
            parsed.push_back(Binding{source, code});
        }

        if (!ok) {
            BRACKEN_LOG_WARN("input", "{}:{}: malformed bindings for '{}'; keeping defaults", path,
                             line_no, name);
            continue;
        }
        // Only replace once the whole line parsed: a half-applied rebind would
        // leave the action partly on its defaults and partly on the file.
        actions_[id.value].bindings = std::move(parsed);
    }
    return true;
}

} // namespace bracken
