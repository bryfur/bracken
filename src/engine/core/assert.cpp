#include "engine/core/assert.hpp"

#include <atomic>
#include <thread>

namespace mmo::engine::core {

namespace {

std::atomic<std::thread::id>& main_thread_id_slot() noexcept {
    // Default-constructed thread::id == "no thread" == not yet bound.
    static std::atomic<std::thread::id> id{};
    return id;
}

} // namespace

void bind_main_thread() noexcept {
    main_thread_id_slot().store(std::this_thread::get_id(), std::memory_order_relaxed);
}

bool is_main_thread() noexcept {
    auto& slot = main_thread_id_slot();
    std::thread::id expected{}; // unset
    const std::thread::id self = std::this_thread::get_id();
    // First caller wins the capture; afterwards the CAS fails and `expected`
    // holds the bound id to compare against.
    if (slot.compare_exchange_strong(expected, self, std::memory_order_relaxed)) {
        return true;
    }
    return expected == self;
}

} // namespace mmo::engine::core
