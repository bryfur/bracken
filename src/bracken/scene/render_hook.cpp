#include "bracken/scene/render_hook.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/logger.hpp"

#include <algorithm>
#include <utility>

namespace bracken::scene {

namespace {
constexpr const char* LOG_TAG = "render";
} // namespace

const char* render_hook_slot_to_string(RenderHookSlot slot) {
    switch (slot) {
        case RenderHookSlot::PreForward:
            return "PreForward";
        case RenderHookSlot::AfterOpaque:
            return "AfterOpaque";
        case RenderHookSlot::BeforeTaa:
            return "BeforeTaa";
        case RenderHookSlot::AfterComposite:
            return "AfterComposite";
        case RenderHookSlot::AfterUi:
            return "AfterUi";
    }
    return "Unknown";
}

RenderHookHandle RenderHookRegistry::add(RenderHookSlot slot, std::string name, int order, RenderHookFn fn) {
    // Registration is main-thread tier; the sim job overlaps the previous
    // frame's recording, so mutating this list from on_simulate would race
    // dispatch.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    // Host-boundary precondition, live in release: mutating from inside a hook
    // reallocates the list the dispatcher is walking.
    BRACKEN_CONTRACT(!is_dispatching(), "RenderHookRegistry::add called from inside a render hook");
    if (!fn) {
        BRACKEN_LOG_ERROR(LOG_TAG, "RenderHookRegistry::add('{}'): callback is empty", name);
        return RenderHookHandle::invalid();
    }
    const size_t slot_index = index_of(slot);
    BRACKEN_CONTRACT(slot_index < k_render_hook_slot_count, "RenderHookSlot out of range");

    auto entry = std::make_unique<Entry>();
    entry->name = std::move(name);
    // Intern before the entry can ever be handed to the GPU timer pool, which
    // keeps the pointer past this registration's life (Entry::stable_name).
    entry->stable_name = name_pool_.insert(entry->name).first->c_str();
    entry->order = order;
    entry->sequence = next_sequence_++;
    entry->slot = slot;
    entry->fn = std::move(fn);
    Entry* raw = entry.get();

    // Reuse the lowest free slot so the index space stays dense; the
    // generation on that slot was already bumped when it was freed.
    uint32_t index = static_cast<uint32_t>(slots_.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(slots_.size()); ++i) {
        if (slots_[i].entry == nullptr) {
            index = i;
            break;
        }
    }
    if (index == static_cast<uint32_t>(slots_.size())) {
        slots_.push_back(Slot{});
    }
    slots_[index].entry = std::move(entry);
    ++live_;

    // Ordered insert. upper_bound on `order` alone puts a new entry AFTER every
    // entry with the same order, which is exactly the documented tie-break
    // (registration sequence) — and it is stable without a re-sort.
    std::vector<Entry*>& list = order_[slot_index];
    const auto pos = std::upper_bound(list.begin(), list.end(), order,
                                      [](int value, const Entry* e) { return value < e->order; });
    list.insert(pos, raw);

    return RenderHookHandle{index, slots_[index].generation};
}

bool RenderHookRegistry::remove(RenderHookHandle handle) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    BRACKEN_CONTRACT(!is_dispatching(), "RenderHookRegistry::remove called from inside a render hook");
    if (!handle.is_valid() || handle.index >= slots_.size()) {
        return false;
    }
    Slot& slot = slots_[handle.index];
    if (slot.entry == nullptr || slot.generation != handle.generation) {
        return false; // stale copy of a handle whose registration is gone
    }

    std::vector<Entry*>& list = order_[index_of(slot.entry->slot)];
    const auto it = std::find(list.begin(), list.end(), slot.entry.get());
    BRACKEN_ASSERT(it != list.end(), "render hook entry missing from its slot's order list");
    if (it != list.end()) {
        list.erase(it); // erase preserves the relative order of the rest
    }
    slot.entry.reset();
    slot.generation += 1; // a retained copy of the handle now validates false
    --live_;
    return true;
}

void RenderHookRegistry::clear() {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    BRACKEN_CONTRACT(!is_dispatching(), "RenderHookRegistry::clear called from inside a render hook");
    for (auto& list : order_) {
        list.clear();
    }
    for (Slot& slot : slots_) {
        if (slot.entry != nullptr) {
            slot.entry.reset();
            slot.generation += 1;
        }
    }
    live_ = 0;
}

} // namespace bracken::scene
