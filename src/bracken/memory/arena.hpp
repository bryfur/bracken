#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <atomic>
#include <thread>

namespace bracken::memory {

class Arena {
public:
    static constexpr std::size_t k_default_chunk_size = 64 * 1024;

    Arena() = default;
    explicit Arena(std::size_t initial_chunk_size);
    Arena(void* buffer, std::size_t size);

    ~Arena();

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;

    // Precondition: `align` must be a power of two (asserted in debug builds);
    // 0 is accepted and remapped to alignof(std::max_align_t).
    // A size of 0 succeeds and returns a valid non-null pointer that must not
    // be dereferenced (and carries no alignment guarantee), so nullptr
    // uniquely signals allocation failure.
    [[nodiscard]] void* allocate(std::size_t size, std::size_t align);

    template<typename T, typename... Args> [[nodiscard]] T* create(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        if (!mem) {
            return nullptr;
        }
        T* obj = new (mem) T(std::forward<Args>(args)...);
        if constexpr (!std::is_trivially_destructible_v<T>) {
            register_destructor(obj, [](void* p) { static_cast<T*>(p)->~T(); });
        }
        return obj;
    }

    void reset();

    // --- Owner-thread enforcement (thread-safety tier "owned") ---
    // The arena binds to the first thread that calls allocate()/reset() on it
    // (or an explicit bind_owner()/rebind_owner() call); any allocate()/reset()
    // from a different thread aborts. reset() re-opens the binding, so the next
    // allocating thread becomes the new owner — this is how per-frame handoff
    // between pipeline stages is expressed.
    //
    // Compiled unconditionally. It used to be !NDEBUG-only, which made
    // sizeof(Arena) — and therefore sizeof(FramePacket), which embeds one —
    // depend on the build flags of whichever TU did the including. Two TUs that
    // disagreed silently corrupted every member after owner_.

    // Bind the calling thread as owner; aborts if already bound to another thread.
    void bind_owner() noexcept;
    // Force-bind the calling thread as owner (explicit cross-thread handoff).
    void rebind_owner() noexcept;

    std::size_t bytes_used() const noexcept;
    std::size_t bytes_capacity() const noexcept;

    bool owns_memory() const noexcept { return owns_memory_; }

private:
    struct Chunk {
        std::byte* base = nullptr;
        std::size_t size = 0;
        std::size_t used = 0;
    };

    struct Destructor {
        void* obj;
        void (*fn)(void*);
    };

    void destroy_all() noexcept;
    void free_chunks() noexcept;
    bool grow(std::size_t min_bytes);

    void register_destructor(void* obj, void (*fn)(void*));

    // Verifies the calling thread owns this arena; a first-toucher CAS makes
    // the caller the owner when the binding is open. `op` names the entry point
    // in the failure message.
    void check_owner(const char* op) noexcept;

    // Owner-thread id; a default-constructed id means "binding open".
    // Protocol: publishes only the owning thread's id — never data. Ordering is
    // relaxed everywhere: correctness of the memory the arena hands out relies
    // on the caller's own handoff synchronization (join / frame fence) that
    // orders reset() before the next owner's first allocate(); this atomic
    // exists solely so the *detector* itself is race-free (TSan-clean) when the
    // single-owner contract is violated. No release/acquire pairing partner.
    // Not padded: single-owner by contract, so never legitimately contended.
    std::atomic<std::thread::id> owner_{};

    std::vector<Chunk> chunks_;
    std::vector<Destructor> destructors_;
    std::size_t default_chunk_size_ = k_default_chunk_size;
    bool owns_memory_ = true; // false when constructed from caller-supplied buffer
    bool fixed_buffer_ = false;
};

} // namespace bracken::memory
