#pragma once

#include <cstdint>
#include <exception>
#include <utility>

namespace bracken::core::jobs {

struct JobNode; // pooled job record; defined in job_system.hpp

namespace detail {

// Out-of-line JobNode accessors (defined in job_system.cpp). All reference
// operations are generation-checked in a single CAS on JobNode::state (see
// its protocol comment), so operations through stale handles are no-ops that
// can never disturb the node's current task.
[[nodiscard]] bool node_try_add_ref(JobNode* n, std::uint32_t generation) noexcept;
void node_release_ref(JobNode* n, std::uint32_t generation) noexcept; // may recycle the node
bool node_is_done(const JobNode* n, std::uint32_t generation) noexcept;
std::exception_ptr node_error(const JobNode* n, std::uint32_t generation) noexcept;

} // namespace detail

// TaskHandle — intrusive handle to a pooled job node.
//
// Holds one generation-checked reference on the node (added on
// construction/copy, dropped on destruction). The node recycles only when it
// is done AND its matching-generation refcount reaches zero, so a held handle
// pins its task's node. A COPY taken through a handle whose task already
// completed and recycled cannot re-reference the node — the
// generation-checked CAS fails — so it becomes a *detached-done* handle that
// reports completion with no error, which is exactly what the completed task
// would report. No heap allocation, no shared_ptr.
//
// Lifetime contract: a TaskHandle must not outlive the JobSystem that issued
// it (node storage is freed in ~JobSystem). Handles remain valid and
// queryable across JobSystem::shutdown()/init() cycles.
class TaskHandle {
public:
    TaskHandle() = default;

    // Internal (JobSystem + tests): wraps a node, adding a generation-checked
    // reference. If the referenced task already completed and recycled, the
    // handle constructs detached-done (queries report completion, no error).
    // Host code never constructs these.
    TaskHandle(JobNode* node, std::uint32_t generation) noexcept : gen_(generation) {
        if (node != nullptr) {
            if (detail::node_try_add_ref(node, generation)) {
                node_ = node;
            } else {
                detached_done_ = true;
            }
        }
    }

    TaskHandle(const TaskHandle& other) noexcept
        : gen_(other.gen_), detached_done_(other.detached_done_) {
        if (other.node_ != nullptr) {
            // Cannot fail while `other` holds its reference (count ≥ 1 with a
            // matching generation), but stays defensive: a failed
            // re-reference still proves "that task completed".
            if (detail::node_try_add_ref(other.node_, other.gen_)) {
                node_ = other.node_;
            } else {
                detached_done_ = true;
            }
        }
    }

    TaskHandle(TaskHandle&& other) noexcept
        : node_(std::exchange(other.node_, nullptr)),
          gen_(other.gen_),
          detached_done_(std::exchange(other.detached_done_, false)) {}

    TaskHandle& operator=(const TaskHandle& other) noexcept {
        if (this != &other) {
            TaskHandle copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    TaskHandle& operator=(TaskHandle&& other) noexcept {
        if (this != &other) {
            if (node_ != nullptr) {
                detail::node_release_ref(node_, gen_);
            }
            node_ = std::exchange(other.node_, nullptr);
            gen_ = other.gen_;
            detached_done_ = std::exchange(other.detached_done_, false);
        }
        return *this;
    }

    ~TaskHandle() {
        if (node_ != nullptr) {
            detail::node_release_ref(node_, gen_);
        }
    }

    bool valid() const noexcept { return node_ != nullptr || detached_done_; }

    // True once the task has completed (ran, was discarded at shutdown, or
    // its node was recycled — which implies completion).
    bool is_done() const noexcept {
        return detached_done_ || (node_ != nullptr && detail::node_is_done(node_, gen_));
    }

    bool has_error() const noexcept { return error() != nullptr; }

    // Valid only once is_done() is true; while the task is still in flight
    // this returns nullptr (JobNode::error is non-atomic and may only be
    // read after acquiring `done`, which the worker release-stores last).
    // Safe because a ref-holding handle pins the node against recycling; a
    // detached-done handle reports nullptr (the error, if any, was owned by
    // the original task's referenced observers).
    std::exception_ptr error() const noexcept {
        return node_ != nullptr ? detail::node_error(node_, gen_) : nullptr;
    }

    // Internal/debug accessors; used by JobSystem and the jobs tests. Not
    // part of the host-facing surface.
    JobNode* node() const noexcept { return node_; }
    std::uint32_t generation() const noexcept { return gen_; }

private:
    JobNode* node_ = nullptr;
    std::uint32_t gen_ = 0;
    // Set when this handle refers to a task whose node already recycled: the
    // handle holds no reference but must still answer queries as "completed,
    // no error".
    bool detached_done_ = false;
};

} // namespace bracken::core::jobs
