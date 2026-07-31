#pragma once

#include <cstddef>

/**
 * HLSL cbuffer packing rules, expressed once as compile-time predicates.
 *
 * `CLAUDE.md` names "editing one side of a CPU<->GPU uniform struct pair
 * without the other" as the most common way to silently break rendering, and
 * before this header the rule that prevents it existed only as prose repeated
 * in gpu_uniforms.hpp ("every vec3 is followed by a scalar so no field
 * straddles a 16-byte register boundary", "an array would stride at 16 bytes
 * on the HLSL side", ...) plus ~225 runtime `EXPECT_EQ(offsetof(...))`s in
 * gpu_uniforms.test.cpp. Prose is not checked, and a runtime test fails in a
 * different translation unit — and a different build — from the edit that
 * broke it.
 *
 * The macros below turn both into compile errors **in the same TU as the
 * edit**: adding, removing, reordering or resizing a field in a uniform struct
 * stops the build with the field's own name in the message. They pin the CPU
 * side only; the CPU<->SPIR-V agreement is checked by uniform_layout.test.cpp,
 * which reads the compiled shaders' declared block sizes.
 *
 * Usage (see gpu_uniforms.hpp for the 37 live examples):
 *
 *     struct alignas(16) FooUniforms { glm::vec3 bar; float baz; };
 *     BRACKEN_HLSL_CBUFFER(FooUniforms, 16);
 *     BRACKEN_HLSL_FIELD(FooUniforms, bar, 0);
 *     BRACKEN_HLSL_FIELD(FooUniforms, baz, 12);
 */
namespace bracken::gpu::hlsl {

/// A cbuffer "register" — the 16-byte granule HLSL packs into.
inline constexpr std::size_t REGISTER_SIZE = 16;

/**
 * The HLSL packing rule, as a predicate: a member of 16 bytes or fewer may not
 * cross a 16-byte register boundary. (HLSL would silently bump such a member
 * to the next register; the C++ struct would not, and the two layouts diverge
 * from that field onward.) Members larger than one register — mat4, arrays,
 * nested blocks — are register-aligned by construction and are not subject to
 * the rule, so they are reported as non-straddling.
 *
 * consteval: this is a build-time assertion helper and must never be reachable
 * at runtime, where "the layout is wrong" is not an actionable condition.
 */
consteval bool straddles_register(std::size_t offset, std::size_t size) {
    if (size == 0 || size > REGISTER_SIZE) {
        return false;
    }
    return (offset / REGISTER_SIZE) != ((offset + size - 1) / REGISTER_SIZE);
}

} // namespace bracken::gpu::hlsl

/**
 * Pin a cbuffer-facing struct's total size + alignment. `expected_size` is the
 * byte size the matching HLSL cbuffer declares (verified against the compiled
 * SPIR-V by gpu/uniform_layout.test.cpp).
 */
#define BRACKEN_HLSL_CBUFFER(T, expected_size)                                                                         \
    static_assert(sizeof(T) == (expected_size),                                                                        \
                  #T " changed size: the matching HLSL cbuffer must change with it (see the struct's doc comment)");    \
    static_assert(alignof(T) == ::bracken::gpu::hlsl::REGISTER_SIZE, #T " must be 16-byte aligned (alignas(16))");      \
    static_assert(sizeof(T) % ::bracken::gpu::hlsl::REGISTER_SIZE == 0,                                                 \
                  #T " size must be a multiple of 16 — GPU uniform buffers are register-granular")

/**
 * Pin a storage-buffer record's total size. These are indexed as a
 * StructuredBuffer element rather than pushed as a cbuffer, so they are packed
 * tightly and carry no alignas(16) requirement — only the size (== the HLSL
 * struct's stride) is load-bearing.
 */
#define BRACKEN_HLSL_STRUCT(T, expected_size)                                                                          \
    static_assert(sizeof(T) == (expected_size),                                                                        \
                  #T " changed size: the matching HLSL StructuredBuffer element must change with it")

/**
 * Pin one field's byte offset, and assert it does not straddle a 16-byte
 * register boundary. Padding members are pinned too: they exist precisely to
 * hold the following field's offset, so an unpinned pad is a hole in the net.
 */
#define BRACKEN_HLSL_FIELD(T, field, expected_offset)                                                                  \
    static_assert(offsetof(T, field) == (expected_offset),                                                             \
                  #T "::" #field " moved — the matching HLSL cbuffer field must move with it");                        \
    static_assert(!::bracken::gpu::hlsl::straddles_register(offsetof(T, field), sizeof(T::field)),                      \
                  #T "::" #field " straddles a 16-byte register boundary; HLSL would push it to the next register")
