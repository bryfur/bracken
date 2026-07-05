#include <gtest/gtest.h>

#include "bracken/core/result.hpp"

#include <string>

using namespace bracken::core;

TEST(Result, OkPath) {
    Result<int> r = 42;
    EXPECT_TRUE(r.is_ok());
    EXPECT_FALSE(r.is_err());
    EXPECT_EQ(r.value(), 42);
}

TEST(Result, ErrPath) {
    Result<int> r = Err<std::string>{"boom"};
    EXPECT_TRUE(r.is_err());
    EXPECT_FALSE(r.is_ok());
    EXPECT_EQ(r.error(), "boom");
}

TEST(Result, ValueOrFallback) {
    Result<int> ok = 7;
    Result<int> err = Err<std::string>{"no"};
    EXPECT_EQ(ok.value_or(0), 7);
    EXPECT_EQ(err.value_or(99), 99);
}

TEST(Result, MapTransformsOk) {
    Result<int> r = 3;
    auto s = r.map([](int v) { return v * 2; });
    EXPECT_TRUE(s.is_ok());
    EXPECT_EQ(s.value(), 6);
}

TEST(Result, MapPropagatesErr) {
    Result<int> r = Err<std::string>{"fail"};
    auto s = r.map([](int v) { return v * 2; });
    EXPECT_TRUE(s.is_err());
    EXPECT_EQ(s.error(), "fail");
}

TEST(Result, AndThenChainsOk) {
    Result<int> r = 5;
    auto s = r.and_then([](int v) -> Result<int> { return v + 10; });
    EXPECT_TRUE(s.is_ok());
    EXPECT_EQ(s.value(), 15);
}

TEST(Result, AndThenShortCircuitsOnErr) {
    Result<int> r = Err<std::string>{"stop"};
    auto s = r.and_then([](int v) -> Result<int> { return v + 10; });
    EXPECT_TRUE(s.is_err());
    EXPECT_EQ(s.error(), "stop");
}

TEST(Result, AndThenCanReturnError) {
    Result<int> r = 5;
    auto s = r.and_then([](int) -> Result<int> { return Err<std::string>{"bad"}; });
    EXPECT_TRUE(s.is_err());
    EXPECT_EQ(s.error(), "bad");
}

TEST(Result, OkFactory) {
    auto r = Ok(123);
    EXPECT_TRUE(r.is_ok());
    EXPECT_EQ(r.value(), 123);
}

TEST(Result, BoolConversion) {
    Result<int> ok = 1;
    Result<int> err = Err<std::string>{"x"};
    EXPECT_TRUE(static_cast<bool>(ok));
    EXPECT_FALSE(static_cast<bool>(err));
}

// --- Result<void, E> specialization ---

TEST(ResultVoid, DefaultConstructedIsOk) {
    Result<void> r;
    EXPECT_TRUE(r.is_ok());
    EXPECT_FALSE(r.is_err());
    EXPECT_TRUE(static_cast<bool>(r));
}

TEST(ResultVoid, OkFactory) {
    Result<void> r = Ok();
    EXPECT_TRUE(r.is_ok());
}

TEST(ResultVoid, ErrPath) {
    Result<void> r = Err<std::string>{"boom"};
    EXPECT_TRUE(r.is_err());
    EXPECT_FALSE(r.is_ok());
    EXPECT_FALSE(static_cast<bool>(r));
    EXPECT_EQ(r.error(), "boom");
}

TEST(ResultVoid, ErrConvertibleErrorType) {
    Result<void> r = make_err("literal decays to std::string");
    EXPECT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), "literal decays to std::string");
}

TEST(ResultVoid, AndThenChainsOk) {
    Result<void> r;
    auto s = r.and_then([]() -> Result<int> { return 7; });
    EXPECT_TRUE(s.is_ok());
    EXPECT_EQ(s.value(), 7);
}

TEST(ResultVoid, AndThenShortCircuitsOnErr) {
    Result<void> r = Err<std::string>{"stop"};
    bool called = false;
    auto s = r.and_then([&]() -> Result<void> {
        called = true;
        return {};
    });
    EXPECT_TRUE(s.is_err());
    EXPECT_EQ(s.error(), "stop");
    EXPECT_FALSE(called);
}

TEST(ResultVoid, MoveOutError) {
    Result<void> r = Err<std::string>{"movable"};
    std::string e = std::move(r).error();
    EXPECT_EQ(e, "movable");
}

// The class carries [[nodiscard]]: discarding a Result is a compiler warning.
// (Compile-time property; exercised implicitly by every test above consuming
// its Result.)
