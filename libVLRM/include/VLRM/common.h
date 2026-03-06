#pragma once

#if !defined(VLRM_Device)
#   define VLRM_Host
#endif

// Platform defines
#if defined(VLRM_Host)
#   if defined(_WIN32) || defined(_WIN64)
#       define VLRM_Platform_Windows
#       if defined(_MSC_VER)
#           define VLRM_Platform_Windows_MSVC
#       endif
#   elif defined(__APPLE__)
#       define VLRM_Platform_macOS
#   endif
#endif

#if defined(VLRM_Platform_Windows_MSVC)
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#   if defined(VLRM_API_EXPORTS)
#       define VLRM_CPP_API __declspec(dllexport)
#   else
#       define VLRM_CPP_API __declspec(dllimport)
#   endif
#else
#   define VLRM_CPP_API
#endif

#define VLRM_M_PI 3.14159265358979323846f
#ifndef VLRM_HUGE_ENUF
#   define VLRM_HUGE_ENUF  1e+300
#endif
#define VLRM_INFINITY   ((float)(VLRM_HUGE_ENUF * VLRM_HUGE_ENUF))
#define VLRM_NAN        ((float)(VLRM_INFINITY * 0.0f))

#define VLRM_USE_DEVPRINTF

#if defined(VLRM_Host)

#include <cstdio>
#include <cstdlib>
#include <cstdarg>

#endif

#include <cstdint>
#include <cmath>
#include <cfloat>
#include <utility>

#if defined(DEBUG)
#   define ENABLE_ASSERT
#endif

// vlrmDevPrintf / vlrmprintf
#if defined(VLRM_Host)
#   if defined(VLRM_Platform_Windows_MSVC)
VLRM_CPP_API void vlrmDevPrintf(const char* fmt, ...);
#   else
#       define vlrmDevPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   endif

VLRM_CPP_API void vlrmprintf(const char* fmt, ...);
#endif

#if defined(ENABLE_ASSERT)
#   if defined(VLRM_Host)
#       define VLRMAssert(expr, fmt, ...) do { if (!(expr)) { vlrmDevPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); vlrmDevPrintf(fmt"\n", ##__VA_ARGS__); abort(); } } while (0)
#   else
#       define VLRMAssert(expr, fmt, ...) do { if (!(expr)) { vlrmDevPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); vlrmDevPrintf(fmt"\n", ##__VA_ARGS__); assert(false); } } while (0)
#   endif
#else
#   define VLRMAssert(expr, fmt, ...)
#endif

#define VLRMUnused(var) (void)var

#define VLRMAssert_ShouldNotBeCalled() VLRMAssert(false, "Should not be called!")
#define VLRMAssert_NotImplemented() VLRMAssert(false, "Not implemented yet!")

#define VLRM3DPrint(v) v.x, v.y, v.z



#if defined(VLRM_Host)
#   define CUDA_DEVICE_FUNCTION inline
#   define HOST_STATIC_CONSTEXPR static constexpr
#else
#   define CUDA_DEVICE_FUNCTION __device__ __forceinline__
#   define HOST_STATIC_CONSTEXPR
#endif

namespace vlrm {
    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr void _swap(T &a, T &b) {
        T temp = std::move(a);
        a = std::move(b);
        b = std::move(temp);
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T min(const T a, const T b) {
        return a < b ? a : b;
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T max(const T a, const T b) {
        return a > b ? a : b;
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T clamp(const T v, const T minv, const T maxv) {
        return ::vlrm::min(::vlrm::max(v, minv), maxv);
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION T floor(T x) {
        return std::floor(x);
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION T isinf(T x) {
        return std::isinf(x);
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION T isnan(T x) {
        return std::isnan(x);
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION T isfinite(T x) {
        return std::isfinite(x);
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType lerp(RealType a, RealType b, RealType t) {
        return a * (1 - t) + b * t;
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T pow2(T x) {
        if constexpr (std::is_same_v<T, int32_t>)
            VLRMAssert(x >= -46340 && x <= 46340, "pow2(): int32_t Overflow.");
        if constexpr (std::is_same_v<T, uint32_t>)
            VLRMAssert(x <= 65535, "pow2(): uint32_t Overflow.");
        return x * x;
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T pow3(T x) {
        return x * x * x;
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T pow4(T x) {
        return x * x * x * x;
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T pow5(T x) {
        return x * x * x * x * x;
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType saturate(RealType x) {
        return min(max(x, static_cast<RealType>(0)), static_cast<RealType>(1));
    }

    CUDA_DEVICE_FUNCTION uint32_t lzcnt(uint32_t x) {
#if defined(VLRM_Host)
        return _lzcnt_u32(x);
#else
        return __clz(x);
#endif
    }

    CUDA_DEVICE_FUNCTION uint32_t tzcnt(uint32_t x) {
#if defined(VLRM_Host)
        return _tzcnt_u32(x);
#else
        return __clz(__brev(x));
#endif
    }

    CUDA_DEVICE_FUNCTION int32_t popcnt(uint32_t x) {
#if defined(VLRM_Host)
        return _mm_popcnt_u32(x);
#else
        return __popc(x);
#endif
    }

#if defined(VLRM_Device) || defined(__INTELLISENSE__)
    template <>
    CUDA_DEVICE_FUNCTION float floor(float x) {
        return ::floorf(x);
    }

    template <>
    CUDA_DEVICE_FUNCTION float isinf(float x) {
        return ::isinf(x);
    }

    template <>
    CUDA_DEVICE_FUNCTION float isnan(float x) {
        return ::isnan(x);
    }

    template <>
    CUDA_DEVICE_FUNCTION float isfinite(float x) {
        return ::isfinite(x);
    }
#endif

#if defined(VLRM_Host)
    template <typename T, typename ...Args>
    inline std::array<T, sizeof...(Args)> make_array(Args &&...args) {
        return std::array<T, sizeof...(Args)>{ std::forward<Args>(args)... };
    }

    template <typename T, typename ...ArgTypes>
    std::shared_ptr<T> createShared(ArgTypes&&... args) {
        return std::shared_ptr<T>(new T(std::forward<ArgTypes>(args)...));
    }

    template <typename T, typename ...ArgTypes>
    std::unique_ptr<T> createUnique(ArgTypes&&... args) {
        return std::unique_ptr<T>(new T(std::forward<ArgTypes>(args)...));
    }

    inline std::string tolower(std::string str) {
        const auto tolower = [](unsigned char c) { return std::tolower(c); };
        std::transform(str.cbegin(), str.cend(), str.begin(), tolower);
        return str;
    }
#endif
}

// filesystem
#if defined(VLRM_Host)
namespace vlrm {
    std::filesystem::path getExecutableDirectory();
}
#endif
