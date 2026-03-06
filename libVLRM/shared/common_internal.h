#pragma once

#if defined(__CUDA_ARCH__)
#   define VLRM_Device

#   define vlrmDevPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   define vlrmprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#   define VLRM_Host
#endif

#include "../include/VLRM/common.h"

#define VLRM_Color_System_CIE_1931_2deg  0
#define VLRM_Color_System_CIE_1964_10deg 1
#define VLRM_Color_System_CIE_2012_2deg  2
#define VLRM_Color_System_CIE_2012_10deg 3

// JP: RGB レンダリングを使用（スペクトラルレンダリングは無効）
// EN: Use RGB rendering (spectral rendering disabled)
#define VLRM_Color_System_is_based_on VLRM_Color_System_CIE_1931_2deg
static constexpr uint32_t NumSpectralSamples = 3;

#if defined(VLRM_Host)

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <filesystem>

#include <array>
#include <vector>
#include <deque>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <stack>

#include <chrono>
#include <limits>
#include <memory>
#include <functional>
#include <random>

#include <immintrin.h>

#endif

#include <algorithm>

#undef CUDA_DEVICE_FUNCTION
#include "../utils/optixu_on_cudau.h"
#include <vector_types.h>
#include <vector_functions.h>

#if defined(DEBUG)
#   define VLRM_DEBUG_SELECT(A, B) A
#else
#   define VLRM_DEBUG_SELECT(A, B) B
#endif

#define VLRM_Minimum_Machine_Alignment 16
#define VLRM_L1_Cacheline_Size 64

// For memalign, free, alignof
#if defined(VLRM_Platform_Windows_MSVC)
#   include <malloc.h>
#   define VLRM_memalign(size, alignment) _aligned_malloc(size, alignment)
#   define VLRM_freealign(ptr) _aligned_free(ptr)
#   define VLRM_alignof(T) __alignof(T)
#elif defined(VLRM_Platform_macOS)
inline void* VLRM_memalign(size_t size, size_t alignment) {
    void* ptr;
    if (posix_memalign(&ptr, alignment, size))
        ptr = nullptr;
    return ptr;
}
#   define VLRM_freealign(ptr) ::free(ptr)
#   define VLRM_alignof(T) alignof(T)
#endif
