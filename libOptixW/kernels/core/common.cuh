// UTF-8 BOM - 确保 MSVC 正确识别中文注释
// common.cuh - GPU 层通用工具函数
// 职责：通用的设备函数（随机数、数学工具等）
#pragma once

#include <cuda_runtime.h>
#include "vector_math.cuh"

namespace optixw {
namespace device {

// 常量定义
static constexpr float kPi = 3.14159265f;
static constexpr float kRayEps = 0.001f;
static constexpr uint32_t kInvalidTextureId = 0xFFFFFFFFu;

// 随机数生成器（LCG）
__device__ inline float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 16) / 65536.0f;
}

// 数学工具函数
__device__ inline float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

__device__ inline float saturate(float x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

__device__ inline float max3(const float3& v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

// float3 工具函数
__device__ inline float3 black3() { 
    return make_float3(0.0f, 0.0f, 0.0f); 
}

__device__ inline float3 white3() { 
    return make_float3(1.0f, 1.0f, 1.0f); 
}

__device__ inline bool hasValue(const float3& v) { 
    return v.x > 0.0f || v.y > 0.0f || v.z > 0.0f; 
}

__device__ inline float3 lerp3(const float3& a, const float3& b, float t) { 
    return a * (1.0f - t) + b * t; 
}

__device__ inline float3 cdiv3(const float3& a, const float3& b) {
    return make_float3(
        a.x / fmaxf(b.x, 1e-6f),
        a.y / fmaxf(b.y, 1e-6f),
        a.z / fmaxf(b.z, 1e-6f));
}

// float2 工具函数
__device__ inline float2 make_float2_lerp(const float2& a, const float2& b, float t) {
    return make_float2(a.x * (1.0f - t) + b.x * t, a.y * (1.0f - t) + b.y * t);
}

// UV 坐标包裹
__device__ inline float wrap01(float x) {
    return x - floorf(x);
}

} // namespace device
} // namespace optixw
