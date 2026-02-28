#pragma once

#include <cuda_runtime.h>

// Vector math operators for CUDA

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, float b) {
    float inv = 1.0f / b;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
