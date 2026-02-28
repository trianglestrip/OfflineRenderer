#pragma once

#include <cuda_runtime.h>
#include <cmath>

// CPU-side vector math helpers
// These are in a namespace to avoid conflicts with CUDA's built-in vector functions
namespace cpu_math {

inline float3 make_float3(float x, float y, float z) {
    float3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline float3 operator/(const float3& a, float b) {
    float inv = 1.0f / b;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

inline float3 normalize(const float3& v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

} // namespace cpu_math
