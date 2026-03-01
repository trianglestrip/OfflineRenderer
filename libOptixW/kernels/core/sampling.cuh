// UTF-8 BOM - 确保 MSVC 正确识别中文注释
// sampling.cuh - GPU 层采样函数
// 职责：各种采样算法（余弦半球采样、GGX 采样等）
#pragma once

#include <cuda_runtime.h>
#include "../vector_math.cuh"
#include "../core/common.cuh"

namespace optixw {
namespace device {

// 余弦加权半球采样
__device__ inline float3 sampleCosineHemisphere(float u1, float u2, float* pdf) {
    const float r = sqrtf(u1);
    const float phi = 2.0f * kPi * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    *pdf = z / kPi;
    return make_float3(x, y, z);
}

// 局部坐标转世界坐标
__device__ inline float3 toWorld(const float3& local, const float3& normal) {
    float3 tangent, bitangent;
    if (fabsf(normal.y) > 0.999f) {
        tangent = make_float3(1.0f, 0.0f, 0.0f);
    } else {
        tangent = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), normal));
    }
    bitangent = cross(normal, tangent);
    return local.x * tangent + local.y * bitangent + local.z * normal;
}

// GGX 半向量采样
__device__ inline float3 sampleGGXHalf(const float3& n, float alpha, uint32_t& seed, float* outPdfH) {
    const float u1 = randf(seed);
    const float u2 = randf(seed);
    
    const float theta = atanf(alpha * sqrtf(u1) / sqrtf(1.0f - u1));
    const float phi = 2.0f * kPi * u2;
    
    const float sinTheta = sinf(theta);
    const float cosTheta = cosf(theta);
    const float3 localH = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
    
    if (outPdfH) {
        const float D = (alpha * alpha) / (kPi * powf(cosTheta * cosTheta * (alpha * alpha - 1.0f) + 1.0f, 2.0f));
        *outPdfH = D * cosTheta;
    }
    
    return toWorld(localH, n);
}

} // namespace device
} // namespace optixw
