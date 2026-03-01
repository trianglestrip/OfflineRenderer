// UTF-8 BOM - 确保 MSVC 正确识别中文注释
// nee.cuh - GPU 层 Next Event Estimation (NEE) 函数
// 职责：直接光照采样、阴影光线生成
#pragma once

#include <cuda_runtime.h>
#include <optixw/wavefront_kernel_params.h>
#include "../vector_math.cuh"
#include "../core/common.cuh"
#include "material.cuh"

namespace optixw {
namespace device {

// 加载顶点数据
__device__ inline float3 loadVertex(const float* vertices, uint32_t vertexIndex) {
    const float* v = vertices + vertexIndex * 3;
    return make_float3(v[0], v[1], v[2]);
}

// 采样发光三角形
__device__ bool sampleEmissiveTriangle(
    const ShadeKernelParams& params,
    const float3& shadingPoint,
    uint32_t& seed,
    float3* lightPoint,
    float3* lightNormal,
    float3* radiance,
    float* pdf)
{
    // 简化实现：随机选择一个发光三角形
    // 实际应该根据面积和辐射度加权采样
    
    if (params.numTriangles == 0) return false;
    
    // 随机选择三角形
    const uint32_t triIdx = static_cast<uint32_t>(randf(seed) * params.numTriangles) % params.numTriangles;
    
    // 加载三角形顶点
    const uint32_t* indices = params.indices + triIdx * 3;
    const float3 v0 = loadVertex(params.vertices, indices[0]);
    const float3 v1 = loadVertex(params.vertices, indices[1]);
    const float3 v2 = loadVertex(params.vertices, indices[2]);
    
    // 重心坐标采样
    const float u1 = randf(seed);
    const float u2 = randf(seed);
    const float su1 = sqrtf(u1);
    const float b0 = 1.0f - su1;
    const float b1 = su1 * (1.0f - u2);
    const float b2 = su1 * u2;
    
    *lightPoint = v0 * b0 + v1 * b1 + v2 * b2;
    
    // 计算法线
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;
    *lightNormal = normalize(cross(e1, e2));
    
    // 计算面积
    const float area = 0.5f * length(cross(e1, e2));
    
    // 获取材质
    const uint32_t matId = params.materialIds ? params.materialIds[triIdx] : 0;
    if (matId >= params.numMaterials) return false;
    
    const MaterialData& mat = params.materials[matId];
    if (!isEmitterMaterial(mat)) return false;
    
    *radiance = mat.emission;
    
    // PDF = 1 / (numEmissiveTriangles * area)
    // 简化：假设所有三角形都是发光的
    *pdf = 1.0f / (params.numTriangles * area);
    
    return true;
}

// 将阴影光线加入队列
__device__ inline void queueShadowRay(
    RayState* rayPool,
    uint32_t rayIdx,
    const float3& origin,
    const float3& direction,
    float distance,
    const float3& contribution)
{
    RayState& ray = rayPool[rayIdx];
    
    // 存储阴影光线信息
    // 注意：这里需要根据实际的 RayState 结构调整
    ray.origin = origin;
    ray.direction = direction;
    ray.tMax = distance - kRayEps;
    ray.throughput = contribution;
    ray.isShadowRay = 1;
}

// 俄罗斯轮盘赌终止
__device__ inline void maybeRussianRoulette(RayState& ray, bool* terminate) {
    if (ray.depth < 3) {
        *terminate = false;
        return;
    }
    
    const float survivalProb = fminf(0.95f, max3(ray.throughput));
    const float xi = randf(ray.seed);
    
    if (xi > survivalProb) {
        *terminate = true;
    } else {
        ray.throughput = ray.throughput / survivalProb;
        *terminate = false;
    }
}

} // namespace device
} // namespace optixw
