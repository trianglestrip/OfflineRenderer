// UTF-8 BOM - 确保 MSVC 正确识别中文注释
// material.cuh - GPU 层材质评估函数
// 职责：材质类型定义、材质参数解析、材质 BSDF 评估
#pragma once

#include <cuda_runtime.h>
#include <optixw/wavefront_kernel_params.h>
#include "../vector_math.cuh"
#include "../core/common.cuh"

namespace optixw {
namespace device {

// 材质类型枚举
enum MaterialTag : uint32_t {
    kMatte = 0,
    kLambertianScattering = 1,
    kSpecularReflection = 2,
    kSpecularScattering = 3,
    kMicrofacetReflection = 4,
    kMicrofacetScattering = 5,
    kUE4 = 6,
    kOldStyle = 7,
    kDiffuseEmitter = 8,
    kDirectionalEmitter = 9,
    kPointEmitter = 10,
    kMulti = 11,
    kEnvironmentEmitter = 12
};

// 纹理双线性采样
__device__ inline float4 sampleTexture2DBilinear(const Texture2DData& tex, float2 uv) {
    const uint32_t w = tex.width;
    const uint32_t h = tex.height;
    if (tex.pixels == nullptr || w == 0 || h == 0) {
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    float u = wrap01(uv.x);
    float v = wrap01(uv.y);
    float x = u * static_cast<float>(w - 1u);
    float y = (1.0f - v) * static_cast<float>(h - 1u);

    uint32_t x0 = static_cast<uint32_t>(floorf(x));
    uint32_t y0 = static_cast<uint32_t>(floorf(y));
    uint32_t x1 = min(x0 + 1u, w - 1u);
    uint32_t y1 = min(y0 + 1u, h - 1u);

    float tx = x - static_cast<float>(x0);
    float ty = y - static_cast<float>(y0);

    const float4 c00 = tex.pixels[y0 * w + x0];
    const float4 c10 = tex.pixels[y0 * w + x1];
    const float4 c01 = tex.pixels[y1 * w + x0];
    const float4 c11 = tex.pixels[y1 * w + x1];

    const float4 cx0 = make_float4(
        c00.x * (1.0f - tx) + c10.x * tx,
        c00.y * (1.0f - tx) + c10.y * tx,
        c00.z * (1.0f - tx) + c10.z * tx,
        c00.w * (1.0f - tx) + c10.w * tx);

    const float4 cx1 = make_float4(
        c01.x * (1.0f - tx) + c11.x * tx,
        c01.y * (1.0f - tx) + c11.y * tx,
        c01.z * (1.0f - tx) + c11.z * tx,
        c01.w * (1.0f - tx) + c11.w * tx);

    return make_float4(
        cx0.x * (1.0f - ty) + cx1.x * ty,
        cx0.y * (1.0f - ty) + cx1.y * ty,
        cx0.z * (1.0f - ty) + cx1.z * ty,
        cx0.w * (1.0f - ty) + cx1.w * ty);
}

// 环境贴图采样
__device__ inline float3 sampleEnvironment(const ShadeKernelParams& params, const float3& dir) {
    if (params.environmentMap.pixels == nullptr) {
        return params.environmentRadiance;
    }
    
    // 方向转 UV
    const float phi = atan2f(dir.z, dir.x);
    const float theta = acosf(saturate(dir.y));
    const float u = (phi + kPi) / (2.0f * kPi);
    const float v = theta / kPi;
    
    const float4 color = sampleTexture2DBilinear(params.environmentMap, make_float2(u, v));
    return make_float3(color.x, color.y, color.z) * params.environmentMapScale;
}

// 解析材质数据（处理纹理）
__device__ inline MaterialData resolveMaterial(
    const MaterialData& mat,
    const Texture2DData* textures,
    const float2& uv)
{
    MaterialData resolved = mat;
    
    // 采样反照率纹理
    if (mat.albedoTextureId != kInvalidTextureId) {
        const float4 texColor = sampleTexture2DBilinear(textures[mat.albedoTextureId], uv);
        resolved.albedo = make_float3(texColor.x, texColor.y, texColor.z);
    }
    
    return resolved;
}

// 材质粗糙度转 alpha
__device__ inline float materialAlpha(const MaterialData& mat) {
    const float roughness = fmaxf(mat.roughness, 0.001f);
    return roughness * roughness;
}

// IOR 转 F0
__device__ inline float iorToF0(float etaI, float etaT) {
    const float r = (etaI - etaT) / (etaI + etaT);
    return r * r;
}

// UE4 材质参数解析
__device__ inline void ue4Params(const MaterialData& mat, float3* kd, float3* F0, float* alpha) {
    *kd = mat.albedo * (1.0f - mat.metallic);
    *F0 = lerp3(make_float3(0.04f, 0.04f, 0.04f), mat.albedo, mat.metallic);
    *alpha = materialAlpha(mat);
}

// 旧式材质参数解析
__device__ inline void oldStyleParams(const MaterialData& mat, float3* kd, float3* F0, float* alpha) {
    *kd = mat.albedo;
    const float f0 = iorToF0(1.0f, mat.ior);
    *F0 = make_float3(f0, f0, f0);
    *alpha = materialAlpha(mat);
}

// 检查是否为发光材质
__device__ inline bool isEmitterMaterial(const MaterialData& mat) {
    return mat.type == kDiffuseEmitter || 
           mat.type == kDirectionalEmitter || 
           mat.type == kPointEmitter;
}

// 检查是否为 Delta 材质（镜面反射/折射）
__device__ inline bool isDeltaMaterial(const MaterialData& mat) {
    return mat.type == kSpecularReflection || 
           mat.type == kSpecularScattering;
}

// 发光材质辐射度
__device__ inline float3 emitterRadiance(const MaterialData& mat, const float3& lightToTargetDir) {
    if (mat.type == kDiffuseEmitter) {
        return mat.emission;
    } else if (mat.type == kDirectionalEmitter) {
        // 方向光源（未实现）
        return black3();
    } else if (mat.type == kPointEmitter) {
        // 点光源（未实现）
        return black3();
    }
    return black3();
}

} // namespace device
} // namespace optixw
