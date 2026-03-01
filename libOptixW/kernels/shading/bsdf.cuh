// UTF-8 BOM - 确保 MSVC 正确识别中文注释
// bsdf.cuh - GPU 层 BSDF 函数
// 职责：BSDF 评估和采样（Lambertian、GGX 等）
#pragma once

#include <cuda_runtime.h>
#include <optixw/wavefront_kernel_params.h>
#include "../vector_math.cuh"
#include "../core/common.cuh"
#include "../core/sampling.cuh"

namespace optixw {
namespace device {

// BSDF 采样结果
struct BSDFSample {
    float3 wi;          // 入射方向
    float3 weight;      // f * cos / pdf
    float pdf;          // 概率密度
    uint32_t isDelta;   // 是否为 delta 分布
    uint32_t valid;     // 是否有效
};

// 反射方向
__device__ inline float3 reflectDir(const float3& wi, const float3& n) {
    return wi - 2.0f * dot(wi, n) * n;
}

// 折射方向
__device__ inline bool refractDir(const float3& wi, const float3& n, float eta, float3* wt) {
    const float cosThetaI = dot(wi, n);
    const float sin2ThetaI = fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
    const float sin2ThetaT = eta * eta * sin2ThetaI;
    
    if (sin2ThetaT >= 1.0f) return false;  // 全内反射
    
    const float cosThetaT = sqrtf(1.0f - sin2ThetaT);
    *wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

// Fresnel 项（电介质）
__device__ inline float dielectricFresnel(float cosThetaI, float etaI, float etaT) {
    cosThetaI = saturate(cosThetaI);
    
    const float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    const float sinThetaT = etaI / etaT * sinThetaI;
    
    if (sinThetaT >= 1.0f) return 1.0f;  // 全内反射
    
    const float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    const float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / 
                        ((etaT * cosThetaI) + (etaI * cosThetaT));
    const float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / 
                        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
}

// Fresnel 项（导体）
__device__ inline float3 fresnelConductor(float cosThetaI, const float3& eta, const float3& k) {
    cosThetaI = saturate(cosThetaI);
    const float cos2 = cosThetaI * cosThetaI;
    const float sin2 = 1.0f - cos2;
    const float3 eta2 = eta * eta;
    const float3 k2 = k * k;
    
    const float3 t0 = eta2 - k2 - make_float3(sin2, sin2, sin2);
    const float3 a2pb2 = sqrt(t0 * t0 + 4.0f * eta2 * k2);
    const float3 t1 = a2pb2 + make_float3(cos2, cos2, cos2);
    const float3 a = sqrt(0.5f * (a2pb2 + t0));
    const float3 t2 = 2.0f * cosThetaI * a;
    const float3 Rs = (t1 - t2) / (t1 + t2);
    
    const float3 t3 = cos2 * a2pb2 + make_float3(sin2 * sin2, sin2 * sin2, sin2 * sin2);
    const float3 t4 = t2 * sin2;
    const float3 Rp = Rs * (t3 - t4) / (t3 + t4);
    
    return 0.5f * (Rp + Rs);
}

// Schlick Fresnel 近似
__device__ inline float3 schlickFresnel(float cosTheta, const float3& F0) {
    return F0 + (white3() - F0) * pow5(1.0f - cosTheta);
}

// GGX 法线分布函数
__device__ inline float ggxD(float alpha, float NoH) {
    const float alpha2 = alpha * alpha;
    const float NoH2 = NoH * NoH;
    const float denom = NoH2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (kPi * denom * denom);
}

// GGX 几何项（单向）
__device__ inline float ggxG1(float alpha, float NoV) {
    const float alpha2 = alpha * alpha;
    const float NoV2 = NoV * NoV;
    return 2.0f * NoV / (NoV + sqrtf(alpha2 + (1.0f - alpha2) * NoV2));
}

// GGX 几何项（双向）
__device__ inline float ggxG(float alpha, float NoV, float NoL) {
    return ggxG1(alpha, NoV) * ggxG1(alpha, NoL);
}

// Lambertian 漫反射采样
__device__ inline BSDFSample sampleDiffuse(const float3& coeff, const float3& n, uint32_t& seed) {
    BSDFSample result;
    
    const float u1 = randf(seed);
    const float u2 = randf(seed);
    
    result.wi = toWorld(sampleCosineHemisphere(u1, u2, &result.pdf), n);
    result.weight = coeff;  // f * cos / pdf = (coeff / pi) * cos / (cos / pi) = coeff
    result.isDelta = 0;
    result.valid = 1;
    
    return result;
}

// GGX 镜面反射采样
__device__ inline BSDFSample sampleGGXSpecular(
    const float3& wo, const float3& n, const float3& F0, float alpha, uint32_t& seed)
{
    BSDFSample result;
    result.valid = 0;
    
    // 采样半向量
    float pdfH;
    const float3 h = sampleGGXHalf(n, alpha, seed, &pdfH);
    
    // 计算反射方向
    result.wi = reflectDir(-wo, h);
    
    const float NoL = dot(n, result.wi);
    const float NoV = dot(n, wo);
    
    if (NoL <= 0.0f || NoV <= 0.0f) return result;
    
    const float VoH = dot(wo, h);
    const float NoH = dot(n, h);
    
    // 计算 BRDF
    const float D = ggxD(alpha, NoH);
    const float G = ggxG(alpha, NoV, NoL);
    const float3 F = schlickFresnel(VoH, F0);
    
    const float3 spec = F * D * G / (4.0f * NoV * NoL);
    
    // 计算 PDF
    result.pdf = pdfH / (4.0f * VoH);
    result.weight = spec * NoL / result.pdf;
    result.isDelta = 0;
    result.valid = 1;
    
    return result;
}

} // namespace device
} // namespace optixw
