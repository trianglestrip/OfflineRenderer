// 材质 BSDF 与 NEE/MIS 辅助函数（阶段 1：Diffuse / GGX / Dielectric / Emissive）
#pragma once
#include "wavefront_types.cuh"

namespace wpt {

// ---- Diffuse (Lambert) ----
__device__ float3_rgb evalDiffuse(float3_rgb albedo) {
    return albedo * (1.0f / 3.14159265f);
}

__device__ float pdfDiffuse(float cosTheta) {
    return fmaxf(cosTheta, 0.0f) / 3.14159265f;
}

__device__ float3 sampleDiffuse(float u1, float u2) {
    float phi = 2.0f * 3.14159265f * u1;
    float r = sqrtf(u2);
    float x = r * cosf(phi);
    float z = r * sinf(phi);
    float y = sqrtf(fmaxf(0.0f, 1.0f - u2));
    return make_float3(x, y, z);
}

// ---- GGX（简化占位：当前按 diffuse，后续实现微表面） ----
__device__ float3_rgb evalGGX(float3_rgb albedo, float roughness, float NdotH, float NdotL, float NdotV) {
    // 占位：简化为 diffuse，阶段 1 后期或阶段 2 实现真 GGX
    return albedo * (1.0f / 3.14159265f);
}

__device__ float pdfGGX(float roughness, float NdotH) {
    return 1.0f / 3.14159265f; // 占位
}

__device__ float3 sampleGGX(float u1, float u2, float roughness) {
    return sampleDiffuse(u1, u2); // 占位
}

// ---- Dielectric（简化占位：折射，后续实现） ----
__device__ float3_rgb evalDielectric(float3_rgb albedo, float ior, float cosTheta) {
    return albedo; // 占位
}

__device__ float3 sampleDielectric(float u1, float u2, float3 N, float3 wo, float ior, float& pdf) {
    pdf = 1.0f;
    return sampleDiffuse(u1, u2); // 占位
}

// ---- MIS：power heuristic ----
__device__ float powerHeuristic(float pdfA, float pdfB) {
    float a = pdfA * pdfA;
    float b = pdfB * pdfB;
    return a / fmaxf(a + b, 1e-7f);
}

// ---- 坐标系转换 ----
__device__ float3 toWorld(float3 local, float3 N) {
    float3 T = fabsf(N.x) < 0.9f ? make_float3(1, 0, 0) : make_float3(0, 1, 0);
    T = normalize(cross(T, N));
    float3 B = cross(N, T);
    return local.x * T + local.y * N + local.z * B;
}

__device__ float3 toLocal(float3 world, float3 N) {
    float3 T = fabsf(N.x) < 0.9f ? make_float3(1, 0, 0) : make_float3(0, 1, 0);
    T = normalize(cross(T, N));
    float3 B = cross(N, T);
    return make_float3(dot(world, T), dot(world, N), dot(world, B));
}

} // namespace wpt
