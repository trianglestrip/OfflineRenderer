#pragma once
#include <cuda_runtime.h>

namespace wr {
namespace internal {

// Fresnel equations for dielectric materials (glass, water, etc.)
__device__ __forceinline__ float fresnelDielectric(float cosI, float etaI, float etaT) {
    float sinT2 = etaI / etaT * etaI / etaT * (1.0f - cosI * cosI);
    if (sinT2 > 1.0f) return 1.0f;  // Total internal reflection
    
    float cosT = sqrtf(1.0f - sinT2);
    float rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    float rp = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
    return (rs * rs + rp * rp) * 0.5f;
}

// Schlick approximation (faster, less accurate)
__device__ __forceinline__ float fresnelSchlick(float cosI, float F0) {
    float x = 1.0f - cosI;
    float x2 = x * x;
    return F0 + (1.0f - F0) * x2 * x2 * x;
}

// Fresnel for conductors (metals) - full spectral version
__device__ __forceinline__ float3 fresnelConductor(float cosI, float3 eta, float3 k) {
    float cosI2 = cosI * cosI;
    float sinI2 = 1.0f - cosI2;
    
    float3 eta2 = make_float3(eta.x * eta.x, eta.y * eta.y, eta.z * eta.z);
    float3 k2 = make_float3(k.x * k.x, k.y * k.y, k.z * k.z);
    
    float3 t0 = make_float3(
        eta2.x - k2.x - sinI2,
        eta2.y - k2.y - sinI2,
        eta2.z - k2.z - sinI2
    );
    
    float3 a2plusb2 = make_float3(
        sqrtf(t0.x * t0.x + 4.0f * eta2.x * k2.x),
        sqrtf(t0.y * t0.y + 4.0f * eta2.y * k2.y),
        sqrtf(t0.z * t0.z + 4.0f * eta2.z * k2.z)
    );
    
    float3 t1 = make_float3(
        a2plusb2.x + cosI2,
        a2plusb2.y + cosI2,
        a2plusb2.z + cosI2
    );
    
    float3 a = make_float3(
        sqrtf(0.5f * (a2plusb2.x + t0.x)),
        sqrtf(0.5f * (a2plusb2.y + t0.y)),
        sqrtf(0.5f * (a2plusb2.z + t0.z))
    );
    
    float3 t2 = make_float3(2.0f * a.x * cosI, 2.0f * a.y * cosI, 2.0f * a.z * cosI);
    
    float3 Rs = make_float3(
        (t1.x - t2.x) / (t1.x + t2.x),
        (t1.y - t2.y) / (t1.y + t2.y),
        (t1.z - t2.z) / (t1.z + t2.z)
    );
    
    float sinI2_2 = sinI2 * sinI2;
    float3 t3 = make_float3(
        cosI2 * a2plusb2.x + sinI2_2,
        cosI2 * a2plusb2.y + sinI2_2,
        cosI2 * a2plusb2.z + sinI2_2
    );
    
    float3 t4 = make_float3(t2.x * sinI2, t2.y * sinI2, t2.z * sinI2);
    
    float3 Rp = make_float3(
        Rs.x * (t3.x - t4.x) / (t3.x + t4.x),
        Rs.y * (t3.y - t4.y) / (t3.y + t4.y),
        Rs.z * (t3.z - t4.z) / (t3.z + t4.z)
    );
    
    return make_float3(
        0.5f * (Rp.x + Rs.x),
        0.5f * (Rp.y + Rs.y),
        0.5f * (Rp.z + Rs.z)
    );
}

} // namespace internal
} // namespace wr
