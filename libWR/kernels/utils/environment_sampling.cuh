#pragma once
#include <cuda_runtime.h>
#include "internal/gpu_types.h"
#include "vector_math.cuh"

namespace wr {
namespace internal {

// Convert direction to spherical coordinates (latitude-longitude)
__device__ __forceinline__ float2 directionToUV(float3 dir) {
    // Normalize direction
    dir = normalize(dir);
    
    // Spherical coordinates
    float phi = atan2f(dir.z, dir.x);  // [-pi, pi]
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));  // [0, pi]
    
    // Convert to UV [0, 1]
    float u = (phi + 3.14159265f) / (2.0f * 3.14159265f);  // [0, 1]
    float v = theta / 3.14159265f;  // [0, 1]
    
    return make_float2(u, v);
}

// Sample environment map
__device__ __forceinline__ float3 sampleEnvironmentMap(
    float3 direction,
    cudaTextureObject_t envMap
) {
    if (envMap == 0) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float2 uv = directionToUV(direction);
    float4 sample = tex2D<float4>(envMap, uv.x, uv.y);
    
    return make_float3(sample.x, sample.y, sample.z);
}

// Get environment radiance (HDR map or uniform)
__device__ __forceinline__ float3 getEnvironmentRadiance(
    float3 direction,
    const LaunchParams* p
) {
    if (p->envMap != 0) {
        // Use HDR environment map
        return sampleEnvironmentMap(direction, p->envMap);
    } else {
        // Use uniform environment radiance
        return make_float3(p->environmentRadiance.x, p->environmentRadiance.y, p->environmentRadiance.z);
    }
}

} // namespace internal
} // namespace wr
