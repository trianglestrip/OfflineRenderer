#include <optix.h>
#include <cuda_runtime.h>
#include <optixw/types.h>
#include "vector_math.cuh"

using namespace optixw;

// Launch parameters for ray generation
struct RayGenParams {
    RayState* rayPool;
    uint32_t* activeIndices;
    CameraData camera;
    uint32_t width;
    uint32_t height;
    uint32_t sampleIndex;
};

extern "C" {
    __constant__ RayGenParams params;
}

// Simple random number generator (LCG)
__device__ __forceinline__ float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (seed & 0x00FFFFFF) / 16777216.0f;
}

// Generate primary rays
extern "C" __global__ void __raygen__generate_primary() {
    const uint32_t idx = optixGetLaunchIndex().x;
    const uint32_t rayIndex = params.activeIndices[idx];
    
    // Pixel coordinates
    uint32_t px = rayIndex % params.width;
    uint32_t py = rayIndex / params.width;
    
    // Subpixel jitter
    uint32_t seed = (rayIndex * 1664525u + params.sampleIndex * 1013904223u) ^ 0x9e3779b9u;
    float jitterX = randf(seed) - 0.5f;
    float jitterY = randf(seed) - 0.5f;
    
    // NDC coordinates [-1, 1]
    float ndcX = (2.0f * (px + 0.5f + jitterX) / params.width - 1.0f) * params.camera.aspect;
    float ndcY = 1.0f - 2.0f * (py + 0.5f + jitterY) / params.height;
    
    // Ray direction in world space
    float3 rayDir = normalize(
        params.camera.forward * params.camera.tanHalfFovY +
        params.camera.right * ndcX * params.camera.tanHalfFovY +
        params.camera.up * ndcY * params.camera.tanHalfFovY
    );
    
    // Initialize ray state
    RayState& ray = params.rayPool[rayIndex];
    ray.origin = params.camera.position;
    ray.direction = rayDir;
    ray.throughput = make_float3(1.0f, 1.0f, 1.0f);
    ray.radiance = make_float3(0.0f, 0.0f, 0.0f);
    ray.pendingDirect = make_float3(0.0f, 0.0f, 0.0f);
    ray.nextOrigin = make_float3(0.0f, 0.0f, 0.0f);
    ray.nextDirection = make_float3(0.0f, 0.0f, 0.0f);
    ray.nextThroughput = make_float3(0.0f, 0.0f, 0.0f);
    ray.pixelIndex = rayIndex;
    ray.depth = 0;
    ray.materialId = 0;
    ray.stage = RayState::Trace;
    ray.terminateAfterShadow = 0;
    ray.insideMedium = 0;
    ray.seed = seed;
    ray.tMin = 0.001f;
    ray.tMax = 1e20f;
}
