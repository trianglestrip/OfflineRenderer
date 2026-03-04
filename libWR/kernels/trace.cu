#include <optix.h>
#include <cuda_runtime.h>
#include <wr/types.h>
#include "vector_math.cuh"

using namespace wr;

static constexpr float kPi = 3.14159265f;

extern "C" {
    __constant__ const LaunchParams* params;
}

__device__ inline float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 8) / 16777216.0f;
}

extern "C" __global__ void __raygen__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    if (idx >= params->numActive) return;
    
    const uint32_t rayIndex = params->activeIndices[idx];
    RayState& ray = params->rayPool[rayIndex];
    
    if (ray.stage == RayStage::Terminated) return;
    
    if (ray.stage == RayStage::Trace) {
        if (ray.depth == 0) {
            uint32_t px = rayIndex % params->width;
            uint32_t py = rayIndex / params->width;
            
            uint32_t seed = (rayIndex * 1664525u + params->sampleIndex * 1013904223u) ^ 0x9e3779b9u;
            
            float r1 = randf(seed);
            float r2 = randf(seed);
            
            float ndcX = (2.0f * (px + r1) / params->width - 1.0f) * params->camera.aspect;
            float ndcY = 1.0f - 2.0f * (py + r2) / params->height;
            
            float3 rayDir = params->camera.forward + 
                           params->camera.right * ndcX * params->camera.tanHalfFovY +
                           params->camera.up * ndcY * params->camera.tanHalfFovY;
            rayDir = normalize(rayDir);
            
            ray.origin = params->camera.position;
            ray.direction = rayDir;
            ray.throughput = make_float3(1.0f, 1.0f, 1.0f);
            ray.radiance = make_float3(0.0f, 0.0f, 0.0f);
            ray.pixelIndex = rayIndex;
            ray.depth = 0;
            ray.stage = RayStage::Trace;
            ray.seed = seed;
            ray.tMin = 0.001f;
            ray.tMax = 1e20f;
        }
    }
    
    uint32_t hitFlag = 0;
    optixTrace(
        params->traversable,
        ray.origin,
        ray.direction,
        ray.tMin,
        ray.tMax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        hitFlag
    );
    
    if (hitFlag == 0) {
        ray.radiance = ray.radiance + ray.throughput * params->environmentRadiance;
        params->accumBuffer[ray.pixelIndex] = params->accumBuffer[ray.pixelIndex] + ray.radiance;
        ray.stage = RayStage::Terminated;
    } else {
        ray.stage = RayStage::Shade;
    }
}

extern "C" __global__ void __closesthit__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    if (idx >= params->numActive) return;
    
    const uint32_t rayIndex = params->activeIndices[idx];
    RayState& ray = params->rayPool[rayIndex];
    
    if (ray.stage == RayStage::Shadow) {
        optixSetPayload_0(1);
        return;
    }
    
    float t = optixGetRayTmax();
    float3 origin = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float3 hitPos = origin + t * direction;
    
    uint32_t primIdx = optixGetPrimitiveIndex();
    
    uint32_t i0 = params->geometry.indices[primIdx * 3 + 0];
    uint32_t i1 = params->geometry.indices[primIdx * 3 + 1];
    uint32_t i2 = params->geometry.indices[primIdx * 3 + 2];
    
    float3 v0 = make_float3(
        params->geometry.vertices[i0 * 3 + 0],
        params->geometry.vertices[i0 * 3 + 1],
        params->geometry.vertices[i0 * 3 + 2]
    );
    float3 v1 = make_float3(
        params->geometry.vertices[i1 * 3 + 0],
        params->geometry.vertices[i1 * 3 + 1],
        params->geometry.vertices[i1 * 3 + 2]
    );
    float3 v2 = make_float3(
        params->geometry.vertices[i2 * 3 + 0],
        params->geometry.vertices[i2 * 3 + 1],
        params->geometry.vertices[i2 * 3 + 2]
    );
    
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 geometricNormal = normalize(cross(e1, e2));
    float3 normal = dot(geometricNormal, direction) < 0.0f ? geometricNormal : -geometricNormal;
    
    HitInfo& hit = params->hitBuffer[rayIndex];
    hit.position = hitPos;
    hit.normal = normal;
    hit.materialId = params->geometry.triangleMaterialIds[primIdx];
    hit.primIndex = primIdx;
    
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__trace() {
    optixSetPayload_0(0);
}
