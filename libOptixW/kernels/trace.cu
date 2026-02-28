#include <optix.h>
#include <cuda_runtime.h>
#include <optixw/types.h>
#include "vector_math.cuh"
#include "launch_params.cuh"

using namespace optixw;

extern "C" {
    __constant__ LaunchParams params;
}

// Raygen program for tracing
extern "C" __global__ void __raygen__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    if (idx >= params.numActive) return;
    
    const uint32_t rayIndex = params.activeIndices[idx];
    RayState& ray = params.rayPool[rayIndex];
    
    // If ray is not initialized (first iteration), initialize it
    if (ray.depth == 0 && ray.stage != RayState::Trace) {
        // Pixel coordinates
        uint32_t px = rayIndex % params.width;
        uint32_t py = rayIndex / params.width;
        
        // NDC coordinates [-1, 1]
        float ndcX = (2.0f * (px + 0.5f) / params.width - 1.0f) * params.camera.aspect;
        float ndcY = 1.0f - 2.0f * (py + 0.5f) / params.height;
        
        // Ray direction
        float3 rayDir = params.camera.forward + 
                       params.camera.right * ndcX * params.camera.tanHalfFovY +
                       params.camera.up * ndcY * params.camera.tanHalfFovY;
        rayDir = normalize(rayDir);
        
        ray.origin = params.camera.position;
        ray.direction = rayDir;
        ray.throughput = make_float3(1.0f, 1.0f, 1.0f);
        ray.radiance = make_float3(0.0f, 0.0f, 0.0f);
        ray.pixelIndex = rayIndex;
        ray.depth = 0;
        ray.stage = RayState::Trace;
        ray.seed = (rayIndex * 1664525u + params.sampleIndex * 1013904223u) ^ 0x9e3779b9u;
        ray.tMin = 0.001f;
        ray.tMax = 1e20f;
    }
    
    // Trace ray using OptiX
    uint32_t hitFlag = 0;
    optixTrace(
        params.traversable,
        ray.origin,
        ray.direction,
        ray.tMin,
        ray.tMax,
        0.0f,  // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset
        1,  // SBT stride
        0,  // missSBTIndex
        hitFlag  // payload
    );
    
    if (hitFlag) {
        ray.stage = RayState::Shade;
    } else {
        // Miss: accumulate background color and terminate
        ray.radiance = ray.radiance + ray.throughput * make_float3(0.0f, 0.0f, 0.0f);
        ray.stage = RayState::Terminated;
        // Write to accumulation buffer
        params.rayPool[rayIndex].radiance = ray.radiance;
    }
}

// Closest hit program
extern "C" __global__ void __closesthit__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    if (idx >= params.numActive) return;
    
    const uint32_t rayIndex = params.activeIndices[idx];
    
    // Get hit information
    float t = optixGetRayTmax();
    float3 origin = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float3 hitPos = origin + t * direction;
    
    uint32_t primIdx = optixGetPrimitiveIndex();
    
    // Get triangle vertices
    uint32_t i0 = params.indices[primIdx * 3 + 0];
    uint32_t i1 = params.indices[primIdx * 3 + 1];
    uint32_t i2 = params.indices[primIdx * 3 + 2];
    
    float3 v0 = make_float3(
        params.vertices[i0 * 3 + 0],
        params.vertices[i0 * 3 + 1],
        params.vertices[i0 * 3 + 2]
    );
    float3 v1 = make_float3(
        params.vertices[i1 * 3 + 0],
        params.vertices[i1 * 3 + 1],
        params.vertices[i1 * 3 + 2]
    );
    float3 v2 = make_float3(
        params.vertices[i2 * 3 + 0],
        params.vertices[i2 * 3 + 1],
        params.vertices[i2 * 3 + 2]
    );
    
    // Compute geometric normal
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 normal = normalize(cross(e1, e2));
    
    // Flip normal if needed
    if (dot(normal, direction) > 0.0f) {
        normal = -normal;
    }
    
    // Store hit information
    HitInfo& hit = params.hitBuffer[rayIndex];
    hit.position = hitPos;
    hit.normal = normal;
    hit.texCoord = make_float2(0, 0);  // TODO: compute barycentric UVs
    hit.materialId = params.triangleMaterialIds[primIdx];
    hit.primIndex = primIdx;
    
    // Set payload to indicate hit
    optixSetPayload_0(1);
}

// Miss program
extern "C" __global__ void __miss__trace() {
    optixSetPayload_0(0);
}
