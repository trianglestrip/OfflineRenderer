#include <optix.h>
#include <cuda_runtime.h>
#include <optixw/types.h>
#include "vector_math.cuh"
#include "launch_params.cuh"

using namespace optixw;

static constexpr float kPi = 3.14159265f;

extern "C" {
    __constant__ LaunchParams params;
}

__device__ inline float wrap01(float x) {
    return x - floorf(x);
}

__device__ inline float3 sampleEnvironment(const float3& dir) {
    if (params.environmentMap == nullptr || params.environmentMapWidth == 0 || params.environmentMapHeight == 0) {
        return params.environmentRadiance;
    }

    const float3 d = normalize(dir);
    const float u = wrap01(atan2f(d.z, d.x) * (0.5f / kPi) + 0.5f);
    const float v = acosf(fminf(1.0f, fmaxf(-1.0f, d.y))) / kPi;
    const uint32_t x = min(static_cast<uint32_t>(u * params.environmentMapWidth), params.environmentMapWidth - 1u);
    const uint32_t y = min(static_cast<uint32_t>(v * params.environmentMapHeight), params.environmentMapHeight - 1u);
    const float4 c = params.environmentMap[y * params.environmentMapWidth + x];
    return make_float3(c.x, c.y, c.z) * fmaxf(params.environmentMapScale, 0.0f);
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
        ray.pendingDirect = make_float3(0.0f, 0.0f, 0.0f);
        ray.nextOrigin = make_float3(0.0f, 0.0f, 0.0f);
        ray.nextDirection = make_float3(0.0f, 0.0f, 0.0f);
        ray.nextThroughput = make_float3(0.0f, 0.0f, 0.0f);
        ray.pixelIndex = rayIndex;
        ray.depth = 0;
        ray.stage = RayState::Trace;
        ray.terminateAfterShadow = 0;
        ray.insideMedium = 0;
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
    
    if (ray.stage == RayState::Shadow) {
        if (!hitFlag) {
            ray.radiance = ray.radiance + ray.pendingDirect;
        }
        ray.pendingDirect = make_float3(0.0f, 0.0f, 0.0f);

        if (ray.terminateAfterShadow) {
            ray.terminateAfterShadow = 0;
            ray.stage = RayState::Terminated;
            return;
        }

        // Restore next bounce path state after shadow visibility test.
        ray.origin = ray.nextOrigin;
        ray.direction = ray.nextDirection;
        ray.throughput = ray.nextThroughput;
        ray.terminateAfterShadow = 0;
        ray.tMin = 0.001f;
        ray.tMax = 1e20f;
        ray.stage = RayState::Trace;
    } else if (hitFlag) {
        ray.stage = RayState::Shade;
    } else {
        // Miss: accumulate environment radiance and terminate
        ray.radiance = ray.radiance + ray.throughput * sampleEnvironment(ray.direction);
        ray.stage = RayState::Terminated;
    }
}

// Closest hit program
extern "C" __global__ void __closesthit__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    if (idx >= params.numActive) return;
    
    const uint32_t rayIndex = params.activeIndices[idx];
    
    RayState& ray = params.rayPool[rayIndex];
    if (ray.stage == RayState::Shadow) {
        optixSetPayload_0(1);
        return;
    }

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
    float3 geometricNormal = normalize(cross(e1, e2));
    uint32_t frontFace = dot(geometricNormal, direction) < 0.0f ? 1u : 0u;
    float3 normal = frontFace ? geometricNormal : -geometricNormal;
    
    float2 uv = make_float2(0.0f, 0.0f);
    if (params.texcoords != nullptr) {
        const float2 bary = optixGetTriangleBarycentrics();
        const float b1 = bary.x;
        const float b2 = bary.y;
        const float b0 = 1.0f - b1 - b2;
        const float2 t0 = make_float2(
            params.texcoords[i0 * 2 + 0],
            params.texcoords[i0 * 2 + 1]);
        const float2 t1 = make_float2(
            params.texcoords[i1 * 2 + 0],
            params.texcoords[i1 * 2 + 1]);
        const float2 t2 = make_float2(
            params.texcoords[i2 * 2 + 0],
            params.texcoords[i2 * 2 + 1]);
        uv = make_float2(
            t0.x * b0 + t1.x * b1 + t2.x * b2,
            t0.y * b0 + t1.y * b1 + t2.y * b2);
    }

    // Store hit information
    HitInfo& hit = params.hitBuffer[rayIndex];
    hit.position = hitPos;
    hit.normal = normal;
    hit.texCoord = uv;
    hit.materialId = params.triangleMaterialIds[primIdx];
    hit.primIndex = primIdx;
    hit.frontFace = frontFace;
    
    // Set payload to indicate hit
    optixSetPayload_0(1);
}

// Miss program
extern "C" __global__ void __miss__trace() {
    optixSetPayload_0(0);
}
