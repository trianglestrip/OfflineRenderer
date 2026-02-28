#include <optix.h>
#include <optixw/types.h>

using namespace optixw;

// Launch parameters for ray tracing
struct TraceParams {
    RayState* rayPool;
    uint32_t* activeIndices;
    HitInfo* hitBuffer;
    const float* vertices;
    const uint32_t* indices;
    const uint32_t* triangleMaterialIds;
};

extern "C" {
    __constant__ TraceParams params;
}

// Raygen program for tracing
extern "C" __global__ void __raygen__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    const uint32_t rayIndex = params.activeIndices[idx];
    
    RayState& ray = params.rayPool[rayIndex];
    
    // Trace ray using OptiX
    uint32_t hitFlag = 0;
    optixTrace(
        optixGetPayloadTypeFromReservedSpaceOpaque(uint32_t),
        ray.origin,
        ray.direction,
        ray.tMin,
        ray.tMax,
        0.0f,  // rayTime
        OptixVisibilityMask(1),
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
    }
}

// Closest hit program
extern "C" __global__ void __closesthit__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
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
