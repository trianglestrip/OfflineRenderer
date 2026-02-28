// 主路径追踪：OptiX trace primary/bounce rays，写入 hitBuffer 供 shade 使用
#include "optix_common.cuh"

struct TraceLaunchParams {
    OptixTraversableHandle traversable;
    wpt::RayState* rayPool;
    uint32_t* activeQueue;
    wpt::HitInfo* hitBuffer;
    wpt::float3_rgb* accumBuffer;
    const float* vertices;
    const uint32_t* indices;
    const uint32_t* triangleMaterialIds;
    int debugMode;  // 0=default, 1=normal visualization
};

extern "C" {
__constant__ TraceLaunchParams tracePlp;
}

extern "C" __global__ void __raygen__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    const uint32_t rayIndex = tracePlp.activeQueue[idx];
    wpt::RayState& ray = tracePlp.rayPool[rayIndex];

    uint32_t hitFlag = 0;
    uint32_t pad = 0;
    optixTrace(
        tracePlp.traversable,
        ray.origin,
        ray.direction,
        ray.tmin,
        ray.tmax,
        0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        pad, hitFlag, pad
    );

    if (hitFlag) {
        // Debug mode: visualize normals directly
        if (tracePlp.debugMode == 1) {
            wpt::HitInfo& hit = tracePlp.hitBuffer[rayIndex];
            // Convert normal from [-1,1] to [0,1] for RGB visualization
            wpt::float3_rgb normalColor = wpt::make_rgb(
                hit.normal.x * 0.5f + 0.5f,
                hit.normal.y * 0.5f + 0.5f,
                hit.normal.z * 0.5f + 0.5f
            );
            if (tracePlp.accumBuffer) {
                tracePlp.accumBuffer[ray.pixel_index] = tracePlp.accumBuffer[ray.pixel_index] + normalColor;
            }
            ray.stage = wpt::Stage_Terminated;
        } else {
            ray.material_id = tracePlp.hitBuffer[rayIndex].material_id;
            ray.stage = wpt::Stage_Shade;
        }
    } else {
        wpt::float3_rgb bg = wpt::make_rgb(0.0f, 0.0f, 0.0f);  // black background
        ray.radiance = ray.radiance + ray.throughput * bg;
        if (tracePlp.accumBuffer) {
            tracePlp.accumBuffer[ray.pixel_index] = tracePlp.accumBuffer[ray.pixel_index] + ray.radiance;
        }
        ray.stage = wpt::Stage_Terminated;
    }
}

extern "C" __global__ void __miss__trace() {
    optixSetPayload_1(0);
}

extern "C" __global__ void __closesthit__trace() {
    const uint32_t launchIdx = optixGetLaunchIndex().x;
    const uint32_t rayIndex = tracePlp.activeQueue[launchIdx];
    float t = optixGetRayTmax();
    float3 O = optixGetWorldRayOrigin();
    float3 D = optixGetWorldRayDirection();
    float3 position = make_float3(O.x + t * D.x, O.y + t * D.y, O.z + t * D.z);

    uint32_t primIdx = optixGetPrimitiveIndex();
    uint32_t i0 = tracePlp.indices[primIdx * 3 + 0];
    uint32_t i1 = tracePlp.indices[primIdx * 3 + 1];
    uint32_t i2 = tracePlp.indices[primIdx * 3 + 2];
    float3 v0 = make_float3(
        tracePlp.vertices[i0 * 3 + 0], tracePlp.vertices[i0 * 3 + 1], tracePlp.vertices[i0 * 3 + 2]);
    float3 v1 = make_float3(
        tracePlp.vertices[i1 * 3 + 0], tracePlp.vertices[i1 * 3 + 1], tracePlp.vertices[i1 * 3 + 2]);
    float3 v2 = make_float3(
        tracePlp.vertices[i2 * 3 + 0], tracePlp.vertices[i2 * 3 + 1], tracePlp.vertices[i2 * 3 + 2]);
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 N = wpt::normalize(wpt::cross(e1, e2));
    if (wpt::dot(N, D) > 0.0f) N = make_float3(-N.x, -N.y, -N.z);

    tracePlp.hitBuffer[rayIndex].position = position;
    tracePlp.hitBuffer[rayIndex].normal = N;
    tracePlp.hitBuffer[rayIndex].material_id = tracePlp.triangleMaterialIds
        ? tracePlp.triangleMaterialIds[primIdx] : 0u;

    // Debug: visualize normals directly in closesthit
    if (tracePlp.debugMode == 1) {
        wpt::RayState& ray = tracePlp.rayPool[rayIndex];
        wpt::float3_rgb normalColor = wpt::make_rgb(
            N.x * 0.5f + 0.5f,
            N.y * 0.5f + 0.5f,
            N.z * 0.5f + 0.5f
        );
        if (tracePlp.accumBuffer) {
            tracePlp.accumBuffer[ray.pixel_index] = normalColor;
        }
    }

    optixSetPayload_1(1);
}
