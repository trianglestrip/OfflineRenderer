// 主路径追踪：OptiX trace primary/bounce rays
#include "optix_common.cuh"

struct TraceLaunchParams {
    OptixTraversableHandle traversable;
    wpt::RayState* rayPool;
    uint32_t* activeQueue;
    uint32_t* queueCounters;
};

extern "C" {
__constant__ TraceLaunchParams tracePlp;
}

extern "C" __global__ void __raygen__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    const uint32_t rayIndex = tracePlp.activeQueue[idx];
    
    wpt::RayState& ray = tracePlp.rayPool[rayIndex];
    
    uint32_t hitFlag = 0;
    uint32_t materialID = 0;
    
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
        hitFlag, materialID
    );
    
    if (hitFlag) {
        ray.material_id = materialID;
        ray.stage = wpt::Stage_Shade;
    } else {
        ray.radiance = ray.radiance + ray.throughput * wpt::make_rgb(0, 0, 0);
        ray.stage = wpt::Stage_Terminated;
    }
}

extern "C" __global__ void __miss__trace() {
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__trace() {
    optixSetPayload_0(1);
    optixSetPayload_1(0);
}
