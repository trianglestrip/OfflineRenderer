// Shadow 队列：OptiX trace shadow rays
#include "optix_common.cuh"

struct ShadowTraceLaunchParams {
    OptixTraversableHandle traversable;
    wpt::RayState* rayPool;
    uint32_t* shadowQueue;
    uint32_t* queueCounters;
};

extern "C" {
__constant__ ShadowTraceLaunchParams shadowPlp;
}

extern "C" __global__ void __raygen__shadowTrace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    const uint32_t rayIndex = shadowPlp.shadowQueue[idx];
    
    wpt::RayState& ray = shadowPlp.rayPool[rayIndex];
    
    uint32_t visible = 1;
    
    optixTrace(
        shadowPlp.traversable,
        ray.origin,
        ray.direction,
        ray.tmin,
        ray.tmax,
        0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0, 1, 0,
        visible
    );
    
    if (visible) {
        ray.stage = wpt::Stage_Terminated;
    } else {
        ray.stage = wpt::Stage_Terminated;
    }
}

extern "C" __global__ void __miss__shadowTrace() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __anyhit__shadowTrace() {
    optixSetPayload_0(0);
    optixTerminateRay();
}
