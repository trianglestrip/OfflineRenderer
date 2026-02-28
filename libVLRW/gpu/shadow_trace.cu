// Shadow ray tracing: anyhit program for visibility test, terminates on first hit
#include "optix_common.cuh"

struct ShadowLaunchParams {
    OptixTraversableHandle traversable;
    wpt::HitInfo* hitBuffer;
    wpt::PointLight* lights;
    uint32_t numLights;
    uint32_t* activeQueue;
    uint32_t numActive;
    uint32_t* visibilityBuffer;  // [rayIdx * numLights + lightIdx]: 1=visible, 0=occluded
};

extern "C" {
__constant__ ShadowLaunchParams shadowPlp;
}

extern "C" __global__ void __raygen__shadowTrace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    const uint32_t numLights = shadowPlp.numLights;
    if (numLights == 0) return;

    const uint32_t rayIdx = idx / numLights;
    const uint32_t lightIdx = idx % numLights;
    if (rayIdx >= shadowPlp.numActive) return;

    const uint32_t rayIndex = shadowPlp.activeQueue[rayIdx];
    const wpt::HitInfo& hit = shadowPlp.hitBuffer[rayIndex];
    const wpt::PointLight& light = shadowPlp.lights[lightIdx];

    float3 position = hit.position;
    float3 N = hit.normal;
    float3 toLight = make_float3(
        light.position.x - position.x,
        light.position.y - position.y,
        light.position.z - position.z);
    float distSq = wpt::dot(toLight, toLight);
    float dist = sqrtf(distSq);
    if (dist < 1e-6f) {
        shadowPlp.visibilityBuffer[idx] = 1u;
        return;
    }
    float3 L = make_float3(toLight.x / dist, toLight.y / dist, toLight.z / dist);

    float NdotL = wpt::dot(N, L);
    if (NdotL <= 0.0f) {
        shadowPlp.visibilityBuffer[idx] = 0u;
        return;
    }

    // Shadow ray: origin offset along normal to avoid self-intersection
    const float eps = 1e-4f;
    float3 origin = make_float3(
        position.x + eps * N.x,
        position.y + eps * N.y,
        position.z + eps * N.z);
    float3 direction = L;
    float tmin = eps;
    float tmax = dist - eps;  // Stop just before light
    if (tmax <= tmin) {
        shadowPlp.visibilityBuffer[idx] = 1u;
        return;
    }

    uint32_t payload0 = 1u;  // Assume visible; anyhit will set 0 if occluded
    uint32_t payload1 = 0u;
    uint32_t payload2 = 0u;
    optixTrace(
        shadowPlp.traversable,
        origin,
        direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0, 1, 0,
        payload0, payload1, payload2
    );

    shadowPlp.visibilityBuffer[idx] = payload0;
}

extern "C" __global__ void __miss__shadowTrace() {
    optixSetPayload_0(1u);  // Visible - no occlusion
}

extern "C" __global__ void __anyhit__shadowTrace() {
    optixSetPayload_0(0u);  // Occluded
    optixTerminateRay();
}
