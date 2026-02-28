#pragma once

#include <optix.h>
#include <optixw/types.h>

namespace optixw {

// Launch parameters shared between host and device.
struct LaunchParams {
    // Scene data
    OptixTraversableHandle traversable;
    const float* vertices;
    const uint32_t* indices;
    const uint32_t* triangleMaterialIds;

    // Ray pool and buffers
    RayState* rayPool;
    uint32_t* activeIndices;
    HitInfo* hitBuffer;

    // Camera
    CameraData camera;

    // Render settings
    uint32_t width;
    uint32_t height;
    uint32_t sampleIndex;
    uint32_t numActive;
    float3 environmentRadiance;
};

} // namespace optixw
