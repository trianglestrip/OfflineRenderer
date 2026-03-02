#pragma once

#include <optix.h>
#include <optixw/types.h>

namespace optixw {

// Launch parameters shared between host and device.
struct LaunchParams {
    // Scene data
    OptixTraversableHandle traversable;
    
    // Geometry data
    GeometryBuffers geometry;
    
    // Ray tracing buffers
    RenderBufferDataRW renderBuffers;
    
    // Camera
    CameraData camera;
    
    // Render settings
    uint32_t width;
    uint32_t height;
    uint32_t sampleIndex;
    uint32_t numActive;
    
    // Environment data
    EnvironmentMappingData environment;
};

} // namespace optixw
