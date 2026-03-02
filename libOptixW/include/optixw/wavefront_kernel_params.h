#pragma once

#include <optixw/types.h>

namespace optixw {

struct ShadeKernelParams {
    RenderBufferDataRO renderBuffers;
    GeometryBuffers geometry;
    MaterialTextureData materialsTextures;
    LightingData lighting;
    EnvironmentMappingData environment;
};

struct CompactKernelParams {
    const RayState* rayPool;
    const uint32_t* activeIndicesIn;
    uint32_t* activeIndicesOut;
    uint32_t* counter;
    uint32_t numActive;
};

} // namespace optixw
