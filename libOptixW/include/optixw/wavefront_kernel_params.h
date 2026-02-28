#pragma once

#include <optixw/types.h>

namespace optixw {

struct ShadeKernelParams {
    RayState* rayPool;
    const uint32_t* activeIndices;
    const HitInfo* hitBuffer;
    const float* vertices;
    const uint32_t* indices;
    const uint32_t* triangleMaterialIds;
    const MaterialData* materials;
    float3* accumBuffer;
    uint32_t numTriangles;
    uint32_t numMaterials;
    uint32_t numActive;
};

struct CompactKernelParams {
    const RayState* rayPool;
    const uint32_t* activeIndicesIn;
    uint32_t* activeIndicesOut;
    uint32_t* counter;
    uint32_t numActive;
};

} // namespace optixw
