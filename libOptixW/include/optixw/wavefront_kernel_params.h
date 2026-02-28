#pragma once

#include <optixw/types.h>

namespace optixw {

struct ShadeKernelParams {
    RayState* rayPool;
    const uint32_t* activeIndices;
    const HitInfo* hitBuffer;
    const float* vertices;
    const float* texcoords;
    const uint32_t* indices;
    const uint32_t* triangleMaterialIds;
    const MaterialData* materials;
    const Texture2DData* textures;
    float3* accumBuffer;
    uint32_t numTriangles;
    uint32_t numMaterials;
    uint32_t numTextures;
    uint32_t numActive;
    float3 environmentRadiance;
    const float4* environmentMap;
    uint32_t environmentMapWidth;
    uint32_t environmentMapHeight;
    float environmentMapScale;
};

struct CompactKernelParams {
    const RayState* rayPool;
    const uint32_t* activeIndicesIn;
    uint32_t* activeIndicesOut;
    uint32_t* counter;
    uint32_t numActive;
};

} // namespace optixw
