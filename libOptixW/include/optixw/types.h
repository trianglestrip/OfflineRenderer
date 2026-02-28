#pragma once

#include <cuda_runtime.h>
#include <optix.h>

namespace optixw {

// Wavefront ray state
struct RayState {
    float3 origin;
    float3 direction;
    float3 throughput;  // Path throughput
    float3 radiance;    // Accumulated radiance
    float3 pendingDirect;    // Direct light contribution awaiting visibility test
    float3 nextOrigin;       // Next-bounce path state stored while tracing shadow ray
    float3 nextDirection;
    float3 nextThroughput;
    
    uint32_t pixelIndex;
    uint32_t depth;
    uint32_t materialId;
    uint32_t seed;      // Random seed
    
    enum Stage : uint32_t {
        GeneratePrimary = 0,
        Trace = 1,
        Shade = 2,
        Shadow = 3,
        Terminated = 4
    };
    uint32_t stage;
    uint32_t terminateAfterShadow;
    uint32_t insideMedium;
    
    float tMin, tMax;
};

// Hit information
struct HitInfo {
    float3 position;
    float3 normal;
    float2 texCoord;
    uint32_t materialId;
    uint32_t primIndex;
    uint32_t frontFace;
};

// Material payload mapped from libVLR surface material families.
struct MaterialData {
    float3 baseColor;
    float3 emission;
    float3 specularColor;
    float3 eta;
    float3 k;

    float roughness;
    float anisotropy;
    float rotation;

    float metallic;
    float iorExt;
    float iorInt;
    float specularF0;
    float glossiness;
    float occlusion;
    float emitterScale;

    float3 emitterDirection;
    uint32_t type;  // MaterialType
    uint32_t subMaterialIndices[4];
    uint32_t numSubMaterials;
};

// Light data
struct PointLightData {
    float3 position;
    float3 intensity;
};

struct AreaLightData {
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float width;
    float height;
    float3 emission;
    uint32_t doubleSided;
};

// Camera data
struct CameraData {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float tanHalfFovY;
    float aspect;
};

// Wavefront queues
struct WavefrontQueues {
    RayState* rayPool;           // All rays
    uint32_t* activeIndices;     // Active ray indices
    uint32_t* compactIndices;    // Compacted indices
    uint32_t numActive;
    uint32_t capacity;
};

} // namespace optixw
