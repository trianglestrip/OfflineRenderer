#pragma once

#include <cuda_runtime.h>
#include <optix.h>

namespace optixw {

// Wavefront ray state
struct RayState {
    float3 origin;
    float3 direction;
    float3 throughput;  // Path throughput (RGB for now)
    float3 radiance;    // Accumulated radiance
    
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
    
    float tMin, tMax;
};

// Hit information
struct HitInfo {
    float3 position;
    float3 normal;
    float2 texCoord;
    uint32_t materialId;
    uint32_t primIndex;
};

// Material data (simplified, will reference libVLR later)
struct MaterialData {
    float3 albedo;
    float3 emission;
    float roughness;
    float metallic;
    float ior;
    uint32_t type;  // MaterialType
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
