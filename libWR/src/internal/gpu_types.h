#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace wr {
namespace internal {

// Material types
enum MaterialType : uint32_t {
    Lambertian = 0,
    Emissive = 1,
    Glass = 2
};

// Ray stages
enum RayStage : uint32_t {
    Trace = 0,
    Shade = 1,
    Shadow = 2,
    Terminated = 3
};

// Material data (GPU)
struct MaterialData {
    float3 albedo;
    float3 emission;
    float ior;
    uint32_t type;
};

// Light sample data (for NEE)
struct LightSample {
    uint32_t triangleIndex;
    float area;
    float3 emission;
    float pdf;
};

// Ray state (GPU)
struct RayState {
    float3 origin;
    float3 direction;
    float3 throughput;
    float3 radiance;
    
    uint32_t pixelIndex;
    uint32_t depth;
    uint32_t seed;
    uint32_t stage;
    
    float tMin;
    float tMax;
    
    // For denoiser guide layers (first hit only)
    bool isFirstHit;
    
    // For MIS (track previous PDF)
    float prevPdf;
    bool prevWasDelta;
};

// Hit information (GPU)
struct HitInfo {
    float3 position;
    float3 normal;
    uint32_t materialId;
    uint32_t primIndex;
};

// Camera data (GPU)
struct CameraData {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float tanHalfFovY;
    float aspect;
};

// Geometry buffers
struct GeometryBuffers {
    const float* vertices;
    const uint32_t* indices;
    const uint32_t* triangleMaterialIds;
};

// Launch parameters for OptiX
struct LaunchParams {
    OptixTraversableHandle traversable;
    
    GeometryBuffers geometry;
    
    RayState* rayPool;
    uint32_t* activeIndices;
    HitInfo* hitBuffer;
    float3* accumBuffer;
    
    // Denoiser guide buffers (accumulated over samples)
    float3* albedoBuffer;      // First hit albedo
    float3* normalBuffer;      // First hit normal
    
    const MaterialData* materials;
    uint32_t numMaterials;
    
    // Light sampling for NEE
    const uint32_t* emissiveTriangles;  // List of emissive triangle indices
    const float* emissiveTriangleCDF;   // CDF for importance sampling
    uint32_t numEmissiveTriangles;
    
    CameraData camera;
    float3 environmentRadiance;
    
    uint32_t width;
    uint32_t height;
    uint32_t sampleIndex;
    uint32_t numActive;
    
    // Rendering config
    bool useNEE;               // Enable Next Event Estimation
    uint32_t maxBounces;
    float rrStartDepth;        // Russian Roulette start depth
};

// Compact kernel parameters
struct CompactParams {
    const RayState* rayPool;
    const uint32_t* activeIndicesIn;
    uint32_t* activeIndicesOut;
    uint32_t* counter;
    uint32_t numActive;
};

} // namespace internal
} // namespace wr
