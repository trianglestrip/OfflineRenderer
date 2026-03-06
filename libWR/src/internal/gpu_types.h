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
    Glass = 2,
    GGXReflection = 3,      // Microfacet reflection (metal/dielectric)
    GGXTransmission = 4     // Microfacet transmission (rough glass)
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
    float4 albedo;       // xyz = albedo, w unused (used when albedoTextureId == 0)
    float4 emission;     // xyz = emission, w unused
    float ior;
    uint32_t type;
    float roughness;     // Alpha parameter for GGX
    float metallic;      // 0 = dielectric, 1 = metal
    uint32_t albedoTextureId;  // 0 = use albedo, else index into textures array
    uint32_t normalTextureId;  // 0 = no normal map, else index into textures array
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
    
    // For shadow rays (NEE visibility test)
    float3 shadowContribution;  // Pending contribution from NEE
    float3 savedDirection;      // Incoming direction (restored after shadow test)
    uint32_t neeDone;           // 1 = NEE done, skip to BSDF when resuming
    
    // RNG dimension counter (to avoid screen-space correlation)
    uint32_t rngDimension;
    
    // Current medium IOR for nested dielectrics (1.0 = air/vacuum)
    // This must be initialized to 1.0f when ray is created
    float currentIOR;
};

// Hit information (GPU)
struct HitInfo {
    float3 position;
    float3 normal;           // Shading normal (interpolated from vertices or geometric)
    float3 geometricNormal;  // True geometric normal (for offset and consistency)
    float2 uv;               // Interpolated UV coordinates
    uint32_t materialId;
    uint32_t primIndex;
};

// Camera data (GPU)
struct CameraData {
    float4 position;    // xyz = position, w unused
    float4 forward;     // xyz = forward, w unused
    float4 right;       // xyz = right, w unused
    float4 up;          // xyz = up, w unused
    float tanHalfFovY;
    float aspect;
    float focalDistance;  // For DOF: 0 = infinite focus
    float lensRadius;     // For DOF: 0 = pinhole camera
};

// Geometry buffers
struct GeometryBuffers {
    const float* vertices;          // 3 floats per vertex (x, y, z)
    const float* normals;            // 3 floats per vertex (nx, ny, nz), null if no normals
    const uint32_t* indices;         // 3 indices per triangle
    const uint32_t* triangleMaterialIds;
    const float* uvs;                // 2 floats per vertex, null if no UVs
};

// Photon structure for photon mapping (must be defined before LaunchParams)
struct Photon {
    float3 position;
    float3 direction;  // Incoming direction
    float3 power;      // Photon energy
    uint32_t flags;    // Type flags
};

// Photon map parameters
struct PhotonMapParams {
    const Photon* photons;           // Photon buffer
    uint32_t numPhotons;             // Current photon count
    uint32_t maxPhotons;             // Maximum photons
    float searchRadius;              // Search radius for k-NN
    uint32_t maxPhotonsPerQuery;     // Max photons to gather
};

// Caustic photon map (separate for quality)
struct CausticPhotonMapParams {
    const Photon* photons;
    uint32_t numPhotons;
    uint32_t maxPhotons;
    float searchRadius;
    uint32_t maxPhotonsPerQuery;
};

// Flags for photon types
enum PhotonFlags : uint32_t {
    PHOTON_DIRECT = 1 << 0,
    PHOTON_INDIRECT = 1 << 1,
    PHOTON_CAUSTIC = 1 << 2,
    PHOTON_SHADOW = 1 << 3
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

    // Textures (cudaTextureObject_t array, indexed by material albedoTextureId)
    const void* textures;       // cudaTextureObject_t* on device
    uint32_t numTextures;
    
    // Light sampling for NEE
    const uint32_t* emissiveTriangles;  // List of emissive triangle indices
    const float* emissiveTriangleCDF;   // CDF for importance sampling
    uint32_t numEmissiveTriangles;
    
    CameraData camera;
    float4 environmentRadiance;  // xyz = radiance, w unused (for uniform environment)
    
    uint32_t width;
    uint32_t height;
    uint32_t sampleIndex;
    uint32_t numActive;
    
    // Rendering config
    uint32_t useNEE;           // Changed from bool for alignment
    uint32_t maxBounces;
    float rrStartDepth;
    float fireflyClamp;        // Max luminance per sample (0 = disabled)
    
    // HDR Environment Map (for IBL) - 8-byte aligned
    cudaTextureObject_t envMap;  // 0 = use uniform environmentRadiance (8 bytes)
    uint32_t envMapWidth;
    uint32_t envMapHeight;
    uint32_t _envPadding;      // Padding for alignment
    
    // Photon mapping
    PhotonMapParams photonMap;
    CausticPhotonMapParams causticMap;
    uint32_t usePhotonMapping;  // 0 = disabled, 1 = enabled
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
