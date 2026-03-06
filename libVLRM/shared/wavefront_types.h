#pragma once

#include "basic_types_internal.h"

namespace vlrm {
    using SampledSpectrum = RGB;

    enum class RayStage : uint32_t {
        RayGen = 0,
        Trace,
        Shade,
        Shadow,
        Terminated
    };

    enum class MaterialType : uint32_t {
        Lambertian = 0,
        Mirror,
        Glass,
        Emissive
    };

    struct MaterialData {
        MaterialType type;
        RGB albedo;
        RGB emission;
        float ior;
        float roughness;
        uint32_t padding[2];
    };

    struct RayState {
        float3 origin;
        float3 direction;
        RGB throughput;
        RGB radiance;
        uint32_t seed;
        uint32_t rngDimension;
        uint32_t pixelIndex;
        uint32_t depth;
        float tMin;
        float tMax;
        RayStage stage;
        uint32_t padding;
    };

    struct HitInfo {
        Point3D position;
        Normal3D normal;
        uint32_t materialId;
        uint32_t primitiveId;
        float2 barycentrics;
        float t;
    };

    struct WavefrontParams {
        OptixTraversableHandle traversable;
        
        uint32_t width;
        uint32_t height;
        uint32_t spp;
        uint32_t maxDepth;
        
        float3 cameraPosition;
        float3 cameraForward;
        float3 cameraRight;
        float3 cameraUp;
        float fovY;
        
        RayState* rayQueue;
        uint32_t* rayQueueSize;
        uint32_t maxRayQueueSize;
        
        RGB* accumBuffer;
        float4* outputBuffer;
        
        MaterialData* materials;
        uint32_t numMaterials;
        
        uint32_t* emissiveTriangles;
        uint32_t numEmissiveTriangles;
        
        float3* vertices;
        uint3* indices;
        uint32_t* materialIndices;
        
        bool useNEE;
        uint32_t rrStartDepth;
        
        uint32_t frameIndex;
    };

    struct GeometryData {
        float3* vertices;
        uint3* indices;
        uint32_t* materialIndices;
        uint32_t numTriangles;
    };

    struct PTReadOnlyPayload {
        float3 hitPosition;
        float3 hitNormal;
        uint32_t materialId;
        uint32_t primitiveId;
        float t;
    };

    struct ShadowPayload {
        float visibility;
    };
}
