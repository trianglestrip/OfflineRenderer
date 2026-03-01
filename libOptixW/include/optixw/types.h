#pragma once

#include <optixw/core/math_types.h>
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
    uint32_t baseColorTextureId;
};

struct Texture2DData {
    const float4* pixels;
    uint32_t width;
    uint32_t height;
    uint32_t isSRGB;
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

// ==================== 材质参数结构体 ====================
struct MatteMaterialParams {
    Vec3 albedo;
};

struct UE4MaterialParams {
    Vec3 baseColor;
    float occlusion;
    float roughness;
    float metallic;
};

struct SpecularReflectionParams {
    Vec3 coeff;
    Vec3 eta;
    Vec3 k;
};

struct SpecularScatteringParams {
    Vec3 coeff;
    float iorExt;
    float iorInt;
};

struct MicrofacetReflectionParams {
    Vec3 eta;
    Vec3 k;
    float roughness;
    float anisotropy = 0.0f;
    float rotation = 0.0f;
};

struct MicrofacetScatteringParams {
    Vec3 coeff;
    float iorExt;
    float iorInt;
    float roughness;
    float anisotropy = 0.0f;
    float rotation = 0.0f;
};

struct OldStyleMaterialParams {
    Vec3 diffuseColor;
    Vec3 specularColor;
    float glossiness;
};

struct DiffuseEmitterParams {
    Vec3 emittance;
    float scale = 1.0f;
};

struct DirectionalEmitterParams {
    Vec3 emittance;
    float scale;
    Vec3 direction;
};

struct PointEmitterParams {
    Vec3 intensity;
    float scale = 1.0f;
};

struct EnvironmentEmitterParams {
    Vec3 emittance;
    float scale = 1.0f;
};

struct MetalMaterialParams {
    Vec3 albedo;
    float roughness;
};

struct GlassMaterialParams {
    Vec3 albedo;
    float ior;
};

// ==================== 材质参数变体 ====================
using MaterialParams = std::variant<
    MatteMaterialParams,
    UE4MaterialParams,
    SpecularReflectionParams,
    SpecularScatteringParams,
    MicrofacetReflectionParams,
    MicrofacetScatteringParams,
    OldStyleMaterialParams,
    DiffuseEmitterParams,
    DirectionalEmitterParams,
    PointEmitterParams,
    EnvironmentEmitterParams,
    MetalMaterialParams,
    GlassMaterialParams
>;

// ==================== 其他参数结构体 ====================
struct TriangleMeshParams {
    std::span<const float> vertices;
    std::span<const float> texcoords;
    std::span<const uint32_t> indices;
    uint32_t materialId;
    
    TriangleMeshParams() : materialId(0) {}
};

struct EnvironmentMapParams {
    const float* pixels;
    uint32_t width;
    uint32_t height;
    float scale = 1.0f;
    
    EnvironmentMapParams() : pixels(nullptr), width(0), height(0), scale(1.0f) {}
};

struct TextureLoadParams {
    const char* filePath;
    bool sRGB = true;
    
    TextureLoadParams() : filePath(nullptr), sRGB(true) {}
};

struct RenderConfig {
    uint32_t spp;
    bool enableDenoiser;
    float denoiserBlend;
    bool enableTiling;
    uint32_t tileWidth;
    uint32_t tileHeight;
    
    RenderConfig() 
        : spp(1), enableDenoiser(false), denoiserBlend(0.0f),
          enableTiling(false), tileWidth(0), tileHeight(0) {}
};

// ==================== 降噪器参数结构体 ====================
struct DenoiserSetupParams {
    uint32_t width;
    uint32_t height;
    bool useAlbedo = true;
    bool useNormal = true;
    
    DenoiserSetupParams() : width(0), height(0), useAlbedo(true), useNormal(true) {}
};

struct DenoiserParams {
    CUdeviceptr inputColor;
    CUdeviceptr inputAlbedo;
    CUdeviceptr inputNormal;
    CUdeviceptr output;
    CUstream stream = 0;
    
    DenoiserParams() : inputColor(0), inputAlbedo(0), inputNormal(0), output(0), stream(0) {}
};

struct DenoiserTiledParams {
    CUdeviceptr inputColor;
    CUdeviceptr inputAlbedo;
    CUdeviceptr inputNormal;
    CUdeviceptr output;
    uint32_t tileWidth;
    uint32_t tileHeight;
    uint32_t overlap;
    CUstream stream = 0;
    
    DenoiserTiledParams() 
        : inputColor(0), inputAlbedo(0), inputNormal(0), output(0),
          tileWidth(0), tileHeight(0), overlap(0), stream(0) {}
};

// ==================== 纹理加载参数结构体 ====================
struct TextureLoadSingleParams {
    std::string path;
    bool decodeSRGB = true;
    
    TextureLoadSingleParams() : decodeSRGB(true) {}
    TextureLoadSingleParams(const std::string& p, bool sRGB = true) 
        : path(p), decodeSRGB(sRGB) {}
};

struct TextureLoadBatchParams {
    std::vector<std::string> paths;
    bool decodeSRGB = true;
    
    TextureLoadBatchParams() : decodeSRGB(true) {}
};

} // namespace optixw
