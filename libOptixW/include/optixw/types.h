#pragma once

#include <optixw/core/math_types.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <cuda.h>

// Host-only STL includes
#ifndef __CUDACC__
#include <vector>
#include <string>
#include <variant>
#include <span>
#endif

namespace optixw {

// Forward declarations (host-only)
#ifndef __CUDACC__
class Scene;
struct Camera;
#endif

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
    
    float initImportance;  // Initial path importance for Russian Roulette
    
    // MIS data for BSDF sampling hit light
    float prevBsdfPdf;
    float prevLightPdf;
    uint32_t prevDeltaSample;
    
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

// ==================== 其他参数结构体（不使用STL的，主机和设备都可用）====================
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

// ==================== 通用管理结构体 ====================

// 波前渲染队列缓冲
struct WavefrontBuffers {
    CUdeviceptr d_rayPool = 0;
    CUdeviceptr d_activeIndices = 0;
    CUdeviceptr d_compactIndices = 0;
    CUdeviceptr d_hitBuffer = 0;
    CUdeviceptr d_compactCounter = 0;

#ifndef __CUDACC__
    void free() {
        if (d_rayPool) { cudaFree(reinterpret_cast<void*>(d_rayPool)); d_rayPool = 0; }
        if (d_activeIndices) { cudaFree(reinterpret_cast<void*>(d_activeIndices)); d_activeIndices = 0; }
        if (d_compactIndices) { cudaFree(reinterpret_cast<void*>(d_compactIndices)); d_compactIndices = 0; }
        if (d_hitBuffer) { cudaFree(reinterpret_cast<void*>(d_hitBuffer)); d_hitBuffer = 0; }
        if (d_compactCounter) { cudaFree(reinterpret_cast<void*>(d_compactCounter)); d_compactCounter = 0; }
    }
    
    void allocate(size_t rayPoolSize, size_t indicesSize, size_t hitBufferSize) {
        free();
        cudaMalloc(reinterpret_cast<void**>(&d_rayPool), rayPoolSize);
        cudaMalloc(reinterpret_cast<void**>(&d_activeIndices), indicesSize);
        cudaMalloc(reinterpret_cast<void**>(&d_compactIndices), indicesSize);
        cudaMalloc(reinterpret_cast<void**>(&d_hitBuffer), hitBufferSize);
        cudaMalloc(reinterpret_cast<void**>(&d_compactCounter), sizeof(uint32_t));
    }
#endif
};

// 渲染累积缓冲
struct RenderAccumBuffers {
    CUdeviceptr d_accumBuffer = 0;
    CUdeviceptr d_albedoBuffer = 0;
    CUdeviceptr d_normalBuffer = 0;

#ifndef __CUDACC__
    void free() {
        if (d_accumBuffer) { cudaFree(reinterpret_cast<void*>(d_accumBuffer)); d_accumBuffer = 0; }
        if (d_albedoBuffer) { cudaFree(reinterpret_cast<void*>(d_albedoBuffer)); d_albedoBuffer = 0; }
        if (d_normalBuffer) { cudaFree(reinterpret_cast<void*>(d_normalBuffer)); d_normalBuffer = 0; }
    }
    
    void allocate(size_t accumSize) {
        free();
        cudaMalloc(reinterpret_cast<void**>(&d_accumBuffer), accumSize);
        cudaMalloc(reinterpret_cast<void**>(&d_albedoBuffer), accumSize);
        cudaMalloc(reinterpret_cast<void**>(&d_normalBuffer), accumSize);
    }
#endif
};

// 分块合并缓冲
struct MergeBuffers {
    CUdeviceptr d_mergeAccum = 0;
    CUdeviceptr d_mergeWeight = 0;

#ifndef __CUDACC__
    void free() {
        if (d_mergeAccum) { cudaFree(reinterpret_cast<void*>(d_mergeAccum)); d_mergeAccum = 0; }
        if (d_mergeWeight) { cudaFree(reinterpret_cast<void*>(d_mergeWeight)); d_mergeWeight = 0; }
    }
    
    void allocate(size_t accumSize, size_t weightSize) {
        free();
        cudaMalloc(reinterpret_cast<void**>(&d_mergeAccum), accumSize);
        cudaMalloc(reinterpret_cast<void**>(&d_mergeWeight), weightSize);
    }
#endif
};

// 材质纹理缓冲
struct MaterialTextureBuffers {
    CUdeviceptr d_materials = 0;
    CUdeviceptr d_texcoords = 0;
    CUdeviceptr d_textures = 0;
    uint32_t numMaterials = 0;
    uint32_t numTextures = 0;
    uint32_t numTriangles = 0;

#ifndef __CUDACC__
    void free() {
        d_materials = 0;
        d_texcoords = 0;
        d_textures = 0;
    }
#endif
};

// 环境贴图参数
struct EnvironmentMap {
    CUdeviceptr d_map = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    float scale = 1.0f;

#ifndef __CUDACC__
    void free() {
        d_map = 0;
    }
#endif
};

// Launch 参数缓冲
struct LaunchParamsBuffer {
    CUdeviceptr d_launchParams = 0;

#ifndef __CUDACC__
    void free() {
        if (d_launchParams) { cudaFree(reinterpret_cast<void*>(d_launchParams)); d_launchParams = 0; }
    }
    
    void allocate(size_t size) {
        free();
        cudaMalloc(reinterpret_cast<void**>(&d_launchParams), size);
    }
#endif
};

// 降噪器缓冲
struct DenoiserBuffers {
    CUdeviceptr d_state = 0;
    CUdeviceptr d_scratch = 0;
    CUdeviceptr d_intensity = 0;
    size_t stateSize = 0;
    size_t scratchSize = 0;

#ifndef __CUDACC__
    void free() {
        if (d_state) { cudaFree(reinterpret_cast<void*>(d_state)); d_state = 0; }
        if (d_scratch) { cudaFree(reinterpret_cast<void*>(d_scratch)); d_scratch = 0; }
        if (d_intensity) { cudaFree(reinterpret_cast<void*>(d_intensity)); d_intensity = 0; }
    }
    
    void allocate(size_t stateSize_, size_t scratchSize_) {
        free();
        stateSize = stateSize_;
        scratchSize = scratchSize_;
        cudaMalloc(reinterpret_cast<void**>(&d_state), stateSize);
        cudaMalloc(reinterpret_cast<void**>(&d_scratch), scratchSize);
        cudaMalloc(reinterpret_cast<void**>(&d_intensity), sizeof(float));
    }
#endif
};

// 降噪器主要图像缓冲
struct DenoiserImageBuffers {
    CUdeviceptr d_input = 0;
    CUdeviceptr d_output = 0;
    CUdeviceptr d_albedoInput = 0;
    CUdeviceptr d_normalInput = 0;

#ifndef __CUDACC__
    void free() {
        if (d_input) { cudaFree(reinterpret_cast<void*>(d_input)); d_input = 0; }
        if (d_output) { cudaFree(reinterpret_cast<void*>(d_output)); d_output = 0; }
        if (d_albedoInput) { cudaFree(reinterpret_cast<void*>(d_albedoInput)); d_albedoInput = 0; }
        if (d_normalInput) { cudaFree(reinterpret_cast<void*>(d_normalInput)); d_normalInput = 0; }
    }
    
    void allocate(size_t imageSize) {
        free();
        cudaMalloc(reinterpret_cast<void**>(&d_input), imageSize);
        cudaMalloc(reinterpret_cast<void**>(&d_output), imageSize);
        cudaMalloc(reinterpret_cast<void**>(&d_albedoInput), imageSize);
        cudaMalloc(reinterpret_cast<void**>(&d_normalInput), imageSize);
    }
#endif
};

// 分块降噪临时缓冲
struct DenoiserTileBuffers {
    CUdeviceptr tileBuffer = 0;
    CUdeviceptr tileInputBuffer = 0;
    CUdeviceptr tileAlbedoBuffer = 0;
    CUdeviceptr tileNormalBuffer = 0;

#ifndef __CUDACC__
    void free() {
        if (tileBuffer) { cudaFree(reinterpret_cast<void*>(tileBuffer)); tileBuffer = 0; }
        if (tileInputBuffer) { cudaFree(reinterpret_cast<void*>(tileInputBuffer)); tileInputBuffer = 0; }
        if (tileAlbedoBuffer) { cudaFree(reinterpret_cast<void*>(tileAlbedoBuffer)); tileAlbedoBuffer = 0; }
        if (tileNormalBuffer) { cudaFree(reinterpret_cast<void*>(tileNormalBuffer)); tileNormalBuffer = 0; }
    }
    
    void allocate(size_t imageSize) {
        free();
        cudaMalloc(reinterpret_cast<void**>(&tileBuffer), imageSize);
        cudaMalloc(reinterpret_cast<void**>(&tileInputBuffer), imageSize);
        cudaMalloc(reinterpret_cast<void**>(&tileAlbedoBuffer), imageSize);
        cudaMalloc(reinterpret_cast<void**>(&tileNormalBuffer), imageSize);
    }
#endif
};

// 降噪器完整状态
struct DenoiserState {
    OptixDenoiser handle = nullptr;
    DenoiserBuffers buffers;
    DenoiserImageBuffers imageBuffers;
    DenoiserTileBuffers tileBuffers;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t overlap = 0;

#ifndef __CUDACC__
    void destroy() {
        if (handle) {
            optixDenoiserDestroy(handle);
            handle = nullptr;
        }
        buffers.free();
        imageBuffers.free();
        tileBuffers.free();
        width = 0;
        height = 0;
        overlap = 0;
    }
#endif
};

// 降噪器参数（内部）
struct DenoiserParamsInternal {
    uint32_t width = 0;
    uint32_t height = 0;
    bool useAlbedo = false;
    bool useNormal = false;
    bool setup = false;
};

// 环境数据
struct EnvironmentData {
    EnvironmentMap map;
    float3 radiance;

#ifndef __CUDACC__
    EnvironmentData() : radiance{0.0f, 0.0f, 0.0f} {}
#endif
};

// 管线程序组
struct PipelineProgramGroups {
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;

#ifndef __CUDACC__
    void destroy(OptixDeviceContext context) {
        if (raygenPG) { optixProgramGroupDestroy(raygenPG); raygenPG = nullptr; }
        if (missPG) { optixProgramGroupDestroy(missPG); missPG = nullptr; }
        if (hitgroupPG) { optixProgramGroupDestroy(hitgroupPG); hitgroupPG = nullptr; }
    }
#endif
};

// 管线 SBT 记录
struct PipelineSBTRecords {
    CUdeviceptr raygenRecord = 0;
    CUdeviceptr missRecord = 0;
    CUdeviceptr hitgroupRecord = 0;

#ifndef __CUDACC__
    void free() {
        if (raygenRecord) { cudaFree(reinterpret_cast<void*>(raygenRecord)); raygenRecord = 0; }
        if (missRecord) { cudaFree(reinterpret_cast<void*>(missRecord)); missRecord = 0; }
        if (hitgroupRecord) { cudaFree(reinterpret_cast<void*>(hitgroupRecord)); hitgroupRecord = 0; }
    }
#endif
};

// 核模块
struct KernelModules {
    CUmodule shadeModule = nullptr;
    CUmodule compactModule = nullptr;
    CUmodule scaleModule = nullptr;
    CUmodule mergeModule = nullptr;

#ifndef __CUDACC__
    void unload() {
        if (shadeModule) { cuModuleUnload(shadeModule); shadeModule = nullptr; }
        if (compactModule) { cuModuleUnload(compactModule); compactModule = nullptr; }
        if (scaleModule) { cuModuleUnload(scaleModule); scaleModule = nullptr; }
        if (mergeModule) { cuModuleUnload(mergeModule); mergeModule = nullptr; }
    }
#endif
};

// 光源缓冲
struct LightBuffers {
    CUdeviceptr d_pointLights = 0;
    CUdeviceptr d_areaLights = 0;
    uint32_t numPointLights = 0;
    uint32_t numAreaLights = 0;

#ifndef __CUDACC__
    void free() {
        d_pointLights = 0;
        d_areaLights = 0;
    }
#endif
};

// 核函数
struct KernelFunctions {
    CUfunction shadeKernel = nullptr;
    CUfunction compactKernel = nullptr;
    CUfunction scaleKernel = nullptr;
    CUfunction mergeKernel = nullptr;
    CUfunction normalizeKernel = nullptr;
};

// ==================== 主机端资源管理辅助（仅主机端可用）====================
#ifndef __CUDACC__

// 用于管理 CUDA 设备指针的 RAII 包装器
struct DevicePtrWrapper {
    CUdeviceptr ptr = 0;
    
    DevicePtrWrapper() = default;
    explicit DevicePtrWrapper(CUdeviceptr p) : ptr(p) {}
    
    // 自动转换为 CUdeviceptr
    operator CUdeviceptr() const { return ptr; }
    
    // 辅助方法
    bool valid() const { return ptr != 0; }
    
    // 分配内存
    bool allocate(size_t size) {
        if (ptr != 0) free();
        return cudaMalloc(reinterpret_cast<void**>(&ptr), size) == cudaSuccess;
    }
    
    // 释放内存
    void free() {
        if (ptr != 0) {
            cudaFree(reinterpret_cast<void*>(ptr));
            ptr = 0;
        }
    }
    
    ~DevicePtrWrapper() {
        free();
    }
    
    // 禁止拷贝，允许移动
    DevicePtrWrapper(const DevicePtrWrapper&) = delete;
    DevicePtrWrapper& operator=(const DevicePtrWrapper&) = delete;
    DevicePtrWrapper(DevicePtrWrapper&& other) noexcept : ptr(other.ptr) {
        other.ptr = 0;
    }
    DevicePtrWrapper& operator=(DevicePtrWrapper&& other) noexcept {
        if (this != &other) {
            free();
            ptr = other.ptr;
            other.ptr = 0;
        }
        return *this;
    }
};

#endif // __CUDACC__

// ==================== 材质参数变体（仅主机端）====================
#ifndef __CUDACC__
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

// ==================== 其他参数结构体（仅主机端，使用STL）====================
struct TriangleMeshParams {
    std::span<const float> vertices;
    std::span<const float> texcoords;
    std::span<const uint32_t> indices;
    uint32_t materialId;
    
    TriangleMeshParams() : materialId(0) {}
};

struct RenderParams {
    Scene* scene;
    const Camera* camera;
    Vec3* outputBuffer;
    uint32_t width;
    uint32_t height;
    RenderConfig config;
    
    RenderParams()
        : scene(nullptr), camera(nullptr), outputBuffer(nullptr),
          width(0), height(0), config() {}
};

// ==================== 纹理加载参数结构体（仅主机端，使用STL）====================
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
#endif

// ==================== 设备端参数结构体（主机和设备都可用）====================

// 几何数据缓冲
struct GeometryBuffers {
    const float* vertices;
    const float* texcoords;
    const uint32_t* indices;
    const uint32_t* triangleMaterialIds;
};

// 材质纹理数据
struct MaterialTextureData {
    const MaterialData* materials;
    const Texture2DData* textures;
    uint32_t numMaterials;
    uint32_t numTextures;
    uint32_t numTriangles;
};

// 光照数据
struct LightingData {
    const PointLightData* pointLights;
    const AreaLightData* areaLights;
    uint32_t numPointLights;
    uint32_t numAreaLights;
};

// 环境映射数据
struct EnvironmentMappingData {
    const float4* environmentMap;
    uint32_t environmentMapWidth;
    uint32_t environmentMapHeight;
    float environmentMapScale;
    float3 environmentRadiance;
};

// 渲染缓冲数据（只读）
struct RenderBufferDataRO {
    RayState* rayPool;
    const uint32_t* activeIndices;
    const HitInfo* hitBuffer;
    float3* accumBuffer;
    float3* albedoBuffer;
    float3* normalBuffer;
    uint32_t numActive;
};

// 渲染缓冲数据（读写）
struct RenderBufferDataRW {
    RayState* rayPool;
    uint32_t* activeIndices;
    HitInfo* hitBuffer;
    float3* accumBuffer;
    float3* albedoBuffer;
    float3* normalBuffer;
    uint32_t numActive;
};

} // namespace optixw
