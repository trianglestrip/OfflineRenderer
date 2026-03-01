#include "optixw/optixw.h"
#include "optixw/types.h"
#include "optixw/core/task_scheduler.h"
#include "optixw/scene/geometry_manager.h"
#include "optixw/scene/material_manager.h"
#include "optixw/scene/texture_manager.h"
#include "optixw/scene/light_manager.h"
#include "scene_internal.h"
#include "utils/checks.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <windows.h>
#include <wincodec.h>

namespace optixw {

// Forward declare context impl to get OptiX context
extern OptixDeviceContext g_optixContext;

namespace {

inline float3 toFloat3(const Vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline float3 normalize3(const float3& v) {
    const float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len2 <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float invLen = 1.0f / sqrtf(len2);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

inline std::wstring utf8ToWide(const char* text) {
    if (text == nullptr || text[0] == '\0') {
        return std::wstring();
    }
    const int len = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (len <= 0) {
        throw std::runtime_error("Failed to convert UTF-8 path to wide string.");
    }
    std::wstring wide(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, wide.data(), len);
    wide.pop_back();
    return wide;
}

inline float srgbToLinear(float x) {
    x = fmaxf(0.0f, fminf(1.0f, x));
    return (x <= 0.04045f) ? (x / 12.92f) : powf((x + 0.055f) / 1.055f, 2.4f);
}

inline bool loadImageRGBA32F(const char* filePath, bool decodeSRGB, std::vector<float4>* outPixels, uint32_t* outWidth, uint32_t* outHeight) {
    if (!outPixels || !outWidth || !outHeight) {
        return false;
    }

    std::wstring path = utf8ToWide(filePath);
    if (path.empty()) {
        return false;
    }

    HRESULT hrInit = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool didInit = SUCCEEDED(hrInit);

    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;
    bool ok = false;
    UINT w = 0;
    UINT h = 0;
    std::vector<uint8_t> rgba;

    HRESULT hr = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = factory->CreateDecoderFromFilename(
        path.c_str(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnDemand,
        &decoder);
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = factory->CreateFormatConverter(&converter);
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = converter->Initialize(
        frame,
        GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone,
        nullptr,
        0.0,
        WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = converter->GetSize(&w, &h);
    if (FAILED(hr) || w == 0 || h == 0) {
        goto Cleanup;
    }

    rgba.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4u);
    hr = converter->CopyPixels(
        nullptr,
        w * 4u,
        static_cast<UINT>(rgba.size()),
        rgba.data());
    if (FAILED(hr)) {
        goto Cleanup;
    }

    outPixels->resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (size_t i = 0; i < outPixels->size(); ++i) {
        float r = rgba[i * 4 + 0] / 255.0f;
        float g = rgba[i * 4 + 1] / 255.0f;
        float b = rgba[i * 4 + 2] / 255.0f;
        float a = rgba[i * 4 + 3] / 255.0f;
        if (decodeSRGB) {
            r = srgbToLinear(r);
            g = srgbToLinear(g);
            b = srgbToLinear(b);
        }
        (*outPixels)[i] = make_float4(r, g, b, a);
    }

    *outWidth = static_cast<uint32_t>(w);
    *outHeight = static_cast<uint32_t>(h);
    ok = true;

Cleanup:
    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (didInit) CoUninitialize();
    return ok;
}

} // namespace

// Scene implementation
class Scene::Impl {
public:
    // Managers for different aspects of the scene
    std::unique_ptr<GeometryManager> geometryManager;
    std::unique_ptr<MaterialManager> materialManager;
    std::unique_ptr<TextureManager> textureManager;
    std::unique_ptr<LightManager> lightManager;
    
    bool finalized = false;
    
    // Getters for accessing internal managers
    GeometryManager* getGeometryManager() const { return geometryManager.get(); }
    MaterialManager* getMaterialManager() const { return materialManager.get(); }
    TextureManager* getTextureManager() const { return textureManager.get(); }
    LightManager* getLightManager() const { return lightManager.get(); }
    
    ~Impl() = default;
};

Scene::Scene(TaskScheduler* scheduler) : m_impl(std::make_unique<Impl>()) {
    // Initialize managers with default parameters
    m_impl->geometryManager = std::make_unique<GeometryManager>(g_optixContext);
    m_impl->materialManager = std::make_unique<MaterialManager>();
    
    // Initialize TextureManager with the provided TaskScheduler
    if (scheduler) {
        m_impl->textureManager = std::make_unique<TextureManager>(scheduler);
    } else {
        // If no TaskScheduler provided, create a default one (though this is not ideal)
        static thread_local TaskScheduler defaultScheduler;
        m_impl->textureManager = std::make_unique<TextureManager>(&defaultScheduler);
    }
    
    m_impl->lightManager = std::make_unique<LightManager>();
}

Scene::~Scene() = default;

void Scene::addTriangleMesh(
    std::span<const float> verts,
    std::span<const uint32_t> inds,
    uint32_t materialId)
{
    m_impl->geometryManager->addTriangleMesh(
        verts.data(),
        static_cast<uint32_t>(verts.size() / 3),
        inds.data(),
        static_cast<uint32_t>(inds.size() / 3),
        nullptr,  // no texcoords
        materialId
    );
}

void Scene::addTriangleMeshWithTexcoords(
    std::span<const float> verts,
    std::span<const float> texcoords,
    std::span<const uint32_t> inds,
    uint32_t materialId)
{
    if (verts.size() % 3 != 0) {
        throw std::runtime_error("Vertices must be packed as float3.");
    }
    const uint32_t vertexCount = static_cast<uint32_t>(verts.size() / 3);
    if (!texcoords.empty() && texcoords.size() != static_cast<size_t>(vertexCount) * 2u) {
        throw std::runtime_error("Texcoords must be packed as float2 per vertex.");
    }

    m_impl->geometryManager->addTriangleMesh(
        verts.data(),
        vertexCount,
        inds.data(),
        static_cast<uint32_t>(inds.size() / 3),
        texcoords.empty() ? nullptr : texcoords.data(),
        materialId
    );
}

uint32_t Scene::addMaterial(const MaterialData& material) {
    return m_impl->materialManager->addMaterial(material);
}

uint32_t Scene::addMatteMaterial(const Vec3& albedo) {
    return m_impl->materialManager->addMatteMaterial(albedo);
}

uint32_t Scene::addLambertianScatteringMaterial(const Vec3& coeff, float f0) {
    // Using matte material as fallback for now
    return m_impl->materialManager->addMatteMaterial(coeff);
}

uint32_t Scene::addSpecularReflectionMaterial(const Vec3& coeff, const Vec3& eta, const Vec3& k) {
    return m_impl->materialManager->addSpecularReflectionMaterial(coeff);
}

uint32_t Scene::addSpecularScatteringMaterial(const Vec3& coeff, float iorExt, float iorInt) {
    // Using microfacet reflection as fallback for now
    return m_impl->materialManager->addMicrofacetReflectionMaterial(coeff, 0.0f);
}

uint32_t Scene::addMicrofacetReflectionMaterial(
    const Vec3& eta, const Vec3& k, float roughness, float anisotropy, float rotation) {
    return m_impl->materialManager->addMicrofacetReflectionMaterial(eta, roughness);
}

uint32_t Scene::addMicrofacetScatteringMaterial(
    const Vec3& coeff, float iorExt, float iorInt, float roughness, float anisotropy, float rotation) {
    return m_impl->materialManager->addMicrofacetReflectionMaterial(coeff, roughness);
}

uint32_t Scene::addUE4Material(const Vec3& baseColor, float occlusion, float roughness, float metallic) {
    return m_impl->materialManager->addUE4Material(baseColor, roughness, metallic);
}

uint32_t Scene::addOldStyleMaterial(const Vec3& diffuseColor, const Vec3& specularColor, float glossiness) {
    return m_impl->materialManager->addMatteMaterial(diffuseColor);
}

uint32_t Scene::addDiffuseEmitterMaterial(const Vec3& emittance, float scale) {
    return m_impl->materialManager->addDiffuseEmitterMaterial(emittance);
}

uint32_t Scene::addDirectionalEmitterMaterial(const Vec3& emittance, float scale, const Vec3& direction) {
    return m_impl->materialManager->addDiffuseEmitterMaterial(emittance);
}

uint32_t Scene::addPointEmitterMaterial(const Vec3& intensity, float scale) {
    return m_impl->materialManager->addDiffuseEmitterMaterial(intensity);
}

uint32_t Scene::addMultiMaterial(std::span<const uint32_t> subMaterials) {
    return m_impl->materialManager->addMatteMaterial(Vec3(1.0f, 1.0f, 1.0f));
}

uint32_t Scene::addEnvironmentEmitterMaterial(const Vec3& emittance, float scale) {
    return m_impl->materialManager->addDiffuseEmitterMaterial(emittance);
}

uint32_t Scene::addLambertianMaterial(const Vec3& albedo) {
    return addMatteMaterial(albedo);
}

uint32_t Scene::addEmissiveMaterial(const Vec3& emission) {
    return addDiffuseEmitterMaterial(emission, 1.0f);
}

uint32_t Scene::addMetalMaterial(const Vec3& albedo, float roughness) {
    const Vec3 eta(0.17f, 0.35f, 1.5f);
    const Vec3 kk(3.1f, 2.7f, 1.9f);
    uint32_t matId = addMicrofacetReflectionMaterial(eta, kk, roughness, 0.0f, 0.0f);
    // Update material base color
    return matId;
}

uint32_t Scene::addGlassMaterial(const Vec3& albedo, float ior) {
    return addSpecularScatteringMaterial(albedo, 1.0f, ior);
}

uint32_t Scene::loadTexture2D(const char* filePath, bool sRGB) {
    if (!filePath || filePath[0] == '\0') {
        throw std::runtime_error("Texture path is empty.");
    }
    if (m_impl->finalized) {
        throw std::runtime_error("loadTexture2D must be called before finalize().");
    }

    return m_impl->textureManager->loadTexture2D(filePath, sRGB);
}

void Scene::setMaterialBaseColorTexture(uint32_t materialId, uint32_t textureId) {
    if (materialId >= m_impl->materialManager->getNumMaterials()) {
        throw std::runtime_error("Material id out of range.");
    }
    if (textureId >= 0) { // Assuming texture exists if id >= 0
        m_impl->materialManager->setMaterialAlbedoTexture(materialId, textureId);
    }
}

void Scene::setEnvironmentRadiance(const Vec3& radiance) {
    m_impl->lightManager->setEnvironmentRadiance(radiance);
}

void Scene::setEnvironmentMap(const char* filePath, float scale) {
    if (!filePath || filePath[0] == '\0') {
        throw std::runtime_error("Environment map path is empty.");
    }
    if (m_impl->finalized) {
        throw std::runtime_error("setEnvironmentMap must be called before finalize().");
    }

    std::vector<float4> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!loadImageRGBA32F(filePath, true, &pixels, &width, &height)) {
        throw std::runtime_error(std::string("Failed to load environment map: ") + filePath);
    }

    // Convert float4 pixels to float array
    std::vector<float> floatPixels(pixels.size() * 4);
    for (size_t i = 0; i < pixels.size(); ++i) {
        floatPixels[i * 4 + 0] = pixels[i].x;
        floatPixels[i * 4 + 1] = pixels[i].y;
        floatPixels[i * 4 + 2] = pixels[i].z;
        floatPixels[i * 4 + 3] = pixels[i].w;
    }

    m_impl->lightManager->setEnvironmentMap(floatPixels.data(), width, height, scale);
}

void Scene::addPointLight(const PointLight& light) {
    PointLightData data;
    data.position = toFloat3(light.position);
    data.intensity = toFloat3(light.intensity);
    m_impl->lightManager->addPointLight(data);
}

void Scene::addAreaLight(const AreaLight& light) {
    AreaLightData data;
    data.position = toFloat3(light.position);
    data.normal = toFloat3(light.normal);
    data.emission = toFloat3(light.emission);
    data.width = light.width;
    data.height = light.height;
    data.doubleSided = light.doubleSided ? 1 : 0;
    
    // Compute tangent and bitangent
    float3 up = fabsf(data.normal.y) < 0.9f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    data.tangent = normalize3(cross3(up, data.normal));
    data.bitangent = cross3(data.normal, data.tangent);
    
    m_impl->lightManager->addAreaLight(data);
}

void Scene::finalize() {
    if (m_impl->finalized) return;
    
    std::cout << "[Scene] Finalizing scene with new managers..." << std::endl;
    
    // Upload data to GPU using managers
    m_impl->materialManager->uploadToDevice();
    m_impl->textureManager->uploadToDevice();
    m_impl->lightManager->uploadToDevice();
    
    // Build geometry acceleration structure
    m_impl->geometryManager->buildGAS();
    
    std::cout << "[Scene] Scene finalized successfully" << std::endl;
    
    m_impl->finalized = true;
}

} // namespace optixw

// ==================== 结构体版本的方法实现 ====================
namespace optixw {

uint32_t Scene::addMaterial(const MaterialParams& params) {
    return std::visit([this](const auto& p) -> uint32_t {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, MatteMaterialParams>) {
            return addMatteMaterial(p.albedo);
        } else if constexpr (std::is_same_v<T, UE4MaterialParams>) {
            return addUE4Material(p.baseColor, p.occlusion, p.roughness, p.metallic);
        } else if constexpr (std::is_same_v<T, SpecularReflectionParams>) {
            return addSpecularReflectionMaterial(p.coeff, p.eta, p.k);
        } else if constexpr (std::is_same_v<T, SpecularScatteringParams>) {
            return addSpecularScatteringMaterial(p.coeff, p.iorExt, p.iorInt);
        } else if constexpr (std::is_same_v<T, MicrofacetReflectionParams>) {
            return addMicrofacetReflectionMaterial(p.eta, p.k, p.roughness, p.anisotropy, p.rotation);
        } else if constexpr (std::is_same_v<T, MicrofacetScatteringParams>) {
            return addMicrofacetScatteringMaterial(p.coeff, p.iorExt, p.iorInt, p.roughness, p.anisotropy, p.rotation);
        } else if constexpr (std::is_same_v<T, OldStyleMaterialParams>) {
            return addOldStyleMaterial(p.diffuseColor, p.specularColor, p.glossiness);
        } else if constexpr (std::is_same_v<T, DiffuseEmitterParams>) {
            return addDiffuseEmitterMaterial(p.emittance, p.scale);
        } else if constexpr (std::is_same_v<T, DirectionalEmitterParams>) {
            return addDirectionalEmitterMaterial(p.emittance, p.scale, p.direction);
        } else if constexpr (std::is_same_v<T, PointEmitterParams>) {
            return addPointEmitterMaterial(p.intensity, p.scale);
        } else if constexpr (std::is_same_v<T, EnvironmentEmitterParams>) {
            return addEnvironmentEmitterMaterial(p.emittance, p.scale);
        } else if constexpr (std::is_same_v<T, MetalMaterialParams>) {
            return addMetalMaterial(p.albedo, p.roughness);
        } else if constexpr (std::is_same_v<T, GlassMaterialParams>) {
            return addGlassMaterial(p.albedo, p.ior);
        } else {
            static_assert(std::false_type_v<T>, "未处理的材质参数类型");
            return 0;
        }
    }, params);
}

void Scene::addTriangleMesh(const TriangleMeshParams& params) {
    addTriangleMesh(params.vertices, params.indices, params.materialId);
}

void Scene::addTriangleMeshWithTexcoords(const TriangleMeshParams& params) {
    addTriangleMeshWithTexcoords(params.vertices, params.texcoords, params.indices, params.materialId);
}

uint32_t Scene::loadTexture2D(const TextureLoadParams& params) {
    return loadTexture2D(params.filePath, params.sRGB);
}

void Scene::setEnvironmentMap(const EnvironmentMapParams& params) {
    if (!params.pixels || params.pixels == nullptr) {
        return;
    }
    m_impl->lightManager->setEnvironmentMap(params.pixels, params.width, params.height, params.scale);
}

} // namespace optixw

// Implement accessor
namespace optixw {

OptixTraversableHandle SceneAccessor::getGasHandle(Scene* scene) {
    return scene->m_impl->geometryManager->getGASHandle();
}

CUdeviceptr SceneAccessor::getVerticesPtr(Scene* scene) {
    return scene->m_impl->geometryManager->getVerticesBuffer();
}

CUdeviceptr SceneAccessor::getIndicesPtr(Scene* scene) {
    return scene->m_impl->geometryManager->getIndicesBuffer();
}

CUdeviceptr SceneAccessor::getTexcoordsPtr(Scene* scene) {
    return scene->m_impl->geometryManager->getTexcoordsBuffer();
}

CUdeviceptr SceneAccessor::getMaterialsPtr(Scene* scene) {
    return scene->m_impl->materialManager->getDeviceBuffer();
}

CUdeviceptr SceneAccessor::getTexturesPtr(Scene* scene) {
    return scene->m_impl->textureManager->getDeviceBuffer();
}

CUdeviceptr SceneAccessor::getTriangleMaterialIdsPtr(Scene* scene) {
    return scene->m_impl->geometryManager->getMaterialIdsBuffer();
}

uint32_t SceneAccessor::getMaterialCount(Scene* scene) {
    return scene->m_impl->materialManager->getNumMaterials();
}

uint32_t SceneAccessor::getTriangleCount(Scene* scene) {
    return scene->m_impl->geometryManager->getNumTriangles();
}

uint32_t SceneAccessor::getTextureCount(Scene* scene) {
    return scene->m_impl->textureManager->getNumTextures();
}

Vec3 SceneAccessor::getEnvironmentRadiance(Scene* scene) {
    const Vec3& radiance = scene->m_impl->lightManager->getEnvironmentRadiance();
    return radiance;
}

CUdeviceptr SceneAccessor::getEnvironmentMapPtr(Scene* scene) {
    return scene->m_impl->lightManager->getEnvironmentMapBuffer();
}

uint32_t SceneAccessor::getEnvironmentMapWidth(Scene* scene) {
    return scene->m_impl->lightManager->getEnvironmentMapWidth();
}

uint32_t SceneAccessor::getEnvironmentMapHeight(Scene* scene) {
    return scene->m_impl->lightManager->getEnvironmentMapHeight();
}

float SceneAccessor::getEnvironmentMapScale(Scene* scene) {
    return scene->m_impl->lightManager->getEnvironmentMapScale();
}

} // namespace optixw
