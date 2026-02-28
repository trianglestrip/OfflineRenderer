#include "optixw/optixw.h"
#include "optixw/types.h"
#include "scene_internal.h"
#include "checks.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

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

} // namespace

// Scene implementation
class Scene::Impl {
public:
    // Geometry data
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> triangleMaterialIds;
    
    // Materials
    std::vector<MaterialData> materials;
    
    // Lights
    std::vector<PointLightData> pointLights;
    std::vector<AreaLightData> areaLights;
    
    // OptiX acceleration structure
    OptixTraversableHandle gasHandle = 0;
    CUdeviceptr d_gasOutputBuffer = 0;
    
    // Device buffers
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_materials = 0;
    CUdeviceptr d_triangleMaterialIds = 0;
    
    bool finalized = false;
    
    OptixTraversableHandle getGasHandle() const { return gasHandle; }
    CUdeviceptr getVerticesPtr() const { return d_vertices; }
    CUdeviceptr getIndicesPtr() const { return d_indices; }
    CUdeviceptr getMaterialsPtr() const { return d_materials; }
    CUdeviceptr getTriangleMaterialIdsPtr() const { return d_triangleMaterialIds; }
    
    ~Impl() {
        if (d_gasOutputBuffer) cudaFree((void*)d_gasOutputBuffer);
        if (d_vertices) cudaFree((void*)d_vertices);
        if (d_indices) cudaFree((void*)d_indices);
        if (d_materials) cudaFree((void*)d_materials);
        if (d_triangleMaterialIds) cudaFree((void*)d_triangleMaterialIds);
    }
};

Scene::Scene() : m_impl(std::make_unique<Impl>()) {}
Scene::~Scene() = default;

void Scene::addTriangleMesh(
    std::span<const float> verts,
    std::span<const uint32_t> inds,
    uint32_t materialId)
{
    uint32_t baseVertex = m_impl->vertices.size() / 3;
    
    m_impl->vertices.insert(m_impl->vertices.end(), verts.begin(), verts.end());
    
    for (uint32_t idx : inds) {
        m_impl->indices.push_back(baseVertex + idx);
    }
    
    uint32_t numTriangles = inds.size() / 3;
    for (uint32_t i = 0; i < numTriangles; ++i) {
        m_impl->triangleMaterialIds.push_back(materialId);
    }
}

uint32_t Scene::addMaterial(const MaterialData& material) {
    m_impl->materials.push_back(material);
    return m_impl->materials.size() - 1;
}

static MaterialData createBaseMaterial() {
    MaterialData mat;
    mat.baseColor = make_float3(1, 1, 1);
    mat.emission = make_float3(0, 0, 0);
    mat.specularColor = make_float3(0.04f, 0.04f, 0.04f);
    mat.eta = make_float3(1.5f, 1.5f, 1.5f);
    mat.k = make_float3(0.0f, 0.0f, 0.0f);
    mat.roughness = 0.3f;
    mat.anisotropy = 0.0f;
    mat.rotation = 0.0f;
    mat.metallic = 0.0f;
    mat.iorExt = 1.0f;
    mat.iorInt = 1.5f;
    mat.specularF0 = 0.04f;
    mat.glossiness = 0.5f;
    mat.occlusion = 1.0f;
    mat.emitterScale = 1.0f;
    mat.emitterDirection = make_float3(0, 0, 1);
    mat.type = static_cast<uint32_t>(MaterialType::Matte);
    mat.subMaterialIndices[0] = 0;
    mat.subMaterialIndices[1] = 0;
    mat.subMaterialIndices[2] = 0;
    mat.subMaterialIndices[3] = 0;
    mat.numSubMaterials = 0;
    return mat;
}

uint32_t Scene::addMatteMaterial(const Vec3& albedo) {
    MaterialData mat = createBaseMaterial();
    mat.baseColor = toFloat3(albedo);
    mat.roughness = 1.0f;
    mat.type = static_cast<uint32_t>(MaterialType::Matte);
    return addMaterial(mat);
}

uint32_t Scene::addLambertianScatteringMaterial(const Vec3& coeff, float f0) {
    MaterialData mat = createBaseMaterial();
    mat.baseColor = toFloat3(coeff);
    mat.specularF0 = f0;
    mat.roughness = 1.0f;
    mat.type = static_cast<uint32_t>(MaterialType::LambertianScattering);
    return addMaterial(mat);
}

uint32_t Scene::addSpecularReflectionMaterial(const Vec3& coeff, const Vec3& eta, const Vec3& k) {
    MaterialData mat = createBaseMaterial();
    mat.baseColor = toFloat3(coeff);
    mat.eta = toFloat3(eta);
    mat.k = toFloat3(k);
    mat.roughness = 0.0f;
    mat.type = static_cast<uint32_t>(MaterialType::SpecularReflection);
    return addMaterial(mat);
}

uint32_t Scene::addSpecularScatteringMaterial(const Vec3& coeff, float iorExt, float iorInt) {
    MaterialData mat = createBaseMaterial();
    mat.baseColor = toFloat3(coeff);
    mat.iorExt = iorExt;
    mat.iorInt = iorInt;
    mat.roughness = 0.0f;
    mat.type = static_cast<uint32_t>(MaterialType::SpecularScattering);
    return addMaterial(mat);
}

uint32_t Scene::addMicrofacetReflectionMaterial(
    const Vec3& eta, const Vec3& k, float roughness, float anisotropy, float rotation) {
    MaterialData mat = createBaseMaterial();
    mat.eta = toFloat3(eta);
    mat.k = toFloat3(k);
    mat.roughness = roughness;
    mat.anisotropy = anisotropy;
    mat.rotation = rotation;
    mat.type = static_cast<uint32_t>(MaterialType::MicrofacetReflection);
    return addMaterial(mat);
}

uint32_t Scene::addMicrofacetScatteringMaterial(
    const Vec3& coeff, float iorExt, float iorInt, float roughness, float anisotropy, float rotation) {
    MaterialData mat = createBaseMaterial();
    mat.baseColor = toFloat3(coeff);
    mat.iorExt = iorExt;
    mat.iorInt = iorInt;
    mat.roughness = roughness;
    mat.anisotropy = anisotropy;
    mat.rotation = rotation;
    mat.type = static_cast<uint32_t>(MaterialType::MicrofacetScattering);
    return addMaterial(mat);
}

uint32_t Scene::addUE4Material(const Vec3& baseColor, float occlusion, float roughness, float metallic) {
    MaterialData mat = createBaseMaterial();
    mat.baseColor = toFloat3(baseColor);
    mat.occlusion = occlusion;
    mat.roughness = roughness;
    mat.metallic = metallic;
    mat.type = static_cast<uint32_t>(MaterialType::UE4);
    return addMaterial(mat);
}

uint32_t Scene::addOldStyleMaterial(const Vec3& diffuseColor, const Vec3& specularColor, float glossiness) {
    MaterialData mat = createBaseMaterial();
    mat.baseColor = toFloat3(diffuseColor);
    mat.specularColor = toFloat3(specularColor);
    mat.glossiness = glossiness;
    mat.type = static_cast<uint32_t>(MaterialType::OldStyle);
    return addMaterial(mat);
}

uint32_t Scene::addDiffuseEmitterMaterial(const Vec3& emittance, float scale) {
    MaterialData mat = createBaseMaterial();
    mat.emission = toFloat3(emittance);
    mat.emitterScale = scale;
    mat.type = static_cast<uint32_t>(MaterialType::DiffuseEmitter);
    return addMaterial(mat);
}

uint32_t Scene::addDirectionalEmitterMaterial(const Vec3& emittance, float scale, const Vec3& direction) {
    MaterialData mat = createBaseMaterial();
    mat.emission = toFloat3(emittance);
    mat.emitterScale = scale;
    mat.emitterDirection = normalize3(toFloat3(direction));
    mat.type = static_cast<uint32_t>(MaterialType::DirectionalEmitter);
    return addMaterial(mat);
}

uint32_t Scene::addPointEmitterMaterial(const Vec3& intensity, float scale) {
    MaterialData mat = createBaseMaterial();
    mat.emission = toFloat3(intensity);
    mat.emitterScale = scale;
    mat.type = static_cast<uint32_t>(MaterialType::PointEmitter);
    return addMaterial(mat);
}

uint32_t Scene::addMultiMaterial(std::span<const uint32_t> subMaterials) {
    MaterialData mat = createBaseMaterial();
    mat.type = static_cast<uint32_t>(MaterialType::Multi);
    uint32_t count = static_cast<uint32_t>(subMaterials.size());
    if (count > 4) {
        count = 4;
    }
    mat.numSubMaterials = count;
    for (uint32_t i = 0; i < count; ++i) {
        mat.subMaterialIndices[i] = subMaterials[i];
    }
    for (uint32_t i = count; i < 4; ++i) {
        mat.subMaterialIndices[i] = 0;
    }
    return addMaterial(mat);
}

uint32_t Scene::addEnvironmentEmitterMaterial(const Vec3& emittance, float scale) {
    MaterialData mat = createBaseMaterial();
    mat.emission = toFloat3(emittance);
    mat.emitterScale = scale;
    mat.type = static_cast<uint32_t>(MaterialType::EnvironmentEmitter);
    return addMaterial(mat);
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
    MaterialData& mat = m_impl->materials[matId];
    mat.baseColor = toFloat3(albedo);
    return matId;
}

uint32_t Scene::addGlassMaterial(const Vec3& albedo, float ior) {
    return addSpecularScatteringMaterial(albedo, 1.0f, ior);
}

void Scene::addPointLight(const PointLight& light) {
    PointLightData data;
    data.position = toFloat3(light.position);
    data.intensity = toFloat3(light.intensity);
    m_impl->pointLights.push_back(data);
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
    
    m_impl->areaLights.push_back(data);
}

void Scene::finalize() {
    if (m_impl->finalized) return;
    
    std::cout << "[Scene] Finalizing scene..." << std::endl;
    std::cout << "[Scene] Vertices: " << m_impl->vertices.size() / 3 << std::endl;
    std::cout << "[Scene] Triangles: " << m_impl->indices.size() / 3 << std::endl;
    std::cout << "[Scene] Materials: " << m_impl->materials.size() << std::endl;
    
    // Upload geometry to GPU
    size_t verticesSize = m_impl->vertices.size() * sizeof(float);
    size_t indicesSize = m_impl->indices.size() * sizeof(uint32_t);
    size_t materialsSize = m_impl->materials.size() * sizeof(MaterialData);
    size_t triangleMaterialIdsSize = m_impl->triangleMaterialIds.size() * sizeof(uint32_t);
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_impl->d_vertices), verticesSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_impl->d_indices), indicesSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_impl->d_materials), materialsSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_impl->d_triangleMaterialIds), triangleMaterialIdsSize));
    
    CUDA_CHECK(cudaMemcpy((void*)m_impl->d_vertices, m_impl->vertices.data(), 
                          verticesSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)m_impl->d_indices, m_impl->indices.data(), 
                          indicesSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)m_impl->d_materials, m_impl->materials.data(), 
                          materialsSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)m_impl->d_triangleMaterialIds, m_impl->triangleMaterialIds.data(), 
                          triangleMaterialIdsSize, cudaMemcpyHostToDevice));
    
    // Build OptiX GAS (Geometry Acceleration Structure)
    std::cout << "[Scene] Building OptiX GAS..." << std::endl;
    
    // Setup build input
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    OptixBuildInputTriangleArray& triangleArray = buildInput.triangleArray;
    triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    triangleArray.numVertices = m_impl->vertices.size() / 3;
    triangleArray.vertexBuffers = &m_impl->d_vertices;
    
    triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    triangleArray.numIndexTriplets = m_impl->indices.size() / 3;
    triangleArray.indexBuffer = m_impl->d_indices;
    
    uint32_t buildFlags = OPTIX_GEOMETRY_FLAG_NONE;
    triangleArray.flags = &buildFlags;
    triangleArray.numSbtRecords = 1;
    
    // Compute memory requirements
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        g_optixContext,
        &accelOptions,
        &buildInput,
        1,  // num build inputs
        &gasBufferSizes
    ));
    
    std::cout << "[Scene] GAS buffer sizes:" << std::endl;
    std::cout << "  Temp: " << gasBufferSizes.tempSizeInBytes / 1024 << " KB" << std::endl;
    std::cout << "  Output: " << gasBufferSizes.outputSizeInBytes / 1024 << " KB" << std::endl;
    
    // Allocate temporary and output buffers
    CUdeviceptr d_tempBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), gasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_impl->d_gasOutputBuffer), gasBufferSizes.outputSizeInBytes));
    
    // Build GAS
    OPTIX_CHECK(optixAccelBuild(
        g_optixContext,
        0,  // CUDA stream
        &accelOptions,
        &buildInput,
        1,  // num build inputs
        d_tempBuffer,
        gasBufferSizes.tempSizeInBytes,
        m_impl->d_gasOutputBuffer,
        gasBufferSizes.outputSizeInBytes,
        &m_impl->gasHandle,
        nullptr,  // emitted property list
        0         // num emitted properties
    ));
    
    CUDA_CHECK(cudaFree((void*)d_tempBuffer));
    
    std::cout << "[Scene] GAS built successfully, handle: " << m_impl->gasHandle << std::endl;
    
    m_impl->finalized = true;
}

} // namespace optixw

// Implement accessor
namespace optixw {

OptixTraversableHandle SceneAccessor::getGasHandle(Scene* scene) {
    return scene->m_impl->getGasHandle();
}

CUdeviceptr SceneAccessor::getVerticesPtr(Scene* scene) {
    return scene->m_impl->getVerticesPtr();
}

CUdeviceptr SceneAccessor::getIndicesPtr(Scene* scene) {
    return scene->m_impl->getIndicesPtr();
}

CUdeviceptr SceneAccessor::getMaterialsPtr(Scene* scene) {
    return scene->m_impl->getMaterialsPtr();
}

CUdeviceptr SceneAccessor::getTriangleMaterialIdsPtr(Scene* scene) {
    return scene->m_impl->getTriangleMaterialIdsPtr();
}

uint32_t SceneAccessor::getMaterialCount(Scene* scene) {
    return static_cast<uint32_t>(scene->m_impl->materials.size());
}

uint32_t SceneAccessor::getTriangleCount(Scene* scene) {
    return static_cast<uint32_t>(scene->m_impl->indices.size() / 3);
}

Vec3 SceneAccessor::getEnvironmentRadiance(Scene* scene) {
    for (const MaterialData& mat : scene->m_impl->materials) {
        if (mat.type == static_cast<uint32_t>(MaterialType::EnvironmentEmitter)) {
            const float scale = (mat.emitterScale > 0.0f) ? mat.emitterScale : 0.0f;
            return Vec3(mat.emission.x * scale, mat.emission.y * scale, mat.emission.z * scale);
        }
    }
    return Vec3(0.0f, 0.0f, 0.0f);
}

} // namespace optixw
