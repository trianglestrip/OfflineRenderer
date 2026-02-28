#include "optixw/optixw.h"
#include "optixw/types.h"
#include "scene_internal.h"
#include "cpu_vector_math.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            throw std::runtime_error(                                          \
                std::string("OptiX call failed: ") +                           \
                optixGetErrorName(res) + " (" +                                \
                optixGetErrorString(res) + ")");                               \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            throw std::runtime_error(                                          \
                std::string("CUDA call failed: ") +                            \
                cudaGetErrorString(error));                                    \
        }                                                                      \
    } while (0)

namespace optixw {

// Forward declare context impl to get OptiX context
extern OptixDeviceContext g_optixContext;

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

// Helper method for simple materials
static MaterialData createLambertianMaterial(const RGB& albedo) {
    MaterialData mat;
    mat.albedo = cpu_math::make_float3(albedo.r, albedo.g, albedo.b);
    mat.emission = cpu_math::make_float3(0, 0, 0);
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.ior = 1.5f;
    mat.type = static_cast<uint32_t>(MaterialType::Lambertian);
    return mat;
}

uint32_t Scene::addLambertianMaterial(const RGB& albedo) {
    return addMaterial(createLambertianMaterial(albedo));
}

uint32_t Scene::addEmissiveMaterial(const RGB& emission) {
    MaterialData mat;
    mat.albedo = cpu_math::make_float3(0, 0, 0);
    mat.emission = cpu_math::make_float3(emission.r, emission.g, emission.b);
    mat.roughness = 0.0f;
    mat.metallic = 0.0f;
    mat.ior = 1.0f;
    mat.type = static_cast<uint32_t>(MaterialType::Emissive);
    
    m_impl->materials.push_back(mat);
    return m_impl->materials.size() - 1;
}

uint32_t Scene::addMetalMaterial(const RGB& albedo, float roughness) {
    MaterialData mat;
    mat.albedo = cpu_math::make_float3(albedo.r, albedo.g, albedo.b);
    mat.emission = cpu_math::make_float3(0, 0, 0);
    mat.roughness = roughness;
    mat.metallic = 1.0f;
    mat.ior = 1.0f;
    mat.type = static_cast<uint32_t>(MaterialType::Metal);
    
    m_impl->materials.push_back(mat);
    return m_impl->materials.size() - 1;
}

uint32_t Scene::addGlassMaterial(const RGB& albedo, float ior) {
    MaterialData mat;
    mat.albedo = cpu_math::make_float3(albedo.r, albedo.g, albedo.b);
    mat.emission = cpu_math::make_float3(0, 0, 0);
    mat.roughness = 0.0f;
    mat.metallic = 0.0f;
    mat.ior = ior;
    mat.type = static_cast<uint32_t>(MaterialType::Glass);
    
    m_impl->materials.push_back(mat);
    return m_impl->materials.size() - 1;
}

void Scene::addPointLight(const PointLight& light) {
    PointLightData data;
    data.position = cpu_math::make_float3(light.position.r, light.position.g, light.position.b);
    data.intensity = cpu_math::make_float3(light.intensity.r, light.intensity.g, light.intensity.b);
    m_impl->pointLights.push_back(data);
}

void Scene::addAreaLight(const AreaLight& light) {
    AreaLightData data;
    data.position = cpu_math::make_float3(light.position.r, light.position.g, light.position.b);
    data.normal = cpu_math::make_float3(light.normal.r, light.normal.g, light.normal.b);
    data.emission = cpu_math::make_float3(light.emission.r, light.emission.g, light.emission.b);
    data.width = light.width;
    data.height = light.height;
    data.doubleSided = light.doubleSided ? 1 : 0;
    
    // Compute tangent and bitangent
    float3 up = fabsf(data.normal.y) < 0.9f ? cpu_math::make_float3(0, 1, 0) : cpu_math::make_float3(1, 0, 0);
    data.tangent = cpu_math::normalize(cpu_math::cross(up, data.normal));
    data.bitangent = cpu_math::cross(data.normal, data.tangent);
    
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

} // namespace optixw
