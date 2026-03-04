#include "wr/wr.h"
#include "wr/types.h"
#include "utils/cuda_utils.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace wr {

extern OptixDeviceContext getOptixContext();

struct SceneImpl {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> triangleMaterialIds;
    std::vector<MaterialData> materials;
    Vec3 environmentRadiance{0.0f, 0.0f, 0.0f};
    
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_triangleMaterialIds = 0;
    CUdeviceptr d_materials = 0;
    CUdeviceptr d_gasOutput = 0;
    OptixTraversableHandle gasHandle = 0;
    
    uint32_t numVertices = 0;
    uint32_t numTriangles = 0;
    bool finalized = false;
    
    ~SceneImpl() {
        if (d_vertices) cuMemFree(d_vertices);
        if (d_indices) cuMemFree(d_indices);
        if (d_triangleMaterialIds) cuMemFree(d_triangleMaterialIds);
        if (d_materials) cuMemFree(d_materials);
        if (d_gasOutput) cuMemFree(d_gasOutput);
    }
};

Scene::Scene() : m_impl(new SceneImpl()) {}
Scene::~Scene() { delete m_impl; }

uint32_t Scene::addLambertianMaterial(const Vec3& albedo) {
    MaterialData mat = {};
    mat.albedo = make_float3(albedo.x, albedo.y, albedo.z);
    mat.emission = make_float3(0.0f, 0.0f, 0.0f);
    mat.ior = 1.0f;
    mat.type = MaterialType::Lambertian;
    
    m_impl->materials.push_back(mat);
    return static_cast<uint32_t>(m_impl->materials.size() - 1);
}

uint32_t Scene::addEmissiveMaterial(const Vec3& emission) {
    MaterialData mat = {};
    mat.albedo = make_float3(0.0f, 0.0f, 0.0f);
    mat.emission = make_float3(emission.x, emission.y, emission.z);
    mat.ior = 1.0f;
    mat.type = MaterialType::Emissive;
    
    m_impl->materials.push_back(mat);
    return static_cast<uint32_t>(m_impl->materials.size() - 1);
}

uint32_t Scene::addGlassMaterial(const Vec3& albedo, float ior) {
    MaterialData mat = {};
    mat.albedo = make_float3(albedo.x, albedo.y, albedo.z);
    mat.emission = make_float3(0.0f, 0.0f, 0.0f);
    mat.ior = ior;
    mat.type = MaterialType::Glass;
    
    m_impl->materials.push_back(mat);
    return static_cast<uint32_t>(m_impl->materials.size() - 1);
}

void Scene::addTriangleMesh(std::span<const float> verts,
                            std::span<const uint32_t> inds,
                            uint32_t materialId) {
    uint32_t vertexOffset = static_cast<uint32_t>(m_impl->vertices.size() / 3);
    
    for (float v : verts) {
        m_impl->vertices.push_back(v);
    }
    
    for (uint32_t idx : inds) {
        m_impl->indices.push_back(vertexOffset + idx);
    }
    
    uint32_t numTris = static_cast<uint32_t>(inds.size() / 3);
    for (uint32_t i = 0; i < numTris; ++i) {
        m_impl->triangleMaterialIds.push_back(materialId);
    }
    
    m_impl->numTriangles += numTris;
    m_impl->numVertices = static_cast<uint32_t>(m_impl->vertices.size() / 3);
}

void Scene::setEnvironmentRadiance(const Vec3& radiance) {
    m_impl->environmentRadiance = radiance;
}

void Scene::finalize() {
    if (m_impl->finalized) return;
    
    std::cout << "[Scene] Finalizing scene: " << m_impl->numTriangles 
              << " triangles, " << m_impl->numVertices << " vertices" << std::endl;
    
    if (m_impl->vertices.empty() || m_impl->indices.empty()) {
        throw std::runtime_error("Scene has no geometry");
    }
    
    size_t vertSize = m_impl->vertices.size() * sizeof(float);
    CU_CHECK(cuMemAlloc(&m_impl->d_vertices, vertSize));
    CU_CHECK(cuMemcpyHtoD(m_impl->d_vertices, m_impl->vertices.data(), vertSize));
    
    size_t indSize = m_impl->indices.size() * sizeof(uint32_t);
    CU_CHECK(cuMemAlloc(&m_impl->d_indices, indSize));
    CU_CHECK(cuMemcpyHtoD(m_impl->d_indices, m_impl->indices.data(), indSize));
    
    size_t matIdSize = m_impl->triangleMaterialIds.size() * sizeof(uint32_t);
    CU_CHECK(cuMemAlloc(&m_impl->d_triangleMaterialIds, matIdSize));
    CU_CHECK(cuMemcpyHtoD(m_impl->d_triangleMaterialIds, m_impl->triangleMaterialIds.data(), matIdSize));
    
    size_t matSize = m_impl->materials.size() * sizeof(MaterialData);
    CU_CHECK(cuMemAlloc(&m_impl->d_materials, matSize));
    CU_CHECK(cuMemcpyHtoD(m_impl->d_materials, m_impl->materials.data(), matSize));
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    CUdeviceptr d_vertexBuffers[1] = { m_impl->d_vertices };
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexBuffers = d_vertexBuffers;
    buildInput.triangleArray.numVertices = m_impl->numVertices;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    
    buildInput.triangleArray.indexBuffer = m_impl->d_indices;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    buildInput.triangleArray.numIndexTriplets = m_impl->numTriangles;
    
    uint32_t buildInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.triangleArray.flags = buildInputFlags;
    buildInput.triangleArray.numSbtRecords = 1;
    
    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        getOptixContext(),
        &accelOptions,
        &buildInput,
        1,
        &bufferSizes
    ));
    
    CUdeviceptr d_tempBuffer;
    CU_CHECK(cuMemAlloc(&d_tempBuffer, bufferSizes.tempSizeInBytes));
    CU_CHECK(cuMemAlloc(&m_impl->d_gasOutput, bufferSizes.outputSizeInBytes));
    
    OPTIX_CHECK(optixAccelBuild(
        getOptixContext(),
        0,
        &accelOptions,
        &buildInput,
        1,
        d_tempBuffer,
        bufferSizes.tempSizeInBytes,
        m_impl->d_gasOutput,
        bufferSizes.outputSizeInBytes,
        &m_impl->gasHandle,
        nullptr,
        0
    ));
    
    CU_CHECK(cuMemFree(d_tempBuffer));
    
    std::cout << "[Scene] GAS built, handle: " << m_impl->gasHandle << std::endl;
    
    m_impl->finalized = true;
}

CUdeviceptr Scene::getVerticesBuffer() const { return m_impl->d_vertices; }
CUdeviceptr Scene::getIndicesBuffer() const { return m_impl->d_indices; }
CUdeviceptr Scene::getTriangleMaterialIdsBuffer() const { return m_impl->d_triangleMaterialIds; }
CUdeviceptr Scene::getMaterialsBuffer() const { return m_impl->d_materials; }
OptixTraversableHandle Scene::getGASHandle() const { return m_impl->gasHandle; }
uint32_t Scene::getNumMaterials() const { return static_cast<uint32_t>(m_impl->materials.size()); }
Vec3 Scene::getEnvironmentRadiance() const { return m_impl->environmentRadiance; }

} // namespace wr
