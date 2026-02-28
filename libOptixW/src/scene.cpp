#include "optixw/optixw.h"
#include "optixw/types.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace optixw {

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

uint32_t Scene::addMaterial(MaterialType type, const RGB& albedo) {
    MaterialData mat;
    mat.albedo = make_float3(albedo.r, albedo.g, albedo.b);
    mat.emission = make_float3(0, 0, 0);
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.ior = 1.5f;
    mat.type = static_cast<uint32_t>(type);
    
    m_impl->materials.push_back(mat);
    return m_impl->materials.size() - 1;
}

uint32_t Scene::addEmissiveMaterial(const RGB& emission) {
    MaterialData mat;
    mat.albedo = make_float3(0, 0, 0);
    mat.emission = make_float3(emission.r, emission.g, emission.b);
    mat.roughness = 0.0f;
    mat.metallic = 0.0f;
    mat.ior = 1.0f;
    mat.type = static_cast<uint32_t>(MaterialType::Emissive);
    
    m_impl->materials.push_back(mat);
    return m_impl->materials.size() - 1;
}

void Scene::addPointLight(const PointLight& light) {
    PointLightData data;
    data.position = make_float3(light.position.r, light.position.g, light.position.b);
    data.intensity = make_float3(light.intensity.r, light.intensity.g, light.intensity.b);
    m_impl->pointLights.push_back(data);
}

void Scene::addAreaLight(const AreaLight& light) {
    AreaLightData data;
    data.position = make_float3(light.position.r, light.position.g, light.position.b);
    data.normal = make_float3(light.normal.r, light.normal.g, light.normal.b);
    data.emission = make_float3(light.emission.r, light.emission.g, light.emission.b);
    data.width = light.width;
    data.height = light.height;
    data.doubleSided = light.doubleSided ? 1 : 0;
    
    // Compute tangent and bitangent
    float3 up = fabsf(data.normal.y) < 0.9f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    data.tangent = normalize(cross(up, data.normal));
    data.bitangent = cross(data.normal, data.tangent);
    
    m_impl->areaLights.push_back(data);
}

void Scene::finalize() {
    if (m_impl->finalized) return;
    
    // Upload geometry to GPU
    size_t verticesSize = m_impl->vertices.size() * sizeof(float);
    size_t indicesSize = m_impl->indices.size() * sizeof(uint32_t);
    size_t materialsSize = m_impl->materials.size() * sizeof(MaterialData);
    size_t triangleMaterialIdsSize = m_impl->triangleMaterialIds.size() * sizeof(uint32_t);
    
    cudaMalloc(&m_impl->d_vertices, verticesSize);
    cudaMalloc(&m_impl->d_indices, indicesSize);
    cudaMalloc(&m_impl->d_materials, materialsSize);
    cudaMalloc(&m_impl->d_triangleMaterialIds, triangleMaterialIdsSize);
    
    cudaMemcpy((void*)m_impl->d_vertices, m_impl->vertices.data(), verticesSize, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)m_impl->d_indices, m_impl->indices.data(), indicesSize, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)m_impl->d_materials, m_impl->materials.data(), materialsSize, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)m_impl->d_triangleMaterialIds, m_impl->triangleMaterialIds.data(), 
               triangleMaterialIdsSize, cudaMemcpyHostToDevice);
    
    // TODO: Build OptiX GAS (next step)
    
    m_impl->finalized = true;
}

} // namespace optixw
