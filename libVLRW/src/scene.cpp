#include "scene.h"
#include "optix_context.h"
#include "cuda_check.h"
#include <stdexcept>

namespace vlrw {

SceneImpl::SceneImpl(ContextImpl* context) : m_context(context) {}

SceneImpl::~SceneImpl() {
    for (auto& mesh : m_meshes) {
        if (mesh.d_vertices) cuMemFree(mesh.d_vertices);
        if (mesh.d_indices) cuMemFree(mesh.d_indices);
        if (mesh.d_gasBuffer) cuMemFree(mesh.d_gasBuffer);
    }
    if (d_iasBuffer) cuMemFree(d_iasBuffer);
    if (d_materials) cuMemFree(d_materials);
    if (d_mergedVertices) cuMemFree(d_mergedVertices);
    if (d_mergedIndices) cuMemFree(d_mergedIndices);
    if (d_triangleMaterialIds) cuMemFree(d_triangleMaterialIds);
    if (d_mergedGasBuffer) cuMemFree(d_mergedGasBuffer);
}

void SceneImpl::addTriangleMesh(
    std::span<const float> vertices,
    std::span<const uint32_t> indices,
    uint32_t materialId)
{
    MeshData mesh = {};
    mesh.numVertices = (uint32_t)vertices.size() / 3;
    mesh.numIndices = (uint32_t)indices.size();
    mesh.materialId = materialId;

    CUDA_CHECK(cuMemAlloc(&mesh.d_vertices, vertices.size_bytes()));
    CUDA_CHECK(cuMemcpyHtoD(mesh.d_vertices, vertices.data(), vertices.size_bytes()));

    CUDA_CHECK(cuMemAlloc(&mesh.d_indices, indices.size_bytes()));
    CUDA_CHECK(cuMemcpyHtoD(mesh.d_indices, indices.data(), indices.size_bytes()));

    m_meshes.push_back(mesh);
}

void SceneImpl::addTriangleMeshMove(
    std::vector<float>&& vertices,
    std::vector<uint32_t>&& indices,
    uint32_t materialId)
{
    addTriangleMesh(
        std::span<const float>(vertices),
        std::span<const uint32_t>(indices),
        materialId);
}

void SceneImpl::setMaterial(uint32_t id, const MaterialDesc& desc) {
    if (id >= m_materials.size())
        m_materials.resize(id + 1);
    m_materials[id] = desc;
}

void SceneImpl::addPointLight(const PointLightDesc& light) {
    m_lights.push_back(light);
}

void SceneImpl::addAreaLight(const AreaLightDesc& light) {
    m_areaLights.push_back(light);
}

void SceneImpl::finalize() {
    if (m_meshes.empty()) return;

    if (!m_materials.empty()) {
        struct MaterialGPU { float r, g, b; float er, eg, eb; };
        std::vector<MaterialGPU> matData(m_materials.size());
        for (size_t i = 0; i < m_materials.size(); ++i) {
            matData[i].r = m_materials[i].albedo.r;
            matData[i].g = m_materials[i].albedo.g;
            matData[i].b = m_materials[i].albedo.b;
            matData[i].er = m_materials[i].emission.r;
            matData[i].eg = m_materials[i].emission.g;
            matData[i].eb = m_materials[i].emission.b;
        }
        CUDA_CHECK(cuMemAlloc(&d_materials, matData.size() * sizeof(MaterialGPU)));
        CUDA_CHECK(cuMemcpyHtoD(d_materials, matData.data(), matData.size() * sizeof(MaterialGPU)));
    }

    if (m_meshes.size() == 1) {
        MeshData& mesh = m_meshes[0];
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        CUdeviceptr d_vertices = mesh.d_vertices;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexBuffers = &d_vertices;
        buildInput.triangleArray.numVertices = mesh.numVertices;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexBuffer = mesh.d_indices;
        buildInput.triangleArray.numIndexTriplets = mesh.numIndices / 3;
        uint32_t inputFlags = OPTIX_GEOMETRY_FLAG_NONE;
        buildInput.triangleArray.flags = &inputFlags;
        buildInput.triangleArray.numSbtRecords = 1;
        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_context->getOptixContext(), &accelOptions, &buildInput, 1, &gasBufferSizes));
        CUDA_CHECK(cuMemAlloc(&mesh.d_gasBuffer, gasBufferSizes.outputSizeInBytes));
        CUdeviceptr d_temp;
        CUDA_CHECK(cuMemAlloc(&d_temp, gasBufferSizes.tempSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(
            m_context->getOptixContext(), 0, &accelOptions, &buildInput, 1,
            d_temp, gasBufferSizes.tempSizeInBytes,
            mesh.d_gasBuffer, gasBufferSizes.outputSizeInBytes,
            &mesh.gasHandle, nullptr, 0));
        CUDA_CHECK(cuMemFree(d_temp));
        m_iasHandle = mesh.gasHandle;
        uint32_t numTri = mesh.numIndices / 3;
        std::vector<uint32_t> triMatIds(numTri, mesh.materialId);
        CUDA_CHECK(cuMemAlloc(&d_triangleMaterialIds, numTri * sizeof(uint32_t)));
        CUDA_CHECK(cuMemcpyHtoD(d_triangleMaterialIds, triMatIds.data(), numTri * sizeof(uint32_t)));
    } else {
        std::vector<float> mergedVertices;
        std::vector<uint32_t> mergedIndices;
        std::vector<uint32_t> mergedTriangleMaterialIds;
        uint32_t vertexOffset = 0;
        for (const auto& mesh : m_meshes) {
            std::vector<float> v(mesh.numVertices * 3);
            CUDA_CHECK(cuMemcpyDtoH(v.data(), mesh.d_vertices, v.size() * sizeof(float)));
            std::vector<uint32_t> idx(mesh.numIndices);
            CUDA_CHECK(cuMemcpyDtoH(idx.data(), mesh.d_indices, idx.size() * sizeof(uint32_t)));
            for (float f : v) mergedVertices.push_back(f);
            for (uint32_t i : idx) mergedIndices.push_back(i + vertexOffset);
            vertexOffset += mesh.numVertices;
            uint32_t numTri = mesh.numIndices / 3;
            for (uint32_t t = 0; t < numTri; ++t) mergedTriangleMaterialIds.push_back(mesh.materialId);
        }
        CUDA_CHECK(cuMemAlloc(&d_mergedVertices, mergedVertices.size() * sizeof(float)));
        CUDA_CHECK(cuMemcpyHtoD(d_mergedVertices, mergedVertices.data(), mergedVertices.size() * sizeof(float)));
        CUDA_CHECK(cuMemAlloc(&d_mergedIndices, mergedIndices.size() * sizeof(uint32_t)));
        CUDA_CHECK(cuMemcpyHtoD(d_mergedIndices, mergedIndices.data(), mergedIndices.size() * sizeof(uint32_t)));
        CUDA_CHECK(cuMemAlloc(&d_triangleMaterialIds, mergedTriangleMaterialIds.size() * sizeof(uint32_t)));
        CUDA_CHECK(cuMemcpyHtoD(d_triangleMaterialIds, mergedTriangleMaterialIds.data(), mergedTriangleMaterialIds.size() * sizeof(uint32_t)));
        m_mergedNumVertices = (uint32_t)mergedVertices.size() / 3;
        m_mergedNumIndices = (uint32_t)mergedIndices.size();
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexBuffers = &d_mergedVertices;
        buildInput.triangleArray.numVertices = (uint32_t)mergedVertices.size() / 3;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexBuffer = d_mergedIndices;
        buildInput.triangleArray.numIndexTriplets = (uint32_t)(mergedIndices.size() / 3);
        uint32_t inputFlags = OPTIX_GEOMETRY_FLAG_NONE;
        buildInput.triangleArray.flags = &inputFlags;
        buildInput.triangleArray.numSbtRecords = 1;
        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_context->getOptixContext(), &accelOptions, &buildInput, 1, &gasBufferSizes));
        CUDA_CHECK(cuMemAlloc(&d_mergedGasBuffer, gasBufferSizes.outputSizeInBytes));
        CUdeviceptr d_temp;
        CUDA_CHECK(cuMemAlloc(&d_temp, gasBufferSizes.tempSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(
            m_context->getOptixContext(), 0, &accelOptions, &buildInput, 1,
            d_temp, gasBufferSizes.tempSizeInBytes,
            d_mergedGasBuffer, gasBufferSizes.outputSizeInBytes,
            &m_mergedGasHandle, nullptr, 0));
        CUDA_CHECK(cuMemFree(d_temp));
        m_iasHandle = m_mergedGasHandle;
    }
}

CUdeviceptr SceneImpl::getFirstMeshVertices() const {
    if (m_meshes.empty()) return 0;
    return d_mergedVertices ? d_mergedVertices : m_meshes[0].d_vertices;
}
CUdeviceptr SceneImpl::getFirstMeshIndices() const {
    if (m_meshes.empty()) return 0;
    return d_mergedIndices ? d_mergedIndices : m_meshes[0].d_indices;
}
uint32_t SceneImpl::getFirstMeshNumVertices() const {
    if (m_meshes.empty()) return 0;
    if (d_mergedVertices) return m_mergedNumVertices;
    return m_meshes[0].numVertices;
}
uint32_t SceneImpl::getFirstMeshNumIndices() const {
    if (m_meshes.empty()) return 0;
    if (d_mergedIndices) return m_mergedNumIndices;
    return m_meshes[0].numIndices;
}

} // namespace vlrw
