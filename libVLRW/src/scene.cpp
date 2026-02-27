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

void SceneImpl::finalize() {
    if (m_meshes.empty()) return;

    for (auto& mesh : m_meshes) {
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        CUdeviceptr d_vertices = mesh.d_vertices;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexBuffers = &d_vertices;
        buildInput.triangleArray.numVertices = mesh.numVertices;

        CUdeviceptr d_indices = mesh.d_indices;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexBuffer = d_indices;
        buildInput.triangleArray.numIndexTriplets = mesh.numIndices / 3;

        uint32_t inputFlags = OPTIX_GEOMETRY_FLAG_NONE;
        buildInput.triangleArray.flags = &inputFlags;
        buildInput.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_context->getOptixContext(),
            &accelOptions,
            &buildInput, 1,
            &gasBufferSizes));

        CUDA_CHECK(cuMemAlloc(&mesh.d_gasBuffer, gasBufferSizes.outputSizeInBytes));

        CUdeviceptr d_temp;
        CUDA_CHECK(cuMemAlloc(&d_temp, gasBufferSizes.tempSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            m_context->getOptixContext(),
            0,
            &accelOptions,
            &buildInput, 1,
            d_temp, gasBufferSizes.tempSizeInBytes,
            mesh.d_gasBuffer, gasBufferSizes.outputSizeInBytes,
            &mesh.gasHandle,
            nullptr, 0));

        CUDA_CHECK(cuMemFree(d_temp));
    }

    if (m_meshes.size() == 1) {
        m_iasHandle = m_meshes[0].gasHandle;
    } else {
        // TODO: Build IAS for multiple GAS instances
        m_iasHandle = m_meshes[0].gasHandle;
    }
}

} // namespace vlrw
