// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/scene/geometry_manager.h>
#include "../utils/checks.h"
#include <iostream>

namespace optixw {

GeometryManager::GeometryManager(OptixDeviceContext context)
    : m_context(context)
    , m_d_vertices(0)
    , m_d_indices(0)
    , m_d_texcoords(0)
    , m_d_materialIds(0)
    , m_gasHandle(0)
    , m_d_gasOutputBuffer(0)
    , m_numVertices(0)
    , m_numTriangles(0)
    , m_built(false)
{
}

GeometryManager::~GeometryManager() {
    clear();
}

void GeometryManager::addTriangleMesh(
    const float* vertices,
    uint32_t numVertices,
    const uint32_t* indices,
    uint32_t numTriangles,
    const float* texcoords,
    uint32_t materialId)
{
    if (m_built) {
        throw std::runtime_error("Cannot add mesh after GAS is built");
    }

    // 添加顶点
    const size_t vertexOffset = m_vertices.size();
    m_vertices.insert(m_vertices.end(), vertices, vertices + numVertices * 3);

    // 添加索引（调整偏移）
    const uint32_t indexOffset = static_cast<uint32_t>(vertexOffset / 3);
    for (uint32_t i = 0; i < numTriangles * 3; ++i) {
        m_indices.push_back(indices[i] + indexOffset);
    }

    // 添加纹理坐标
    if (texcoords) {
        m_texcoords.insert(m_texcoords.end(), texcoords, texcoords + numVertices * 2);
    } else {
        // 默认纹理坐标
        for (uint32_t i = 0; i < numVertices; ++i) {
            m_texcoords.push_back(0.0f);
            m_texcoords.push_back(0.0f);
        }
    }

    // 添加材质 ID
    for (uint32_t i = 0; i < numTriangles; ++i) {
        m_materialIds.push_back(materialId);
    }

    m_numVertices += numVertices;
    m_numTriangles += numTriangles;
}

void GeometryManager::buildGAS() {
    if (m_built) {
        std::cout << "[GeometryManager] GAS 已构建，跳过" << std::endl;
        return;
    }

    if (m_numTriangles == 0) {
        throw std::runtime_error("No geometry to build GAS");
    }

    std::cout << "[GeometryManager] 构建 GAS: " << m_numTriangles << " 个三角形" << std::endl;

    // 上传数据到设备
    uploadToDevice();

    // 构建输入
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr d_vertexBuffers[1] = { m_d_vertices };
    buildInput.triangleArray.vertexBuffers = d_vertexBuffers;
    buildInput.triangleArray.numVertices = m_numVertices;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float) * 3;

    buildInput.triangleArray.indexBuffer = m_d_indices;
    buildInput.triangleArray.numIndexTriplets = m_numTriangles;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;

    uint32_t buildInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.triangleArray.flags = buildInputFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    // 加速结构选项
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // 查询内存需求
    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_context,
        &accelOptions,
        &buildInput,
        1,
        &bufferSizes
    ));

    // 分配临时缓冲
    CUdeviceptr d_tempBuffer;
    CU_CHECK(cuMemAlloc(&d_tempBuffer, bufferSizes.tempSizeInBytes));

    // 分配输出缓冲
    CU_CHECK(cuMemAlloc(&m_d_gasOutputBuffer, bufferSizes.outputSizeInBytes));

    // 构建 GAS
    OPTIX_CHECK(optixAccelBuild(
        m_context,
        0,  // stream
        &accelOptions,
        &buildInput,
        1,
        d_tempBuffer,
        bufferSizes.tempSizeInBytes,
        m_d_gasOutputBuffer,
        bufferSizes.outputSizeInBytes,
        &m_gasHandle,
        nullptr,
        0
    ));

    // 释放临时缓冲
    CU_CHECK(cuMemFree(d_tempBuffer));

    m_built = true;
    std::cout << "[GeometryManager] GAS 构建完成" << std::endl;
}

void GeometryManager::clear() {
    m_vertices.clear();
    m_indices.clear();
    m_texcoords.clear();
    m_materialIds.clear();

    freeDeviceBuffers();

    m_numVertices = 0;
    m_numTriangles = 0;
    m_built = false;
}

void GeometryManager::uploadToDevice() {
    // 上传顶点
    if (!m_vertices.empty()) {
        const size_t size = m_vertices.size() * sizeof(float);
        CU_CHECK(cuMemAlloc(&m_d_vertices, size));
        CU_CHECK(cuMemcpyHtoD(m_d_vertices, m_vertices.data(), size));
    }

    // 上传索引
    if (!m_indices.empty()) {
        const size_t size = m_indices.size() * sizeof(uint32_t);
        CU_CHECK(cuMemAlloc(&m_d_indices, size));
        CU_CHECK(cuMemcpyHtoD(m_d_indices, m_indices.data(), size));
    }

    // 上传纹理坐标
    if (!m_texcoords.empty()) {
        const size_t size = m_texcoords.size() * sizeof(float);
        CU_CHECK(cuMemAlloc(&m_d_texcoords, size));
        CU_CHECK(cuMemcpyHtoD(m_d_texcoords, m_texcoords.data(), size));
    }

    // 上传材质 ID
    if (!m_materialIds.empty()) {
        const size_t size = m_materialIds.size() * sizeof(uint32_t);
        CU_CHECK(cuMemAlloc(&m_d_materialIds, size));
        CU_CHECK(cuMemcpyHtoD(m_d_materialIds, m_materialIds.data(), size));
    }
}

void GeometryManager::freeDeviceBuffers() {
    if (m_d_vertices) {
        CU_CHECK(cuMemFree(m_d_vertices));
        m_d_vertices = 0;
    }
    if (m_d_indices) {
        CU_CHECK(cuMemFree(m_d_indices));
        m_d_indices = 0;
    }
    if (m_d_texcoords) {
        CU_CHECK(cuMemFree(m_d_texcoords));
        m_d_texcoords = 0;
    }
    if (m_d_materialIds) {
        CU_CHECK(cuMemFree(m_d_materialIds));
        m_d_materialIds = 0;
    }
    if (m_d_gasOutputBuffer) {
        CU_CHECK(cuMemFree(m_d_gasOutputBuffer));
        m_d_gasOutputBuffer = 0;
    }
}

} // namespace optixw
