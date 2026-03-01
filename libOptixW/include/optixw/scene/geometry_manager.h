// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <vector>
#include <cstdint>

namespace optixw {

// GeometryManager - CPU 层几何数据管理器
// 职责：
//   - 管理三角形网格数据
//   - 构建 GAS (Geometry Acceleration Structure)
//   - 管理顶点、索引、纹理坐标等缓冲
// 注意：
//   - 这是 CPU 层代码，不包含 GPU 计算逻辑
//   - GAS 构建需要 OptiX 上下文
class GeometryManager {
public:
    explicit GeometryManager(OptixDeviceContext context);
    ~GeometryManager();

    // 添加三角形网格
    // vertices: 顶点数组 (x, y, z, x, y, z, ...)
    // numVertices: 顶点数量
    // indices: 索引数组 (i0, i1, i2, i0, i1, i2, ...)
    // numTriangles: 三角形数量
    // texcoords: 纹理坐标数组 (u, v, u, v, ...) 可选
    // materialId: 材质 ID
    void addTriangleMesh(
        const float* vertices,
        uint32_t numVertices,
        const uint32_t* indices,
        uint32_t numTriangles,
        const float* texcoords = nullptr,
        uint32_t materialId = 0);

    // 构建 GAS（必须在添加所有网格后调用）
    void buildGAS();

    // 访问器
    OptixTraversableHandle getGASHandle() const { return m_gasHandle; }
    CUdeviceptr getVerticesBuffer() const { return m_d_vertices; }
    CUdeviceptr getIndicesBuffer() const { return m_d_indices; }
    CUdeviceptr getTexcoordsBuffer() const { return m_d_texcoords; }
    CUdeviceptr getMaterialIdsBuffer() const { return m_d_materialIds; }
    uint32_t getNumTriangles() const { return m_numTriangles; }
    uint32_t getNumVertices() const { return m_numVertices; }
    bool isBuilt() const { return m_built; }
    
    // 新增结构体版本
    void addTriangleMesh(const TriangleMeshParams& params);

    // 清空几何数据
    void clear();

private:
    // OptiX 上下文
    OptixDeviceContext m_context;

    // 主机端数据（构建前临时存储）
    std::vector<float> m_vertices;      // 顶点 (x, y, z)
    std::vector<uint32_t> m_indices;    // 索引
    std::vector<float> m_texcoords;     // 纹理坐标 (u, v)
    std::vector<uint32_t> m_materialIds; // 材质 ID（每个三角形）

    // 设备端缓冲
    CUdeviceptr m_d_vertices;
    CUdeviceptr m_d_indices;
    CUdeviceptr m_d_texcoords;
    CUdeviceptr m_d_materialIds;

    // GAS
    OptixTraversableHandle m_gasHandle;
    CUdeviceptr m_d_gasOutputBuffer;

    // 统计信息
    uint32_t m_numVertices;
    uint32_t m_numTriangles;
    bool m_built;

    // 内部方法
    void uploadToDevice();
    void freeDeviceBuffers();

    // 禁止拷贝和赋值
    GeometryManager(const GeometryManager&) = delete;
    GeometryManager& operator=(const GeometryManager&) = delete;
};

} // namespace optixw
