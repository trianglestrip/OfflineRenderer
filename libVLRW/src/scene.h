#pragma once

#include "vlrw/vlrw.h"
#include <vector>
#include <span>
#include <cuda.h>
#include <optix.h>

namespace vlrw {

struct MeshData {
    CUdeviceptr d_vertices;
    CUdeviceptr d_indices;
    uint32_t numVertices;
    uint32_t numIndices;
    uint32_t materialId;
    OptixTraversableHandle gasHandle;
    CUdeviceptr d_gasBuffer;
};

class SceneImpl {
public:
    SceneImpl(ContextImpl* context);
    ~SceneImpl();

    void addTriangleMesh(
        std::span<const float> vertices,
        std::span<const uint32_t> indices,
        uint32_t materialId);
    
    void addTriangleMeshMove(
        std::vector<float>&& vertices,
        std::vector<uint32_t>&& indices,
        uint32_t materialId);

    void setMaterial(uint32_t id, const MaterialDesc& desc);
    void addPointLight(const PointLightDesc& light);

    void finalize();

    OptixTraversableHandle getTraversable() const { return m_iasHandle; }
    const std::vector<MaterialDesc>& getMaterials() const { return m_materials; }

private:
    ContextImpl* m_context;
    std::vector<MeshData> m_meshes;
    std::vector<MaterialDesc> m_materials;
    std::vector<PointLightDesc> m_lights;
    
    OptixTraversableHandle m_iasHandle = 0;
    CUdeviceptr d_iasBuffer = 0;
};

} // namespace vlrw
