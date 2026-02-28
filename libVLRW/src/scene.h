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
    void addAreaLight(const AreaLightDesc& light);

    void finalize();

    OptixTraversableHandle getTraversable() const { return m_iasHandle; }
    const std::vector<MaterialDesc>& getMaterials() const { return m_materials; }
    CUdeviceptr getMaterialBuffer() const { return d_materials; }
    uint32_t getNumMaterials() const { return (uint32_t)m_materials.size(); }
    const std::vector<PointLightDesc>& getLights() const { return m_lights; }
    const std::vector<AreaLightDesc>& getAreaLights() const { return m_areaLights; }
    CUdeviceptr getFirstMeshVertices() const;
    CUdeviceptr getFirstMeshIndices() const;
    uint32_t getFirstMeshNumVertices() const;
    uint32_t getFirstMeshNumIndices() const;
    /** Per-triangle material ID for the (possibly merged) mesh; nullptr if not used. */
    CUdeviceptr getTriangleMaterialIds() const { return d_triangleMaterialIds; }

private:
    ContextImpl* m_context;
    std::vector<MeshData> m_meshes;
    std::vector<MaterialDesc> m_materials;
    std::vector<PointLightDesc> m_lights;
    std::vector<AreaLightDesc> m_areaLights;

    OptixTraversableHandle m_iasHandle = 0;
    CUdeviceptr d_iasBuffer = 0;
    CUdeviceptr d_materials = 0;

    /** Merged mesh (when multiple meshes): single GAS + per-triangle material IDs */
    CUdeviceptr d_mergedVertices = 0;
    CUdeviceptr d_mergedIndices = 0;
    CUdeviceptr d_triangleMaterialIds = 0;
    CUdeviceptr d_mergedGasBuffer = 0;
    OptixTraversableHandle m_mergedGasHandle = 0;
    uint32_t m_mergedNumVertices = 0;
    uint32_t m_mergedNumIndices = 0;
};

} // namespace vlrw
