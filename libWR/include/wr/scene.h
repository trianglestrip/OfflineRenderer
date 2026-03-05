#pragma once

#include "wr/types.h"
#include <span>
#include <cstdint>

namespace wr {

// Scene build configuration
struct SceneBuildConfig {
    bool allowUpdate = false;
    bool allowCompaction = true;
    bool preferFastTrace = true;
};

// Forward declarations
struct SceneImpl;

// Opaque types for internal GPU handles (no GPU headers in public API)
using DevicePtr = unsigned long long;
using TraversableHandle = unsigned long long;

// Scene manages geometry and materials
class Scene {
public:
    Scene();
    ~Scene();

    // Material creation
    uint32_t addLambertianMaterial(const Vec3& albedo);
    uint32_t addEmissiveMaterial(const Vec3& emission);
    uint32_t addGlassMaterial(const Vec3& albedo, float ior);

    // Geometry
    void addTriangleMesh(std::span<const float> vertices,
                         std::span<const uint32_t> indices,
                         uint32_t materialId);

    // Environment
    void setEnvironmentRadiance(const Vec3& radiance);
    
    // Finalize scene (build acceleration structure)
    void finalize(const SceneBuildConfig& config = {});
    
    // Internal accessors (used by Renderer)
    DevicePtr getVerticesBuffer() const;
    DevicePtr getIndicesBuffer() const;
    DevicePtr getTriangleMaterialIdsBuffer() const;
    DevicePtr getMaterialsBuffer() const;
    TraversableHandle getGASHandle() const;
    uint32_t getNumMaterials() const;
    Vec3 getEnvironmentRadiance() const;
    
    // Light sampling accessors
    DevicePtr getEmissiveTrianglesBuffer() const;
    DevicePtr getEmissiveTriangleCDFBuffer() const;
    uint32_t getNumEmissiveTriangles() const;

private:
    SceneImpl* m_impl;
};

} // namespace wr
