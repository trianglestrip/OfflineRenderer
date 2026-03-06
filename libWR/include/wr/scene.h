#pragma once

#include "wr/types.h"
#include <span>
#include <cstdint>
#include <string>

namespace wr {

// Scene build configuration
struct SceneBuildConfig {
    bool allowUpdate = false;
    bool allowCompaction = true;
    bool preferFastTrace = true;
};

// Forward declarations
struct SceneImpl;
class Scene;

// Opaque types for internal GPU handles (no GPU headers in public API)
using DevicePtr = unsigned long long;
using TraversableHandle = unsigned long long;
using TextureHandle = unsigned long long;

// Texture2D - RGBA8 format, supports PNG loading
class Texture2D {
public:
    // Create from PNG file via Scene, returns texture id for use in materials
    static uint32_t create(Scene* scene, const std::string& path);
    static void destroy(Scene* scene, uint32_t textureId);
};

// Scene manages geometry and materials
class Scene {
public:
    Scene();
    ~Scene();

    // Material creation
    uint32_t addLambertianMaterial(const Vec3& albedo);
    uint32_t addLambertianMaterial(const Vec3& albedo, uint32_t albedoTextureId);
    uint32_t addEmissiveMaterial(const Vec3& emission);
    uint32_t addGlassMaterial(const Vec3& albedo, float ior);
    uint32_t addGGXReflectionMaterial(const Vec3& albedo, float roughness, float metallic = 0.0f);
    uint32_t addGGXTransmissionMaterial(const Vec3& albedo, float roughness, float ior);

    // Geometry
    void addTriangleMesh(std::span<const float> vertices,
                         std::span<const uint32_t> indices,
                         uint32_t materialId);
    void addTriangleMesh(std::span<const float> vertices,
                         std::span<const uint32_t> indices,
                         std::span<const float> uvs,
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

    // Texture accessors (for Renderer)
    DevicePtr getTexturesBuffer() const;
    uint32_t getNumTextures() const;
    DevicePtr getUVsBuffer() const;

    // Texture loading (used by Texture2D::create)
    uint32_t loadTexture(const std::string& path);
    void destroyTexture(uint32_t textureId);

private:
    SceneImpl* m_impl;
};

} // namespace wr
