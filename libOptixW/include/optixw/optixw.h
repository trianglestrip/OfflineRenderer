#pragma once

#include <optixw/types.h>
#include <memory>
#include <span>

namespace optixw {

// Forward declarations
class Scene;
class Renderer;

// 3D vector type for positions/directions
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Camera parameters
struct Camera {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    float fovY;
    float aspect;
};

// Point light
struct PointLight {
    Vec3 position;
    Vec3 intensity;
};

// Area light
struct AreaLight {
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
    float width;
    float height;
    Vec3 emission;
    bool doubleSided;
};

// Material type enum
enum class MaterialType : uint32_t {
    Matte = 0,
    LambertianScattering = 1,
    SpecularReflection = 2,
    SpecularScattering = 3,
    MicrofacetReflection = 4,
    MicrofacetScattering = 5,
    UE4 = 6,
    OldStyle = 7,
    DiffuseEmitter = 8,
    DirectionalEmitter = 9,
    PointEmitter = 10,
    Multi = 11,
    EnvironmentEmitter = 12,

    // Backward-compatible aliases.
    Lambertian = Matte,
    Emissive = DiffuseEmitter,
    Metal = MicrofacetReflection,
    Glass = SpecularScattering
};

// Context: CUDA and OptiX device initialization
class Context {
public:
    Context(int deviceId = 0);
    ~Context();
    
    // Factory methods
    Scene* createScene();
    Renderer* createRenderer();
    
private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Scene: Geometry and material management
class Scene {
public:
    Scene();
    ~Scene();
    
    // Add geometry
    void addTriangleMesh(
        std::span<const float> vertices,
        std::span<const uint32_t> indices,
        uint32_t materialId
    );
    
    // Add materials
    uint32_t addMaterial(const MaterialData& material);
    uint32_t addMatteMaterial(const Vec3& albedo);
    uint32_t addLambertianScatteringMaterial(const Vec3& coeff, float f0 = 0.04f);
    uint32_t addSpecularReflectionMaterial(const Vec3& coeff, const Vec3& eta, const Vec3& k);
    uint32_t addSpecularScatteringMaterial(const Vec3& coeff, float iorExt, float iorInt);
    uint32_t addMicrofacetReflectionMaterial(
        const Vec3& eta, const Vec3& k, float roughness, float anisotropy = 0.0f, float rotation = 0.0f);
    uint32_t addMicrofacetScatteringMaterial(
        const Vec3& coeff, float iorExt, float iorInt, float roughness, float anisotropy = 0.0f, float rotation = 0.0f);
    uint32_t addUE4Material(const Vec3& baseColor, float occlusion, float roughness, float metallic);
    uint32_t addOldStyleMaterial(const Vec3& diffuseColor, const Vec3& specularColor, float glossiness);
    uint32_t addDiffuseEmitterMaterial(const Vec3& emittance, float scale = 1.0f);
    uint32_t addDirectionalEmitterMaterial(const Vec3& emittance, float scale, const Vec3& direction);
    uint32_t addPointEmitterMaterial(const Vec3& intensity, float scale = 1.0f);
    uint32_t addMultiMaterial(std::span<const uint32_t> subMaterials);
    uint32_t addEnvironmentEmitterMaterial(const Vec3& emittance, float scale = 1.0f);

    uint32_t addLambertianMaterial(const Vec3& albedo);
    uint32_t addEmissiveMaterial(const Vec3& emission);
    uint32_t addMetalMaterial(const Vec3& albedo, float roughness);
    uint32_t addGlassMaterial(const Vec3& albedo, float ior);
    
    // Add lights
    void addPointLight(const PointLight& light);
    void addAreaLight(const AreaLight& light);
    
    // Finalize scene (build acceleration structures)
    void finalize();
    
private:
    friend class Renderer;
    friend class SceneAccessor;
    
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Renderer: Wavefront path tracing
class Renderer {
public:
    Renderer();
    ~Renderer();
    
    // Render scene
    void render(
        Scene* scene,
        const Camera& camera,
        Vec3* outputBuffer,
        uint32_t width,
        uint32_t height,
        uint32_t spp = 1,
        bool enableDenoiser = false
    );
    
private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace optixw
