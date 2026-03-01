#pragma once

// disable warning about characters outside current code page
#pragma warning(push)
#pragma warning(disable:4819)

#include <optixw/types.h>
#include <optixw/core/math_types.h>
#include <optixw/core/task_scheduler.h>
#include <memory>
#include <span>

namespace optixw {

// Global OptiX device context (defined in context.cpp)
extern OptixDeviceContext g_optixContext;

// Pimpl implementation type (defined in renderer_impl.h)
class Impl;

// Forward declarations
class Scene;
class Renderer;

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
    
    // Accessors
    TaskScheduler* getTaskScheduler() { return m_taskScheduler.get(); }
    
private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
    std::unique_ptr<TaskScheduler> m_taskScheduler;
};

// Scene: Geometry and material management
class Scene {
public:
    explicit Scene(TaskScheduler* scheduler = nullptr);
    ~Scene();
    
    // Add geometry
    void addTriangleMesh(
        std::span<const float> vertices,
        std::span<const uint32_t> indices,
        uint32_t materialId
    );
    void addTriangleMeshWithTexcoords(
        std::span<const float> vertices,
        std::span<const float> texcoords,
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
    uint32_t loadTexture2D(const char* filePath, bool sRGB = true);
    void setMaterialBaseColorTexture(uint32_t materialId, uint32_t textureId);
    void setEnvironmentRadiance(const Vec3& radiance);
    void setEnvironmentMap(const char* filePath, float scale = 1.0f);
    
    // Add lights
    void addPointLight(const PointLight& light);
    void addAreaLight(const AreaLight& light);
    
    // Finalize scene (build acceleration structures)
    void finalize();
    
    // ==================== 结构体版本的方法 ====================
    uint32_t addMaterial(const MaterialParams& params);
    
    void addTriangleMesh(const TriangleMeshParams& params);
    void addTriangleMeshWithTexcoords(const TriangleMeshParams& params);
    uint32_t loadTexture2D(const TextureLoadParams& params);
    void setEnvironmentMap(const EnvironmentMapParams& params);
    
private:
    friend class Renderer;
    friend class SceneAccessor;
    
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Renderer: Wavefront path tracing
class Renderer {
public:
    explicit Renderer(TaskScheduler* scheduler = nullptr);
    ~Renderer();
    
    // Render scene
    // Render the provided scene into `outputBuffer` (width*height pixels).
    // spp: samples per pixel.  enableDenoiser toggles the OptiX denoiser, and
    // blendFactor is passed through to `OptixDenoiserParams.blendFactor` to
    // mix the original beauty buffer with the denoised result (0.0 = full
    // denoiser output, 1.0 = no denoising).  Useful when the network is over‑
    // aggressive at higher sample counts.
    void render(
        Scene* scene,
        const Camera& camera,
        Vec3* outputBuffer,
        uint32_t width,
        uint32_t height,
        uint32_t spp = 1,
        bool enableDenoiser = false,
        float denoiserBlend = 0.0f,
            bool enableTiling = false,
        uint32_t tileWidth = 0,
        uint32_t tileHeight = 0
    );
    
    void render(
        Scene* scene,
        const Camera& camera,
        Vec3* outputBuffer,
        uint32_t width,
        uint32_t height,
        const RenderConfig& config);
    
    void render(const RenderParams& params);
    
private:
    std::unique_ptr<Impl> m_impl;
};

} // namespace optixw

#pragma warning(pop)
