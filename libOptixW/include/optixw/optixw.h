#pragma once

#include <optixw/types.h>
#include <memory>
#include <span>

namespace optixw {

// Forward declarations
class Scene;
class Renderer;

// RGB color type
struct RGB {
    float r, g, b;
    RGB() : r(0), g(0), b(0) {}
    RGB(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
};

// Camera parameters
struct Camera {
    RGB position;
    RGB target;
    RGB up;
    float fovY;
    float aspect;
};

// Point light
struct PointLight {
    RGB position;
    RGB intensity;
};

// Area light
struct AreaLight {
    RGB position;
    RGB normal;
    RGB tangent;
    RGB bitangent;
    float width;
    float height;
    RGB emission;
    bool doubleSided;
};

// Material type enum
enum class MaterialType : uint32_t {
    Lambertian = 0,
    Emissive = 1,
    Specular = 2
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
    uint32_t addLambertianMaterial(const RGB& albedo);
    uint32_t addEmissiveMaterial(const RGB& emission);
    
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
        RGB* outputBuffer,
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
