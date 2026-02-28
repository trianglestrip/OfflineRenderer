#pragma once

#include <optixw/types.h>
#include <memory>
#include <span>

namespace optixw {

// Forward declarations
class Scene;
class Renderer;

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
