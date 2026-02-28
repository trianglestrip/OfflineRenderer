#pragma once

#include "types.h"
#include <cstdint>
#include <vector>
#include <string_view>
#include <span>

// libVLRW - VLR Wavefront Public API (C++20)
namespace vlrw {

// Forward declarations
class ContextImpl;
class SceneImpl;
class RendererImpl;
class Scene;
class Renderer;

// ---- Context ----
class Context {
public:
    static Context* create(void* cuContext);
    ~Context();

    Scene* createScene();
    Renderer* createRenderer();

private:
    Context(ContextImpl* impl) : m_impl(impl) {}
    ContextImpl* m_impl;
};

// ---- Material ----
struct MaterialDesc {
    RGB albedo;
    float roughness;
    float metallic;
    float ior;
    RGB emission;
};

// ---- Light ----
struct PointLightDesc {
    RGB position;
    RGB intensity;
};

struct AreaLightDesc {
    RGB position;   // center of the rectangle
    RGB normal;     // surface normal (emission direction)
    RGB tangent;   // tangent vector (defines width axis)
    float width;   // size along tangent
    float height;  // size along bitangent (cross(normal, tangent))
    RGB emission;  // radiance (W/sr/m^2)
    bool doubleSided = false;  // if true, emit from both sides
};

// ---- Scene ----
class Scene {
public:
    ~Scene();

    // C++20 zero-copy API: std::span for read-only views (Taskflow-friendly)
    void addTriangleMesh(
        std::span<const float> vertices,
        std::span<const uint32_t> indices,
        uint32_t materialId);
    
    // Explicit ownership transfer (backward compatibility)
    void addTriangleMeshMove(
        std::vector<float>&& vertices,
        std::vector<uint32_t>&& indices,
        uint32_t materialId);

    void setMaterial(uint32_t id, const MaterialDesc& desc);
    void addPointLight(const PointLightDesc& light);
    void addAreaLight(const AreaLightDesc& light);

    void finalize();

private:
    Scene(SceneImpl* impl) : m_impl(impl) {}
    SceneImpl* m_impl;
    friend class Context;
    friend class ContextImpl;
    friend class Renderer;
    friend class RendererImpl;
};

// ---- Camera ----
struct Camera {
    RGB position;
    RGB target;
    RGB up;
    float fovY;
    float aspect;
};

// ---- Renderer ----
class Renderer {
public:
    ~Renderer();

    void render(
        Scene* scene,
        const Camera& camera,
        RGB* outputBuffer,
        uint32_t width,
        uint32_t height,
        uint32_t spp = 1,
        bool enableDenoiser = false,
        int debugMode = 0);

private:
    Renderer(RendererImpl* impl) : m_impl(impl) {}
    RendererImpl* m_impl;
    friend class Context;
};

} // namespace vlrw
