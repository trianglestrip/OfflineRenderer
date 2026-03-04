#pragma once

#include <span>
#include <cstdint>
#include <optix.h>
#include <cuda.h>

namespace wr {

struct Vec3 {
    float x{0}, y{0}, z{0};
    Vec3() = default;
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};

struct Camera {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    float fovY{0};
    float aspect{0};
};

struct SceneImpl;

class Scene {
public:
    Scene();
    ~Scene();

    uint32_t addLambertianMaterial(const Vec3&);
    uint32_t addEmissiveMaterial(const Vec3&);
    uint32_t addGlassMaterial(const Vec3&, float);

    void addTriangleMesh(std::span<const float> vertices,
                         std::span<const uint32_t> indices,
                         uint32_t materialId);

    void setEnvironmentRadiance(const Vec3&);
    void finalize();
    
    CUdeviceptr getVerticesBuffer() const;
    CUdeviceptr getIndicesBuffer() const;
    CUdeviceptr getTriangleMaterialIdsBuffer() const;
    CUdeviceptr getMaterialsBuffer() const;
    OptixTraversableHandle getGASHandle() const;
    uint32_t getNumMaterials() const;
    Vec3 getEnvironmentRadiance() const;

private:
    SceneImpl* m_impl;
};

struct RendererImpl;

class Renderer {
public:
    Renderer();
    ~Renderer();

    void render(Scene* scene,
                const Camera& camera,
                Vec3* outputBuffer,
                uint32_t width,
                uint32_t height,
                uint32_t spp = 1,
                bool denoiser = false);

private:
    RendererImpl* m_impl;
};

class Context {
public:
    Context(int deviceId = 0);
    ~Context();

    Scene* createScene();
    Renderer* createRenderer();
};

} // namespace wr
