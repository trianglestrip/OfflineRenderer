#pragma once

#include "common.h"
#include "basic_types.h"

#include <span>
#include <functional>

// Forward declaration for PIMPL
struct ContextImpl;
struct SceneImpl;
struct RendererImpl;

namespace vlrm {
    class VLRM_CPP_API Context {
    private:
        ContextImpl* m_impl;  // PIMPL idiom
        
    public:
        Context();
        ~Context();

        void initialize();
        void finalize();
        
        OptixDeviceContext getOptixDeviceContext() const;
    
    private:
        static void optixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata);
    };

    class VLRM_CPP_API Scene {
    private:
        SceneImpl* m_impl;  // PIMPL idiom
        
    public:
        Scene(Context* context);
        ~Scene();

        void addTriangleMesh(
            std::span<const float> vertices,
            std::span<const uint32_t> indices,
            uint32_t materialId
        );

        void addMaterial(
            const RGB& albedo,
            const RGB& emission = RGB::Zero()
        );

        void buildAccelerationStructure();
    };

    class VLRM_CPP_API Renderer {
    private:
        RendererImpl* m_impl;  // PIMPL idiom
        
    public:
        Renderer(Context* context, Scene* scene);
        ~Renderer();

        void setCamera(
            const Point3D& position,
            const Vector3D& forward,
            const Vector3D& up,
            float fovY
        );

        void render(
            uint32_t width,
            uint32_t height,
            uint32_t spp,
            uint32_t maxDepth,
            RGB* outputBuffer
        );
    };
}
