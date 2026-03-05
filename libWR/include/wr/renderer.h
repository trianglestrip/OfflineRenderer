#pragma once

#include "wr/types.h"
#include "wr/scene.h"
#include <cstdint>

namespace wr {

// Renderer configuration
struct RendererConfig {
    uint32_t maxBounces = 8;
    bool useNEE = true;  // Next Event Estimation
    float russianRouletteDepth = 3;
};

// Denoiser configuration
struct DenoiserConfig {
    bool enabled = false;
    bool useAlbedo = false;
    bool useNormal = false;
    float hdrIntensity = 1.0f;
};

// Render parameters
struct RenderParams {
    uint32_t width;
    uint32_t height;
    uint32_t spp = 1;  // Samples per pixel
    DenoiserConfig denoiser;
    
    // Advanced rendering options
    bool useNEE = true;                  // Next Event Estimation
    uint32_t maxBounces = 8;
    float russianRouletteDepth = 3.0f;
};

// Forward declaration
struct RendererImpl;

// Renderer performs path tracing
class Renderer {
public:
    Renderer(const RendererConfig& config = {});
    ~Renderer();

    // Main render function
    void render(Scene* scene,
                const Camera& camera,
                Vec3* outputBuffer,
                const RenderParams& params);
    
    // Convenience overload (backward compatible)
    void render(Scene* scene,
                const Camera& camera,
                Vec3* outputBuffer,
                uint32_t width,
                uint32_t height,
                uint32_t spp = 1,
                bool denoiser = false);

private:
    RendererImpl* m_impl;
    RendererConfig m_config;
};

} // namespace wr
