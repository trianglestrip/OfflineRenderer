#pragma once

#include "vlrw/vlrw.h"
#include "optix_context.h"

namespace vlrw {

class SceneImpl;
class ContextImpl;

class RendererImpl {
public:
    RendererImpl(ContextImpl* context);
    ~RendererImpl();

    void render(
        SceneImpl* scene,
        const Camera& camera,
        RGB* outputBuffer,
        uint32_t width,
        uint32_t height,
        uint32_t spp,
        bool enableDenoiser = false,
        int debugMode = 0);

private:
    void setupDenoiser(uint32_t width, uint32_t height);
    void denoise(CUdeviceptr d_beauty, uint32_t width, uint32_t height);
    void cleanupDenoiser();

    ContextImpl* m_context;
    TracePipelineHandle m_traceHandle;
    ShadowPipelineHandle m_shadowHandle;
    bool m_tracePipelineCreated = false;
    bool m_shadowPipelineCreated = false;

    // Denoiser
    OptixDenoiser m_denoiser = nullptr;
    CUdeviceptr d_denoiserState = 0;
    CUdeviceptr d_denoiserScratch = 0;
    uint32_t m_denoiserWidth = 0;
    uint32_t m_denoiserHeight = 0;
    bool m_denoiserCreated = false;
};

} // namespace vlrw
