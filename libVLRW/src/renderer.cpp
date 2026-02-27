#include "renderer.h"
#include "optix_context.h"
#include "scene.h"
#include "cuda_check.h"
#include "../gpu/wavefront_types.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <iostream>

namespace vlrw {

// Host-side types matching GPU types
struct float3_rgb_host { float r, g, b; };

// CameraParams for kernel launch
namespace wpt {
    struct CameraParams {
        float3 position;
        float3 target;
        float3 up;
        float fov;
        uint32_t width;
        uint32_t height;
    };
}

RendererImpl::RendererImpl(ContextImpl* context)
    : m_context(context)
{
}

RendererImpl::~RendererImpl() = default;

void RendererImpl::render(
    SceneImpl* scene, const Camera& camera, RGB* outputBuffer,
    uint32_t width, uint32_t height, uint32_t spp)
{
    std::cout << "[Renderer] Starting render " << width << "x" << height << std::endl;
    
    const uint32_t numPixels = width * height;
    const uint32_t maxDepth = 8;
    
    std::cout << "[Renderer] Allocating GPU buffers..." << std::endl;
    
    // 1. Allocate GPU buffers (rayPool, queue, accum)
    CUdeviceptr d_rayPool;
    CUdeviceptr d_activeQueue;
    CUdeviceptr d_accumBuffer;
    
    const size_t rayStateSize = 128;  // Approximate size of wpt::RayState
    CUDA_CHECK(cuMemAlloc(&d_rayPool, numPixels * rayStateSize));
    CUDA_CHECK(cuMemAlloc(&d_activeQueue, numPixels * sizeof(uint32_t)));
    CUDA_CHECK(cuMemAlloc(&d_accumBuffer, numPixels * sizeof(float3_rgb_host)));
    CUDA_CHECK(cuMemsetD8(d_accumBuffer, 0, numPixels * sizeof(float3_rgb_host)));
    
    std::cout << "[Renderer] GPU buffers allocated" << std::endl;
    
    // 2. Generate primary rays
    wpt::CameraParams camParams;
    camParams.position = make_float3(camera.position[0], camera.position[1], camera.position[2]);
    camParams.target = make_float3(camera.target[0], camera.target[1], camera.target[2]);
    camParams.up = make_float3(camera.up[0], camera.up[1], camera.up[2]);
    camParams.fov = camera.fovY;
    camParams.width = width;
    camParams.height = height;
    
    std::cout << "Rendering " << width << "x" << height << " image..." << std::endl;
    
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (numPixels + blockSize - 1) / blockSize;
    
    // 2. Call external CUDA kernels (compiled into libVLRW)
    // Declare external functions
    extern "C" void wpt_launchGeneratePrimaryRays(
        wpt::RayState* rayPool,
        uint32_t* activeQueue,
        const wpt::CameraParams& camera,
        uint32_t numPixels,
        unsigned int gridSize,
        unsigned int blockSize);
    
    extern "C" void wpt_launchVisualizeRays(
        const wpt::RayState* rayPool,
        const uint32_t* activeQueue,
        float3_rgb_host* accumBuffer,
        uint32_t numActive,
        unsigned int gridSize,
        unsigned int blockSize);
    
    // Generate primary rays
    wpt_launchGeneratePrimaryRays(
        reinterpret_cast<wpt::RayState*>(d_rayPool),
        reinterpret_cast<uint32_t*>(d_activeQueue),
        camParams,
        numPixels,
        gridSize,
        blockSize);
    CUDA_CHECK(cuCtxSynchronize());
    std::cout << "Generated " << numPixels << " primary rays" << std::endl;
    
    // Visualize ray directions
    wpt_launchVisualizeRays(
        reinterpret_cast<const wpt::RayState*>(d_rayPool),
        reinterpret_cast<const uint32_t*>(d_activeQueue),
        reinterpret_cast<float3_rgb_host*>(d_accumBuffer),
        numPixels,
        gridSize,
        blockSize);
    CUDA_CHECK(cuCtxSynchronize());
    std::cout << "Ray directions visualized" << std::endl;
    
    // 4. Copy accumBuffer to CPU
    std::vector<float3_rgb_host> hostAccum(numPixels);
    CUDA_CHECK(cuMemcpyDtoH(hostAccum.data(), d_accumBuffer, numPixels * sizeof(float3_rgb_host)));
    
    // 5. Convert to output buffer
    for (uint32_t i = 0; i < numPixels; ++i) {
        outputBuffer[i][0] = hostAccum[i].r;
        outputBuffer[i][1] = hostAccum[i].g;
        outputBuffer[i][2] = hostAccum[i].b;
    }
    
    // 6. Free buffers
    CUDA_CHECK(cuMemFree(d_rayPool));
    CUDA_CHECK(cuMemFree(d_activeQueue));
    CUDA_CHECK(cuMemFree(d_accumBuffer));
}

Renderer::~Renderer() {
    delete m_impl;
}

void Renderer::render(
    Scene* scene, const Camera& camera, RGB* outputBuffer,
    uint32_t width, uint32_t height, uint32_t spp)
{
    m_impl->render(scene->m_impl, camera, outputBuffer, width, height, spp);
}

} // namespace vlrw
