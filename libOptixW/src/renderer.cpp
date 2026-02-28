#include "optixw/optixw.h"
#include "optixw/types.h"
#include "scene_internal.h"
#include "cpu_vector_math.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            throw std::runtime_error(                                          \
                std::string("OptiX call failed: ") +                           \
                optixGetErrorName(res) + " (" +                                \
                optixGetErrorString(res) + ")");                               \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            throw std::runtime_error(                                          \
                std::string("CUDA call failed: ") +                            \
                cudaGetErrorString(error));                                    \
        }                                                                      \
    } while (0)

namespace optixw {

// Forward declarations
extern OptixDeviceContext g_optixContext;

// Load PTX file
static std::vector<char> loadPTX(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("Failed to open PTX file: ") + filename);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> data(size);
    file.read(data.data(), size);
    
    return data;
}

// Renderer implementation
class Renderer::Impl {
public:
    // OptiX pipeline
    OptixModule traceModule = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;
    OptixShaderBindingTable sbt = {};
    
    // Wavefront queues
    CUdeviceptr d_rayPool = 0;
    CUdeviceptr d_activeIndices = 0;
    CUdeviceptr d_accumBuffer = 0;
    CUdeviceptr d_hitBuffer = 0;
    
    // Launch parameters
    CUdeviceptr d_launchParams = 0;
    
    uint32_t numPixels = 0;
    uint32_t maxRays = 0;
    
    bool pipelineCreated = false;
    
#include "renderer_pipeline.inl"
#include "renderer_loop.inl"
    
    Impl() {}
    
    ~Impl() {
        if (raygenPG) optixProgramGroupDestroy(raygenPG);
        if (missPG) optixProgramGroupDestroy(missPG);
        if (hitgroupPG) optixProgramGroupDestroy(hitgroupPG);
        if (pipeline) optixPipelineDestroy(pipeline);
        if (traceModule) optixModuleDestroy(traceModule);
        
        if (sbt.raygenRecord) cudaFree((void*)sbt.raygenRecord);
        if (sbt.missRecordBase) cudaFree((void*)sbt.missRecordBase);
        if (sbt.hitgroupRecordBase) cudaFree((void*)sbt.hitgroupRecordBase);
        
        if (d_rayPool) cudaFree((void*)d_rayPool);
        if (d_activeIndices) cudaFree((void*)d_activeIndices);
        if (d_accumBuffer) cudaFree((void*)d_accumBuffer);
        if (d_hitBuffer) cudaFree((void*)d_hitBuffer);
        if (d_launchParams) cudaFree((void*)d_launchParams);
    }
    
    void allocateBuffers(uint32_t width, uint32_t height) {
        numPixels = width * height;
        maxRays = numPixels;  // One ray per pixel for now
        
        size_t rayPoolSize = maxRays * sizeof(RayState);
        size_t indicesSize = maxRays * sizeof(uint32_t);
        size_t accumSize = numPixels * sizeof(float3);
        size_t hitBufferSize = maxRays * sizeof(HitInfo);
        
        if (d_rayPool) cudaFree((void*)d_rayPool);
        if (d_activeIndices) cudaFree((void*)d_activeIndices);
        if (d_accumBuffer) cudaFree((void*)d_accumBuffer);
        if (d_hitBuffer) cudaFree((void*)d_hitBuffer);
        if (d_launchParams) cudaFree((void*)d_launchParams);
        
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayPool), rayPoolSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_activeIndices), indicesSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_accumBuffer), accumSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitBuffer), hitBufferSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launchParams), 4096));  // Large enough for any params
        
        // Initialize accumulation buffer to zero
        CUDA_CHECK(cudaMemset((void*)d_accumBuffer, 0, accumSize));
        
        std::cout << "[Renderer] Allocated buffers: " 
                  << numPixels << " pixels, " 
                  << maxRays << " rays" << std::endl;
    }
};

Renderer::Renderer() : m_impl(std::make_unique<Impl>()) {}
Renderer::~Renderer() = default;

void Renderer::render(
    Scene* scene,
    const Camera& camera,
    RGB* outputBuffer,
    uint32_t width,
    uint32_t height,
    uint32_t spp,
    bool enableDenoiser)
{
    if (!scene) {
        throw std::runtime_error("Scene is null");
    }
    
    std::cout << "[Renderer] Starting render: " 
              << width << "x" << height 
              << " @ " << spp << " spp" << std::endl;
    
    // Create pipeline if needed
    if (!m_impl->pipelineCreated) {
        m_impl->createPipeline();
    }
    
    // Allocate buffers
    m_impl->allocateBuffers(width, height);
    
    // Setup camera data
    CameraData camData;
    camData.position = cpu_math::make_float3(camera.position.r, camera.position.g, camera.position.b);

    float3 target = cpu_math::make_float3(camera.target.r, camera.target.g, camera.target.b);
    camData.forward = cpu_math::normalize(cpu_math::operator-(target, camData.position));

    float3 up = cpu_math::make_float3(camera.up.r, camera.up.g, camera.up.b);
    camData.right = cpu_math::normalize(cpu_math::cross(camData.forward, up));
    camData.up = cpu_math::cross(camData.right, camData.forward);
    
    camData.tanHalfFovY = tanf(camera.fovY * 0.5f);
    camData.aspect = camera.aspect;
    
    // Get scene data
    OptixTraversableHandle gasHandle = SceneAccessor::getGasHandle(scene);
    CUdeviceptr d_vertices = SceneAccessor::getVerticesPtr(scene);
    CUdeviceptr d_indices = SceneAccessor::getIndicesPtr(scene);
    CUdeviceptr d_triangleMaterialIds = SceneAccessor::getTriangleMaterialIdsPtr(scene);
    
    std::cout << "[Renderer] GAS handle: " << gasHandle << std::endl;
    
    // Render samples
    for (uint32_t sample = 0; sample < spp; ++sample) {
        std::cout << "\r[Renderer] Sample " << (sample + 1) << "/" << spp << std::flush;
        
        m_impl->renderSample(
            gasHandle,
            d_vertices,
            d_indices,
            d_triangleMaterialIds,
            camData,
            width,
            height,
            sample
        );
    }
    
    std::cout << std::endl;
    
    // Download result
    std::vector<float3> accumBuffer(m_impl->numPixels);
    CUDA_CHECK(cudaMemcpy(
        accumBuffer.data(),
        (void*)m_impl->d_accumBuffer,
        m_impl->numPixels * sizeof(float3),
        cudaMemcpyDeviceToHost
    ));
    
    // Average and copy to output
    float invSpp = 1.0f / spp;
    for (uint32_t i = 0; i < m_impl->numPixels; ++i) {
        outputBuffer[i].r = accumBuffer[i].x * invSpp;
        outputBuffer[i].g = accumBuffer[i].y * invSpp;
        outputBuffer[i].b = accumBuffer[i].z * invSpp;
    }
    
    std::cout << "[Renderer] Render complete" << std::endl;
}

} // namespace optixw
