#pragma once

#include "optixw/types.h"
#include <optix.h>
#include <cuda.h>
#include <vector>
#include <memory>

namespace optixw {

// Forward declaration
class TaskScheduler;

// Implementation class forward declaration
class Impl {
public:
    // Constructor
    Impl();

    // Destructor  
    ~Impl();

    // Method to create the OptiX pipeline
    void createPipeline();

    // Method to render a single sample
    void renderSample(
        OptixTraversableHandle gasHandle,
        CUdeviceptr d_vertices,
        CUdeviceptr d_texcoords,
        CUdeviceptr d_indices,
        CUdeviceptr d_triangleMaterialIds,
        const CameraData& camData,
        uint32_t width,
        uint32_t height,
        uint32_t sample);

    // Allocate buffers for rendering
    void allocateBuffers(uint32_t width, uint32_t height);

    // Ensure denoiser buffers
    void ensureDenoiserBuffers(uint32_t width, uint32_t height,
                               uint32_t tileWidth, uint32_t tileHeight);

    // Other members...
    OptixModule traceModule;
    OptixPipeline pipeline;
    OptixProgramGroup raygenPG;
    OptixProgramGroup missPG;
    OptixProgramGroup hitgroupPG;
    OptixShaderBindingTable sbt;

    // Ray tracing buffers
    RayTracingBuffers wavefrontBuffers;
    
    // Kernel modules and functions
    KernelModules kernelModules;
    KernelFunctions kernelFunctions;
    
    // Launch params buffer
    LaunchParamsBuffer launchParamsBuffer;
    
    // Accumulation buffers
    RenderAccumBuffers accumulationBuffers;
    
    // Material and texture buffers
    MaterialTextureBuffers materialTextureBuffers;
    
    // Environment data
    EnvironmentData environmentData;
    
    // Light buffers
    LightBuffers lightBuffers;
    
    // Denoiser
    DenoiserState denoiserState;
    
    // Scene data counts
    SceneCounts sceneCounts;
    
    // Dimensions and counters
    uint32_t numPixels;
    uint32_t maxRays;
    
    bool pipelineCreated;
    
    // Task scheduler for CPU-side parallelism
    TaskScheduler* scheduler;
};

} // namespace optixw