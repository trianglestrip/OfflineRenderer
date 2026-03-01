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

    CUdeviceptr d_rayPool;
    CUdeviceptr d_activeIndices;
    CUdeviceptr d_compactIndices;
    CUdeviceptr d_accumBuffer;
    CUdeviceptr d_albedoBuffer;
    CUdeviceptr d_normalBuffer;
    CUdeviceptr d_hitBuffer;
    CUdeviceptr d_compactCounter;
    CUdeviceptr d_launchParams;
    
    // Kernel modules
    CUmodule shadeModule;
    CUmodule compactModule;
    CUmodule scaleModule;
    
    // Kernel functions
    CUfunction shadeKernel;
    CUfunction compactKernel;
    CUfunction scaleKernel;
    CUfunction mergeKernel;
    CUfunction normalizeKernel;

    // Denoiser
    OptixDenoiser* denoiser;
    CUdeviceptr d_denoiserState;
    CUdeviceptr d_denoiserScratch;
    CUdeviceptr d_denoiserInput;
    CUdeviceptr d_denoiserOutput;
    CUdeviceptr d_tileBuffer;
    CUdeviceptr d_tileInputBuffer;
    CUdeviceptr d_tileAlbedoInput;
    CUdeviceptr d_tileNormalInput;
    CUdeviceptr d_denoiserIntensity;
    CUdeviceptr d_denoiserAlbedoInput;
    CUdeviceptr d_denoiserNormalInput;
    CUdeviceptr d_mergeAccum;
    CUdeviceptr d_mergeWeight;

    size_t denoiserStateSize;
    size_t denoiserScratchSize;
    uint32_t denoiserWidth;
    uint32_t denoiserHeight;
    uint32_t denoiserOverlap;

    uint32_t numPixels;
    uint32_t maxRays;
    uint32_t numMaterials;
    uint32_t numTextures;
    uint32_t numTriangles;
    float3 environmentRadiance;
    CUdeviceptr d_environmentMap;
    uint32_t environmentMapWidth;
    uint32_t environmentMapHeight;
    float environmentMapScale;
    
    bool pipelineCreated;

    // Task scheduler for CPU-side parallelism
    TaskScheduler* scheduler;
};

} // namespace optixw