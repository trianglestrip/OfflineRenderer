#pragma once

#include "optixw/optixw.h"
#include "optixw/types.h"
#include "optixw/launch_params.h"
#include "optixw/wavefront_kernel_params.h"
#include "scene_internal.h"
#include "checks.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <filesystem>

namespace optixw {

class Impl {
public:
    // OptiX pipeline
    OptixModule traceModule;
    OptixPipeline pipeline;
    OptixProgramGroup raygenPG;
    OptixProgramGroup missPG;
    OptixProgramGroup hitgroupPG;
    OptixShaderBindingTable sbt;

    // Wavefront queues
    CUdeviceptr d_rayPool;
    CUdeviceptr d_activeIndices;
    CUdeviceptr d_compactIndices;
    CUdeviceptr d_accumBuffer;
    // guidance buffers used by the denoiser
    CUdeviceptr d_albedoBuffer;
    CUdeviceptr d_normalBuffer;
    CUdeviceptr d_hitBuffer;
    CUdeviceptr d_compactCounter;
    CUdeviceptr d_materials;
    CUdeviceptr d_texcoords;
    CUdeviceptr d_textures;

    // Launch parameters
    CUdeviceptr d_launchParams;

    // CUDA kernels
    CUmodule shadeModule;
    CUmodule compactModule;
    CUfunction shadeKernel;
    CUfunction compactKernel;
    CUmodule scaleModule;
    CUfunction scaleKernel;
    // merge kernels for tiled denoiser overlap-weighted blending
    CUmodule mergeModule;
    CUfunction mergeKernel;
    CUfunction normalizeKernel;

    // OptiX denoiser
    OptixDenoiser denoiser;
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
    // accumulation buffers used to merge overlapping denoised tiles
    CUdeviceptr d_mergeAccum; // float3 per-pixel
    CUdeviceptr d_mergeWeight; // float per-pixel
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

    // lifecycle
    Impl();
    ~Impl();

    // methods moved to separate compilation units
    void createPipeline();
    void createWavefrontKernels();
    void buildSBT();
    void renderSample(
        OptixTraversableHandle gasHandle,
        CUdeviceptr d_vertices,
        CUdeviceptr d_texcoords,
        CUdeviceptr d_indices,
        CUdeviceptr d_triangleMaterialIds,
        const CameraData& camera,
        uint32_t width,
        uint32_t height,
        uint32_t sampleIndex);

    // buffer management (left in renderer.cpp)
    void allocateBuffers(uint32_t width, uint32_t height);
    void ensureDenoiserBuffers(uint32_t width, uint32_t height,
                               uint32_t tileWidth, uint32_t tileHeight);
};

} // namespace optixw
