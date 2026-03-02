#pragma once

#include "optixw/optixw.h"
#include "optixw/types.h"
#include "optixw/launch_params.h"
#include "optixw/wavefront_kernel_params.h"
#include "scene_internal.h"
#include "../src/utils/checks.h"
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
    PipelineProgramGroups programGroups;
    OptixShaderBindingTable sbt;
    PipelineSBTRecords sbtRecords;

    // Wavefront queues
    WavefrontBuffers wavefrontBuffers;
    RenderAccumBuffers accumulationBuffers;
    MergeBuffers mergeBuffers;
    LaunchParamsBuffer launchParamsBuffer;
    MaterialTextureBuffers materialTextureBuffers;

    // CUDA kernels
    KernelModules kernelModules;
    KernelFunctions kernelFunctions;

    // OptiX denoiser
    DenoiserState denoiser;

    uint32_t numPixels;
    uint32_t maxRays;
    EnvironmentData environmentData;

    bool pipelineCreated;

    LightBuffers lightBuffers;

    // Task scheduler for CPU-side parallelism
    class TaskScheduler* scheduler;

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