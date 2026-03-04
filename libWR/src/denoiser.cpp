#include "denoiser.h"
#include "utils/cuda_utils.h"
#include <optix_stubs.h>
#include <optix_host.h>
#include <iostream>
#include <stdexcept>

namespace wr {

// Forward declaration
extern OptixDeviceContext getOptixContext();

Denoiser::~Denoiser() {
    cleanup();
}

void Denoiser::initialize(uint32_t width, uint32_t height, const DenoiserConfig& config) {
    m_width = width;
    m_height = height;
    m_config = config;
    
    OptixDeviceContext context = getOptixContext();
    
    // Create denoiser options based on config
    OptixDenoiserOptions options = {};
    options.guideAlbedo = m_config.useAlbedo ? 1 : 0;
    options.guideNormal = m_config.useNormal ? 1 : 0;
    
    // Create OptiX denoiser
    OPTIX_CHECK(optixDenoiserCreate(
        context,
        OPTIX_DENOISER_MODEL_KIND_HDR,
        &options,
        &m_denoiser
    ));
    
    // Get denoiser memory requirements
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(
        m_denoiser,
        m_width,
        m_height,
        &m_denoiserSizes
    ));
    
    // Allocate memory
    size_t stateSize = m_denoiserSizes.stateSizeInBytes;
    size_t scratchSize = m_denoiserSizes.withoutOverlapScratchSizeInBytes;
    size_t bufferSize = m_width * m_height * sizeof(float3);
    
    CUDA_MALLOC(&m_d_denoiserState, stateSize);
    CUDA_MALLOC(&m_d_denoiserScratch, scratchSize);
    CUDA_MALLOC(&m_d_denoiseInput, bufferSize);
    CUDA_MALLOC(&m_d_denoiseOutput, bufferSize);
    
    if (m_config.useAlbedo) {
        CUDA_MALLOC(&m_d_denoiseAlbedo, bufferSize);
    }
    if (m_config.useNormal) {
        CUDA_MALLOC(&m_d_denoiseNormal, bufferSize);
    }
    
    // Setup denoiser
    OPTIX_CHECK(optixDenoiserSetup(
        m_denoiser,
        nullptr, // Use default stream
        m_width,
        m_height,
        m_d_denoiserState,
        stateSize,
        m_d_denoiserScratch,
        scratchSize
    ));
    
    std::cout << "[Denoiser] Initialized for " << m_width << "x" << m_height << " resolution" << std::endl;
}

void Denoiser::denoise(CUdeviceptr input, CUdeviceptr output) {
    // Copy input data
    size_t bufferSize = m_width * m_height * sizeof(float3);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_d_denoiseInput),
        reinterpret_cast<void*>(input),
        bufferSize,
        cudaMemcpyDeviceToDevice
    ));
    
    // Prepare denoiser params
    OptixDenoiserParams denoiserParams = {};
    // hdrIntensity is a CUdeviceptr, set to null for auto exposure
    denoiserParams.hdrIntensity = 0;
    denoiserParams.blendFactor = 0.0f;
    
    // Prepare guide layer (empty if not using albedo/normal)
    OptixDenoiserGuideLayer guideLayer = {};
    if (m_config.useAlbedo && m_d_denoiseAlbedo) {
        guideLayer.albedo.data = m_d_denoiseAlbedo;
        guideLayer.albedo.width = m_width;
        guideLayer.albedo.height = m_height;
        guideLayer.albedo.rowStrideInBytes = m_width * sizeof(float3);
        guideLayer.albedo.pixelStrideInBytes = sizeof(float3);
        guideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    }
    if (m_config.useNormal && m_d_denoiseNormal) {
        guideLayer.normal.data = m_d_denoiseNormal;
        guideLayer.normal.width = m_width;
        guideLayer.normal.height = m_height;
        guideLayer.normal.rowStrideInBytes = m_width * sizeof(float3);
        guideLayer.normal.pixelStrideInBytes = sizeof(float3);
        guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    }
    
    // Prepare input/output layer
    OptixDenoiserLayer layer = {};
    layer.input.data = m_d_denoiseInput;
    layer.input.width = m_width;
    layer.input.height = m_height;
    layer.input.rowStrideInBytes = m_width * sizeof(float3);
    layer.input.pixelStrideInBytes = sizeof(float3);
    layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    
    layer.output.data = m_d_denoiseOutput;
    layer.output.width = m_width;
    layer.output.height = m_height;
    layer.output.rowStrideInBytes = m_width * sizeof(float3);
    layer.output.pixelStrideInBytes = sizeof(float3);
    layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    
    // Execute denoising
    OPTIX_CHECK(optixDenoiserInvoke(
        m_denoiser,
        nullptr, // Use default stream
        &denoiserParams,
        m_d_denoiserState,
        m_denoiserSizes.stateSizeInBytes,
        &guideLayer,
        &layer,
        1, // Process one layer
        0, // inputOffsetX
        0, // inputOffsetY
        m_d_denoiserScratch,
        m_denoiserSizes.withoutOverlapScratchSizeInBytes
    ));
    
    // Copy result to output buffer
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(output),
        reinterpret_cast<void*>(m_d_denoiseOutput),
        bufferSize,
        cudaMemcpyDeviceToDevice
    ));
    
    std::cout << "[Denoiser] Denoising completed" << std::endl;
}

void Denoiser::cleanup() {
    CUDA_FREE(m_d_denoiserState);
    CUDA_FREE(m_d_denoiserScratch);
    CUDA_FREE(m_d_denoiseInput);
    CUDA_FREE(m_d_denoiseOutput);
    CUDA_FREE(m_d_denoiseAlbedo);
    CUDA_FREE(m_d_denoiseNormal);
    OPTIX_DESTROY(m_denoiser, optixDenoiserDestroy);
    
    m_width = 0;
    m_height = 0;
}

} // namespace wr
