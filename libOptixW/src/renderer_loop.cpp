#include "optixw/optixw.h"
#include "optixw/renderer_impl_public.h"
#include <vector>
#include <iostream>

// forward declaration of helper in renderer.cpp
namespace optixw {
std::vector<char> loadPTX(const char* filename);
}

using namespace optixw;

void Impl::renderSample(
    OptixTraversableHandle gasHandle,
    CUdeviceptr d_vertices,
    CUdeviceptr d_texcoords,
    CUdeviceptr d_indices,
    CUdeviceptr d_triangleMaterialIds,
    const CameraData& camera,
    uint32_t width,
    uint32_t height,
    uint32_t sampleIndex)
{
    std::cout << "[Debug] renderSample start" << std::endl;
    
    if (materialTextureBuffers.numMaterials == 0 || materialTextureBuffers.d_materials == 0) {
        std::cout << "[Warning] No materials available for shading" << std::endl;
        return;
    }
    
    std::cout << "[Debug] materials available: " << materialTextureBuffers.numMaterials << std::endl;

    // Initialize active rays (all pixels)
    std::vector<uint32_t> activeIndices(numPixels);
    for (uint32_t i = 0; i < numPixels; ++i) {
        activeIndices[i] = i;
    }
    
    CUDA_CHECK(cudaMemcpy(
        (void*)wavefrontBuffers.d_activeIndices,
        activeIndices.data(),
        numPixels * sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));
    
    CUdeviceptr activeIn = wavefrontBuffers.d_activeIndices;
    CUdeviceptr activeOut = wavefrontBuffers.d_compactIndices;
    uint32_t numActive = numPixels;
    
    // Wavefront rendering loop.
    // Shadow visibility uses extra trace stages, so allocate more iteration budget.
    const uint32_t maxDepth = 8;
    const uint32_t maxIterations = maxDepth * 2;
    const uint32_t blockSize = 256;
    for (uint32_t depth = 0; depth < maxIterations && numActive > 0; ++depth) {
        // Initialize all ray states to Trace with depth=0 for primary ray generation
        // Only initialize on the FIRST iteration of each sample
        if (depth == 0) {
            std::vector<RayState> hostRays(numPixels);
            for (uint32_t i = 0; i < numPixels; ++i) {
                hostRays[i].stage = RayState::Trace;
                hostRays[i].depth = 0;
                hostRays[i].pixelIndex = i;
                hostRays[i].initImportance = 1.0f;
                hostRays[i].radiance = make_float3(0.0f, 0.0f, 0.0f);
                hostRays[i].throughput = make_float3(1.0f, 1.0f, 1.0f);
                hostRays[i].pendingDirect = make_float3(0.0f, 0.0f, 0.0f);
                hostRays[i].nextOrigin = make_float3(0.0f, 0.0f, 0.0f);
                hostRays[i].nextDirection = make_float3(0.0f, 0.0f, 0.0f);
                hostRays[i].nextThroughput = make_float3(0.0f, 0.0f, 0.0f);
                hostRays[i].terminateAfterShadow = 0;
                hostRays[i].insideMedium = 0;
                hostRays[i].seed = (i * 1664525u + sampleIndex * 1013904223u) ^ 0x9e3779b9u;
                hostRays[i].prevBsdfPdf = 0.0f;
                hostRays[i].prevLightPdf = 0.0f;
                hostRays[i].prevDeltaSample = 0;
                hostRays[i].tMin = 0.001f;
                hostRays[i].tMax = 1e20f;
            }
            CUDA_CHECK(cudaMemcpy(
                (void*)wavefrontBuffers.d_rayPool,
                hostRays.data(),
                numPixels * sizeof(RayState),
                cudaMemcpyHostToDevice
            ));
        }
        
        LaunchParams launchParams = {};
        launchParams.traversable = gasHandle;
        
        // Set geometry data
        launchParams.geometry.vertices = reinterpret_cast<const float*>(d_vertices);
        launchParams.geometry.texcoords = reinterpret_cast<const float*>(materialTextureBuffers.d_texcoords);
        launchParams.geometry.indices = reinterpret_cast<const uint32_t*>(d_indices);
        launchParams.geometry.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
        
        // Set ray tracing buffers
        launchParams.renderBuffers.rayPool = reinterpret_cast<RayState*>(wavefrontBuffers.d_rayPool);
        launchParams.renderBuffers.activeIndices = reinterpret_cast<uint32_t*>(activeIn);
        launchParams.renderBuffers.hitBuffer = reinterpret_cast<HitInfo*>(wavefrontBuffers.d_hitBuffer);
        
        launchParams.camera = camera;
        launchParams.width = width;
        launchParams.height = height;
        launchParams.sampleIndex = sampleIndex;
        launchParams.numActive = numActive;
        
        // Set environment data
        launchParams.environment.environmentRadiance = environmentData.radiance;
        launchParams.environment.environmentMap = reinterpret_cast<const float4*>(environmentData.map.d_map);
        launchParams.environment.environmentMapWidth = environmentData.map.width;
        launchParams.environment.environmentMapHeight = environmentData.map.height;
        launchParams.environment.environmentMapScale = environmentData.map.scale;
        
        CUDA_CHECK(cudaMemcpy(
            (void*)launchParamsBuffer.d_launchParams,
            &launchParams,
            sizeof(LaunchParams),
            cudaMemcpyHostToDevice
        ));
        
        // Launch OptiX trace
        OPTIX_CHECK(optixLaunch(
            pipeline,
            0,  // CUDA stream
            launchParamsBuffer.d_launchParams,
            sizeof(LaunchParams),
            &sbt,
            numActive,
            1,
            1
        ));
        
        // Synchronize to ensure OptiX launch completes before shade kernel
        CUDA_CHECK(cudaDeviceSynchronize());

        const uint32_t numBlocks = (numActive + blockSize - 1) / blockSize;

        // Allocate device memory for shade parameters
        ShadeKernelParams* d_shadeParams;
        CUDA_CHECK(cudaMalloc(&d_shadeParams, sizeof(ShadeKernelParams)));
        
        ShadeKernelParams shadeParams = {};
        // Set render buffers
        shadeParams.renderBuffers.rayPool = reinterpret_cast<RayState*>(wavefrontBuffers.d_rayPool);
        shadeParams.renderBuffers.activeIndices = reinterpret_cast<const uint32_t*>(activeIn);
        shadeParams.renderBuffers.hitBuffer = reinterpret_cast<const HitInfo*>(wavefrontBuffers.d_hitBuffer);
        shadeParams.renderBuffers.accumBuffer = reinterpret_cast<float3*>(accumulationBuffers.d_accumBuffer);
        shadeParams.renderBuffers.albedoBuffer = reinterpret_cast<float3*>(accumulationBuffers.d_albedoBuffer);
        shadeParams.renderBuffers.normalBuffer = reinterpret_cast<float3*>(accumulationBuffers.d_normalBuffer);
        shadeParams.renderBuffers.numActive = numActive;
        
        // Set geometry buffers
        shadeParams.geometry.vertices = reinterpret_cast<const float*>(d_vertices);
        shadeParams.geometry.texcoords = reinterpret_cast<const float*>(materialTextureBuffers.d_texcoords);
        shadeParams.geometry.indices = reinterpret_cast<const uint32_t*>(d_indices);
        shadeParams.geometry.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
        
        // Set material and texture data
        shadeParams.materialsTextures.materials = reinterpret_cast<const MaterialData*>(materialTextureBuffers.d_materials);
        shadeParams.materialsTextures.textures = reinterpret_cast<const Texture2DData*>(materialTextureBuffers.d_textures);
        shadeParams.materialsTextures.numMaterials = materialTextureBuffers.numMaterials;
        shadeParams.materialsTextures.numTextures = materialTextureBuffers.numTextures;
        shadeParams.materialsTextures.numTriangles = materialTextureBuffers.numTriangles;
        
        // Set lighting data
        shadeParams.lighting.pointLights = reinterpret_cast<const PointLightData*>(lightBuffers.d_pointLights);
        shadeParams.lighting.areaLights = reinterpret_cast<const AreaLightData*>(lightBuffers.d_areaLights);
        shadeParams.lighting.numPointLights = lightBuffers.numPointLights;
        shadeParams.lighting.numAreaLights = lightBuffers.numAreaLights;
        
        // Set environment mapping data
        shadeParams.environment.environmentMap = reinterpret_cast<const float4*>(environmentData.map.d_map);
        shadeParams.environment.environmentMapWidth = environmentData.map.width;
        shadeParams.environment.environmentMapHeight = environmentData.map.height;
        shadeParams.environment.environmentMapScale = environmentData.map.scale;
        shadeParams.environment.environmentRadiance = environmentData.radiance;

        // Copy parameters to device
        CUDA_CHECK(cudaMemcpy(d_shadeParams, &shadeParams, sizeof(ShadeKernelParams), cudaMemcpyHostToDevice));

        void* shadeArgs[] = { d_shadeParams };
        CU_CHECK(cuLaunchKernel(
            kernelFunctions.shadeKernel,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            shadeArgs,
            nullptr
        ));

        // Free device memory
        CUDA_CHECK(cudaFree(d_shadeParams));

        CUDA_CHECK(cudaMemset((void*)wavefrontBuffers.d_compactCounter, 0, sizeof(uint32_t)));

        // Allocate device memory for compact parameters
        CompactKernelParams* d_compactParams;
        CUDA_CHECK(cudaMalloc(&d_compactParams, sizeof(CompactKernelParams)));
        
        CompactKernelParams compactParams = {};
        compactParams.rayPool = reinterpret_cast<const RayState*>(wavefrontBuffers.d_rayPool);
        compactParams.activeIndicesIn = reinterpret_cast<const uint32_t*>(activeIn);
        compactParams.activeIndicesOut = reinterpret_cast<uint32_t*>(activeOut);
        compactParams.counter = reinterpret_cast<uint32_t*>(wavefrontBuffers.d_compactCounter);
        compactParams.numActive = numActive;

        // Copy parameters to device
        CUDA_CHECK(cudaMemcpy(d_compactParams, &compactParams, sizeof(CompactKernelParams), cudaMemcpyHostToDevice));

        void* compactArgs[] = { d_compactParams };
        CU_CHECK(cuLaunchKernel(
            kernelFunctions.compactKernel,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            compactArgs,
            nullptr
        ));

        // Free device memory
        CUDA_CHECK(cudaFree(d_compactParams));

        CUDA_CHECK(cudaMemcpy(
            &numActive,
            (void*)wavefrontBuffers.d_compactCounter,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ));

        CUdeviceptr temp = activeIn;
    activeIn = activeOut;
    activeOut = temp;
}

// After all iterations, make sure all terminated rays have their radiance accumulated
// We need to run a final pass to accumulate radiance from all rays
{
    // 不再需要final pass，因为在shade内核中，当光线被标记为Terminated时已经累积了辐射度
    // 移除final pass以避免重复累积
    CUDA_CHECK(cudaDeviceSynchronize());
}
}
