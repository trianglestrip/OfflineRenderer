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
    if (materialTextureBuffers.numMaterials == 0 || materialTextureBuffers.d_materials == 0) {
        std::cout << "[Warning] No materials available for shading" << std::endl;
        return;
    }

    // Each sample must restart path state from primary rays.
    // 只在第一次迭代时清零 rayPool，而不是每次迭代都清零
    // CUDA_CHECK(cudaMemset((void*)wavefrontBuffers.d_rayPool, 0, maxRays * sizeof(RayState)));

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
        LaunchParams launchParams = {};
        launchParams.traversable = gasHandle;
        launchParams.vertices = reinterpret_cast<const float*>(d_vertices);
        launchParams.texcoords = reinterpret_cast<const float*>(materialTextureBuffers.d_texcoords);
        launchParams.indices = reinterpret_cast<const uint32_t*>(d_indices);
        launchParams.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
        launchParams.rayPool = reinterpret_cast<RayState*>(wavefrontBuffers.d_rayPool);
        launchParams.activeIndices = reinterpret_cast<uint32_t*>(activeIn);
        launchParams.hitBuffer = reinterpret_cast<HitInfo*>(wavefrontBuffers.d_hitBuffer);
        launchParams.camera = camera;
        launchParams.width = width;
        launchParams.height = height;
        launchParams.sampleIndex = sampleIndex;
        launchParams.numActive = numActive;
        launchParams.environmentRadiance = environmentRadiance;
        launchParams.environmentMap = reinterpret_cast<const float4*>(environmentMap.d_map);
        launchParams.environmentMapWidth = environmentMap.width;
        launchParams.environmentMapHeight = environmentMap.height;
        launchParams.environmentMapScale = environmentMap.scale;
        
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

        const uint32_t numBlocks = (numActive + blockSize - 1) / blockSize;

        ShadeKernelParams shadeParams = {};
        shadeParams.rayPool = reinterpret_cast<RayState*>(wavefrontBuffers.d_rayPool);
        shadeParams.activeIndices = reinterpret_cast<const uint32_t*>(activeIn);
        shadeParams.hitBuffer = reinterpret_cast<const HitInfo*>(wavefrontBuffers.d_hitBuffer);
        shadeParams.vertices = reinterpret_cast<const float*>(d_vertices);
        shadeParams.texcoords = reinterpret_cast<const float*>(materialTextureBuffers.d_texcoords);
        shadeParams.indices = reinterpret_cast<const uint32_t*>(d_indices);
        shadeParams.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
        shadeParams.materials = reinterpret_cast<const MaterialData*>(materialTextureBuffers.d_materials);
        shadeParams.textures = reinterpret_cast<const Texture2DData*>(materialTextureBuffers.d_textures);
        shadeParams.accumBuffer = reinterpret_cast<float3*>(accumBuffers.d_accumBuffer);
        // newly added guidance buffers - renderer will fill them before
        // invoking the denoiser if it is enabled.  They may remain null
        // when denoising is off.
        shadeParams.albedoBuffer = reinterpret_cast<float3*>(accumBuffers.d_albedoBuffer);
        shadeParams.normalBuffer = reinterpret_cast<float3*>(accumBuffers.d_normalBuffer);
        shadeParams.numTriangles = materialTextureBuffers.numTriangles;
        shadeParams.numMaterials = materialTextureBuffers.numMaterials;
        shadeParams.numTextures = materialTextureBuffers.numTextures;
        shadeParams.numActive = numActive;
        shadeParams.environmentRadiance = environmentRadiance;
        shadeParams.environmentMap = reinterpret_cast<const float4*>(environmentMap.d_map);
        shadeParams.environmentMapWidth = environmentMap.width;
        shadeParams.environmentMapHeight = environmentMap.height;
        shadeParams.environmentMapScale = environmentMap.scale;
        shadeParams.pointLights = reinterpret_cast<const PointLightData*>(lightBuffers.d_pointLights);
        shadeParams.areaLights = reinterpret_cast<const AreaLightData*>(lightBuffers.d_areaLights);
        shadeParams.numPointLights = lightBuffers.numPointLights;
        shadeParams.numAreaLights = lightBuffers.numAreaLights;

        void* shadeArgs[] = { &shadeParams };
        CU_CHECK(cuLaunchKernel(
            kernelFunctions.shadeKernel,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            shadeArgs,
            nullptr
        ));

        CUDA_CHECK(cudaMemset((void*)wavefrontBuffers.d_compactCounter, 0, sizeof(uint32_t)));

        CompactKernelParams compactParams = {};
        compactParams.rayPool = reinterpret_cast<const RayState*>(wavefrontBuffers.d_rayPool);
        compactParams.activeIndicesIn = reinterpret_cast<const uint32_t*>(activeIn);
        compactParams.activeIndicesOut = reinterpret_cast<uint32_t*>(activeOut);
        compactParams.counter = reinterpret_cast<uint32_t*>(wavefrontBuffers.d_compactCounter);
        compactParams.numActive = numActive;

        void* compactArgs[] = { &compactParams };
        CU_CHECK(cuLaunchKernel(
            kernelFunctions.compactKernel,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            compactArgs,
            nullptr
        ));

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
    const uint32_t numBlocks = (numPixels + blockSize - 1) / blockSize;
    
    ShadeKernelParams finalShadeParams = {};
    finalShadeParams.rayPool = reinterpret_cast<RayState*>(wavefrontBuffers.d_rayPool);
    finalShadeParams.activeIndices = reinterpret_cast<const uint32_t*>(wavefrontBuffers.d_activeIndices);
    finalShadeParams.hitBuffer = reinterpret_cast<const HitInfo*>(wavefrontBuffers.d_hitBuffer);
    finalShadeParams.vertices = reinterpret_cast<const float*>(d_vertices);
    finalShadeParams.texcoords = reinterpret_cast<const float*>(materialTextureBuffers.d_texcoords);
    finalShadeParams.indices = reinterpret_cast<const uint32_t*>(d_indices);
    finalShadeParams.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
    finalShadeParams.materials = reinterpret_cast<const MaterialData*>(materialTextureBuffers.d_materials);
    finalShadeParams.textures = reinterpret_cast<const Texture2DData*>(materialTextureBuffers.d_textures);
    finalShadeParams.accumBuffer = reinterpret_cast<float3*>(accumBuffers.d_accumBuffer);
    finalShadeParams.albedoBuffer = reinterpret_cast<float3*>(accumBuffers.d_albedoBuffer);
    finalShadeParams.normalBuffer = reinterpret_cast<float3*>(accumBuffers.d_normalBuffer);
    finalShadeParams.numTriangles = materialTextureBuffers.numTriangles;
    finalShadeParams.numMaterials = materialTextureBuffers.numMaterials;
    finalShadeParams.numTextures = materialTextureBuffers.numTextures;
    finalShadeParams.numActive = numPixels;
    finalShadeParams.environmentRadiance = environmentRadiance;
    finalShadeParams.environmentMap = reinterpret_cast<const float4*>(environmentMap.d_map);
    finalShadeParams.environmentMapWidth = environmentMap.width;
    finalShadeParams.environmentMapHeight = environmentMap.height;
    finalShadeParams.environmentMapScale = environmentMap.scale;
    finalShadeParams.pointLights = reinterpret_cast<const PointLightData*>(lightBuffers.d_pointLights);
    finalShadeParams.areaLights = reinterpret_cast<const AreaLightData*>(lightBuffers.d_areaLights);
    finalShadeParams.numPointLights = lightBuffers.numPointLights;
    finalShadeParams.numAreaLights = lightBuffers.numAreaLights;

    void* finalShadeArgs[] = { &finalShadeParams };
    CU_CHECK(cuLaunchKernel(
        kernelFunctions.shadeKernel,
        numBlocks, 1, 1,
        blockSize, 1, 1,
        0,
        0,
        finalShadeArgs,
        nullptr
    ));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}
}
