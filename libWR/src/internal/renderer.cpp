#include "wr/renderer.h"
#include "wr/scene.h"
#include "internal/gpu_types.h"
#include "internal/cuda_utils.h"
#include "internal/denoiser.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace wr {

using namespace internal;

struct PipelineImpl;
extern PipelineImpl* createPipeline();
extern void launchPipeline(PipelineImpl* impl, CUdeviceptr paramsPtr, uint32_t numActive);
extern void destroyPipeline(PipelineImpl* impl);

struct RendererImpl {
    PipelineImpl* pipeline = nullptr;
    
    CUmodule shadeModule = nullptr;
    CUmodule compactModule = nullptr;
    CUfunction shadeKernel = nullptr;
    CUfunction compactKernel = nullptr;
    
    CUdeviceptr d_rayPool = 0;
    CUdeviceptr d_activeIndices[2] = {0, 0};
    CUdeviceptr d_hitBuffer = 0;
    CUdeviceptr d_accumBuffer = 0;
    CUdeviceptr d_counter = 0;
    
    // Denoiser guide buffers
    CUdeviceptr d_albedoBuffer = 0;
    CUdeviceptr d_normalBuffer = 0;
    
    CUdeviceptr d_launchParams = 0;
    CUdeviceptr d_launchParamsPtr = 0;
    CUdeviceptr d_compactParams = 0;
    
    Denoiser denoiser;
    
    uint32_t numPixels = 0;
    uint32_t maxRays = 0;
    
    void loadKernels() {
        // CUBIN files are copied to the same directory as the executable by CMake
        CU_CHECK(cuModuleLoad(&shadeModule, "shade.cubin"));
        CU_CHECK(cuModuleLoad(&compactModule, "compact.cubin"));
        
        CU_CHECK(cuModuleGetFunction(&shadeKernel, shadeModule, "shade"));
        CU_CHECK(cuModuleGetFunction(&compactKernel, compactModule, "compact"));
        
        std::cout << "[Renderer] Kernels loaded" << std::endl;
    }
    
    void allocateBuffers(uint32_t width, uint32_t height) {
        numPixels = width * height;
        maxRays = numPixels;
        
        size_t rayPoolSize = maxRays * sizeof(RayState);
        size_t indicesSize = maxRays * sizeof(uint32_t);
        size_t hitBufferSize = maxRays * sizeof(HitInfo);
        size_t accumSize = numPixels * sizeof(float3);
        size_t guideSize = numPixels * sizeof(float3);
        
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayPool), rayPoolSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_activeIndices[0]), indicesSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_activeIndices[1]), indicesSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitBuffer), hitBufferSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_accumBuffer), accumSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counter), sizeof(uint32_t)));
        
        // Allocate denoiser guide buffers
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_albedoBuffer), guideSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_normalBuffer), guideSize));
        
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_rayPool), 0, rayPoolSize));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_hitBuffer), 0, hitBufferSize));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_accumBuffer), 0, accumSize));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_albedoBuffer), 0, guideSize));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_normalBuffer), 0, guideSize));
        
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launchParams), sizeof(LaunchParams)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launchParamsPtr), sizeof(LaunchParams*)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compactParams), sizeof(CompactParams)));
        
        std::cout << "[Renderer] Buffers allocated: " << numPixels << " pixels" << std::endl;
    }
    
    void freeBuffers() {
        if (d_rayPool) cudaFree(reinterpret_cast<void*>(d_rayPool));
        if (d_activeIndices[0]) cudaFree(reinterpret_cast<void*>(d_activeIndices[0]));
        if (d_activeIndices[1]) cudaFree(reinterpret_cast<void*>(d_activeIndices[1]));
        if (d_hitBuffer) cudaFree(reinterpret_cast<void*>(d_hitBuffer));
        if (d_accumBuffer) cudaFree(reinterpret_cast<void*>(d_accumBuffer));
        if (d_albedoBuffer) cudaFree(reinterpret_cast<void*>(d_albedoBuffer));
        if (d_normalBuffer) cudaFree(reinterpret_cast<void*>(d_normalBuffer));
        if (d_counter) cudaFree(reinterpret_cast<void*>(d_counter));
        if (d_launchParams) cudaFree(reinterpret_cast<void*>(d_launchParams));
        if (d_launchParamsPtr) cudaFree(reinterpret_cast<void*>(d_launchParamsPtr));
        if (d_compactParams) cudaFree(reinterpret_cast<void*>(d_compactParams));
    }
    
    ~RendererImpl() {
        freeBuffers();
        if (shadeModule) cuModuleUnload(shadeModule);
        if (compactModule) cuModuleUnload(compactModule);
        if (pipeline) destroyPipeline(pipeline);
    }
};

Renderer::Renderer(const RendererConfig& config) 
    : m_impl(new RendererImpl()), m_config(config) {
    try {
        m_impl->pipeline = createPipeline();
        m_impl->loadKernels();
    } catch (const std::exception& e) {
        std::cerr << "[Renderer] Initialization failed: " << e.what() << std::endl;
        throw;
    }
}

Renderer::~Renderer() {
    delete m_impl;
}

void Renderer::render(Scene* scene,
                      const Camera& camera,
                      Vec3* outputBuffer,
                      const RenderParams& renderParams) {
    uint32_t width = renderParams.width;
    uint32_t height = renderParams.height;
    uint32_t spp = renderParams.spp;
    bool denoiser = renderParams.denoiser.enabled;
    std::cout << "[Renderer] Starting render: " << width << "x" << height << " @ " << spp << " spp" << std::endl;
    
    m_impl->allocateBuffers(width, height);
    
    Vec3 forward = camera.getForward();
    Vec3 right = camera.getRight();
    Vec3 up = camera.getUp();

    CameraData camData;
    camData.position = make_float4(camera.position.x, camera.position.y, camera.position.z, 0.0f);
    camData.forward = make_float4(forward.x, forward.y, forward.z, 0.0f);
    camData.right = make_float4(right.x, right.y, right.z, 0.0f);
    camData.up = make_float4(up.x, up.y, up.z, 0.0f);
    camData.tanHalfFovY = tanf(camera.fovY * 0.5f);
    camData.aspect = camera.aspect;
    
    LaunchParams hostParams = {};
    hostParams.traversable = static_cast<OptixTraversableHandle>(scene->getGASHandle());
    hostParams.geometry.vertices = reinterpret_cast<const float*>(static_cast<CUdeviceptr>(scene->getVerticesBuffer()));
    hostParams.geometry.indices = reinterpret_cast<const uint32_t*>(static_cast<CUdeviceptr>(scene->getIndicesBuffer()));
    hostParams.geometry.triangleMaterialIds = reinterpret_cast<const uint32_t*>(static_cast<CUdeviceptr>(scene->getTriangleMaterialIdsBuffer()));
    hostParams.geometry.uvs = scene->getUVsBuffer() ? reinterpret_cast<const float*>(static_cast<CUdeviceptr>(scene->getUVsBuffer())) : nullptr;
    hostParams.rayPool = reinterpret_cast<RayState*>(m_impl->d_rayPool);
    hostParams.hitBuffer = reinterpret_cast<HitInfo*>(m_impl->d_hitBuffer);
    hostParams.accumBuffer = reinterpret_cast<float3*>(m_impl->d_accumBuffer);
    hostParams.albedoBuffer = reinterpret_cast<float3*>(m_impl->d_albedoBuffer);
    hostParams.normalBuffer = reinterpret_cast<float3*>(m_impl->d_normalBuffer);
    hostParams.materials = reinterpret_cast<const MaterialData*>(static_cast<CUdeviceptr>(scene->getMaterialsBuffer()));
    hostParams.numMaterials = scene->getNumMaterials();
    hostParams.textures = scene->getNumTextures() > 0 ? reinterpret_cast<const void*>(static_cast<CUdeviceptr>(scene->getTexturesBuffer())) : nullptr;
    hostParams.numTextures = scene->getNumTextures();
    hostParams.emissiveTriangles = reinterpret_cast<const uint32_t*>(static_cast<CUdeviceptr>(scene->getEmissiveTrianglesBuffer()));
    hostParams.emissiveTriangleCDF = reinterpret_cast<const float*>(static_cast<CUdeviceptr>(scene->getEmissiveTriangleCDFBuffer()));
    hostParams.numEmissiveTriangles = scene->getNumEmissiveTriangles();
    hostParams.camera = camData;
    Vec3 envRad = scene->getEnvironmentRadiance();
    hostParams.environmentRadiance = make_float4(envRad.x, envRad.y, envRad.z, 0.0f);
    hostParams.width = width;
    hostParams.height = height;
    hostParams.useNEE = renderParams.useNEE;
    hostParams.maxBounces = renderParams.maxBounces;
    hostParams.rrStartDepth = renderParams.russianRouletteDepth;
    
    for (uint32_t s = 0; s < spp; ++s) {
        std::cout << "\r[Renderer] Sample " << (s + 1) << "/" << spp << std::flush;
        
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_impl->d_rayPool), 0, m_impl->maxRays * sizeof(RayState)));
        
        std::vector<uint32_t> hostActiveIndices(m_impl->numPixels);
        for (uint32_t i = 0; i < m_impl->numPixels; ++i) {
            hostActiveIndices[i] = i;
        }
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_impl->d_activeIndices[0]),
            hostActiveIndices.data(),
            m_impl->numPixels * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));
        
        uint32_t numActive = m_impl->numPixels;
        uint32_t activeIn = 0;
        uint32_t activeOut = 1;
        uint32_t depth = 0;
        
        hostParams.sampleIndex = s;
        
        while (numActive > 0 && depth < 8) {
            hostParams.activeIndices = reinterpret_cast<uint32_t*>(m_impl->d_activeIndices[activeIn]);
            hostParams.numActive = numActive;
            
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(m_impl->d_launchParams),
                &hostParams,
                sizeof(LaunchParams),
                cudaMemcpyHostToDevice
            ));
            
            LaunchParams* d_launchParamsValue = reinterpret_cast<LaunchParams*>(m_impl->d_launchParams);
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(m_impl->d_launchParamsPtr),
                &d_launchParamsValue,
                sizeof(LaunchParams*),
                cudaMemcpyHostToDevice
            ));
            
            launchPipeline(m_impl->pipeline, m_impl->d_launchParamsPtr, numActive);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            const uint32_t blockSize = 256;
            const uint32_t numBlocks = (numActive + blockSize - 1) / blockSize;
            
            void* shadeArgs[] = { &m_impl->d_launchParams };
            CU_CHECK(cuLaunchKernel(
                m_impl->shadeKernel,
                numBlocks, 1, 1,
                blockSize, 1, 1,
                0, 0,
                shadeArgs, nullptr
            ));
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_impl->d_counter), 0, sizeof(uint32_t)));
            
            CompactParams compactParams;
            compactParams.rayPool = reinterpret_cast<const RayState*>(m_impl->d_rayPool);
            compactParams.activeIndicesIn = reinterpret_cast<const uint32_t*>(m_impl->d_activeIndices[activeIn]);
            compactParams.activeIndicesOut = reinterpret_cast<uint32_t*>(m_impl->d_activeIndices[activeOut]);
            compactParams.counter = reinterpret_cast<uint32_t*>(m_impl->d_counter);
            compactParams.numActive = numActive;
            
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(m_impl->d_compactParams),
                &compactParams,
                sizeof(CompactParams),
                cudaMemcpyHostToDevice
            ));
            
            void* compactArgs[] = { &m_impl->d_compactParams };
            CU_CHECK(cuLaunchKernel(
                m_impl->compactKernel,
                numBlocks, 1, 1,
                blockSize, 1, 1,
                0, 0,
                compactArgs, nullptr
            ));
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMemcpy(
                &numActive,
                reinterpret_cast<void*>(m_impl->d_counter),
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost
            ));
            
            std::swap(activeIn, activeOut);
            depth++;
        }
    }
    
    std::cout << std::endl;
    
    // 执行降噪
        if (denoiser) {
            m_impl->denoiser.initialize(width, height, renderParams.denoiser);
            m_impl->denoiser.denoise(
                m_impl->d_accumBuffer,
                m_impl->d_accumBuffer,
                m_impl->d_albedoBuffer,
                m_impl->d_normalBuffer,
                spp
            );
        }
    
    std::vector<float3> hostAccum(m_impl->numPixels);
    CUDA_CHECK(cudaMemcpy(
        hostAccum.data(),
        reinterpret_cast<void*>(m_impl->d_accumBuffer),
        m_impl->numPixels * sizeof(float3),
        cudaMemcpyDeviceToHost
    ));
    
    float invSpp = 1.0f / spp;
    for (uint32_t i = 0; i < m_impl->numPixels; ++i) {
        outputBuffer[i].x = hostAccum[i].x * invSpp;
        outputBuffer[i].y = hostAccum[i].y * invSpp;
        outputBuffer[i].z = hostAccum[i].z * invSpp;
    }
    
    std::cout << "[Renderer] Render complete" << std::endl;
}

void Renderer::render(Scene* scene,
                      const Camera& camera,
                      Vec3* outputBuffer,
                      uint32_t width,
                      uint32_t height,
                      uint32_t spp,
                      bool denoiser) {
    // Backward compatibility: call new render with default params
    RenderParams params;
    params.width = width;
    params.height = height;
    params.spp = spp;
    params.denoiser.enabled = denoiser;
    params.denoiser.useAlbedo = true;
    params.denoiser.useNormal = true;
    
    render(scene, camera, outputBuffer, params);
}

} // namespace wr
