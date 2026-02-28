#include "optixw/optixw.h"
#include "optixw/types.h"
#include "optixw/launch_params.h"
#include "optixw/wavefront_kernel_params.h"
#include "scene_internal.h"
#include "checks.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <iostream>
#include <windows.h>
#include <cmath>

namespace optixw {

// Forward declarations
extern OptixDeviceContext g_optixContext;

namespace {

inline float3 toFloat3(const Vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

inline float3 sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline float3 normalize3(const float3& v) {
    const float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len2 <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float invLen = 1.0f / sqrtf(len2);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

} // namespace

static std::filesystem::path findRuntimeFile(const char* subDir, const char* fileName) {
    std::vector<std::filesystem::path> candidates;
    candidates.emplace_back(std::filesystem::path(subDir) / fileName);

    char exePath[MAX_PATH] = {};
    DWORD len = GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        candidates.emplace_back(exeDir / subDir / fileName);
        candidates.emplace_back(exeDir.parent_path() / subDir / fileName);
    }

    for (const std::filesystem::path& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    throw std::runtime_error(
        std::string("Failed to locate runtime file: ") + subDir + "/" + fileName);
}

static std::filesystem::path findPTXPath(const char* fileName) {
    return findRuntimeFile("ptx", fileName);
}

static std::filesystem::path findCubinPath(const char* fileName) {
    return findRuntimeFile("cubin", fileName);
}

// Load PTX file
static std::vector<char> loadPTX(const char* filename) {
    const std::filesystem::path ptxPath = findPTXPath(filename);
    std::ifstream file(ptxPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("Failed to open PTX file: ") + ptxPath.string());
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
    CUdeviceptr d_compactIndices = 0;
    CUdeviceptr d_accumBuffer = 0;
    CUdeviceptr d_hitBuffer = 0;
    CUdeviceptr d_compactCounter = 0;
    CUdeviceptr d_materials = 0;
    
    // Launch parameters
    CUdeviceptr d_launchParams = 0;
    
    // CUDA kernels
    CUmodule shadeModule = nullptr;
    CUmodule compactModule = nullptr;
    CUfunction shadeKernel = nullptr;
    CUfunction compactKernel = nullptr;
    CUmodule scaleModule = nullptr;
    CUfunction scaleKernel = nullptr;

    // OptiX denoiser
    OptixDenoiser denoiser = nullptr;
    CUdeviceptr d_denoiserState = 0;
    CUdeviceptr d_denoiserScratch = 0;
    CUdeviceptr d_denoiserInput = 0;
    CUdeviceptr d_denoiserOutput = 0;
    size_t denoiserStateSize = 0;
    size_t denoiserScratchSize = 0;
    uint32_t denoiserWidth = 0;
    uint32_t denoiserHeight = 0;
    
    uint32_t numPixels = 0;
    uint32_t maxRays = 0;
    uint32_t numMaterials = 0;
    uint32_t numTriangles = 0;
    float3 environmentRadiance = make_float3(0.0f, 0.0f, 0.0f);
    
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
        if (shadeModule) cuModuleUnload(shadeModule);
        if (compactModule) cuModuleUnload(compactModule);
        if (scaleModule) cuModuleUnload(scaleModule);
        if (denoiser) optixDenoiserDestroy(denoiser);
        
        if (sbt.raygenRecord) cudaFree((void*)sbt.raygenRecord);
        if (sbt.missRecordBase) cudaFree((void*)sbt.missRecordBase);
        if (sbt.hitgroupRecordBase) cudaFree((void*)sbt.hitgroupRecordBase);
        
        if (d_rayPool) cudaFree((void*)d_rayPool);
        if (d_activeIndices) cudaFree((void*)d_activeIndices);
        if (d_compactIndices) cudaFree((void*)d_compactIndices);
        if (d_accumBuffer) cudaFree((void*)d_accumBuffer);
        if (d_hitBuffer) cudaFree((void*)d_hitBuffer);
        if (d_compactCounter) cudaFree((void*)d_compactCounter);
        if (d_launchParams) cudaFree((void*)d_launchParams);
        if (d_denoiserState) cudaFree((void*)d_denoiserState);
        if (d_denoiserScratch) cudaFree((void*)d_denoiserScratch);
        if (d_denoiserInput) cudaFree((void*)d_denoiserInput);
        if (d_denoiserOutput) cudaFree((void*)d_denoiserOutput);
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
        if (d_compactIndices) cudaFree((void*)d_compactIndices);
        if (d_accumBuffer) cudaFree((void*)d_accumBuffer);
        if (d_hitBuffer) cudaFree((void*)d_hitBuffer);
        if (d_compactCounter) cudaFree((void*)d_compactCounter);
        if (d_launchParams) cudaFree((void*)d_launchParams);
        
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayPool), rayPoolSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_activeIndices), indicesSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compactIndices), indicesSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_accumBuffer), accumSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitBuffer), hitBufferSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compactCounter), sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launchParams), sizeof(LaunchParams)));
        
        // Initialize buffers to deterministic values
        CUDA_CHECK(cudaMemset((void*)d_rayPool, 0, rayPoolSize));
        CUDA_CHECK(cudaMemset((void*)d_hitBuffer, 0, hitBufferSize));
        CUDA_CHECK(cudaMemset((void*)d_accumBuffer, 0, accumSize));
        CUDA_CHECK(cudaMemset((void*)d_compactCounter, 0, sizeof(uint32_t)));
        
        std::cout << "[Renderer] Allocated buffers: " 
                  << numPixels << " pixels, " 
                  << maxRays << " rays" << std::endl;
    }

    void ensureDenoiserBuffers(uint32_t width, uint32_t height) {
        if (!denoiser) {
            OptixDenoiserOptions options = {};
            options.guideAlbedo = 0;
            options.guideNormal = 0;
            options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
            OPTIX_CHECK(optixDenoiserCreate(
                g_optixContext,
                OPTIX_DENOISER_MODEL_KIND_HDR,
                &options,
                &denoiser
            ));
        }

        if (width == denoiserWidth && height == denoiserHeight && d_denoiserInput) {
            return;
        }

        if (d_denoiserState) cudaFree((void*)d_denoiserState);
        if (d_denoiserScratch) cudaFree((void*)d_denoiserScratch);
        if (d_denoiserInput) cudaFree((void*)d_denoiserInput);
        if (d_denoiserOutput) cudaFree((void*)d_denoiserOutput);

        OptixDenoiserSizes sizes = {};
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
            denoiser,
            width,
            height,
            &sizes
        ));

        denoiserStateSize = sizes.stateSizeInBytes;
        denoiserScratchSize = sizes.withoutOverlapScratchSizeInBytes;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_denoiserState), denoiserStateSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_denoiserScratch), denoiserScratchSize));

        size_t imageSize = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float4);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_denoiserInput), imageSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_denoiserOutput), imageSize));

        OPTIX_CHECK(optixDenoiserSetup(
            denoiser,
            0,
            width,
            height,
            d_denoiserState,
            denoiserStateSize,
            d_denoiserScratch,
            denoiserScratchSize
        ));

        denoiserWidth = width;
        denoiserHeight = height;
    }
};

Renderer::Renderer() : m_impl(std::make_unique<Impl>()) {}
Renderer::~Renderer() = default;

void Renderer::render(
    Scene* scene,
    const Camera& camera,
    Vec3* outputBuffer,
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
    camData.position = toFloat3(camera.position);

    float3 target = toFloat3(camera.target);
    camData.forward = normalize3(sub3(target, camData.position));

    float3 up = toFloat3(camera.up);
    camData.right = normalize3(cross3(camData.forward, up));
    camData.up = cross3(camData.right, camData.forward);
    
    camData.tanHalfFovY = tanf(camera.fovY * 0.5f);
    camData.aspect = camera.aspect;
    
    // Get scene data
    OptixTraversableHandle gasHandle = SceneAccessor::getGasHandle(scene);
    CUdeviceptr d_vertices = SceneAccessor::getVerticesPtr(scene);
    CUdeviceptr d_indices = SceneAccessor::getIndicesPtr(scene);
    CUdeviceptr d_triangleMaterialIds = SceneAccessor::getTriangleMaterialIdsPtr(scene);
    m_impl->d_materials = SceneAccessor::getMaterialsPtr(scene);
    m_impl->numMaterials = SceneAccessor::getMaterialCount(scene);
    m_impl->numTriangles = SceneAccessor::getTriangleCount(scene);
    const Vec3 env = SceneAccessor::getEnvironmentRadiance(scene);
    m_impl->environmentRadiance = toFloat3(env);
    
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
    
    uint32_t checkCount = (100u < m_impl->numPixels) ? 100u : m_impl->numPixels;

    if (enableDenoiser) {
        m_impl->ensureDenoiserBuffers(width, height);

        const uint32_t count = m_impl->numPixels;
        const float invSpp = 1.0f / spp;
        const float3* d_accum = reinterpret_cast<const float3*>(m_impl->d_accumBuffer);
        float4* d_input = reinterpret_cast<float4*>(m_impl->d_denoiserInput);
        void* scaleArgs[] = { &d_accum, &d_input, (void*)&count, (void*)&invSpp };

        const uint32_t blockSize = 256;
        const uint32_t numBlocks = (count + blockSize - 1) / blockSize;
        CU_CHECK(cuLaunchKernel(
            m_impl->scaleKernel,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            scaleArgs,
            nullptr
        ));

        OptixImage2D input = {};
        input.data = m_impl->d_denoiserInput;
        input.width = width;
        input.height = height;
        input.rowStrideInBytes = width * sizeof(float4);
        input.pixelStrideInBytes = sizeof(float4);
        input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixImage2D output = {};
        output.data = m_impl->d_denoiserOutput;
        output.width = width;
        output.height = height;
        output.rowStrideInBytes = width * sizeof(float4);
        output.pixelStrideInBytes = sizeof(float4);
        output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixDenoiserGuideLayer guide = {};
        guide.albedo = input;
        guide.normal = input;
        OptixDenoiserLayer layer = {};
        layer.input = input;
        layer.output = output;
        layer.previousOutput = {};
        layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;

        OptixDenoiserParams params = {};
        params.hdrAverageColor = 0;
        params.blendFactor = 0.0f;
        params.hdrIntensity = 0;
        params.temporalModeUsePreviousLayers = 0;

        OPTIX_CHECK(optixDenoiserInvoke(
            m_impl->denoiser,
            0,
            &params,
            m_impl->d_denoiserState,
            m_impl->denoiserStateSize,
            &guide,
            &layer,
            1,
            0,
            0,
            m_impl->d_denoiserScratch,
            m_impl->denoiserScratchSize
        ));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float4> denoised(m_impl->numPixels);
        CUDA_CHECK(cudaMemcpy(
            denoised.data(),
            (void*)m_impl->d_denoiserOutput,
            m_impl->numPixels * sizeof(float4),
            cudaMemcpyDeviceToHost
        ));

        float maxOutput = 0.0f;
        for (uint32_t i = 0; i < m_impl->numPixels; ++i) {
            outputBuffer[i].x = denoised[i].x;
            outputBuffer[i].y = denoised[i].y;
            outputBuffer[i].z = denoised[i].z;
            if (i < checkCount) {
                float val = (denoised[i].x > denoised[i].y) ? denoised[i].x : denoised[i].y;
                val = (val > denoised[i].z) ? val : denoised[i].z;
                maxOutput = (maxOutput > val) ? maxOutput : val;
            }
        }
        std::cout << "[Debug] Output buffer max (first 100 pixels, denoised): " << maxOutput << std::endl;
    } else {
        // Download result
        std::vector<float3> accumBuffer(m_impl->numPixels);
        CUDA_CHECK(cudaMemcpy(
            accumBuffer.data(),
            (void*)m_impl->d_accumBuffer,
            m_impl->numPixels * sizeof(float3),
            cudaMemcpyDeviceToHost
        ));

        // Debug: check accumulation buffer
        float maxAccum = 0.0f;
        for (uint32_t i = 0; i < checkCount; ++i) {
            float val = (accumBuffer[i].x > accumBuffer[i].y) ? accumBuffer[i].x : accumBuffer[i].y;
            val = (val > accumBuffer[i].z) ? val : accumBuffer[i].z;
            maxAccum = (maxAccum > val) ? maxAccum : val;
        }
        std::cout << "[Debug] Accum buffer max (first 100 pixels): " << maxAccum << std::endl;

        // Average and copy to output
        float invSpp = 1.0f / spp;
        for (uint32_t i = 0; i < m_impl->numPixels; ++i) {
            outputBuffer[i].x = accumBuffer[i].x * invSpp;
            outputBuffer[i].y = accumBuffer[i].y * invSpp;
            outputBuffer[i].z = accumBuffer[i].z * invSpp;
        }

        // Debug: check output buffer
        float maxOutput = 0.0f;
        for (uint32_t i = 0; i < checkCount; ++i) {
            float val = (outputBuffer[i].x > outputBuffer[i].y) ? outputBuffer[i].x : outputBuffer[i].y;
            val = (val > outputBuffer[i].z) ? val : outputBuffer[i].z;
            maxOutput = (maxOutput > val) ? maxOutput : val;
        }
        std::cout << "[Debug] Output buffer max (first 100 pixels): " << maxOutput << std::endl;
    }
    
    std::cout << "[Renderer] Render complete" << std::endl;
}

} // namespace optixw
