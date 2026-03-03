#include "optixw/optixw.h"
#include "optixw/types.h"
#include "optixw/launch_params.h"
#include "optixw/wavefront_kernel_params.h"
#include "scene_internal.h"
#include "utils/checks.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <iostream>
#include <windows.h>
// windows.h defines min/max macros that conflict with std::min/std::max
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include <cmath>
#include "utils/checks.h"
#include "optixw/renderer_impl_public.h"

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

std::filesystem::path findRuntimeFile(const char* subDir, const char* fileName) {
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

std::filesystem::path findPTXPath(const char* fileName) {
    return findRuntimeFile("ptx", fileName);
}

std::filesystem::path findCubinPath(const char* fileName) {
    return findRuntimeFile("cubin", fileName);
}

// Load PTX file
std::vector<char> loadPTX(const char* filename) {
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

// Implementations for constructor/destructor and chosen methods that remain
// in this translation unit.

Impl::Impl()
    : traceModule(nullptr), pipeline(nullptr), sbt(),
      numPixels(0), maxRays(0),
      pipelineCreated(false),
      scheduler(nullptr)
{
}

Impl::~Impl() {
    programGroups.destroy(g_optixContext);
    if (pipeline) optixPipelineDestroy(pipeline);
    if (traceModule) optixModuleDestroy(traceModule);
    kernelModules.unload();
    
    sbtRecords.free();
    
    wavefrontBuffers.free();
    accumulationBuffers.free();
    mergeBuffers.free();
    launchParamsBuffer.free();
    denoiser.destroy();
}

void Impl::allocateBuffers(uint32_t width, uint32_t height) {
    numPixels = width * height;
    maxRays = numPixels;  // One ray per pixel for now
    
    size_t rayPoolSize = maxRays * sizeof(RayState);
    size_t indicesSize = maxRays * sizeof(uint32_t);
    size_t accumSize = numPixels * sizeof(float3);
    size_t hitBufferSize = maxRays * sizeof(HitInfo);
    size_t weightSize = numPixels * sizeof(float);
    
    std::cout << "[Renderer] allocateBuffers: malloc rayPool " << rayPoolSize << std::endl;
    std::cout << "[Renderer] allocateBuffers: malloc activeIndices " << indicesSize << std::endl;
    std::cout << "[Renderer] allocateBuffers: malloc compactIndices " << indicesSize << std::endl;
    wavefrontBuffers.allocate(rayPoolSize, indicesSize, hitBufferSize);
    
    std::cout << "[Renderer] allocateBuffers: malloc accumBuffer " << accumSize << std::endl;
    std::cout << "[Renderer] allocateBuffers: malloc albedoBuffer " << accumSize << std::endl;
    std::cout << "[Renderer] allocateBuffers: malloc normalBuffer " << accumSize << std::endl;
    accumulationBuffers.allocate(accumSize);
    
    std::cout << "[Renderer] allocateBuffers: malloc mergeAccum " << accumSize << std::endl;
    std::cout << "[Renderer] allocateBuffers: malloc mergeWeight " << weightSize << std::endl;
    mergeBuffers.allocate(accumSize, weightSize);
    
    std::cout << "[Renderer] allocateBuffers: malloc hitBuffer " << hitBufferSize << std::endl;
    std::cout << "[Renderer] allocateBuffers: malloc compactCounter" << std::endl;
    std::cout << "[Renderer] allocateBuffers: malloc launchParams" << std::endl;
    launchParamsBuffer.allocate(sizeof(LaunchParams));
    
    // Initialize buffers to deterministic values
    CUDA_CHECK(cudaMemset((void*)wavefrontBuffers.d_rayPool, 0, rayPoolSize));
    CUDA_CHECK(cudaMemset((void*)wavefrontBuffers.d_hitBuffer, 0, hitBufferSize));
    CUDA_CHECK(cudaMemset((void*)accumulationBuffers.d_accumBuffer, 0, accumSize));
    CUDA_CHECK(cudaMemset((void*)accumulationBuffers.d_albedoBuffer, 0, accumSize));
    CUDA_CHECK(cudaMemset((void*)accumulationBuffers.d_normalBuffer, 0, accumSize));
    CUDA_CHECK(cudaMemset((void*)mergeBuffers.d_mergeAccum, 0, accumSize));
    CUDA_CHECK(cudaMemset((void*)mergeBuffers.d_mergeWeight, 0, weightSize));
    CUDA_CHECK(cudaMemset((void*)wavefrontBuffers.d_compactCounter, 0, sizeof(uint32_t)));
    
    std::cout << "[Renderer] Allocated buffers: " 
              << numPixels << " pixels, " 
              << maxRays << " rays" << std::endl;
}

void Impl::ensureDenoiserBuffers(uint32_t width, uint32_t height,
                                           uint32_t tileWidth, uint32_t tileHeight) {
    if (!denoiser.handle) {
        OptixDenoiserOptions options = {};
        // enable guides so that we can supply albedo/normal buffers
        options.guideAlbedo = 1;
        options.guideNormal = 1;
        options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        OPTIX_CHECK(optixDenoiserCreate(
            g_optixContext,
            OPTIX_DENOISER_MODEL_KIND_HDR,
            &options,
            &denoiser.handle
        ));
    }

    if (width == denoiser.width && height == denoiser.height &&
        denoiser.imageBuffers.d_input && denoiser.imageBuffers.d_albedoInput && denoiser.imageBuffers.d_normalInput) {
        return;
    }

    denoiser.buffers.free();
    denoiser.imageBuffers.free();
    denoiser.tileBuffers.free();

    OptixDenoiserSizes sizes = {};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(
        denoiser.handle,
        width,
        height,
        &sizes
    ));

    denoiser.overlap = sizes.overlapWindowSizeInPixels;

    denoiser.buffers.allocate(sizes.stateSizeInBytes, sizes.withoutOverlapScratchSizeInBytes);

    size_t imageSize = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float4);
    denoiser.imageBuffers.allocate(imageSize);
    denoiser.tileBuffers.allocate(imageSize);

    OPTIX_CHECK(optixDenoiserSetup(
        denoiser.handle,
        0,
        width,
        height,
        denoiser.buffers.d_state,
        denoiser.buffers.stateSize,
        denoiser.buffers.d_scratch,
        denoiser.buffers.scratchSize
    ));

    denoiser.width = width;
    denoiser.height = height;
}

Renderer::Renderer(TaskScheduler* scheduler) : m_impl(std::make_unique<Impl>()) {
    m_impl->scheduler = scheduler;  // Store the scheduler
}

Renderer::~Renderer() = default;

// Main render method
void Renderer::render(
    Scene* scene,
    const Camera& camera,
    Vec3* outputBuffer,
    uint32_t width,
    uint32_t height,
    uint32_t spp,
    bool enableDenoiser,
    float denoiserBlend,
    bool enableTiling,
    uint32_t tileWidth,
    uint32_t tileHeight)
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
    
    // Get scene data from new managers
    OptixTraversableHandle gasHandle = SceneAccessor::getGasHandle(scene);
    CUdeviceptr d_vertices = SceneAccessor::getVerticesPtr(scene);
    CUdeviceptr d_texcoords = SceneAccessor::getTexcoordsPtr(scene);
    CUdeviceptr d_indices = SceneAccessor::getIndicesPtr(scene);
    CUdeviceptr d_triangleMaterialIds = SceneAccessor::getTriangleMaterialIdsPtr(scene);
    m_impl->materialTextureBuffers.d_materials = SceneAccessor::getMaterialsPtr(scene);
    m_impl->materialTextureBuffers.d_texcoords = d_texcoords;
    m_impl->materialTextureBuffers.d_textures = SceneAccessor::getTexturesPtr(scene);
    m_impl->materialTextureBuffers.numMaterials = SceneAccessor::getMaterialCount(scene);
    m_impl->materialTextureBuffers.numTextures = SceneAccessor::getTextureCount(scene);
    m_impl->materialTextureBuffers.numTriangles = SceneAccessor::getTriangleCount(scene);
    const Vec3 env = SceneAccessor::getEnvironmentRadiance(scene);
    m_impl->environmentData.radiance = toFloat3(env);
    m_impl->environmentData.map.d_map = SceneAccessor::getEnvironmentMapPtr(scene);
    m_impl->environmentData.map.width = SceneAccessor::getEnvironmentMapWidth(scene);
    m_impl->environmentData.map.height = SceneAccessor::getEnvironmentMapHeight(scene);
    m_impl->environmentData.map.scale = SceneAccessor::getEnvironmentMapScale(scene);
    
    // Get light data
    m_impl->lightBuffers.d_pointLights = SceneAccessor::getPointLightsPtr(scene);
    m_impl->lightBuffers.d_areaLights = SceneAccessor::getAreaLightsPtr(scene);
    m_impl->lightBuffers.numPointLights = SceneAccessor::getNumPointLights(scene);
    m_impl->lightBuffers.numAreaLights = SceneAccessor::getNumAreaLights(scene);
    
    std::cout << "[Renderer] GAS handle: " << gasHandle << std::endl;

    // Fall back to serial rendering to avoid concurrent access issues
    for (uint32_t sample = 0; sample < spp; ++sample) {
        std::cout << "\r[Renderer] Sample " << (sample + 1) << "/" << spp << std::flush;
        
        m_impl->renderSample(
            gasHandle,
            d_vertices,
            d_texcoords,
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

    // Debug: check accumulation buffer before denoiser
    {
        std::vector<float3> debugAccum(m_impl->numPixels);
        CUDA_CHECK(cudaMemcpy(
            debugAccum.data(),
            (void*)m_impl->accumulationBuffers.d_accumBuffer,
            m_impl->numPixels * sizeof(float3),
            cudaMemcpyDeviceToHost
        ));
        float maxAccum = 0.0f;
        for (uint32_t i = 0; i < checkCount; ++i) {
            float val = std::max({debugAccum[i].x, debugAccum[i].y, debugAccum[i].z});
            maxAccum = std::max(maxAccum, val);
        }
        std::cout << "[Debug] Accum buffer max (first 100 pixels): " << maxAccum << std::endl;
        
        // Check center pixel
        uint32_t centerIdx = (height/2) * width + (width/2);
        std::cout << "[Debug] Center pixel (" << width/2 << "," << height/2 << "): (" 
                  << debugAccum[centerIdx].x << "," << debugAccum[centerIdx].y << "," << debugAccum[centerIdx].z << ")" << std::endl;
    }

    if (enableDenoiser) {
        // determine tile size (0 means whole image). If tiling is disabled
        // via config, force full-image tiles (no tiling flow).
        uint32_t tileW = (!enableTiling || tileWidth == 0) ? width : tileWidth;
        uint32_t tileH = (!enableTiling || tileHeight == 0) ? height : tileHeight;
        m_impl->ensureDenoiserBuffers(width, height, tileW, tileH);
        std::cout << "[Renderer] denoiser overlap = " << m_impl->denoiser.overlap << "\n";
        if ((tileW < width || tileH < height)) {
            // disable tiling if the tile is too small relative to the denoiser
            // overlap.  the overlap window is the radius around each pixel that
            // the denoiser may read; at minimum we need two overlaps per tile to
            // ensure shared pixels between adjacent tasks have identical input
            // context.  when tile dimension <= 2*overlap, the non-overlapping
            // output region would be zero or the seams can diverge, so fall back
            // to single-task rendering.
            uint32_t minSize = m_impl->denoiser.overlap * 2;
            if (tileW <= minSize || tileH <= minSize) {
                std::cout << "[Renderer] requested tile (" << tileW << "x" << tileH
                          << ") is too small for overlap (" << m_impl->denoiser.overlap
                          << "); disabling tiling\n";
                tileW = width;
                tileH = height;
            }
        }

        // prepare beauty/albedo/normal arrays in float4 space (full image)
        const uint32_t count = m_impl->numPixels;
        const float invSpp = 1.0f / spp;
        const float3* d_accum = reinterpret_cast<const float3*>(m_impl->accumulationBuffers.d_accumBuffer);
        float4* d_input = reinterpret_cast<float4*>(m_impl->denoiser.imageBuffers.d_input);
        void* scaleArgs[] = { (void*)d_accum, (void*)d_input, (void*)&count, (void*)&invSpp };

        // zero-out the output buffer to avoid partial-coverage artifacts when
        // copying tiles later.  this prevents the "cross" of black pixels seen
        // when some pixels are never written.
        size_t imageSize = static_cast<size_t>(width) * height * sizeof(float4);
        CUDA_CHECK(cudaMemset((void*)m_impl->denoiser.imageBuffers.d_output, 0, imageSize));

        const uint32_t blockSize = 256;
        const uint32_t numBlocks = (count + blockSize - 1) / blockSize;
        CU_CHECK(cuLaunchKernel(
            m_impl->kernelFunctions.scaleKernel,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            scaleArgs,
            nullptr
        ));

        // Synchronize to ensure scale kernel completes
        CUDA_CHECK(cudaDeviceSynchronize());

        // also guide buffers
        const float3* d_albedo = reinterpret_cast<const float3*>(m_impl->accumulationBuffers.d_albedoBuffer);
        const float3* d_normal = reinterpret_cast<const float3*>(m_impl->accumulationBuffers.d_normalBuffer);
        float4* d_albedoInput = reinterpret_cast<float4*>(m_impl->denoiser.imageBuffers.d_albedoInput);
        float4* d_normalInput = reinterpret_cast<float4*>(m_impl->denoiser.imageBuffers.d_normalInput);
        const float invOne = 1.0f;

        void* scaleAlbedoArgs[] = { (void*)d_albedo, (void*)d_albedoInput, (void*)&count, (void*)&invOne };
        void* scaleNormalArgs[] = { (void*)d_normal, (void*)d_normalInput, (void*)&count, (void*)&invOne };
        CU_CHECK(cuLaunchKernel(
            m_impl->kernelFunctions.scaleKernel,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            scaleAlbedoArgs,
            nullptr
        ));
        CU_CHECK(cuLaunchKernel(
            m_impl->kernelFunctions.scaleKernel,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            scaleNormalArgs,
            nullptr
        ));

        OptixImage2D baseInput = {};
        baseInput.rowStrideInBytes = width * sizeof(float4);
        baseInput.pixelStrideInBytes = sizeof(float4);
        baseInput.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixImage2D baseOutput = baseInput; // same layout

        OptixImage2D albedoImg = {};
        albedoImg.rowStrideInBytes = width * sizeof(float4);
        albedoImg.pixelStrideInBytes = sizeof(float4);
        albedoImg.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixImage2D normalImg = albedoImg;

        // Compute global denoiser intensity once for the full image.  Using
        // a per-tile intensity causes inconsistent normalization across
        // tiles and produces visible seams.
        OptixImage2D fullInput = baseInput;
        fullInput.data = m_impl->denoiser.imageBuffers.d_input;
        fullInput.width = width;
        fullInput.height = height;
        OPTIX_CHECK(optixDenoiserComputeIntensity(
            m_impl->denoiser.handle,
            0,
            &fullInput,
            m_impl->denoiser.buffers.d_intensity,
            m_impl->denoiser.buffers.d_scratch,
            m_impl->denoiser.buffers.scratchSize
        ));

        // 使用单任务降噪，提高质量
        OptixImage2D input = baseInput;
        input.data = m_impl->denoiser.imageBuffers.d_input;
        input.width = width;
        input.height = height;
        
        OptixImage2D output = baseOutput;
        output.data = m_impl->denoiser.imageBuffers.d_output;
        output.width = width;
        output.height = height;
        
        OptixDenoiserLayer layer = {};
        layer.input = input;
        layer.output = output;
        layer.previousOutput = {};
        layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
        
        OptixDenoiserGuideLayer guide = {};
        guide.albedo.data = m_impl->denoiser.imageBuffers.d_albedoInput;
        guide.albedo.width = width;
        guide.albedo.height = height;
        guide.albedo.rowStrideInBytes = width * sizeof(float4);
        guide.albedo.pixelStrideInBytes = sizeof(float4);
        guide.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        
        guide.normal.data = m_impl->denoiser.imageBuffers.d_normalInput;
        guide.normal.width = width;
        guide.normal.height = height;
        guide.normal.rowStrideInBytes = width * sizeof(float4);
        guide.normal.pixelStrideInBytes = sizeof(float4);
        guide.normal.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        
        // 使用传入的 denoiserBlend，与 render_config.ini 一致（libVLR 参考流程）
        OptixDenoiserParams params = {};
        params.hdrAverageColor = 0;
        params.blendFactor = denoiserBlend;
        params.hdrIntensity = m_impl->denoiser.buffers.d_intensity;
        params.temporalModeUsePreviousLayers = 0;
        
        OPTIX_CHECK(optixDenoiserInvoke(
            m_impl->denoiser.handle,
            0,
            &params,
            m_impl->denoiser.buffers.d_state,
            m_impl->denoiser.buffers.stateSize,
            &guide,
            &layer,
            1,
            0,
            0,
            m_impl->denoiser.buffers.d_scratch,
            m_impl->denoiser.buffers.scratchSize
        ));
        
        CUDA_CHECK(cudaDeviceSynchronize());

        // at this point the d_denoiserOutput buffer contains the denoised image

        std::vector<float4> denoised(m_impl->numPixels);
        CUDA_CHECK(cudaMemcpy(
            denoised.data(),
            (void*)m_impl->denoiser.imageBuffers.d_output,
            m_impl->numPixels * sizeof(float4),
            cudaMemcpyDeviceToHost
        ));

        float maxOutput = 0.0f;
        for (uint32_t i = 0; i < m_impl->numPixels; ++i) {
            outputBuffer[i].x = denoised[i].x;
            outputBuffer[i].y = denoised[i].y;
            outputBuffer[i].z = denoised[i].z;
            float val = std::max({denoised[i].x, denoised[i].y, denoised[i].z});
            maxOutput = std::max(maxOutput, val);
        }
        std::cout << "[Debug] Output buffer max (all pixels, denoised): " << maxOutput << std::endl;
    } else {
        // Download result
        std::vector<float3> accumBuffer(m_impl->numPixels);
        CUDA_CHECK(cudaMemcpy(
            accumBuffer.data(),
            (void*)m_impl->accumulationBuffers.d_accumBuffer,
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

void Renderer::render(
    Scene* scene,
    const Camera& camera,
    Vec3* outputBuffer,
    uint32_t width,
    uint32_t height,
    const RenderConfig& config)
{
    render(
        scene,
        camera,
        outputBuffer,
        width,
        height,
        config.spp,
        config.enableDenoiser,
        config.denoiserBlend,
        config.enableTiling,
        config.tileWidth,
        config.tileHeight
    );
}

void Renderer::render(const RenderParams& params) {
    render(
        params.scene,
        *params.camera,
        params.outputBuffer,
        params.width,
        params.height,
        params.config
    );
}

} // namespace optixw

// implementations moved to separate .cpp files
