#include "renderer.h"
#include "optix_context.h"
#include "scene.h"
#include "cuda_check.h"
#include "../gpu/wavefront_types.cuh"
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>

extern "C" void wpt_launchGeneratePrimaryRays(
    void* rayPool,
    uint32_t* activeQueue,
    const void* camera,
    uint32_t numPixels,
    unsigned int gridSize,
    unsigned int blockSize,
    uint32_t sampleIndex);

extern "C" void wpt_launchShadeStage(void* d_globalState, uint32_t numRays, unsigned int gridSize, unsigned int blockSize);

namespace vlrw {

// Host copy of trace launch params (same layout as device TraceLaunchParams)
struct TraceLaunchParamsHost {
    OptixTraversableHandle traversable;
    void* rayPool;
    void* activeQueue;
    void* hitBuffer;
    void* accumBuffer;
    const void* vertices;
    const void* indices;
    const void* triangleMaterialIds;
    int debugMode;  // 0=default, 1=normal visualization
};

// Host copy of shadow launch params (same layout as device ShadowLaunchParams)
struct ShadowLaunchParamsHost {
    OptixTraversableHandle traversable;
    void* hitBuffer;
    void* lights;
    uint32_t numLights;
    void* activeQueue;
    uint32_t numActive;
    void* visibilityBuffer;
};

// Same layout as wpt::CameraParams (float3 x3 + float + uint32 x2) for kernel launch
struct CameraParamsHost {
    float px, py, pz;
    float tx, ty, tz;
    float ux, uy, uz;
    float fov;
    uint32_t width;
    uint32_t height;
};

static CameraParamsHost makeCameraParams(const Camera& camera, uint32_t width, uint32_t height) {
    CameraParamsHost c = {};
    c.px = camera.position[0]; c.py = camera.position[1]; c.pz = camera.position[2];
    c.tx = camera.target[0]; c.ty = camera.target[1]; c.tz = camera.target[2];
    c.ux = camera.up[0]; c.uy = camera.up[1]; c.uz = camera.up[2];
    c.fov = camera.fovY;
    c.width = width;
    c.height = height;
    return c;
}

RendererImpl::RendererImpl(ContextImpl* context)
    : m_context(context)
    , m_tracePipelineCreated(false)
{
}

RendererImpl::~RendererImpl() {
    cleanupDenoiser();
    if (m_tracePipelineCreated) {
        // TODO: free pipeline, SBT, module, program groups
    }
}

static std::string findTracePtxPath() {
    const char* candidates[] = {
        "libVLRW/ptxes/trace_rays.ptx",
        "ptxes/trace_rays.ptx",
        "bin/Release/libVLRW/ptxes/trace_rays.ptx",
        "build/bin/Release/libVLRW/ptxes/trace_rays.ptx",
    };
    for (const char* p : candidates) {
        std::ifstream f(p, std::ios::binary);
        if (f) { f.close(); return p; }
    }
    throw std::runtime_error("Could not find trace_rays.ptx; run from build/bin/Release or set cwd");
}

static std::string findShadowPtxPath() {
    const char* candidates[] = {
        "libVLRW/ptxes/shadow_trace.ptx",
        "ptxes/shadow_trace.ptx",
        "bin/Release/libVLRW/ptxes/shadow_trace.ptx",
        "build/bin/Release/libVLRW/ptxes/shadow_trace.ptx",
    };
    for (const char* p : candidates) {
        std::ifstream f(p, std::ios::binary);
        if (f) { f.close(); return p; }
    }
    throw std::runtime_error("Could not find shadow_trace.ptx; run from build/bin/Release or set cwd");
}

void RendererImpl::render(
    SceneImpl* scene, const Camera& camera, RGB* outputBuffer,
    uint32_t width, uint32_t height, uint32_t spp, bool enableDenoiser, int debugMode)
{
    const uint32_t numPixels = width * height;
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (numPixels + blockSize - 1) / blockSize;

    if (!m_tracePipelineCreated) {
        std::string ptxPath = findTracePtxPath();
        m_traceHandle = m_context->createTracePipelineFromFile(ptxPath.c_str());
        m_tracePipelineCreated = true;
    }
    if (!m_shadowPipelineCreated) {
        std::string shadowPtxPath = findShadowPtxPath();
        m_shadowHandle = m_context->createShadowPipelineFromFile(shadowPtxPath.c_str());
        m_shadowPipelineCreated = true;
    }

    CUdeviceptr d_rayPool = 0;
    CUdeviceptr d_activeIndices0 = 0;
    CUdeviceptr d_activeIndices1 = 0;
    CUdeviceptr d_hitBuffer = 0;
    CUdeviceptr d_accumBuffer = 0;
    CUdeviceptr d_queueCounters = 0;
    CUdeviceptr d_globalState = 0;
    CUdeviceptr d_tracePlp = 0;
    CUdeviceptr d_shadowPlp = 0;
    CUdeviceptr d_lights = 0;
    CUdeviceptr d_areaLights = 0;
    CUdeviceptr d_visibilityBuffer = 0;

    const size_t rayStateSize = sizeof(wpt::RayState);
    const size_t hitInfoSize = sizeof(wpt::HitInfo);

    CUDA_CHECK(cuMemAlloc(&d_rayPool, numPixels * rayStateSize));
    CUDA_CHECK(cuMemAlloc(&d_activeIndices0, numPixels * sizeof(uint32_t)));
    CUDA_CHECK(cuMemAlloc(&d_activeIndices1, numPixels * sizeof(uint32_t)));
    CUDA_CHECK(cuMemAlloc(&d_hitBuffer, numPixels * hitInfoSize));
    CUDA_CHECK(cuMemAlloc(&d_accumBuffer, numPixels * sizeof(::wpt::float3_rgb)));
    CUDA_CHECK(cuMemAlloc(&d_queueCounters, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cuMemAlloc(&d_globalState, sizeof(wpt::GlobalState)));
    CUDA_CHECK(cuMemAlloc(&d_tracePlp, sizeof(TraceLaunchParamsHost)));

    // Upload point lights
    const auto& lights = scene->getLights();
    uint32_t numLights = static_cast<uint32_t>(lights.size());
    if (numLights > 0) {
        std::vector<::wpt::PointLight> gpuLights(numLights);
        for (uint32_t i = 0; i < numLights; ++i) {
            gpuLights[i].position = make_float3(lights[i].position.r, lights[i].position.g, lights[i].position.b);
            gpuLights[i].intensity.r = lights[i].intensity.r;
            gpuLights[i].intensity.g = lights[i].intensity.g;
            gpuLights[i].intensity.b = lights[i].intensity.b;
        }
        CUDA_CHECK(cuMemAlloc(&d_lights, numLights * sizeof(::wpt::PointLight)));
        CUDA_CHECK(cuMemcpyHtoD(d_lights, gpuLights.data(), numLights * sizeof(::wpt::PointLight)));
        CUDA_CHECK(cuMemAlloc(&d_visibilityBuffer, numPixels * numLights * sizeof(uint32_t)));
    }

    // Upload area lights
    const auto& areaLights = scene->getAreaLights();
    uint32_t numAreaLights = static_cast<uint32_t>(areaLights.size());
    if (numAreaLights > 0) {
        std::vector<::wpt::AreaLight> gpuAreaLights(numAreaLights);
        for (uint32_t i = 0; i < numAreaLights; ++i) {
            const auto& al = areaLights[i];
            gpuAreaLights[i].position = make_float3(al.position.r, al.position.g, al.position.b);
            gpuAreaLights[i].normal = make_float3(al.normal.r, al.normal.g, al.normal.b);
            gpuAreaLights[i].tangent = make_float3(al.tangent.r, al.tangent.g, al.tangent.b);
            gpuAreaLights[i].width = al.width;
            gpuAreaLights[i].height = al.height;
            gpuAreaLights[i].emission.r = al.emission.r;
            gpuAreaLights[i].emission.g = al.emission.g;
            gpuAreaLights[i].emission.b = al.emission.b;
            gpuAreaLights[i].doubleSided = al.doubleSided ? 1u : 0u;
        }
        CUDA_CHECK(cuMemAlloc(&d_areaLights, numAreaLights * sizeof(::wpt::AreaLight)));
        CUDA_CHECK(cuMemcpyHtoD(d_areaLights, gpuAreaLights.data(), numAreaLights * sizeof(::wpt::AreaLight)));
    }

    ::wpt::GlobalState hostG = {};
    hostG.rayPool = reinterpret_cast<::wpt::RayState*>(d_rayPool);
    hostG.hitBuffer = reinterpret_cast<::wpt::HitInfo*>(d_hitBuffer);
    hostG.materials = scene->getNumMaterials() ? reinterpret_cast<::wpt::MaterialData*>(scene->getMaterialBuffer()) : nullptr;
    hostG.lights = d_lights ? reinterpret_cast<::wpt::PointLight*>(d_lights) : nullptr;
    hostG.numLights = static_cast<uint32_t>(scene->getLights().size());
    hostG.visibilityBuffer = d_visibilityBuffer ? reinterpret_cast<uint32_t*>(d_visibilityBuffer) : nullptr;
    hostG.areaLights = d_areaLights ? reinterpret_cast<::wpt::AreaLight*>(d_areaLights) : nullptr;
    hostG.numAreaLights = static_cast<uint32_t>(scene->getAreaLights().size());
    hostG.activeQueue.indices = reinterpret_cast<uint32_t*>(d_activeIndices0);
    hostG.activeQueue.capacity = numPixels;
    hostG.nextQueue.indices = reinterpret_cast<uint32_t*>(d_activeIndices1);
    hostG.nextQueue.capacity = numPixels;
    hostG.shadowQueue.indices = nullptr;
    hostG.shadowQueue.capacity = 0;
    hostG.accumBuffer = reinterpret_cast<::wpt::float3_rgb*>(d_accumBuffer);
    hostG.queueCounters = reinterpret_cast<uint32_t*>(d_queueCounters);
    CUDA_CHECK(cuMemcpyHtoD(d_globalState, &hostG, sizeof(::wpt::GlobalState)));

    CameraParamsHost camParams = makeCameraParams(camera, width, height);

    const uint32_t effectiveSpp = (spp < 1u) ? 1u : spp;
    for (uint32_t sampleIndex = 0; sampleIndex < effectiveSpp; ++sampleIndex) {
        if (sampleIndex == 0) {
            CUDA_CHECK(cuMemsetD8(d_accumBuffer, 0, numPixels * sizeof(::wpt::float3_rgb)));
        }

        wpt_launchGeneratePrimaryRays(
            reinterpret_cast<::wpt::RayState*>(d_rayPool),
            reinterpret_cast<uint32_t*>(d_activeIndices0),
            &camParams,
            numPixels,
            gridSize,
            blockSize,
            sampleIndex);
        CUDA_CHECK(cuCtxSynchronize());

        uint32_t numActive = numPixels;
        CUdeviceptr d_activeCur = d_activeIndices0;
        CUdeviceptr d_nextCur = d_activeIndices1;
        const int maxDepth = 8;

        for (int depth = 0; depth < maxDepth && numActive > 0; ++depth) {
        uint32_t zeroCounters[2] = { 0, 0 };
        CUDA_CHECK(cuMemcpyHtoD(d_queueCounters, zeroCounters, sizeof(zeroCounters)));

        TraceLaunchParamsHost plp;
        plp.traversable = scene->getTraversable();
        plp.rayPool = reinterpret_cast<void*>(d_rayPool);
        plp.activeQueue = reinterpret_cast<void*>(d_activeCur);
        plp.hitBuffer = reinterpret_cast<void*>(d_hitBuffer);
        plp.accumBuffer = reinterpret_cast<void*>(d_accumBuffer);
        plp.vertices = reinterpret_cast<const void*>(scene->getFirstMeshVertices());
        plp.indices = reinterpret_cast<const void*>(scene->getFirstMeshIndices());
        plp.triangleMaterialIds = reinterpret_cast<const void*>(scene->getTriangleMaterialIds());
        plp.debugMode = debugMode;
        CUDA_CHECK(cuMemcpyHtoD(d_tracePlp, &plp, sizeof(plp)));

        OPTIX_CHECK(optixLaunch(
            m_traceHandle.pipeline,
            0,
            d_tracePlp,
            sizeof(plp),
            &m_traceHandle.sbt,
            numActive,
            1,
            1
        ));
        CUDA_CHECK(cuCtxSynchronize());

        // Shadow ray pass for NEE visibility (point lights only)
        if (numLights > 0) {
            CUDA_CHECK(cuMemAlloc(&d_shadowPlp, sizeof(ShadowLaunchParamsHost)));

            ShadowLaunchParamsHost shadowPlp;
            shadowPlp.traversable = scene->getTraversable();
            shadowPlp.hitBuffer = reinterpret_cast<void*>(d_hitBuffer);
            shadowPlp.lights = reinterpret_cast<void*>(d_lights);
            shadowPlp.numLights = numLights;
            shadowPlp.activeQueue = reinterpret_cast<uint32_t*>(d_activeCur);
            shadowPlp.numActive = numActive;
            shadowPlp.visibilityBuffer = reinterpret_cast<void*>(d_visibilityBuffer);
            CUDA_CHECK(cuMemcpyHtoD(d_shadowPlp, &shadowPlp, sizeof(shadowPlp)));

            const uint32_t numShadowRays = numActive * numLights;
            OPTIX_CHECK(optixLaunch(
                m_shadowHandle.pipeline,
                0,
                d_shadowPlp,
                sizeof(shadowPlp),
                &m_shadowHandle.sbt,
                numShadowRays,
                1,
                1
            ));
            CUDA_CHECK(cuCtxSynchronize());
            CUDA_CHECK(cuMemFree(d_shadowPlp));
            d_shadowPlp = 0;
        }

        hostG.activeQueue.indices = reinterpret_cast<uint32_t*>(d_activeCur);
        hostG.nextQueue.indices = reinterpret_cast<uint32_t*>(d_nextCur);
        hostG.visibilityBuffer = d_visibilityBuffer ? reinterpret_cast<uint32_t*>(d_visibilityBuffer) : nullptr;
        CUDA_CHECK(cuMemcpyHtoD(d_globalState, &hostG, sizeof(::wpt::GlobalState)));

        wpt_launchShadeStage(
            reinterpret_cast<::wpt::GlobalState*>(d_globalState),
            numActive,
            (numActive + blockSize - 1) / blockSize,
            blockSize);
        CUDA_CHECK(cuCtxSynchronize());

        uint32_t nextCounters[2];
        CUDA_CHECK(cuMemcpyDtoH(nextCounters, d_queueCounters, sizeof(nextCounters)));
        numActive = nextCounters[0];

        std::swap(d_activeCur, d_nextCur);
        }

        std::cout << "[Renderer] Sample " << (sampleIndex + 1) << "/" << effectiveSpp << " done" << std::endl;
    }

    struct Float3Rgb { float r, g, b; };
    std::vector<Float3Rgb> hostAccum(numPixels);
    CUDA_CHECK(cuMemcpyDtoH(hostAccum.data(), d_accumBuffer, numPixels * sizeof(Float3Rgb)));

    // Apply denoiser if enabled
    if (enableDenoiser) {
        setupDenoiser(width, height);
        denoise(d_accumBuffer, width, height);
        // Download denoised result
        CUDA_CHECK(cuMemcpyDtoH(hostAccum.data(), d_accumBuffer, numPixels * sizeof(Float3Rgb)));
    }

    const float invSpp = 1.0f / static_cast<float>(effectiveSpp);
    for (uint32_t i = 0; i < numPixels; ++i) {
        outputBuffer[i][0] = hostAccum[i].r * invSpp;
        outputBuffer[i][1] = hostAccum[i].g * invSpp;
        outputBuffer[i][2] = hostAccum[i].b * invSpp;
    }
    
    CUDA_CHECK(cuMemFree(d_rayPool));
    CUDA_CHECK(cuMemFree(d_activeIndices0));
    CUDA_CHECK(cuMemFree(d_activeIndices1));
    CUDA_CHECK(cuMemFree(d_hitBuffer));
    CUDA_CHECK(cuMemFree(d_accumBuffer));
    CUDA_CHECK(cuMemFree(d_queueCounters));
    CUDA_CHECK(cuMemFree(d_globalState));
    CUDA_CHECK(cuMemFree(d_tracePlp));
    if (d_visibilityBuffer) {
        CUDA_CHECK(cuMemFree(d_visibilityBuffer));
    }
    if (d_lights) {
        CUDA_CHECK(cuMemFree(d_lights));
    }
    if (d_areaLights) {
        CUDA_CHECK(cuMemFree(d_areaLights));
    }
}

void RendererImpl::setupDenoiser(uint32_t width, uint32_t height) {
    if (m_denoiserCreated && m_denoiserWidth == width && m_denoiserHeight == height) {
        return;  // Already setup for this resolution
    }

    cleanupDenoiser();

    // Create denoiser
    OptixDenoiserOptions denoiserOptions = {};
    denoiserOptions.guideAlbedo = 0;  // No albedo guide
    denoiserOptions.guideNormal = 0;  // No normal guide

    OPTIX_CHECK(optixDenoiserCreate(
        m_context->getOptixContext(),
        OPTIX_DENOISER_MODEL_KIND_LDR,
        &denoiserOptions,
        &m_denoiser));

    // Get memory requirements
    OptixDenoiserSizes denoiserSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(
        m_denoiser,
        width,
        height,
        &denoiserSizes));

    // Allocate state and scratch memory
    CUDA_CHECK(cuMemAlloc(&d_denoiserState, denoiserSizes.stateSizeInBytes));
    CUDA_CHECK(cuMemAlloc(&d_denoiserScratch, denoiserSizes.withoutOverlapScratchSizeInBytes));

    // Setup denoiser
    OPTIX_CHECK(optixDenoiserSetup(
        m_denoiser,
        0,  // CUDA stream
        width,
        height,
        d_denoiserState,
        denoiserSizes.stateSizeInBytes,
        d_denoiserScratch,
        denoiserSizes.withoutOverlapScratchSizeInBytes));

    m_denoiserWidth = width;
    m_denoiserHeight = height;
    m_denoiserCreated = true;

    std::cout << "[Denoiser] Setup complete (" << width << "x" << height << ")" << std::endl;
}

void RendererImpl::denoise(CUdeviceptr d_beauty, uint32_t width, uint32_t height) {
    if (!m_denoiserCreated) {
        return;
    }

    // Setup input/output layers
    OptixImage2D inputLayer = {};
    inputLayer.data = d_beauty;
    inputLayer.width = width;
    inputLayer.height = height;
    inputLayer.rowStrideInBytes = width * sizeof(float) * 3;
    inputLayer.pixelStrideInBytes = sizeof(float) * 3;
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixImage2D outputLayer = inputLayer;  // Denoise in-place

    // Denoise parameters (OptiX 8.0)
    OptixDenoiserParams denoiserParams = {};
    denoiserParams.hdrIntensity = 0;  // Auto-exposure (null pointer)
    denoiserParams.blendFactor = 0.0f;  // 100% denoised output

    OptixDenoiserGuideLayer guideLayer = {};  // No guide layers
    OptixDenoiserLayer layer = {};
    layer.input = inputLayer;
    layer.output = outputLayer;

    // Get denoiser sizes for proper memory requirements
    OptixDenoiserSizes denoiserSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(
        m_denoiser,
        width,
        height,
        &denoiserSizes));

    // Invoke denoiser
    OPTIX_CHECK(optixDenoiserInvoke(
        m_denoiser,
        0,  // CUDA stream
        &denoiserParams,
        d_denoiserState,
        denoiserSizes.stateSizeInBytes,
        &guideLayer,
        &layer,
        1,  // num layers
        0,  // input offset X
        0,  // input offset Y
        d_denoiserScratch,
        denoiserSizes.withoutOverlapScratchSizeInBytes));

    std::cout << "[Denoiser] Denoising complete" << std::endl;
}

void RendererImpl::cleanupDenoiser() {
    if (m_denoiserCreated) {
        if (d_denoiserScratch) CUDA_CHECK(cuMemFree(d_denoiserScratch));
        if (d_denoiserState) CUDA_CHECK(cuMemFree(d_denoiserState));
        if (m_denoiser) OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
        
        d_denoiserScratch = 0;
        d_denoiserState = 0;
        m_denoiser = nullptr;
        m_denoiserCreated = false;
    }
}

Renderer::~Renderer() {
    delete m_impl;
}

void Renderer::render(
    Scene* scene, const Camera& camera, RGB* outputBuffer,
    uint32_t width, uint32_t height, uint32_t spp, bool enableDenoiser, int debugMode)
{
    m_impl->render(scene->m_impl, camera, outputBuffer, width, height, spp, enableDenoiser, debugMode);
}

} // namespace vlrw
