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
      numPixels(0), maxRays(0), environmentRadiance(make_float3(0.0f,0.0f,0.0f)),
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
    accumBuffers.free();
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
    accumBuffers.allocate(accumSize);
    
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
    CUDA_CHECK(cudaMemset((void*)accumBuffers.d_accumBuffer, 0, accumSize));
    CUDA_CHECK(cudaMemset((void*)accumBuffers.d_albedoBuffer, 0, accumSize));
    CUDA_CHECK(cudaMemset((void*)accumBuffers.d_normalBuffer, 0, accumSize));
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
    m_impl->environmentRadiance = toFloat3(env);
    m_impl->environmentMap.d_map = SceneAccessor::getEnvironmentMapPtr(scene);
    m_impl->environmentMap.width = SceneAccessor::getEnvironmentMapWidth(scene);
    m_impl->environmentMap.height = SceneAccessor::getEnvironmentMapHeight(scene);
    m_impl->environmentMap.scale = SceneAccessor::getEnvironmentMapScale(scene);
    
    // Get light data
    m_impl->lightBuffers.d_pointLights = SceneAccessor::getPointLightsPtr(scene);
    m_impl->lightBuffers.d_areaLights = SceneAccessor::getAreaLightsPtr(scene);
    m_impl->lightBuffers.numPointLights = SceneAccessor::getNumPointLights(scene);
    m_impl->lightBuffers.numAreaLights = SceneAccessor::getNumAreaLights(scene);
    
    std::cout << "[Renderer] GAS handle: " << gasHandle << std::endl;
    
    // Render samples using Taskflow for parallelization if we have a scheduler
    if (spp > 1 && m_impl->scheduler != nullptr) {
        // Use Taskflow to parallelize sample rendering
        tf::Taskflow taskflow;
        
        // Split samples into chunks for parallel processing
        const uint32_t samplesPerChunk = std::max(1u, spp / std::min(spp, 4u)); // At most 4 chunks or 1 sample per chunk
        
        for (uint32_t chunkStart = 0; chunkStart < spp; chunkStart += samplesPerChunk) {
            uint32_t chunkEnd = std::min(chunkStart + samplesPerChunk, spp);
            
            taskflow.emplace([this, gasHandle, d_vertices, d_texcoords, d_indices, d_triangleMaterialIds, 
                             &camData, width, height, chunkStart, chunkEnd]() {
                for (uint32_t sample = chunkStart; sample < chunkEnd; ++sample) {
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
            });
        }
        
        // Execute the taskflow
        m_impl->scheduler->run(taskflow);
        
        // Print progress for each sample
        for (uint32_t sample = 0; sample < spp; ++sample) {
            std::cout << "\r[Renderer] Sample " << (sample + 1) << "/" << spp << std::flush;
        }
    } else {
        // Fall back to serial rendering if no scheduler or single sample
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
    }
    
    std::cout << std::endl;
    
    uint32_t checkCount = (100u < m_impl->numPixels) ? 100u : m_impl->numPixels;

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
        const float3* d_accum = reinterpret_cast<const float3*>(m_impl->accumBuffers.d_accumBuffer);
        float4* d_input = reinterpret_cast<float4*>(m_impl->denoiser.imageBuffers.d_input);
        void* scaleArgs[] = { &d_accum, &d_input, (void*)&count, (void*)&invSpp };

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

        // also guide buffers
        const float3* d_albedo = reinterpret_cast<const float3*>(m_impl->accumBuffers.d_albedoBuffer);
        const float3* d_normal = reinterpret_cast<const float3*>(m_impl->accumBuffers.d_normalBuffer);
        float4* d_albedoInput = reinterpret_cast<float4*>(m_impl->denoiser.imageBuffers.d_albedoInput);
        float4* d_normalInput = reinterpret_cast<float4*>(m_impl->denoiser.imageBuffers.d_normalInput);
        const float invOne = 1.0f;

        void* scaleAlbedoArgs[] = { &d_albedo, &d_albedoInput, (void*)&count, (void*)&invOne };
        void* scaleNormalArgs[] = { &d_normal, &d_normalInput, (void*)&count, (void*)&invOne };
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

        // compute tile tasks (input region may include overlap, output region
        // is strictly the non‑overlapping tile).  we also store computed input
        // width/height so later the baseInput can be sized correctly per task.
        struct Task { uint32_t inX,inY,inW,inH; uint32_t outX,outY,outW,outH; };
        std::vector<Task> tasks;
        uint32_t ov = m_impl->denoiser.overlap;

        if (tileW < width || tileH < height) {
            // adjust overlap per-dimension so that it never exceeds half the
            // tile size; this guarantees adjacent tiles will have distinct
            // input regions even if the denoiser overlap is huge.
            uint32_t ovx = std::min(ov, tileW/2);
            uint32_t ovy = std::min(ov, tileH/2);

            for (uint32_t y = 0; y < height; y += tileH) {
                uint32_t outH = std::min(tileH, height - y);
                uint32_t topOverlap = (y > ovy) ? ovy : y;
                uint32_t spaceBottom = height - (y + outH);
                uint32_t bottomOverlap = std::min(ovy, spaceBottom);
                uint32_t inY = (y > ovy) ? (y - ovy) : 0;
                uint32_t inH = outH + topOverlap + bottomOverlap;
                if (inY + inH > height) inH = height - inY;

                for (uint32_t x = 0; x < width; x += tileW) {
                    uint32_t outW = std::min(tileW, width - x);
                    uint32_t leftOverlap = (x > ovx) ? ovx : x;
                    uint32_t spaceRight = width - (x + outW);
                    uint32_t rightOverlap = std::min(ovx, spaceRight);
                    uint32_t inX = (x > ovx) ? (x - ovx) : 0;
                    uint32_t inW = outW + leftOverlap + rightOverlap;
                    if (inX + inW > width) inW = width - inX;

                    tasks.push_back({inX, inY, inW, inH, x, y, outW, outH});
                }
            }
        } else {
            // single-task for whole image
            tasks.push_back({0,0,width,height,0,0,width,height});
        }

        // debug: print task count and each entry coordinates
        std::cout << "[Renderer] denoiser tiling: " << tasks.size() << " tasks\n";
        for (size_t ti = 0; ti < tasks.size(); ++ti) {
            auto &t = tasks[ti];
            std::cout << "  task[" << ti << "] in=" << t.inX << "," << t.inY << " (" << t.inW << "x" << t.inH << ")"
                      << " out=" << t.outX << "," << t.outY << " (" << t.outW << "x" << t.outH << ")\n";
        }

        

        // perform denoiser per task
        // If any computed task input covers a full image dimension, fall back
        // to single-task denoising to avoid mismatched input contexts.
        bool invalidMixing = false;
        // Diagnostic override: when collecting low-level diagnostics, allow
        // forcing tiling even if conservative checks would disable it. Set
        // to false for normal operation.
        const bool diagnosticForceTiling = true;
        for (auto &tt : tasks) {
            if (tt.inW == width || tt.inH == height) { invalidMixing = true; break; }
        }
        if (invalidMixing && tasks.size() > 1 && !diagnosticForceTiling) {
            std::cout << "[Renderer] tile input spans full image in at least one task; disabling tiling to avoid seams\n";
            tasks.clear();
            tasks.push_back({0,0,width,height,0,0,width,height});
        } else if (invalidMixing && tasks.size() > 1 && diagnosticForceTiling) {
            std::cout << "[Renderer] WARNING: diagnosticForceTiling active — proceeding with tiled tasks despite conservative mixing checks\n";
        }

        // --- Diagnostic: collect seam sample points and capture four checkpoints
        struct DiagPoint { uint32_t gx, gy; uint32_t taskIdx; uint32_t localX, localY; float4 beforeCopy; float4 afterCopyToTile; float4 afterDenoise; float4 afterCopyBack; };
        std::vector<DiagPoint> diagPoints;

        // For each task boundary, pick representative seam pixels (left/right and top/bottom)
        for (size_t ti = 0; ti < tasks.size(); ++ti) {
            auto &t = tasks[ti];
            if (t.outX > 0) {
                uint32_t seamX = t.outX;
                uint32_t seamY = t.outY + t.outH/2;
                if (seamY >= height) seamY = height-1;
                if (seamX > 0) diagPoints.push_back({seamX-1, seamY, (uint32_t)ti, 0,0});
                if (seamX < width) diagPoints.push_back({seamX, seamY, (uint32_t)ti, 0,0});
            }
            if (t.outY > 0) {
                uint32_t seamY = t.outY;
                uint32_t seamX = t.outX + t.outW/2;
                if (seamX >= width) seamX = width-1;
                if (seamY > 0) diagPoints.push_back({seamX, seamY-1, (uint32_t)ti, 0,0});
                if (seamY < height) diagPoints.push_back({seamX, seamY, (uint32_t)ti, 0,0});
            }
        }

        // Fallback: if no boundaries, sample a few representative pixels
        if (diagPoints.empty()) {
            uint32_t cx = width / 2;
            uint32_t cy = height / 2;
            diagPoints.push_back({cx, cy, 0, 0, 0});
            diagPoints.push_back({(cx>1)?cx-1:0, cy, 0, 0, 0});
            diagPoints.push_back({(cx+1<width)?cx+1:width-1, cy, 0, 0, 0});
            diagPoints.push_back({cx, (cy>1)?cy-1:0, 0, 0, 0});
        }

        // compute local coords and initialize diagnostics
        for (auto &p : diagPoints) {
            if (p.gx >= width) p.gx = width-1;
            if (p.gy >= height) p.gy = height-1;
            auto &t = tasks[p.taskIdx];
            if (p.gx < t.inX) p.localX = 0; else p.localX = p.gx - t.inX;
            if (p.gy < t.inY) p.localY = 0; else p.localY = p.gy - t.inY;
            p.beforeCopy = make_float4(-1.0f,-1.0f,-1.0f,-1.0f);
            p.afterCopyToTile = make_float4(-1.0f,-1.0f,-1.0f,-1.0f);
            p.afterDenoise = make_float4(-1.0f,-1.0f,-1.0f,-1.0f);
            p.afterCopyBack = make_float4(-1.0f,-1.0f,-1.0f,-1.0f);
        }

        // capture "before-copy" values from the full-image contiguous input
        for (auto &p : diagPoints) {
            float4 tmp = {};
            void* src = (void*)(m_impl->denoiser.imageBuffers.d_input + (p.gy * width + p.gx) * sizeof(float4));
            CUDA_CHECK(cudaMemcpy(&tmp, src, sizeof(float4), cudaMemcpyDeviceToHost));
            p.beforeCopy = tmp;
        }

        for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
            auto &t = tasks[taskIndex];
            // copy the tile input region (which may have arbitrary strides in
            // the full-image buffer) into a contiguous temporary buffer so the
            // denoiser can process it correctly
            for (uint32_t ty = 0; ty < t.inH; ++ty) {
                void* srcRow = (void*)(m_impl->denoiser.imageBuffers.d_input + ((t.inY + ty) * width + t.inX) * sizeof(float4));
                void* dstRow = (void*)(m_impl->denoiser.tileBuffers.tileInputBuffer + ty * t.inW * sizeof(float4));
                CUDA_CHECK(cudaMemcpy(dstRow, srcRow, t.inW * sizeof(float4), cudaMemcpyDeviceToDevice));
            }
            // do the same for guidance buffers into their dedicated tile buffers
            if (m_impl->denoiser.imageBuffers.d_albedoInput) {
                for (uint32_t ty = 0; ty < t.inH; ++ty) {
                    void* srcRow = (void*)(m_impl->denoiser.imageBuffers.d_albedoInput + ((t.inY + ty) * width + t.inX) * sizeof(float4));
                    void* dstRow = (void*)(m_impl->denoiser.tileBuffers.tileAlbedoBuffer + ty * t.inW * sizeof(float4));
                    CUDA_CHECK(cudaMemcpy(dstRow, srcRow, t.inW * sizeof(float4), cudaMemcpyDeviceToDevice));
                }
            }
            if (m_impl->denoiser.imageBuffers.d_normalInput) {
                for (uint32_t ty = 0; ty < t.inH; ++ty) {
                    void* srcRow = (void*)(m_impl->denoiser.imageBuffers.d_normalInput + ((t.inY + ty) * width + t.inX) * sizeof(float4));
                    void* dstRow = (void*)(m_impl->denoiser.tileBuffers.tileNormalBuffer + ty * t.inW * sizeof(float4));
                    CUDA_CHECK(cudaMemcpy(dstRow, srcRow, t.inW * sizeof(float4), cudaMemcpyDeviceToDevice));
                }
            }

            // after-copy-to-tile: sample relevant diagnostic points that belong to this task
            for (auto &p : diagPoints) {
                // if this point lies within this task's input region, sample it
                if (p.gx < t.inX || p.gx >= t.inX + t.inW || p.gy < t.inY || p.gy >= t.inY + t.inH) continue;
                float4 tmp = {};
                uint32_t localX = p.gx - t.inX;
                uint32_t localY = p.gy - t.inY;
                void* src = (void*)(m_impl->denoiser.tileBuffers.tileInputBuffer + (localY * t.inW + localX) * sizeof(float4));
                CUDA_CHECK(cudaMemcpy(&tmp, src, sizeof(float4), cudaMemcpyDeviceToHost));
                p.afterCopyToTile = tmp;
            }

            // update baseInput to point to the temporary tile input buffer (now
            // contiguous with correct dimensions)
            baseInput.data = m_impl->denoiser.tileBuffers.tileInputBuffer;
            baseInput.width = t.inW;
            baseInput.height = t.inH;
            baseInput.rowStrideInBytes = t.inW * sizeof(float4); // contiguous

            // configure output to temporary tile buffer sized to the input
            // region; we'll copy the valid subset into the final image later.
            OptixImage2D output = {};
            output.data = m_impl->denoiser.tileBuffers.tileBuffer;
            output.width = t.inW;
            output.height = t.inH;
            output.rowStrideInBytes = t.inW * sizeof(float4);
            output.pixelStrideInBytes = sizeof(float4);
            output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

            OptixDenoiserLayer layer = {};
            layer.input = baseInput;
            layer.output = output;
            layer.previousOutput = {};
            layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;

            OptixDenoiserGuideLayer guide = {};
            guide.albedo.data = 0;
            guide.albedo.width = 0;
            guide.albedo.height = 0;
            guide.albedo.rowStrideInBytes = 0;
            guide.albedo.pixelStrideInBytes = 0;
            guide.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;
            guide.normal = guide.albedo;
            if (m_impl->denoiser.imageBuffers.d_albedoInput) {
                guide.albedo.data = m_impl->denoiser.tileBuffers.tileAlbedoBuffer;
                guide.albedo.width = t.inW;
                guide.albedo.height = t.inH;
                guide.albedo.rowStrideInBytes = t.inW * sizeof(float4);
                guide.albedo.pixelStrideInBytes = sizeof(float4);
            }
            if (m_impl->denoiser.imageBuffers.d_normalInput) {
                guide.normal.data = m_impl->denoiser.tileBuffers.tileNormalBuffer;
                guide.normal.width = t.inW;
                guide.normal.height = t.inH;
                guide.normal.rowStrideInBytes = t.inW * sizeof(float4);
                guide.normal.pixelStrideInBytes = sizeof(float4);
                guide.normal.format = OPTIX_PIXEL_FORMAT_FLOAT4;
            }

            OptixDenoiserParams params = {};
            params.hdrAverageColor = 0;
            params.blendFactor = denoiserBlend;
            // Use the precomputed full-image intensity to keep tiles consistent
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

            // after-denoiser: sample tile output for diagnostic points belonging to this task
            for (auto &p : diagPoints) {
                if (p.gx < t.inX || p.gx >= t.inX + t.inW || p.gy < t.inY || p.gy >= t.inY + t.inH) continue;
                float4 tmp = {};
                uint32_t localX = p.gx - t.inX;
                uint32_t localY = p.gy - t.inY;
                void* src = (void*)(m_impl->denoiser.tileBuffers.tileBuffer + (localY * t.inW + localX) * sizeof(float4));
                CUDA_CHECK(cudaMemcpy(&tmp, src, sizeof(float4), cudaMemcpyDeviceToHost));
                p.afterDenoise = tmp;
            }

            // Instead of direct memcpy into the final image, accumulate the
            // entire tile output into the merge accumulators using the
            // overlap-weighted kernel.  This writes into `d_mergeAccum` and
            // `d_mergeWeight` so overlapping tiles blend smoothly.  The
            // kernel is launched using the CUDA Driver API (cubin).
            uint32_t offX = t.outX - t.inX;
            uint32_t offY = t.outY - t.inY;

            int ovx = static_cast<int>(std::min<uint32_t>(m_impl->denoiser.overlap, t.inW/2));
            int ovy = static_cast<int>(std::min<uint32_t>(m_impl->denoiser.overlap, t.inH/2));
            void* mergeArgs[] = {
                &m_impl->denoiser.tileBuffers.tileBuffer,
                &t.inW,
                &t.inH,
                &t.inX,
                &t.inY,
                &offX,
                &offY,
                &width,
                &height,
                &m_impl->mergeBuffers.d_mergeAccum,
                &m_impl->mergeBuffers.d_mergeWeight,
                &ovx,
                &ovy
            };

            const unsigned int blockX = 16;
            const unsigned int blockY = 16;
            unsigned int gridX = (t.inW + blockX - 1) / blockX;
            unsigned int gridY = (t.inH + blockY - 1) / blockY;

            CU_CHECK(cuLaunchKernel(
            m_impl->kernelFunctions.mergeKernel,
            gridX, gridY, 1,
            blockX, blockY, 1,
            0,
            0,
            mergeArgs,
            nullptr
        ));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Normalize merged accumulators into the denoiser output image
        {
            void* normArgs[] = { &m_impl->mergeBuffers.d_mergeAccum, &m_impl->mergeBuffers.d_mergeWeight, &m_impl->denoiser.imageBuffers.d_output, &width, &height };
            const unsigned int normBlock = 256;
            unsigned int normGrid = (m_impl->numPixels + normBlock - 1) / normBlock;
            CU_CHECK(cuLaunchKernel(
                m_impl->kernelFunctions.normalizeKernel,
                normGrid, 1, 1,
                normBlock, 1, 1,
                0,
                0,
                normArgs,
                nullptr
            ));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // after normalization, sample assembled output buffer for diagnostics
        if (!diagPoints.empty()) {
            for (auto &p : diagPoints) {
                float4 tmp = {};
                void* outAddr = (void*)(m_impl->denoiser.imageBuffers.d_output + (p.gy * width + p.gx) * sizeof(float4));
                CUDA_CHECK(cudaMemcpy(&tmp, outAddr, sizeof(float4), cudaMemcpyDeviceToHost));
                p.afterCopyBack = tmp;
            }

            std::cout << "[Diag] Diagnostic seam samples (gx,gy) -> beforeCopy | afterCopyToTile | afterDenoise | afterCopyBack\n";
            for (size_t i = 0; i < diagPoints.size(); ++i) {
                auto &p = diagPoints[i];
                auto f4s = [](const float4 &f){ char buf[256]; sprintf(buf, "(%0.6f,%0.6f,%0.6f,%0.6f)", f.x, f.y, f.z, f.w); return std::string(buf); };
                std::cout << "[Diag] p[" << i << "] " << p.gx << "," << p.gy << " -> "
                          << f4s(p.beforeCopy) << " | "
                          << f4s(p.afterCopyToTile) << " | "
                          << f4s(p.afterDenoise) << " | "
                          << f4s(p.afterCopyBack) << "\n";
            }
        }

        // at this point the d_denoiserOutput buffer contains the assembled
        // denoised image (tiles copied into place).  download it below.


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
            (void*)m_impl->accumBuffers.d_accumBuffer,
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
