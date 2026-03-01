#include "optixw/optixw.h"
#include "optixw/renderer_impl.h"
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

// forward declarations of helpers defined in renderer.cpp
namespace optixw {
std::vector<char> loadPTX(const char* filename);
std::filesystem::path findCubinPath(const char* fileName);
}

using namespace optixw;

void Impl::createPipeline() {
    if (pipelineCreated) return;
    
    std::cout << "[Renderer] Creating OptiX pipeline..." << std::endl;
    
    // Load PTX
    std::vector<char> tracePTX = loadPTX("trace.ptx");
    
    // Create module
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 1;
    pipelineCompileOptions.numAttributeValues = 0;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    
    char log[2048];
    size_t logSize = sizeof(log);
    
    OPTIX_CHECK(optixModuleCreate(
        g_optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        tracePTX.data(),
        tracePTX.size(),
        log, &logSize,
        &traceModule
    ));
    
    if (logSize > 1) {
        std::cout << "[OptiX Module] " << log << std::endl;
    }
    
    // Create program groups
    OptixProgramGroupOptions pgOptions = {};
    
    // Raygen
    OptixProgramGroupDesc raygenPGDesc = {};
    raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenPGDesc.raygen.module = traceModule;
    raygenPGDesc.raygen.entryFunctionName = "__raygen__trace";
    
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        g_optixContext,
        &raygenPGDesc,
        1,
        &pgOptions,
        log, &logSize,
        &programGroups.raygenPG
    ));
    
    // Miss
    OptixProgramGroupDesc missPGDesc = {};
    missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missPGDesc.miss.module = traceModule;
    missPGDesc.miss.entryFunctionName = "__miss__trace";
    
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        g_optixContext,
        &missPGDesc,
        1,
        &pgOptions,
        log, &logSize,
        &programGroups.missPG
    ));
    
    // Hitgroup
    OptixProgramGroupDesc hitgroupPGDesc = {};
    hitgroupPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupPGDesc.hitgroup.moduleCH = traceModule;
    hitgroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__trace";
    
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        g_optixContext,
        &hitgroupPGDesc,
        1,
        &pgOptions,
        log, &logSize,
        &programGroups.hitgroupPG
    ));
    
    // Link pipeline
    OptixProgramGroup programGroupsArray[] = { programGroups.raygenPG, programGroups.missPG, programGroups.hitgroupPG };
    
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    
    logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        g_optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroupsArray,
        3,
        log, &logSize,
        &pipeline
    ));
    
    if (logSize > 1) {
        std::cout << "[OptiX Pipeline] " << log << std::endl;
    }
    
    // Build SBT
    buildSBT();
    createWavefrontKernels();

    pipelineCreated = true;
    std::cout << "[Renderer] Pipeline created successfully" << std::endl;
}

void Impl::createWavefrontKernels() {
    const std::string shadeCubin = findCubinPath("shade.cubin").string();
    CU_CHECK(cuModuleLoad(&kernelModules.shadeModule, shadeCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&kernelFunctions.shadeKernel, kernelModules.shadeModule, "shade"));

    const std::string compactCubin = findCubinPath("compact.cubin").string();
    CU_CHECK(cuModuleLoad(&kernelModules.compactModule, compactCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&kernelFunctions.compactKernel, kernelModules.compactModule, "compact"));

    const std::string scaleCubin = findCubinPath("scale.cubin").string();
    CU_CHECK(cuModuleLoad(&kernelModules.scaleModule, scaleCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&kernelFunctions.scaleKernel, kernelModules.scaleModule, "scale_to_float4"));

    const std::string mergeCubin = findCubinPath("denoise_merge.cubin").string();
    CU_CHECK(cuModuleLoad(&kernelModules.mergeModule, mergeCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&kernelFunctions.mergeKernel, kernelModules.mergeModule, "merge_tile"));
    CU_CHECK(cuModuleGetFunction(&kernelFunctions.normalizeKernel, kernelModules.mergeModule, "normalize_accum"));

    std::cout << "[Renderer] Wavefront kernels loaded" << std::endl;
}

void Impl::buildSBT() {
    // Raygen record
    struct RaygenRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    RaygenRecord raygenRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.raygenPG, &raygenRecord));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbtRecords.raygenRecord), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(
        (void*)sbtRecords.raygenRecord,
        &raygenRecord,
        sizeof(RaygenRecord),
        cudaMemcpyHostToDevice
    ));
    sbt.raygenRecord = sbtRecords.raygenRecord;
    
    // Miss record
    struct MissRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    MissRecord missRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.missPG, &missRecord));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbtRecords.missRecord), sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(
        (void*)sbtRecords.missRecord,
        &missRecord,
        sizeof(MissRecord),
        cudaMemcpyHostToDevice
    ));
    sbt.missRecordBase = sbtRecords.missRecord;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 1;
    
    // Hitgroup record
    struct HitgroupRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    HitgroupRecord hitgroupRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitgroupPG, &hitgroupRecord));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbtRecords.hitgroupRecord), sizeof(HitgroupRecord)));
    CUDA_CHECK(cudaMemcpy(
        (void*)sbtRecords.hitgroupRecord,
        &hitgroupRecord,
        sizeof(HitgroupRecord),
        cudaMemcpyHostToDevice
    ));
    sbt.hitgroupRecordBase = sbtRecords.hitgroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = 1;
    
    std::cout << "[Renderer] SBT built" << std::endl;
}
