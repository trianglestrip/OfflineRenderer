#include "wr/wr.h"
#include "utils/cuda_utils.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

namespace wr {

extern OptixDeviceContext getOptixContext();

static std::vector<char> loadPTX(const char* filename) {
    std::filesystem::path exePath = std::filesystem::current_path();
    std::filesystem::path ptxPath = exePath / filename;
    
    std::ifstream file(ptxPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open PTX file: ") + ptxPath.string());
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    
    return buffer;
}

struct PipelineImpl {
    OptixDeviceContext context;
    OptixPipeline pipeline = nullptr;
    OptixModule traceModule = nullptr;
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;
    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_raygenRecord = 0;
    CUdeviceptr d_missRecord = 0;
    CUdeviceptr d_hitgroupRecord = 0;
    
    PipelineImpl() : context(getOptixContext()) {}
    
    void create() {
        std::cout << "[Pipeline] Creating OptiX pipeline..." << std::endl;
        
        std::vector<char> tracePTX = loadPTX("trace.ptx");
        
        OptixModuleCompileOptions moduleOptions = {};
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        
        OptixPipelineCompileOptions pipelineOptions = {};
        pipelineOptions.usesMotionBlur = false;
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineOptions.numPayloadValues = 1;
        pipelineOptions.numAttributeValues = 0;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.pipelineLaunchParamsVariableName = "params";
        pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        
        char log[2048];
        size_t logSize = sizeof(log);
        
        OPTIX_CHECK(optixModuleCreate(
            context,
            &moduleOptions,
            &pipelineOptions,
            tracePTX.data(),
            tracePTX.size(),
            log, &logSize,
            &traceModule
        ));
        
        OptixProgramGroupOptions pgOptions = {};
        
        OptixProgramGroupDesc raygenDesc = {};
        raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDesc.raygen.module = traceModule;
        raygenDesc.raygen.entryFunctionName = "__raygen__trace";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context, &raygenDesc, 1, &pgOptions, log, &logSize, &raygenPG));
        
        OptixProgramGroupDesc missDesc = {};
        missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc.miss.module = traceModule;
        missDesc.miss.entryFunctionName = "__miss__trace";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context, &missDesc, 1, &pgOptions, log, &logSize, &missPG));
        
        OptixProgramGroupDesc hitgroupDesc = {};
        hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroupDesc.hitgroup.moduleCH = traceModule;
        hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__trace";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroupDesc, 1, &pgOptions, log, &logSize, &hitgroupPG));
        
        OptixProgramGroup programGroups[] = { raygenPG, missPG, hitgroupPG };
        
        OptixPipelineLinkOptions linkOptions = {};
        linkOptions.maxTraceDepth = 1;
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(
            context,
            &pipelineOptions,
            &linkOptions,
            programGroups,
            3,
            log, &logSize,
            &pipeline
        ));
        
        buildSBT();
        
        std::cout << "[Pipeline] Pipeline created" << std::endl;
    }
    
    void buildSBT() {
        const size_t recordSize = OPTIX_SBT_RECORD_HEADER_SIZE;
        
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), recordSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecord), recordSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroupRecord), recordSize));
        
        std::vector<char> raygenRecord(recordSize);
        std::vector<char> missRecord(recordSize);
        std::vector<char> hitgroupRecord(recordSize);
        
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, raygenRecord.data()));
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG, missRecord.data()));
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, hitgroupRecord.data()));
        
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygenRecord), raygenRecord.data(), recordSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_missRecord), missRecord.data(), recordSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroupRecord), hitgroupRecord.data(), recordSize, cudaMemcpyHostToDevice));
        
        sbt.raygenRecord = d_raygenRecord;
        sbt.missRecordBase = d_missRecord;
        sbt.missRecordStrideInBytes = recordSize;
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_hitgroupRecord;
        sbt.hitgroupRecordStrideInBytes = recordSize;
        sbt.hitgroupRecordCount = 1;
        
        std::cout << "[Pipeline] SBT built" << std::endl;
    }
    
    ~PipelineImpl() {
        if (d_raygenRecord) cudaFree(reinterpret_cast<void*>(d_raygenRecord));
        if (d_missRecord) cudaFree(reinterpret_cast<void*>(d_missRecord));
        if (d_hitgroupRecord) cudaFree(reinterpret_cast<void*>(d_hitgroupRecord));
        if (raygenPG) optixProgramGroupDestroy(raygenPG);
        if (missPG) optixProgramGroupDestroy(missPG);
        if (hitgroupPG) optixProgramGroupDestroy(hitgroupPG);
        if (pipeline) optixPipelineDestroy(pipeline);
        if (traceModule) optixModuleDestroy(traceModule);
    }
};

PipelineImpl* createPipeline() {
    PipelineImpl* impl = new PipelineImpl();
    impl->create();
    return impl;
}

void launchPipeline(PipelineImpl* impl, CUdeviceptr paramsPtr, uint32_t numActive) {
    OPTIX_CHECK(optixLaunch(
        impl->pipeline,
        0,
        paramsPtr,
        sizeof(void*),
        &impl->sbt,
        numActive,
        1,
        1
    ));
}

void destroyPipeline(PipelineImpl* impl) {
    delete impl;
}

} // namespace wr
