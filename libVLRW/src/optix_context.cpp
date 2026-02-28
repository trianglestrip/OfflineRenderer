#include "optix_context.h"
#include "cuda_check.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace vlrw {

static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    fprintf(stderr, "[OptiX][%u][%s]: %s\n", level, tag, message);
}

ContextImpl::ContextImpl(CUcontext cuContext) : m_cuContext(cuContext), m_optixContext(nullptr) {
    fprintf(stderr, "[ContextImpl] Initializing OptiX...\n");
    OPTIX_CHECK(optixInit());
    fprintf(stderr, "[ContextImpl] OptiX initialized\n");
    
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = optixLogCallback;
    options.logCallbackLevel = 4;
    
    fprintf(stderr, "[ContextImpl] Creating OptiX device context...\n");
    OPTIX_CHECK(optixDeviceContextCreate(m_cuContext, &options, &m_optixContext));
    fprintf(stderr, "[ContextImpl] OptiX device context created: %p\n", m_optixContext);
}

ContextImpl::~ContextImpl() {
    if (m_optixContext) {
        optixDeviceContextDestroy(m_optixContext);
    }
}

OptixModule ContextImpl::loadPTXModule(const char* ptxCode, size_t ptxSize) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = 0;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 0;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "tracePlp";

    char log[2048];
    size_t logSize = sizeof(log);

    OptixModule module;
    OPTIX_CHECK(optixModuleCreate(
        m_optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode,
        ptxSize,
        log, &logSize,
        &module
    ));

    return module;
}

TracePipelineHandle ContextImpl::createTracePipelineFromFile(const char* ptxOrOptixirPath) {
    std::ifstream f(ptxOrOptixirPath, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error(std::string("Failed to open trace module: ") + ptxOrOptixirPath);
    size_t size = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> code(size);
    if (!f.read(code.data(), size))
        throw std::runtime_error("Failed to read trace module file");
    f.close();

    OptixModule module = loadPTXModule(code.data(), size);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__trace";

    OptixProgramGroup raygenGroup = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &raygenDesc, 1, &pgOptions, nullptr, nullptr, &raygenGroup));

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__trace";

    OptixProgramGroup missGroup = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &missDesc, 1, &pgOptions, nullptr, nullptr, &missGroup));

    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH = module;
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__trace";

    OptixProgramGroup hitgroupGroup = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &hitgroupDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupGroup));

    OptixPipelineCompileOptions pipeCompileOptions = {};
    pipeCompileOptions.usesMotionBlur = false;
    pipeCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeCompileOptions.numPayloadValues = 3;
    pipeCompileOptions.numAttributeValues = 0;
    pipeCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeCompileOptions.pipelineLaunchParamsVariableName = "tracePlp";

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 2;

    OptixProgramGroup groups[] = { raygenGroup, missGroup, hitgroupGroup };
    OptixPipeline pipeline = nullptr;
    char pipeLog[2048];
    size_t pipeLogSize = sizeof(pipeLog);
    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &pipeCompileOptions,
        &linkOptions,
        groups,
        sizeof(groups) / sizeof(groups[0]),
        pipeLog,
        &pipeLogSize,
        &pipeline
    ));

    size_t raygenRecordSize = OPTIX_SBT_RECORD_HEADER_SIZE;
    size_t missRecordSize = OPTIX_SBT_RECORD_HEADER_SIZE;
    size_t hitgroupRecordSize = OPTIX_SBT_RECORD_HEADER_SIZE;

    std::vector<char> raygenRecord(raygenRecordSize);
    std::vector<char> missRecord(missRecordSize);
    std::vector<char> hitgroupRecord(hitgroupRecordSize);

    OPTIX_CHECK(optixSbtRecordPackHeader(raygenGroup, raygenRecord.data()));
    OPTIX_CHECK(optixSbtRecordPackHeader(missGroup, missRecord.data()));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupGroup, hitgroupRecord.data()));

    TracePipelineHandle handle;
    handle.pipeline = pipeline;
    handle.module = module;
    handle.raygenGroup = raygenGroup;
    handle.missGroup = missGroup;
    handle.hitgroupGroup = hitgroupGroup;

    CUDA_CHECK(cuMemAlloc(&handle.d_raygenRecord, raygenRecordSize));
    CUDA_CHECK(cuMemcpyHtoD(handle.d_raygenRecord, raygenRecord.data(), raygenRecordSize));
    CUDA_CHECK(cuMemAlloc(&handle.d_missRecord, missRecordSize));
    CUDA_CHECK(cuMemcpyHtoD(handle.d_missRecord, missRecord.data(), missRecordSize));
    CUDA_CHECK(cuMemAlloc(&handle.d_hitgroupRecord, hitgroupRecordSize));
    CUDA_CHECK(cuMemcpyHtoD(handle.d_hitgroupRecord, hitgroupRecord.data(), hitgroupRecordSize));

    handle.sbt.raygenRecord = handle.d_raygenRecord;
    handle.sbt.missRecordBase = handle.d_missRecord;
    handle.sbt.missRecordStrideInBytes = missRecordSize;
    handle.sbt.missRecordCount = 1;
    handle.sbt.hitgroupRecordBase = handle.d_hitgroupRecord;
    handle.sbt.hitgroupRecordStrideInBytes = hitgroupRecordSize;
    handle.sbt.hitgroupRecordCount = 1;

    return handle;
}

ShadowPipelineHandle ContextImpl::createShadowPipelineFromFile(const char* ptxOrOptixirPath) {
    std::ifstream f(ptxOrOptixirPath, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error(std::string("Failed to open shadow module: ") + ptxOrOptixirPath);
    size_t size = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> code(size);
    if (!f.read(code.data(), size))
        throw std::runtime_error("Failed to read shadow module file");
    f.close();

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = 0;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 0;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "shadowPlp";

    char log[2048];
    size_t logSize = sizeof(log);

    OptixModule module = nullptr;
    OPTIX_CHECK(optixModuleCreate(
        m_optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        code.data(),
        size,
        log, &logSize,
        &module
    ));

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__shadowTrace";

    OptixProgramGroup raygenGroup = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &raygenDesc, 1, &pgOptions, nullptr, nullptr, &raygenGroup));

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__shadowTrace";

    OptixProgramGroup missGroup = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &missDesc, 1, &pgOptions, nullptr, nullptr, &missGroup));

    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleAH = module;
    hitgroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadowTrace";

    OptixProgramGroup hitgroupGroup = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &hitgroupDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupGroup));

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;

    OptixProgramGroup groups[] = { raygenGroup, missGroup, hitgroupGroup };
    OptixPipeline pipeline = nullptr;
    char pipeLog[2048];
    size_t pipeLogSize = sizeof(pipeLog);
    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &pipelineCompileOptions,
        &linkOptions,
        groups,
        sizeof(groups) / sizeof(groups[0]),
        pipeLog,
        &pipeLogSize,
        &pipeline
    ));

    size_t raygenRecordSize = OPTIX_SBT_RECORD_HEADER_SIZE;
    size_t missRecordSize = OPTIX_SBT_RECORD_HEADER_SIZE;
    size_t hitgroupRecordSize = OPTIX_SBT_RECORD_HEADER_SIZE;

    std::vector<char> raygenRecord(raygenRecordSize);
    std::vector<char> missRecord(missRecordSize);
    std::vector<char> hitgroupRecord(hitgroupRecordSize);

    OPTIX_CHECK(optixSbtRecordPackHeader(raygenGroup, raygenRecord.data()));
    OPTIX_CHECK(optixSbtRecordPackHeader(missGroup, missRecord.data()));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupGroup, hitgroupRecord.data()));

    ShadowPipelineHandle handle;
    handle.pipeline = pipeline;
    handle.module = module;
    handle.raygenGroup = raygenGroup;
    handle.missGroup = missGroup;
    handle.hitgroupGroup = hitgroupGroup;

    CUDA_CHECK(cuMemAlloc(&handle.d_raygenRecord, raygenRecordSize));
    CUDA_CHECK(cuMemcpyHtoD(handle.d_raygenRecord, raygenRecord.data(), raygenRecordSize));
    CUDA_CHECK(cuMemAlloc(&handle.d_missRecord, missRecordSize));
    CUDA_CHECK(cuMemcpyHtoD(handle.d_missRecord, missRecord.data(), missRecordSize));
    CUDA_CHECK(cuMemAlloc(&handle.d_hitgroupRecord, hitgroupRecordSize));
    CUDA_CHECK(cuMemcpyHtoD(handle.d_hitgroupRecord, hitgroupRecord.data(), hitgroupRecordSize));

    handle.sbt.raygenRecord = handle.d_raygenRecord;
    handle.sbt.missRecordBase = handle.d_missRecord;
    handle.sbt.missRecordStrideInBytes = missRecordSize;
    handle.sbt.missRecordCount = 1;
    handle.sbt.hitgroupRecordBase = handle.d_hitgroupRecord;
    handle.sbt.hitgroupRecordStrideInBytes = hitgroupRecordSize;
    handle.sbt.hitgroupRecordCount = 1;

    return handle;
}

} // namespace vlrw
