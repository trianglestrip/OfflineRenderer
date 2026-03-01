// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/rendering/pipeline.h>
#include "../utils/checks.h"
#include "../utils/path_utils.h"
#include <iostream>
#include <vector>

namespace optixw {

Pipeline::Pipeline(OptixDeviceContext context)
    : m_context(context)
    , m_pipeline(nullptr)
    , m_traceModule(nullptr)
    , m_raygenPG(nullptr)
    , m_missPG(nullptr)
    , m_hitgroupPG(nullptr)
    , m_sbtRaygenRecord(0)
    , m_sbtMissRecord(0)
    , m_sbtHitgroupRecord(0)
    , m_created(false)
{
    memset(&m_sbt, 0, sizeof(m_sbt));
}

Pipeline::~Pipeline() {
    destroy();
}

void Pipeline::create() {
    if (m_created) {
        std::cout << "[Pipeline] 管线已创建，跳过" << std::endl;
        return;
    }

    std::cout << "[Pipeline] 创建 OptiX 管线..." << std::endl;

    createModule();
    createProgramGroups();
    linkPipeline();
    buildSBT();

    m_created = true;
    std::cout << "[Pipeline] 管线创建成功" << std::endl;
}

void Pipeline::destroy() {
    if (!m_created) return;

    // 销毁 SBT 缓冲
    if (m_sbtRaygenRecord) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbtRaygenRecord)));
        m_sbtRaygenRecord = 0;
    }
    if (m_sbtMissRecord) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbtMissRecord)));
        m_sbtMissRecord = 0;
    }
    if (m_sbtHitgroupRecord) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbtHitgroupRecord)));
        m_sbtHitgroupRecord = 0;
    }

    // 销毁程序组
    if (m_raygenPG) OPTIX_CHECK(optixProgramGroupDestroy(m_raygenPG));
    if (m_missPG) OPTIX_CHECK(optixProgramGroupDestroy(m_missPG));
    if (m_hitgroupPG) OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroupPG));

    // 销毁管线和模块
    if (m_pipeline) OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
    if (m_traceModule) OPTIX_CHECK(optixModuleDestroy(m_traceModule));

    m_created = false;
    std::cout << "[Pipeline] Destroyed" << std::endl;
}

void Pipeline::launch(CUdeviceptr params, uint32_t width, uint32_t height, CUstream stream) {
    if (!m_created) {
        throw std::runtime_error("Pipeline not created");
    }

    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        stream,
        params,
        sizeof(void*),  // Launch params 大小（实际是指针）
        &m_sbt,
        width,
        height,
        1  // depth
    ));
}

void Pipeline::createModule() {
    // 加载 PTX
    std::vector<char> tracePTX = utils::loadPTX("trace.ptx");

    // 模块编译选项
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    // 管线编译选项
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 1;
    pipelineCompileOptions.numAttributeValues = 0;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    // 创建模块
    char log[2048];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(
        m_context,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        tracePTX.data(),
        tracePTX.size(),
        log, &logSize,
        &m_traceModule
    ));

    if (logSize > 1) {
        std::cout << "[Pipeline] OptiX 模块: " << log << std::endl;
    }
}

void Pipeline::createProgramGroups() {
    OptixProgramGroupOptions pgOptions = {};
    char log[2048];
    size_t logSize;

    // Raygen 程序组
    OptixProgramGroupDesc raygenPGDesc = {};
    raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenPGDesc.raygen.module = m_traceModule;
    raygenPGDesc.raygen.entryFunctionName = "__raygen__trace";

    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        m_context,
        &raygenPGDesc,
        1,
        &pgOptions,
        log, &logSize,
        &m_raygenPG
    ));

    // Miss 程序组
    OptixProgramGroupDesc missPGDesc = {};
    missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missPGDesc.miss.module = m_traceModule;
    missPGDesc.miss.entryFunctionName = "__miss__trace";

    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        m_context,
        &missPGDesc,
        1,
        &pgOptions,
        log, &logSize,
        &m_missPG
    ));

    // Hitgroup 程序组
    OptixProgramGroupDesc hitgroupPGDesc = {};
    hitgroupPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupPGDesc.hitgroup.moduleCH = m_traceModule;
    hitgroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__trace";

    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        m_context,
        &hitgroupPGDesc,
        1,
        &pgOptions,
        log, &logSize,
        &m_hitgroupPG
    ));
}

void Pipeline::linkPipeline() {
    // 链接管线
    OptixProgramGroup programGroups[] = { m_raygenPG, m_missPG, m_hitgroupPG };

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    char log[2048];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixPipelineCreate(
        m_context,
        nullptr,  // 使用默认编译选项
        &pipelineLinkOptions,
        programGroups,
        3,
        log, &logSize,
        &m_pipeline
    ));

    if (logSize > 1) {
        std::cout << "[Pipeline] OptiX 管线: " << log << std::endl;
    }
}

void Pipeline::buildSBT() {
    // Raygen record
    struct RaygenRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    RaygenRecord raygenRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenPG, &raygenRecord));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbtRaygenRecord), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_sbtRaygenRecord),
        &raygenRecord,
        sizeof(RaygenRecord),
        cudaMemcpyHostToDevice
    ));

    m_sbt.raygenRecord = m_sbtRaygenRecord;

    // Miss record
    struct MissRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    MissRecord missRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_missPG, &missRecord));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbtMissRecord), sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_sbtMissRecord),
        &missRecord,
        sizeof(MissRecord),
        cudaMemcpyHostToDevice
    ));

    m_sbt.missRecordBase = m_sbtMissRecord;
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = 1;

    // Hitgroup record
    struct HitgroupRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    HitgroupRecord hitgroupRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupPG, &hitgroupRecord));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbtHitgroupRecord), sizeof(HitgroupRecord)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_sbtHitgroupRecord),
        &hitgroupRecord,
        sizeof(HitgroupRecord),
        cudaMemcpyHostToDevice
    ));

    m_sbt.hitgroupRecordBase = m_sbtHitgroupRecord;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_sbt.hitgroupRecordCount = 1;

    std::cout << "[Pipeline] SBT 构建完成" << std::endl;
}

} // namespace optixw
