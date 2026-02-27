#include "optix_context.h"
#include "cuda_check.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

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
    pipelineCompileOptions.numPayloadValues = 1;
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

OptixPipeline ContextImpl::createTracePipeline() {
    return nullptr;
}

OptixPipeline ContextImpl::createShadowPipeline() {
    return nullptr;
}

} // namespace vlrw
