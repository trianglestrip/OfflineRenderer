#pragma once

#include <optix.h>
#include <cuda.h>
#include <string>
#include <vector>

namespace vlrw {

struct TracePipelineHandle {
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_raygenRecord = 0;
    CUdeviceptr d_missRecord = 0;
    CUdeviceptr d_hitgroupRecord = 0;
    OptixModule module = nullptr;
    OptixProgramGroup raygenGroup = nullptr;
    OptixProgramGroup missGroup = nullptr;
    OptixProgramGroup hitgroupGroup = nullptr;
};

struct ShadowPipelineHandle {
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_raygenRecord = 0;
    CUdeviceptr d_missRecord = 0;
    CUdeviceptr d_hitgroupRecord = 0;
    OptixModule module = nullptr;
    OptixProgramGroup raygenGroup = nullptr;
    OptixProgramGroup missGroup = nullptr;
    OptixProgramGroup hitgroupGroup = nullptr;
};

class ContextImpl {
public:
    ContextImpl(CUcontext cuContext);
    ~ContextImpl();

    OptixDeviceContext getOptixContext() const { return m_optixContext; }
    CUcontext getCUContext() const { return m_cuContext; }

    OptixModule loadPTXModule(const char* ptxCode, size_t ptxSize);
    TracePipelineHandle createTracePipelineFromFile(const char* ptxOrOptixirPath);
    ShadowPipelineHandle createShadowPipelineFromFile(const char* ptxOrOptixirPath);

private:
    CUcontext m_cuContext;
    OptixDeviceContext m_optixContext;
};

} // namespace vlrw
