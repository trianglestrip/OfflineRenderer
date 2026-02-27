#pragma once

#include <optix.h>
#include <cuda.h>

namespace vlrw {

class ContextImpl {
public:
    ContextImpl(CUcontext cuContext);
    ~ContextImpl();

    OptixDeviceContext getOptixContext() const { return m_optixContext; }
    CUcontext getCUContext() const { return m_cuContext; }

    OptixModule loadPTXModule(const char* ptxCode, size_t ptxSize);
    OptixPipeline createTracePipeline();
    OptixPipeline createShadowPipeline();

private:
    CUcontext m_cuContext;
    OptixDeviceContext m_optixContext;
};

} // namespace vlrw
