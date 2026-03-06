#include "../include/VLRM/VLRM.h"
#include "../include/VLRM/common.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <iostream>

namespace vlrm {

class Context::Impl {
public:
    bool initialized = false;
    CUcontext cuda_context = nullptr;
    CUdevice cuda_device = 0;
    OptixDeviceContext optix_context = nullptr;
};
    
OptixDeviceContext Context::getOptixDeviceContext() const {
    if (m_impl) {
        return m_impl->optix_context;
    }
    return nullptr;
}

Context::Context() : m_impl(new Context::Impl()) {
    // 初始化上下文成员变量
}

Context::~Context() {
    // 析构函数，清理资源
    if (m_impl->initialized) {
        finalize();
    }
    delete m_impl;
}

void Context::initialize() {
    // 初始化CUDA
    cudaError_t cudaResult = cudaSetDevice(0);
    if (cudaResult != cudaSuccess) {
        vlrmprintf("CUDA device selection failed: %s\n", cudaGetErrorString(cudaResult));
        return;
    }

    // 获取CUDA上下文
    CUresult cuResult = cuCtxGetCurrent(&m_impl->cuda_context);
    if (cuResult != CUDA_SUCCESS) {
        vlrmprintf("Failed to get CUDA context\n");
        return;
    }

    // 初始化OptiX
    OptixResult optixResult = optixInit();
    if (optixResult != OPTIX_SUCCESS) {
        vlrmprintf("OptiX initialization failed: %u\n", optixResult);
        return;
    }

    // 创建OptiX设备上下文
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    options.logCallbackLevel = 4;

    CUcontext initialContext = 0;
    cuCtxGetCurrent(&initialContext);
    optixResult = optixDeviceContextCreate(m_impl->cuda_context, &options, &m_impl->optix_context);
    if (optixResult != OPTIX_SUCCESS) {
        vlrmprintf("OptiX context creation failed: %u\n", optixResult);
        return;
    }

    m_impl->initialized = true;
    vlrmprintf("Context initialized successfully.\n");
}

void Context::finalize() {
    if (m_impl->optix_context) {
        optixDeviceContextDestroy(m_impl->optix_context);
        m_impl->optix_context = nullptr;
    }

    m_impl->initialized = false;
    vlrmprintf("Context finalized.\n");
}

// 静态日志回调函数
void Context::optixLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata*/) {
    vlrmprintf("OptiX Log [Level %u] [%s]: %s\n", level, tag, message);
}

} // namespace vlrm