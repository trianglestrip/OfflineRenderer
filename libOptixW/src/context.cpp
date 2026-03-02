#include "optixw/optixw.h"
#include "optixw/types.h"
#include "optixw/core/task_scheduler.h"
#include "utils/checks.h"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

namespace optixw {

OptixDeviceContext g_optixContext = nullptr;

class Context::Impl {
public:
    OptixDeviceContext optixContext = nullptr;
    CUcontext cudaContext = nullptr;
    
    void initialize(int deviceId) {
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaFree(0));
        
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
        std::cout << "[OptixW] Using device: " << prop.name << std::endl;
        
        CU_CHECK(cuInit(0));
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, deviceId));
        
        CU_CHECK(cuDevicePrimaryCtxRetain(&cudaContext, device));
        
        CU_CHECK(cuCtxPushCurrent(cudaContext));
        
        OPTIX_CHECK(optixInit());
        
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = [](unsigned int level, const char* tag, 
                                         const char* message, void*) {
            std::cout << "[OptiX][" << level << "][" << tag << "]: " 
                     << message << std::endl;
        };
        options.logCallbackLevel = 4;
        
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
        
        g_optixContext = optixContext;
        
        std::cout << "[OptixW] OptiX context created" << std::endl;
    }
    
    ~Impl() {
        if (optixContext) {
            optixDeviceContextDestroy(optixContext);
        }
        if (cudaContext) {
            CUcontext prevCtx;
            cuCtxPopCurrent(&prevCtx);
            
            CUdevice device;
            if (cuDeviceGet(&device, 0) == CUDA_SUCCESS) {
                cuDevicePrimaryCtxRelease(device);
            }
        }
    }
};

Context::Context(int deviceId) : m_impl(std::make_unique<Impl>()), m_taskScheduler(std::make_unique<TaskScheduler>()) {
    m_impl->initialize(deviceId);
}

Context::~Context() = default;

Scene* Context::createScene() {
    return new Scene(m_taskScheduler.get());
}

Renderer* Context::createRenderer() {
    return new Renderer(m_taskScheduler.get());
}

} // namespace optixw
