#include "optixw/optixw.h"
#include "optixw/types.h"
#include "checks.h"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>

namespace optixw {

// Global OptiX context for access from other modules
OptixDeviceContext g_optixContext = nullptr;

// Context implementation
class Context::Impl {
public:
    OptixDeviceContext optixContext = nullptr;
    CUcontext cudaContext = nullptr;
    
    void initialize(int deviceId) {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));
        
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, deviceId));
        
        char deviceName[256];
        CU_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), device));
        std::cout << "[OptixW] Using device: " << deviceName << std::endl;
        
        CUctxCreateParams params = {};
        CU_CHECK(cuCtxCreate(&cudaContext, &params, 0, device));
        
        // Initialize OptiX
        OPTIX_CHECK(optixInit());
        
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = [](unsigned int level, const char* tag, 
                                         const char* message, void*) {
            std::cout << "[OptiX][" << level << "][" << tag << "]: " 
                     << message << std::endl;
        };
        options.logCallbackLevel = 4;
        
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
        
        g_optixContext = optixContext;  // Store global reference
        
        std::cout << "[OptixW] OptiX context created" << std::endl;
    }
    
    ~Impl() {
        if (optixContext) {
            optixDeviceContextDestroy(optixContext);
        }
        if (cudaContext) {
            cuCtxDestroy(cudaContext);
        }
    }
};

Context::Context(int deviceId) : m_impl(std::make_unique<Impl>()) {
    m_impl->initialize(deviceId);
}

Context::~Context() = default;

Scene* Context::createScene() {
    return new Scene();
}

Renderer* Context::createRenderer() {
    return new Renderer();
}

} // namespace optixw
