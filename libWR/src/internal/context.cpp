#include "wr/context.h"
#include "wr/scene.h"
#include "wr/renderer.h"
#include "internal/cuda_utils.h"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdexcept>

namespace wr {

using namespace internal;

static OptixDeviceContext g_optixContext = nullptr;

struct ContextImpl {
    OptixDeviceContext optixContext = nullptr;
    CUcontext cudaContext = nullptr;
    int deviceId = 0;
    int logLevel = 4;
    
    void initialize(const ContextConfig& config) {
        deviceId = config.deviceId;
        logLevel = config.logLevel;
        
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaFree(0));
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
        std::cout << "[WR] Using device: " << prop.name << std::endl;
        
        CU_CHECK(cuInit(0));
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, deviceId));
        CU_CHECK(cuDevicePrimaryCtxRetain(&cudaContext, device));
        CU_CHECK(cuCtxPushCurrent(cudaContext));
        
        OPTIX_CHECK(optixInit());
        
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = [](unsigned int level, const char* tag, 
                                         const char* message, void*) {
            std::cout << "[OptiX][" << level << "][" << tag << "]: " << message << std::endl;
        };
        options.logCallbackLevel = logLevel;
        options.validationMode = config.enableValidation ? 
            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : 
            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
        
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
        g_optixContext = optixContext;
        
        std::cout << "[WR] OptiX context created" << std::endl;
    }
    
    ~ContextImpl() {
        if (optixContext) {
            optixDeviceContextDestroy(optixContext);
            optixContext = nullptr;
        }
        if (cudaContext) {
            CUcontext prevCtx;
            cuCtxPopCurrent(&prevCtx);
            CUdevice device;
            if (cuDeviceGet(&device, deviceId) == CUDA_SUCCESS) {
                cuDevicePrimaryCtxRelease(device);
            }
            cudaContext = nullptr;
        }
    }
};

static ContextImpl* g_contextImpl = nullptr;

Context::Context(const ContextConfig& config) {
    if (!g_contextImpl) {
        g_contextImpl = new ContextImpl();
        g_contextImpl->initialize(config);
    }
}

Context::~Context() {
}

Scene* Context::createScene() {
    return new Scene();
}

Renderer* Context::createRenderer() {
    return new Renderer();
}

int Context::getDeviceId() const {
    return g_contextImpl ? g_contextImpl->deviceId : 0;
}

OptixDeviceContext getOptixContext() {
    return g_optixContext;
}

} // namespace wr
