#include "wr/wr.h"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

#define CU_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(result, &errStr); \
            throw std::runtime_error(std::string("CU error: ") + errStr); \
        } \
    } while(0)

#define OPTIX_CHECK(call) \
    do { \
        OptixResult result = call; \
        if (result != OPTIX_SUCCESS) { \
            throw std::runtime_error(std::string("OptiX error: ") + optixGetErrorName(result)); \
        } \
    } while(0)

namespace wr {

static OptixDeviceContext g_optixContext = nullptr;

struct ContextImpl {
    OptixDeviceContext optixContext = nullptr;
    CUcontext cudaContext = nullptr;
    int deviceId = 0;
    
    void initialize(int devId) {
        deviceId = devId;
        
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
        options.logCallbackLevel = 4;
        
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

Context::Context(int deviceId) {
    if (!g_contextImpl) {
        g_contextImpl = new ContextImpl();
        g_contextImpl->initialize(deviceId);
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

OptixDeviceContext getOptixContext() {
    return g_optixContext;
}

} // namespace wr
