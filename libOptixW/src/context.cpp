#include "optixw/optixw.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace optixw {

// Helper macros
#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            throw std::runtime_error(                                          \
                std::string("OptiX call failed: ") +                           \
                optixGetErrorName(res) + " (" +                                \
                optixGetErrorString(res) + ")");                               \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            throw std::runtime_error(                                          \
                std::string("CUDA call failed: ") +                            \
                cudaGetErrorString(error));                                    \
        }                                                                      \
    } while (0)

// Context implementation
class Context::Impl {
public:
    OptixDeviceContext optixContext = nullptr;
    CUcontext cudaContext = nullptr;
    
    Impl() {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));
        
        CUdevice device;
        cuDeviceGet(&device, 0);
        
        char deviceName[256];
        cuDeviceGetName(deviceName, sizeof(deviceName), device);
        std::cout << "[OptixW] Using device: " << deviceName << std::endl;
        
        CUctxCreateParams params = {};
        cuCtxCreate(&cudaContext, &params, 0, device);
        
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

Context::Context() : m_impl(std::make_unique<Impl>()) {}
Context::~Context() = default;

std::unique_ptr<Context> Context::create() {
    return std::unique_ptr<Context>(new Context());
}

Scene* Context::createScene() {
    return new Scene();
}

Renderer* Context::createRenderer() {
    return new Renderer();
}

} // namespace optixw
