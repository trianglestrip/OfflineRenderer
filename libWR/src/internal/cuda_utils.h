#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <optix.h>
#include <stdexcept>
#include <string>

namespace wr {
namespace internal {

// OptiX error checking macro
#define OPTIX_CHECK(call) \
    do { \
        OptixResult result = call; \
        if (result != OPTIX_SUCCESS) { \
            throw std::runtime_error(std::string("OptiX error: ") + optixGetErrorName(result)); \
        } \
    } while(0)

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// CUDA driver API error checking macro
#define CU_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(result, &errStr); \
            throw std::runtime_error(std::string("CU error: ") + errStr); \
        } \
    } while(0)

// CUDA memory allocation macro with error checking
#define CUDA_MALLOC(ptr, size) \
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), size))

// CUDA memory deallocation macro with null check
#define CUDA_FREE(ptr) \
    do { \
        if (ptr) { \
            cudaFree(reinterpret_cast<void*>(ptr)); \
            ptr = 0; \
        } \
    } while(0)

// OptiX object destruction macro with null check
#define OPTIX_DESTROY(handle, destroyFunc) \
    do { \
        if (handle) { \
            destroyFunc(handle); \
            handle = nullptr; \
        } \
    } while(0)

} // namespace internal
} // namespace wr
