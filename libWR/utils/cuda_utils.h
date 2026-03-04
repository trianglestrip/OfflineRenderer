#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace wr {
namespace utils {

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

} // namespace utils
} // namespace wr