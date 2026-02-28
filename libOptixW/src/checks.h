#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

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

#define CU_CHECK(call)                                                         \
    do {                                                                       \
        CUresult error = call;                                                 \
        if (error != CUDA_SUCCESS) {                                           \
            const char* errStr = nullptr;                                      \
            cuGetErrorString(error, &errStr);                                  \
            throw std::runtime_error(                                          \
                std::string("CUDA Driver API call failed: ") +                 \
                (errStr ? errStr : "Unknown error"));                          \
        }                                                                      \
    } while (0)
