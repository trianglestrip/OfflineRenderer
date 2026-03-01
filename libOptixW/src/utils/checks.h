// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

// OPTIX_CHECK - OptiX API 调用错误检查宏
// 用法: OPTIX_CHECK(optixDeviceContextCreate(...));
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

// CUDA_CHECK - CUDA Runtime API 调用错误检查宏
// 用法: CUDA_CHECK(cudaMalloc(...));
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            throw std::runtime_error(                                          \
                std::string("CUDA call failed: ") +                            \
                cudaGetErrorString(error));                                    \
        }                                                                      \
    } while (0)

// CU_CHECK - CUDA Driver API 调用错误检查宏
// 用法: CU_CHECK(cuMemAlloc(...));
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
