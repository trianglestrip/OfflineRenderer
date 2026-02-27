#pragma once

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#define OPTIX_CHECK(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call failed: " << #call << " returned error code " \
               << (int)res << " at " << __FILE__ << ":" << __LINE__; \
            const char* errName = optixGetErrorName(res); \
            const char* errMsg = optixGetErrorString(res); \
            if (errName) ss << " [" << errName << "]"; \
            if (errMsg) ss << ": " << errMsg; \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

#define CUDA_CHECK(call) \
    do { \
        CUresult res = call; \
        if (res != CUDA_SUCCESS) { \
            const char* errMsg; \
            cuGetErrorString(res, &errMsg); \
            std::stringstream ss; \
            ss << "CUDA call failed: " << #call << " - " << errMsg; \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)
