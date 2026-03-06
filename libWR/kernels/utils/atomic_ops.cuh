#pragma once
#include <cuda_runtime.h>

namespace wr {
namespace internal {

// Atomic add for float3 (thread-safe accumulation)
__device__ __forceinline__ void atomicAddFloat3(float3* address, float3 value) {
    atomicAdd(&address->x, value.x);
    atomicAdd(&address->y, value.y);
    atomicAdd(&address->z, value.z);
}

// Legacy random function (LCG)
// Note: Prefer rnd_dim() from sampling.cuh for better quality
__device__ __forceinline__ float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 8) / 16777216.0f;
}

} // namespace internal
} // namespace wr
