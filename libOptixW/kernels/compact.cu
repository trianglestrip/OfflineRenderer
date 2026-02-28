#include <optix.h>
#include <cuda_runtime.h>
#include <optixw/types.h>

using namespace optixw;

// Compact kernel parameters
struct CompactParams {
    const RayState* rayPool;
    const uint32_t* activeIndicesIn;
    uint32_t* activeIndicesOut;
    uint32_t* counter;
    uint32_t numActive;
};

// Compact active rays (remove terminated)
extern "C" __global__ void compact(CompactParams params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.numActive) return;
    
    uint32_t rayIndex = params.activeIndicesIn[idx];
    const RayState& ray = params.rayPool[rayIndex];
    
    // Keep only non-terminated rays
    if (ray.stage != RayState::Terminated) {
        uint32_t outIdx = atomicAdd(params.counter, 1);
        params.activeIndicesOut[outIdx] = rayIndex;
    }
}
