#include <cuda_runtime.h>
#include <wr/types.h>

using namespace wr;

extern "C" __global__ void compact(const CompactParams* params) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params->numActive) return;
    
    uint32_t rayIndex = params->activeIndicesIn[idx];
    const RayState& ray = params->rayPool[rayIndex];
    
    if (ray.stage != RayStage::Terminated) {
        uint32_t outIdx = atomicAdd(params->counter, 1);
        params->activeIndicesOut[outIdx] = rayIndex;
    }
}
