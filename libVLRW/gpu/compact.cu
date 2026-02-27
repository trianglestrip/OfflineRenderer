#include "wavefront_types.cuh"

namespace wpt {

__global__ void markActive(
    const RayState* rayPool,
    const uint32_t* inputQueue,
    uint32_t numRays,
    uint32_t* activeMask)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    
    uint32_t rayIndex = inputQueue[idx];
    activeMask[idx] = (rayPool[rayIndex].stage != Stage_Terminated) ? 1u : 0u;
}

__global__ void scatterActiveIndices(
    const RayState* rayPool,
    const uint32_t* inputQueue,
    const uint32_t* prefixSum,
    uint32_t numRays,
    uint32_t* activeIndices)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    
    uint32_t rayIndex = inputQueue[idx];
    if (rayPool[rayIndex].stage != Stage_Terminated) {
        activeIndices[prefixSum[idx] - 1] = rayIndex;
    }
}

} // namespace wpt
