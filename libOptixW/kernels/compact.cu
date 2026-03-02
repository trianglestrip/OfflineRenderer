#include <optix.h>
#include <cuda_runtime.h>
#include <optixw/wavefront_kernel_params.h>

using namespace optixw;

// Compact active rays (keep rays that still require work).
extern "C" __global__ void compact(const CompactKernelParams* params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params->numActive) return;
    
    uint32_t rayIndex = params->activeIndicesIn[idx];
    const RayState& ray = params->rayPool[rayIndex];
    
    // Keep rays that still require work: Trace, Shadow, or Shade
    if (ray.stage == RayState::Trace || ray.stage == RayState::Shadow || ray.stage == RayState::Shade) {
        uint32_t outIdx = atomicAdd(params->counter, 1);
        params->activeIndicesOut[outIdx] = rayIndex;
    }
}
