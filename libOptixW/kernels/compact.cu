#include <optix.h>
#include <cuda_runtime.h>
#include <optixw/wavefront_kernel_params.h>

using namespace optixw;

// Compact active rays (remove terminated or shaded rays).
extern "C" __global__ void compact(CompactKernelParams params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.numActive) return;
    
    uint32_t rayIndex = params.activeIndicesIn[idx];
    const RayState& ray = params.rayPool[rayIndex];
    
    // Keep rays that still require tracing work.
    if (ray.stage == RayState::Trace || ray.stage == RayState::Shadow) {
        uint32_t outIdx = atomicAdd(params.counter, 1);
        params.activeIndicesOut[outIdx] = rayIndex;
    }
}
