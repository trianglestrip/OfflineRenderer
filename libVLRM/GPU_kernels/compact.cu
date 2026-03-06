#include "../shared/wavefront_types.h"

namespace vlrm {
    RT_PIPELINE_LAUNCH_PARAMETERS WavefrontParams params;

    CUDA_DEVICE_KERNEL void compactRayQueue(
        RayState* inputQueue,
        uint32_t inputSize,
        RayState* outputQueue,
        uint32_t* outputSize
    ) {
        uint32_t rayIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (rayIndex >= inputSize)
            return;
        
        RayState& ray = inputQueue[rayIndex];
        
        if (ray.stage != RayStage::Terminated) {
            uint32_t outIndex = atomicAdd(outputSize, 1);
            outputQueue[outIndex] = ray;
        }
    }
}
