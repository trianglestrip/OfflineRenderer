// Gather active rays：简化版，当�?RayState 已经�?pool，无需额外 gather
// 此文件保留用于未来可能的 SoA 优化

#include "wavefront_types.cuh"

namespace wpt {

// 占位 kernel：当前架构下 RayState 已经�?pool 中，无需 gather
__global__ void gatherActiveRays(
    const RayState* rayPool,
    const uint32_t* activeIndices,
    RayState* compactedRayPool,
    uint32_t numActive
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numActive) return;

    uint32_t srcIdx = activeIndices[idx];
    compactedRayPool[idx] = rayPool[srcIdx];
}

} // namespace wpt
