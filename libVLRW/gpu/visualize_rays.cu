// Simple kernel to visualize ray directions as colors
#include "wavefront_types.cuh"

namespace wpt {

// Host-callable wrapper
extern "C" void launchVisualizeRays(
    const RayState* rayPool,
    const uint32_t* activeQueue,
    float3_rgb* accumBuffer,
    uint32_t numActive,
    unsigned int gridSize,
    unsigned int blockSize);

__global__ void visualizeRayDirectionsKernel(
    const RayState* rayPool,
    const uint32_t* activeQueue,
    float3_rgb* accumBuffer,
    uint32_t numActive)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numActive) return;
    
    uint32_t rayIndex = activeQueue[idx];
    const RayState& ray = rayPool[rayIndex];
    
    // Convert ray direction to color (abs values for visibility)
    float3_rgb color;
    color.r = fabsf(ray.direction.x);
    color.g = fabsf(ray.direction.y);
    color.b = fabsf(ray.direction.z);
    
    accumBuffer[ray.pixel_index] = color;
}

extern "C" void launchVisualizeRays(
    const RayState* rayPool,
    const uint32_t* activeQueue,
    float3_rgb* accumBuffer,
    uint32_t numActive,
    unsigned int gridSize,
    unsigned int blockSize)
{
    visualizeRayDirectionsKernel<<<gridSize, blockSize>>>(
        rayPool, activeQueue, accumBuffer, numActive);
}

} // namespace wpt
