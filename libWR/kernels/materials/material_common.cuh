#pragma once
#include "internal/gpu_types.h"
#include "sampling.cuh"
#include "utils/atomic_ops.cuh"
#include "utils/ray_offset.cuh"

namespace wr {
namespace internal {

static constexpr float kPi = 3.14159265f;

// Apply Russian Roulette and update throughput
// Returns false if ray should be terminated
__device__ __forceinline__ bool applyRussianRoulette(
    RayState& ray,
    const LaunchParams* p,
    bool isDelta = false
) {
    float3 newThroughput;
    if (!russianRoulette(ray.throughput, ray.depth, p->rrStartDepth, 
                         rnd_dim(ray.seed, ray.rngDimension++), newThroughput, isDelta)) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
        return false;
    }
    ray.throughput = newThroughput;
    return true;
}

// Setup next ray bounce
__device__ __forceinline__ void setupNextBounce(
    RayState& ray,
    const HitInfo& hit,
    float3 newDirection,
    const LaunchParams* p
) {
    ray.direction = newDirection;
    ray.origin = offsetRayOrigin(hit.position, hit.normal, newDirection);
    ray.tMin = 0.0f;
    ray.tMax = 1e20f;
    ray.depth++;
    ray.stage = RayStage::Trace;
    
    if (ray.depth >= p->maxBounces) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
    }
}

} // namespace internal
} // namespace wr
