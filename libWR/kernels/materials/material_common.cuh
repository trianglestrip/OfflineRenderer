#pragma once
#include "internal/gpu_types.h"
#include "sampling.cuh"
#include "utils/atomic_ops.cuh"
#include "utils/ray_offset.cuh"

namespace wr {
namespace internal {

static constexpr float kPi = 3.14159265f;

// Asymmetric scattering correction for shading normal vs geometric normal
// Reference: Veach thesis, Chapter 5.3
__device__ __forceinline__ float asymmetricScatteringCorrection(
    const float3& wi,
    const float3& shadingNormal,
    const float3& geometricNormal
) {
    float cosShading = fabsf(dot(wi, shadingNormal));
    float cosGeometric = fabsf(dot(wi, geometricNormal));
    
    // Avoid division by zero
    if (cosShading < 1e-6f || cosGeometric < 1e-6f) {
        return 1.0f;
    }
    
    // Adjoint BSDF correction factor
    return cosGeometric / cosShading;
}

// Apply Russian Roulette and update throughput
// Returns false if ray should be terminated
__device__ __forceinline__ void terminateRay(RayState& ray, const LaunchParams* p) {
    float3 clamped = clampFireflies(ray.radiance, p->fireflyClamp);
    atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], clamped);
    ray.stage = RayStage::Terminated;
}

__device__ __forceinline__ bool applyRussianRoulette(
    RayState& ray,
    const LaunchParams* p,
    bool isDelta = false
) {
    float3 newThroughput;
    if (!russianRoulette(ray.throughput, ray.depth, p->rrStartDepth,
                         rnd_dim(ray.seed, ray.rngDimension++), newThroughput, isDelta)) {
        terminateRay(ray, p);
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
        terminateRay(ray, p);
    }
}

} // namespace internal
} // namespace wr
