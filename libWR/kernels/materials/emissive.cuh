#pragma once
#include "material_common.cuh"

namespace wr {
namespace internal {

// Emissive material shader
// Simply terminates the ray and accumulates radiance
__device__ __forceinline__ void shadeEmissive(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p
) {
    atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
    ray.stage = RayStage::Terminated;
}

} // namespace internal
} // namespace wr
