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
    terminateRay(ray, p);
}

} // namespace internal
} // namespace wr
