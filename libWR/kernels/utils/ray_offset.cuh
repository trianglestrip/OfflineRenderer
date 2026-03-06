#pragma once
#include <cuda_runtime.h>

namespace wr {
namespace internal {

// Offset ray origin to avoid self-intersection
// Uses geometric offset based on surface normal
__device__ __forceinline__ float3 offsetRayOrigin(
    float3 position,
    float3 normal,
    float3 direction
) {
    const float offset = 1e-4f;
    float3 offsetDir = dot(direction, normal) > 0.0f ? normal : -normal;
    return make_float3(
        position.x + offsetDir.x * offset,
        position.y + offsetDir.y * offset,
        position.z + offsetDir.z * offset
    );
}

} // namespace internal
} // namespace wr
