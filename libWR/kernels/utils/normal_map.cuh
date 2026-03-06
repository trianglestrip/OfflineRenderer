#pragma once
#include <cuda_runtime.h>
#include "internal/gpu_types.h"
#include "vector_math.cuh"

namespace wr {
namespace internal {

// Sample normal map and transform to world space
// normalMap: tangent-space normal from texture (RGB, [0,1] range)
// Returns: world-space normal
__device__ __forceinline__ float3 applyNormalMap(
    float3 tangentSpaceNormal,  // From texture: (0.5, 0.5, 1.0) = no perturbation
    float3 geometricNormal,
    float3 tangent,
    float3 bitangent
) {
    // Convert from [0,1] to [-1,1]
    float3 n = make_float3(
        tangentSpaceNormal.x * 2.0f - 1.0f,
        tangentSpaceNormal.y * 2.0f - 1.0f,
        tangentSpaceNormal.z * 2.0f - 1.0f
    );
    
    // Transform to world space using TBN matrix
    float3 worldNormal = make_float3(
        tangent.x * n.x + bitangent.x * n.y + geometricNormal.x * n.z,
        tangent.y * n.x + bitangent.y * n.y + geometricNormal.y * n.z,
        tangent.z * n.x + bitangent.z * n.y + geometricNormal.z * n.z
    );
    
    return normalize(worldNormal);
}

// Get effective normal (with normal map if available)
__device__ __forceinline__ float3 getEffectiveNormal(
    const MaterialData& mat,
    const HitInfo& hit,
    const cudaTextureObject_t* textures,
    uint32_t numTextures
) {
    // Check if normal map is available
    if (mat.normalTextureId > 0 && mat.normalTextureId <= numTextures) {
        cudaTextureObject_t normalTex = textures[mat.normalTextureId - 1];
        
        // Sample normal map
        float4 normalSample = tex2D<float4>(normalTex, hit.uv.x, hit.uv.y);
        float3 tangentNormal = make_float3(normalSample.x, normalSample.y, normalSample.z);
        
        // Compute tangent frame
        float3 tangent, bitangent;
        createCoordinateFrame(hit.normal, tangent, bitangent);
        
        // Transform to world space
        return applyNormalMap(tangentNormal, hit.normal, tangent, bitangent);
    }
    
    // No normal map, use geometric normal
    return hit.normal;
}

} // namespace internal
} // namespace wr
