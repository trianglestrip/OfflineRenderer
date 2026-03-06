#pragma once

#include <cuda_runtime.h>
#include "internal/gpu_types.h"

using namespace wr::internal;

// Sample texture with bilinear filtering (uses CUDA texture object)
__device__ __forceinline__ float3 sampleTexture2D(
    cudaTextureObject_t texObj,
    float u, float v
) {
    if (texObj == 0) return make_float3(1.0f, 1.0f, 1.0f);
    float4 c = tex2D<float4>(texObj, u, v);
    return make_float3(c.x, c.y, c.z);
}

// Get albedo from material (texture or constant)
__device__ __forceinline__ float3 getMaterialAlbedo(
    const MaterialData& mat,
    const float2& uv,
    const cudaTextureObject_t* textures,
    uint32_t numTextures
) {
    if (mat.albedoTextureId != 0 && mat.albedoTextureId <= numTextures && textures) {
        cudaTextureObject_t tex = textures[mat.albedoTextureId - 1];
        return sampleTexture2D(tex, uv.x, uv.y);
    }
    return make_float3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
}
