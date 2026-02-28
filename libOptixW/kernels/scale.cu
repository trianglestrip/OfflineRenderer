#include <cuda_runtime.h>
#include <optixw/types.h>

using namespace optixw;

extern "C" __global__ void scale_to_float4(
    const float3* accum,
    float4* output,
    uint32_t count,
    float invSpp)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    float3 v = accum[idx];
    output[idx] = make_float4(v.x * invSpp, v.y * invSpp, v.z * invSpp, 1.0f);
}
