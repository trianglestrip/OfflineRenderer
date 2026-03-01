#include <cuda_runtime.h>

extern "C" __global__ void merge_tile(
    const float4* tile,
    int tileW,
    int tileH,
    int inX,
    int inY,
    int offX,
    int offY,
    int imageW,
    int imageH,
    float3* accumColor,
    float* accumWeight,
    int ovx,
    int ovy)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= tileW || py >= tileH) return;

    int gx = inX + px;
    int gy = inY + py;
    if (gx < 0 || gx >= imageW || gy < 0 || gy >= imageH) return;

    int tid = py * tileW + px;
    float4 c = tile[tid];

    // compute 1D taper factors based on distance to tile input edges
    int left = px;
    int right = tileW - 1 - px;
    int top = py;
    int bottom = tileH - 1 - py;

    int minX = left < right ? left : right;
    int minY = top < bottom ? top : bottom;

    float fx = 1.0f;
    float fy = 1.0f;
    if (ovx > 0) fx = fminf(1.0f, (float)minX / (float)ovx);
    if (ovy > 0) fy = fminf(1.0f, (float)minY / (float)ovy);

    float w = fx * fy;

    int idx = gy * imageW + gx;
    atomicAdd(&accumColor[idx].x, c.x * w);
    atomicAdd(&accumColor[idx].y, c.y * w);
    atomicAdd(&accumColor[idx].z, c.z * w);
    atomicAdd(&accumWeight[idx], w);
}

extern "C" __global__ void normalize_accum(
    float3* accumColor,
    float* accumWeight,
    float4* outImg,
    int imageW,
    int imageH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = imageW * imageH;
    if (idx >= total) return;

    float w = accumWeight[idx];
    float3 c = accumColor[idx];
    float eps = 1e-8f;
    float4 res;
    if (w > eps) {
        res.x = c.x / w;
        res.y = c.y / w;
        res.z = c.z / w;
        res.w = 1.0f;
    } else {
        res.x = 0.0f;
        res.y = 0.0f;
        res.z = 0.0f;
        res.w = 1.0f;
    }
    outImg[idx] = res;
}
