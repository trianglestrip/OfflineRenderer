// Wavefront Kernel - Shade stage
#include "wavefront_types.cuh"
#include "materials.cuh"
#include <optix_device.h>

namespace wpt {

__device__ float randf(uint32_t& seed) {
    seed = seed * 1103515245u + 12345u;
    return (float)(seed >> 16) / 65536.0f;
}

__device__ float3 sampleCosineHemisphere(float u1, float u2, float* pdf) {
    float phi = 2.0f * 3.14159265f * u1;
    float cosTheta = sqrtf(u2);
    float sinTheta = sqrtf(1.0f - u2);
    *pdf = cosTheta / 3.14159265f;
    return make_float3(cosf(phi) * sinTheta, cosTheta, sinf(phi) * sinTheta);
}

__global__ void intersectStage(GlobalState* g, OptixTraversableHandle traversable, uint32_t numRays) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRays) return;
    uint32_t rayIdx = g->activeQueue.indices[tid];
    RayState& ray = g->rayPool[rayIdx];
    if (ray.stage != Stage_Intersect) return;
}

__global__ void shadeStage(GlobalState* g, uint32_t numRays) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRays) return;
    uint32_t rayIdx = g->activeQueue.indices[tid];
    RayState& ray = g->rayPool[rayIdx];
    if (ray.stage != Stage_Shade) return;

    uint32_t pi = ray.pixel_index;
    float3 N = make_float3(0, 1, 0);
    uint32_t materialKind = Material_Diffuse;

    float survivalProb = 0.95f;
    if (ray.depth >= 5) {
        survivalProb = fmaxf(0.05f, fminf(0.95f, fmaxf(ray.throughput.r, fmaxf(ray.throughput.g, ray.throughput.b))));
    }

    if (ray.depth >= 8) {
        g->accumBuffer[pi] += ray.radiance;
        ray.stage = Stage_Terminated;
        return;
    }

    if (survivalProb < 1e-5f || randf(ray.seed) > survivalProb) {
        g->accumBuffer[pi] += ray.radiance;
        ray.stage = Stage_Terminated;
        return;
    }

    float3 lightPos = make_float3(0, 5, 0);
    float3 toLight = lightPos - ray.origin;
    float dist = length(toLight);
    float3 L = toLight / dist;
    float NdotL = fmaxf(dot(N, L), 0.0f);

    if (NdotL > 0 && materialKind != Material_Dielectric) {
        uint32_t shadowIdx = atomicAdd(&g->queueCounters[1], 1u);
        g->shadowQueue.indices[shadowIdx] = rayIdx;
    }

    float3 localDir;
    float pdf;
    float3_rgb brdf;

    switch (materialKind) {
        case Material_Diffuse: {
            float u1 = randf(ray.seed);
            float u2 = randf(ray.seed);
            localDir = sampleCosineHemisphere(u1, u2, &pdf);
            brdf = make_rgb(0.8f, 0.8f, 0.8f);
            break;
        }
        default:
            localDir = make_float3(0, 1, 0);
            pdf = 1.0f;
            brdf = make_rgb(1, 1, 1);
            break;
    }

    float3 wi = toWorld(localDir, N);
    float cosTheta = localDir.y;
    ray.throughput = ray.throughput * brdf * (cosTheta / fmaxf(pdf, 1e-7f)) / survivalProb;
    ray.origin = ray.origin + 1e-4f * N;
    ray.direction = wi;
    ray.depth++;
    ray.stage = Stage_Intersect;

    uint32_t nextIdx = atomicAdd(&g->queueCounters[0], 1u);
    g->nextQueue.indices[nextIdx] = rayIdx;
}

__global__ void shadowStage(GlobalState* g, OptixTraversableHandle traversable, uint32_t numShadowRays) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numShadowRays) return;
    uint32_t rayIdx = g->shadowQueue.indices[tid];
    RayState& ray = g->rayPool[rayIdx];
}

__global__ void compactStage(GlobalState* g, uint32_t numRays, uint32_t* outNumActive) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRays) return;
    uint32_t rayIdx = g->activeQueue.indices[tid];
    RayState& ray = g->rayPool[rayIdx];
    if (ray.stage != Stage_Terminated) {
        uint32_t newIdx = atomicAdd(outNumActive, 1u);
        g->nextQueue.indices[newIdx] = rayIdx;
    }
}

} // namespace wpt
