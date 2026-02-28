#include <optix.h>
#include <cuda_runtime.h>
#include <optixw/types.h>
#include "vector_math.cuh"

using namespace optixw;

// Shading parameters
struct ShadeParams {
    RayState* rayPool;
    uint32_t* activeIndices;
    HitInfo* hitBuffer;
    MaterialData* materials;
    float3* accumBuffer;
    uint32_t numActive;
};

// Simple random number generator
__device__ inline float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 16) / 65536.0f;
}

// Cosine hemisphere sampling
__device__ inline float3 sampleCosineHemisphere(float u1, float u2, float* pdf) {
    float phi = 2.0f * 3.14159265f * u1;
    float cosTheta = sqrtf(u2);
    float sinTheta = sqrtf(1.0f - u2);
    *pdf = cosTheta / 3.14159265f;
    return make_float3(
        cosf(phi) * sinTheta,
        cosTheta,
        sinf(phi) * sinTheta
    );
}

// Transform from local to world space
__device__ inline float3 toWorld(const float3& local, const float3& normal) {
    float3 tangent, bitangent;
    if (fabsf(normal.y) < 0.9f) {
        tangent = normalize(cross(make_float3(0, 1, 0), normal));
    } else {
        tangent = normalize(cross(make_float3(1, 0, 0), normal));
    }
    bitangent = cross(normal, tangent);
    
    return tangent * local.x + normal * local.y + bitangent * local.z;
}

// Shade kernel
extern "C" __global__ void shade(ShadeParams params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.numActive) return;
    
    uint32_t rayIndex = params.activeIndices[idx];
    RayState& ray = params.rayPool[rayIndex];
    const HitInfo& hit = params.hitBuffer[rayIndex];
    const MaterialData& mat = params.materials[hit.materialId];
    
    // Handle emissive materials (check emission values)
    bool isEmissive = (mat.emission.x > 0.0f || mat.emission.y > 0.0f || mat.emission.z > 0.0f);
    if (isEmissive) {
        ray.radiance = ray.radiance + ray.throughput * mat.emission;
        ray.stage = RayState::Terminated;
        params.accumBuffer[ray.pixelIndex] = ray.radiance;
        return;
    }
    
    // Lambertian BRDF
    float3 albedo = mat.albedo;
    float3 brdf = albedo * (1.0f / 3.14159265f);
    
    // Sample new direction (cosine hemisphere)
    float u1 = randf(ray.seed);
    float u2 = randf(ray.seed);
    
    float pdf;
    float3 localDir = sampleCosineHemisphere(u1, u2, &pdf);
    float3 newDir = toWorld(localDir, hit.normal);
    
    // Update throughput
    float cosTheta = localDir.y;  // local y = normal direction
    ray.throughput = ray.throughput * brdf * cosTheta / pdf;
    
    // Russian Roulette
    if (ray.depth >= 3) {
        float3 tp = ray.throughput;
        float survivalProb = fminf(0.95f, fmaxf(0.2f, fmaxf(tp.x, fmaxf(tp.y, tp.z))));
        
        if (randf(ray.seed) > survivalProb) {
            ray.stage = RayState::Terminated;
            params.accumBuffer[ray.pixelIndex] = ray.radiance;
            return;
        }
        
        ray.throughput = ray.throughput / survivalProb;
    }
    
    // Update ray
    ray.origin = hit.position + hit.normal * 0.001f;
    ray.direction = newDir;
    ray.tMin = 0.001f;
    ray.tMax = 1e20f;
    ray.depth++;
    ray.stage = RayState::Trace;
}
