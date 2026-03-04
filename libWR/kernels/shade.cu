#include <cuda_runtime.h>
#include <wr/types.h>
#include "vector_math.cuh"

using namespace wr;

static constexpr float kPi = 3.14159265f;

__device__ inline float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 8) / 16777216.0f;
}

__device__ inline float3 cosineSampleHemisphere(uint32_t& seed) {
    float r1 = randf(seed);
    float r2 = randf(seed);
    
    float phi = 2.0f * kPi * r1;
    float cosTheta = sqrtf(r2);
    float sinTheta = sqrtf(1.0f - r2);
    
    return make_float3(
        cosf(phi) * sinTheta,
        sinTheta * sinf(phi),
        cosTheta
    );
}

__device__ inline void createONB(const float3& n, float3& tangent, float3& bitangent) {
    float3 up = fabsf(n.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(up, n));
    bitangent = cross(n, tangent);
}

__device__ inline float3 toWorld(const float3& v, const float3& n, const float3& t, const float3& b) {
    return t * v.x + b * v.y + n * v.z;
}

__device__ inline float fresnel(float cosI, float etaI, float etaT) {
    float sinT2 = etaI / etaT * etaI / etaT * (1.0f - cosI * cosI);
    if (sinT2 > 1.0f) return 1.0f;
    
    float cosT = sqrtf(1.0f - sinT2);
    float rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    float rp = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
    return (rs * rs + rp * rp) * 0.5f;
}

extern "C" __global__ void shade(const LaunchParams* params) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params->numActive) return;
    
    const uint32_t rayIndex = params->activeIndices[idx];
    RayState& ray = params->rayPool[rayIndex];
    
    if (ray.stage != RayStage::Shade) return;
    
    const HitInfo& hit = params->hitBuffer[rayIndex];
    
    if (hit.materialId >= params->numMaterials) {
        ray.stage = RayStage::Terminated;
        return;
    }
    
    const MaterialData& mat = params->materials[hit.materialId];
    
    if (mat.type == MaterialType::Emissive) {
        ray.radiance = ray.radiance + ray.throughput * mat.emission;
        params->accumBuffer[ray.pixelIndex] = params->accumBuffer[ray.pixelIndex] + ray.radiance;
        ray.stage = RayStage::Terminated;
        return;
    }
    
    if (mat.type == MaterialType::Lambertian) {
        float3 tangent, bitangent;
        createONB(hit.normal, tangent, bitangent);
        
        float3 localDir = cosineSampleHemisphere(ray.seed);
        float3 worldDir = toWorld(localDir, hit.normal, tangent, bitangent);
        
        ray.throughput = ray.throughput * mat.albedo;
        ray.origin = hit.position;
        ray.direction = worldDir;
        ray.tMin = 0.001f;
        ray.tMax = 1e20f;
        ray.depth++;
        ray.stage = RayStage::Trace;
        
        if (ray.depth >= 8) {
            params->accumBuffer[ray.pixelIndex] = params->accumBuffer[ray.pixelIndex] + ray.radiance;
            ray.stage = RayStage::Terminated;
        }
        
        return;
    }
    
    if (mat.type == MaterialType::Glass) {
        float3 wo = -ray.direction;
        float cosI = dot(wo, hit.normal);
        
        float etaI = 1.0f;
        float etaT = mat.ior;
        float3 n = hit.normal;
        
        if (cosI < 0.0f) {
            cosI = -cosI;
            n = -n;
            etaI = mat.ior;
            etaT = 1.0f;
        }
        
        float F = fresnel(cosI, etaI, etaT);
        
        float r = randf(ray.seed);
        if (r < F) {
            float3 reflected = ray.direction - n * (2.0f * dot(ray.direction, n));
            ray.direction = reflected;
        } else {
            float eta = etaI / etaT;
            float k = 1.0f - eta * eta * (1.0f - cosI * cosI);
            if (k < 0.0f) {
                float3 reflected = ray.direction - n * (2.0f * dot(ray.direction, n));
                ray.direction = reflected;
            } else {
                float3 refracted = eta * ray.direction + n * (eta * cosI - sqrtf(k));
                ray.direction = refracted;
            }
        }
        
        ray.throughput = ray.throughput * mat.albedo;
        ray.origin = hit.position;
        ray.tMin = 0.001f;
        ray.tMax = 1e20f;
        ray.depth++;
        ray.stage = RayStage::Trace;
        
        if (ray.depth >= 8) {
            params->accumBuffer[ray.pixelIndex] = params->accumBuffer[ray.pixelIndex] + ray.radiance;
            ray.stage = RayStage::Terminated;
        }
        
        return;
    }
    
    params->accumBuffer[ray.pixelIndex] = params->accumBuffer[ray.pixelIndex] + ray.radiance;
    ray.stage = RayStage::Terminated;
}
