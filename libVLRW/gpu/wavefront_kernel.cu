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

    const HitInfo& hit = g->hitBuffer[rayIdx];
    float3 N = hit.normal;
    float3 position = hit.position;
    uint32_t matId = hit.material_id;
    float3_rgb albedo = make_rgb(0.8f, 0.8f, 0.8f);
    float3_rgb emission = make_rgb(0.0f, 0.0f, 0.0f);
    if (g->materials && matId < 0xFFFFFFFFu) {
        MaterialData& m = g->materials[matId];
        albedo = make_rgb(m.r, m.g, m.b);
        emission = make_rgb(m.er, m.eg, m.eb);
    }

    uint32_t pi = ray.pixel_index;
    
    // Add emission from hit surface
    ray.radiance = ray.radiance + ray.throughput * emission;

    // Next Event Estimation (NEE) - direct lighting from point lights with shadow visibility
    if (g->lights && g->numLights > 0) {
        for (uint32_t li = 0; li < g->numLights; ++li) {
            PointLight& light = g->lights[li];
            float3 toLight = light.position - position;
            float distSq = dot(toLight, toLight);
            float dist = sqrtf(distSq);
            float3 L = toLight / dist;
            
            float NdotL = dot(N, L);
            if (NdotL > 0.0f) {
                // BRDF evaluation (Lambert diffuse)
                float3_rgb brdf = albedo * (1.0f / 3.14159265f);
                
                // Light contribution with inverse square falloff
                float3_rgb Li = light.intensity / distSq;
                
                // Visibility from shadow ray trace (1=visible, 0=occluded)
                float visibility = (g->visibilityBuffer != nullptr)
                    ? (float)g->visibilityBuffer[tid * g->numLights + li]
                    : 1.0f;
                
                float3_rgb directLight = brdf * Li * NdotL * visibility;
                ray.radiance = ray.radiance + ray.throughput * directLight;
            }
        }
    }

    // NEE - area lights: uniform sample on light surface
    if (g->areaLights && g->numAreaLights > 0) {
        for (uint32_t li = 0; li < g->numAreaLights; ++li) {
            AreaLight& light = g->areaLights[li];
            float3 lightN = normalize(light.normal);
            float3 lightT = normalize(light.tangent);
            float3 lightB = normalize(cross(lightN, lightT));
            float area = light.width * light.height;
            if (area < 1e-8f) continue;
            float pdf = 1.0f / area;

            float u1 = randf(ray.seed);
            float u2 = randf(ray.seed);
            float3 lightSample = light.position
                + (u1 - 0.5f) * light.width * lightT
                + (u2 - 0.5f) * light.height * lightB;

            float3 toLight = make_float3(
                lightSample.x - position.x,
                lightSample.y - position.y,
                lightSample.z - position.z);
            float distSq = dot(toLight, toLight);
            float dist = sqrtf(distSq);
            if (dist < 1e-6f) continue;
            float3 L = make_float3(toLight.x / dist, toLight.y / dist, toLight.z / dist);

            float NdotL = dot(N, L);
            if (NdotL <= 0.0f) continue;

            float3 toHit = make_float3(-L.x, -L.y, -L.z);
            float cosThetaLight = dot(lightN, toHit);
            if (light.doubleSided)
                cosThetaLight = fabsf(cosThetaLight);
            else if (cosThetaLight <= 0.0f)
                continue;

            float3_rgb Li = light.emission * (cosThetaLight / (distSq * pdf));
            float3_rgb brdf = albedo * (1.0f / 3.14159265f);
            float visibility = 1.0f;
            float3_rgb directLight = brdf * Li * NdotL * visibility;
            ray.radiance = ray.radiance + ray.throughput * directLight;
        }
    }

    // NO ambient lighting - let indirect illumination be fully visible
    // (Removed ambient term to make color bleeding more prominent)

    // Russian Roulette: delayed to allow more bounces for color bleeding
    float survivalProb = 0.95f;
    if (ray.depth >= 5)
        survivalProb = fmaxf(0.2f, fminf(0.95f, luminance(ray.throughput)));
    if (ray.depth >= 8) {
        g->accumBuffer[pi] += ray.radiance;
        ray.stage = Stage_Terminated;
        return;
    }
    if (randf(ray.seed) > survivalProb) {
        g->accumBuffer[pi] += ray.radiance;
        ray.stage = Stage_Terminated;
        return;
    }

    float u1 = randf(ray.seed);
    float u2 = randf(ray.seed);
    float pdf;
    float3 localDir = sampleCosineHemisphere(u1, u2, &pdf);
    float3_rgb brdf = albedo * (1.0f / 3.14159265f);
    float cosTheta = localDir.y;
    ray.throughput = ray.throughput * brdf * (cosTheta / fmaxf(pdf, 1e-7f)) / survivalProb;
    ray.origin = position + 1e-4f * N;
    ray.direction = toWorld(localDir, N);
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

extern "C" void wpt_launchShadeStage(GlobalState* d_g, uint32_t numRays, unsigned int gridSize, unsigned int blockSize) {
    shadeStage<<<gridSize, blockSize>>>(d_g, numRays);
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
