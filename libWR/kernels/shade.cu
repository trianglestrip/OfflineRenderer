#include <cuda_runtime.h>
#include "internal/gpu_types.h"
#include "vector_math.cuh"
#include "sampling.cuh"
#include "ggx.cuh"

using namespace wr::internal;

static constexpr float kPi = 3.14159265f;

extern "C" {
    __constant__ const LaunchParams* params_shade;
}

// Atomic add for float3
__device__ inline void atomicAddFloat3(float3* address, float3 value) {
    atomicAdd(&address->x, value.x);
    atomicAdd(&address->y, value.y);
    atomicAdd(&address->z, value.z);
}

// Legacy random function (kept for compatibility)
__device__ inline float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 8) / 16777216.0f;
}

__device__ inline float fresnel(float cosI, float etaI, float etaT) {
    float sinT2 = etaI / etaT * etaI / etaT * (1.0f - cosI * cosI);
    if (sinT2 > 1.0f) return 1.0f;
    
    float cosT = sqrtf(1.0f - sinT2);
    float rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    float rp = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
    return (rs * rs + rp * rp) * 0.5f;
}

extern "C" __global__ void shade(const LaunchParams* p) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p->numActive) return;

    const uint32_t rayIndex = p->activeIndices[idx];
    RayState& ray = p->rayPool[rayIndex];

    if (ray.stage != RayStage::Shade) return;

    const HitInfo& hit = p->hitBuffer[rayIndex];
    
    if (hit.materialId >= p->numMaterials) {
        ray.stage = RayStage::Terminated;
        return;
    }
    
    const MaterialData& mat = p->materials[hit.materialId];
    
    // Emissive material hit is handled in trace.cu with MIS
    if (mat.type == MaterialType::Emissive) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
        return;
    }
    
    if (mat.type == MaterialType::Lambertian) {
        // Next Event Estimation (NEE) - Direct light sampling
        if (p->useNEE && p->numEmissiveTriangles > 0) {
            // Sample a light source (use decorrelated RNG)
            float lightU = rnd_dim(ray.seed, ray.rngDimension++);
            uint32_t lightIdx = sampleEmissiveTriangle(lightU, p->emissiveTriangleCDF, p->numEmissiveTriangles);
            uint32_t triIdx = p->emissiveTriangles[lightIdx];
            
            // Get light triangle vertices
            uint32_t i0 = p->geometry.indices[triIdx * 3 + 0];
            uint32_t i1 = p->geometry.indices[triIdx * 3 + 1];
            uint32_t i2 = p->geometry.indices[triIdx * 3 + 2];
            
            float3 v0 = make_float3(
                p->geometry.vertices[i0 * 3 + 0],
                p->geometry.vertices[i0 * 3 + 1],
                p->geometry.vertices[i0 * 3 + 2]
            );
            float3 v1 = make_float3(
                p->geometry.vertices[i1 * 3 + 0],
                p->geometry.vertices[i1 * 3 + 1],
                p->geometry.vertices[i1 * 3 + 2]
            );
            float3 v2 = make_float3(
                p->geometry.vertices[i2 * 3 + 0],
                p->geometry.vertices[i2 * 3 + 1],
                p->geometry.vertices[i2 * 3 + 2]
            );
            
            // Sample point on light (use decorrelated RNG)
            float3 lightNormal;
            float lightArea;
            float3 lightPos = sampleTriangle(
                rnd_dim(ray.seed, ray.rngDimension++), 
                rnd_dim(ray.seed, ray.rngDimension++), 
                v0, v1, v2, lightNormal, lightArea
            );
            
            // Direction to light
            float3 toLight = make_float3(
                lightPos.x - hit.position.x,
                lightPos.y - hit.position.y,
                lightPos.z - hit.position.z
            );
            float distSq = dot(toLight, toLight);
            float dist = sqrtf(distSq);
            toLight = make_float3(toLight.x / dist, toLight.y / dist, toLight.z / dist);
            
            float cosTheta = dot(hit.normal, toLight);
            float cosLightTheta = -dot(lightNormal, toLight);
            
            if (cosTheta > 0.0f && cosLightTheta > 0.0f) {
                // Get light material
                uint32_t lightMatId = p->geometry.triangleMaterialIds[triIdx];
                const MaterialData& lightMat = p->materials[lightMatId];
                
                // PDF for light sampling (solid angle)
                // PDF_area = 1 / (numLights * area)
                // PDF_solidAngle = PDF_area * distSq / cosLightTheta
                float pdfArea = 1.0f / (p->numEmissiveTriangles * lightArea);
                float lightPdf = pdfArea * distSq / fmaxf(cosLightTheta, 1e-8f);
                
                // BSDF PDF for this direction
                float bsdfPdf = cosineHemispherePdf(cosTheta);
                
                // MIS weight (power heuristic)
                float misWeight = powerHeuristic(lightPdf, bsdfPdf);
                
                // Extract float3 from float4
                float3 albedo = make_float3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
                float3 lightEmission = make_float3(lightMat.emission.x, lightMat.emission.y, lightMat.emission.z);
                
                // BSDF evaluation: albedo / pi * cos(theta)
                float3 bsdf = make_float3(
                    albedo.x / kPi * cosTheta,
                    albedo.y / kPi * cosTheta,
                    albedo.z / kPi * cosTheta
                );
                
                // Contribution: throughput * bsdf * emission * misWeight / lightPdf
                float3 contrib = make_float3(
                    ray.throughput.x * bsdf.x * lightEmission.x * misWeight / lightPdf,
                    ray.throughput.y * bsdf.y * lightEmission.y * misWeight / lightPdf,
                    ray.throughput.z * bsdf.z * lightEmission.z * misWeight / lightPdf
                );
                
                // Safety check: skip if contribution is invalid (NaN, Inf, or negative)
                if (!isnan(contrib.x) && !isnan(contrib.y) && !isnan(contrib.z) &&
                    !isinf(contrib.x) && !isinf(contrib.y) && !isinf(contrib.z) &&
                    contrib.x >= 0.0f && contrib.y >= 0.0f && contrib.z >= 0.0f) {
                    ray.radiance = ray.radiance + contrib;
                }
            }
        }
        
        // BSDF sampling (indirect lighting)
        float3 tangent, bitangent;
        createCoordinateFrame(hit.normal, tangent, bitangent);
        
        float bsdfPdf;
        float3 localDir = sampleCosineHemisphere(
            rnd_dim(ray.seed, ray.rngDimension++), 
            rnd_dim(ray.seed, ray.rngDimension++), 
            bsdfPdf
        );
        float3 worldDir = toWorld(localDir, hit.normal, tangent, bitangent);
        
        // Update throughput with BSDF: albedo / pi * cos(theta) / pdf
        // For cosine sampling: pdf = cos(theta) / pi, so this simplifies to albedo
        float3 albedo = make_float3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
        ray.throughput = ray.throughput * albedo;
        
        // Store PDF for MIS on next hit
        ray.prevPdf = bsdfPdf;
        ray.prevWasDelta = false;
        
        // Russian Roulette
        float3 newThroughput;
        if (!russianRoulette(ray.throughput, ray.depth, p->rrStartDepth, rnd_dim(ray.seed, ray.rngDimension++), newThroughput)) {
            atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
            ray.stage = RayStage::Terminated;
            return;
        }
        ray.throughput = newThroughput;
        
        ray.direction = worldDir;
        
        // Offset origin along normal to avoid self-intersection
        // Offset in the direction of the outgoing ray relative to the surface
        float offset = 0.001f;
        float3 offsetDir = dot(worldDir, hit.normal) > 0.0f ? hit.normal : -hit.normal;
        ray.origin = hit.position + offsetDir * offset;
        ray.tMin = 0.0f;
        ray.tMax = 1e20f;
        ray.depth++;
        ray.stage = RayStage::Trace;

        if (ray.depth >= p->maxBounces) {
            atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
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
        
        float r = rnd_dim(ray.seed, ray.rngDimension++);
        bool isReflection = false;
        if (r < F) {
            float3 reflected = ray.direction - n * (2.0f * dot(ray.direction, n));
            ray.direction = reflected;
            isReflection = true;
        } else {
            float eta = etaI / etaT;
            float k = 1.0f - eta * eta * (1.0f - cosI * cosI);
            if (k < 0.0f) {
                float3 reflected = ray.direction - n * (2.0f * dot(ray.direction, n));
                ray.direction = reflected;
                isReflection = true;
            } else {
                float3 refracted = eta * ray.direction + n * (eta * cosI - sqrtf(k));
                ray.direction = refracted;
                isReflection = false;
            }
        }
        
        // Note: Glass is pure specular, no albedo absorption
        // Fresnel equation already handles energy distribution
        ray.prevWasDelta = true;  // Glass is delta distribution
        
        // Russian Roulette
        float3 newThroughput;
        if (!russianRoulette(ray.throughput, ray.depth, p->rrStartDepth, rnd_dim(ray.seed, ray.rngDimension++), newThroughput)) {
            atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
            ray.stage = RayStage::Terminated;
            return;
        }
        ray.throughput = newThroughput;
        
        // Offset origin to avoid self-intersection
        // Use the direction of the new ray to determine offset direction
        float offset = 0.001f;
        float3 offsetDir = dot(ray.direction, hit.normal) > 0.0f ? hit.normal : -hit.normal;
        ray.origin = hit.position + offsetDir * offset;
        ray.tMin = 0.0f;
        ray.tMax = 1e20f;
        ray.depth++;
        ray.stage = RayStage::Trace;

        if (ray.depth >= p->maxBounces) {
            atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
            ray.stage = RayStage::Terminated;
        }
        
        return;
    }
    
    if (mat.type == MaterialType::GGXReflection) {
        // GGX Microfacet BRDF (PBR material)
        float3 wo = -ray.direction;  // View direction
        
        // NEE for GGX (temporarily disabled for debugging)
        if (false && p->useNEE && p->numEmissiveTriangles > 0) {
            float lightU = rnd_dim(ray.seed, ray.rngDimension++);
            uint32_t lightIdx = sampleEmissiveTriangle(lightU, p->emissiveTriangleCDF, p->numEmissiveTriangles);
            uint32_t triIdx = p->emissiveTriangles[lightIdx];
            
            uint32_t i0 = p->geometry.indices[triIdx * 3 + 0];
            uint32_t i1 = p->geometry.indices[triIdx * 3 + 1];
            uint32_t i2 = p->geometry.indices[triIdx * 3 + 2];
            
            float3 v0 = make_float3(
                p->geometry.vertices[i0 * 3 + 0],
                p->geometry.vertices[i0 * 3 + 1],
                p->geometry.vertices[i0 * 3 + 2]
            );
            float3 v1 = make_float3(
                p->geometry.vertices[i1 * 3 + 0],
                p->geometry.vertices[i1 * 3 + 1],
                p->geometry.vertices[i1 * 3 + 2]
            );
            float3 v2 = make_float3(
                p->geometry.vertices[i2 * 3 + 0],
                p->geometry.vertices[i2 * 3 + 1],
                p->geometry.vertices[i2 * 3 + 2]
            );
            
            float3 lightNormal;
            float lightArea;
            float3 lightPos = sampleTriangle(
                rnd_dim(ray.seed, ray.rngDimension++), 
                rnd_dim(ray.seed, ray.rngDimension++), 
                v0, v1, v2, lightNormal, lightArea
            );
            
            float3 toLight = make_float3(
                lightPos.x - hit.position.x,
                lightPos.y - hit.position.y,
                lightPos.z - hit.position.z
            );
            float distSq = dot(toLight, toLight);
            float dist = sqrtf(distSq);
            toLight = make_float3(toLight.x / dist, toLight.y / dist, toLight.z / dist);
            
            float cosTheta = dot(hit.normal, toLight);
            float cosLightTheta = -dot(lightNormal, toLight);
            
            if (cosTheta > 0.0f && cosLightTheta > 0.0f) {
                uint32_t lightMatId = p->geometry.triangleMaterialIds[triIdx];
                const MaterialData& lightMat = p->materials[lightMatId];
                
                // PDF for light sampling (solid angle)
                float pdfArea = 1.0f / (p->numEmissiveTriangles * lightArea);
                float lightPdf = pdfArea * distSq / fmaxf(cosLightTheta, 1e-8f);
                
                // Extract float3 from float4
                float3 albedo = make_float3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
                float3 lightEmission = make_float3(lightMat.emission.x, lightMat.emission.y, lightMat.emission.z);
                
                // Evaluate GGX BRDF (returns BRDF only, not BRDF * cosTheta)
                float3 brdf = evaluateGGXReflection(wo, toLight, hit.normal, albedo, mat.roughness, mat.metallic, mat.ior);
                
                // Calculate GGX PDF for MIS
                float bsdfPdf = ggxReflectionPdf(wo, toLight, hit.normal, mat.roughness);
                float misWeight = powerHeuristic(lightPdf, bsdfPdf);
                
                // Contribution: throughput * BRDF * emission * cosTheta * MIS / lightPdf
                float3 contrib = make_float3(
                    ray.throughput.x * brdf.x * lightEmission.x * cosTheta * misWeight / lightPdf,
                    ray.throughput.y * brdf.y * lightEmission.y * cosTheta * misWeight / lightPdf,
                    ray.throughput.z * brdf.z * lightEmission.z * cosTheta * misWeight / lightPdf
                );
                
                // Safety check: skip if contribution is invalid
                if (!isnan(contrib.x) && !isnan(contrib.y) && !isnan(contrib.z) &&
                    !isinf(contrib.x) && !isinf(contrib.y) && !isinf(contrib.z) &&
                    contrib.x >= 0.0f && contrib.y >= 0.0f && contrib.z >= 0.0f) {
                    ray.radiance = ray.radiance + contrib;
                }
            }
        }
        
        // BSDF sampling using VNDF
        float3 tangent, bitangent;
        createCoordinateFrame(hit.normal, tangent, bitangent);
        
        float3 wi;
        float bsdfPdf;
        sampleGGXReflection(
            wo, hit.normal, tangent, bitangent, mat.roughness, 
            rnd_dim(ray.seed, ray.rngDimension++), 
            rnd_dim(ray.seed, ray.rngDimension++), 
            wi, bsdfPdf
        );
        
        // Check if sampling failed
        float cosTheta = dot(hit.normal, wi);
        if (bsdfPdf < 1e-5f || cosTheta <= 0.0f) {
            // Sampling failed, terminate ray
            atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
            ray.stage = RayStage::Terminated;
            return;
        }
        
        // Extract float3 from float4
        float3 albedo = make_float3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
        
        // Evaluate BRDF (returns BRDF only, not BRDF * cosTheta)
        float3 brdf = evaluateGGXReflection(wo, wi, hit.normal, albedo, mat.roughness, mat.metallic, mat.ior);
        
        // Check for NaN or Inf in BRDF
        if (isnan(brdf.x) || isnan(brdf.y) || isnan(brdf.z) ||
            isinf(brdf.x) || isinf(brdf.y) || isinf(brdf.z)) {
            // Invalid BRDF, terminate ray
            atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
            ray.stage = RayStage::Terminated;
            return;
        }
        
        // Update throughput: brdf * cosTheta / pdf
        ray.throughput = make_float3(
            ray.throughput.x * brdf.x * cosTheta / bsdfPdf,
            ray.throughput.y * brdf.y * cosTheta / bsdfPdf,
            ray.throughput.z * brdf.z * cosTheta / bsdfPdf
        );
        
        ray.prevPdf = bsdfPdf;
        ray.prevWasDelta = false;
        
        // Russian Roulette
        float3 newThroughput;
        if (!russianRoulette(ray.throughput, ray.depth, p->rrStartDepth, rnd_dim(ray.seed, ray.rngDimension++), newThroughput)) {
            atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
            ray.stage = RayStage::Terminated;
            return;
        }
        ray.throughput = newThroughput;
        
        ray.direction = wi;
        
        // Offset origin along normal to avoid self-intersection
        // Offset in the direction of the outgoing ray relative to the surface
        float offset = 0.001f;
        float3 offsetDir = dot(wi, hit.normal) > 0.0f ? hit.normal : -hit.normal;
        ray.origin = hit.position + offsetDir * offset;
        ray.tMin = 0.0f;
        ray.tMax = 1e20f;
        ray.depth++;
        ray.stage = RayStage::Trace;

        if (ray.depth >= p->maxBounces) {
            atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
            ray.stage = RayStage::Terminated;
        }
        
        return;
    }
    
    atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
    ray.stage = RayStage::Terminated;
}
