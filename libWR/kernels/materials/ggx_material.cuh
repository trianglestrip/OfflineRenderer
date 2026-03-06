#pragma once
#include "material_common.cuh"
#include "ggx.cuh"

namespace wr {
namespace internal {

// GGX Microfacet BRDF material shader
// Implements physically-based rough metal reflection
__device__ __forceinline__ void shadeGGX(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p
) {
    float3 wo = -ray.direction;  // View direction
    
    // NEE for GGX
    if (p->useNEE && p->numEmissiveTriangles > 0) {
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
            
            // Evaluate GGX BRDF
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
            
            // Safety check
            if (!isnan(contrib.x) && !isnan(contrib.y) && !isnan(contrib.z) &&
                !isinf(contrib.x) && !isinf(contrib.y) && !isinf(contrib.z) &&
                contrib.x >= 0.0f && contrib.y >= 0.0f && contrib.z >= 0.0f) {
                ray.radiance = make_float3(
                    ray.radiance.x + contrib.x,
                    ray.radiance.y + contrib.y,
                    ray.radiance.z + contrib.z
                );
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
        terminateRay(ray, p);
        return;
    }
    
    // Extract albedo
    float3 albedo = make_float3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
    
    // Evaluate BRDF
    float3 brdf = evaluateGGXReflection(wo, wi, hit.normal, albedo, mat.roughness, mat.metallic, mat.ior);
    
    // Check for NaN or Inf
    if (isnan(brdf.x) || isnan(brdf.y) || isnan(brdf.z) ||
        isinf(brdf.x) || isinf(brdf.y) || isinf(brdf.z)) {
        terminateRay(ray, p);
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
    if (!applyRussianRoulette(ray, p, false)) {
        return;
    }
    
    // Setup next bounce
    setupNextBounce(ray, hit, wi, p);
}

} // namespace internal
} // namespace wr
