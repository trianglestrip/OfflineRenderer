#pragma once
#include "material_common.cuh"

namespace wr {
namespace internal {

// Lambertian (diffuse) material shader
__device__ __forceinline__ void shadeLambertian(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p
) {
    // Next Event Estimation (NEE) - Direct light sampling with shadow visibility test
    if (p->useNEE && p->numEmissiveTriangles > 0 && !ray.neeDone) {
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
            float pdfArea = 1.0f / (p->numEmissiveTriangles * lightArea);
            float lightPdf = pdfArea * distSq / fmaxf(cosLightTheta, 1e-8f);
            
            // BSDF PDF for this direction
            float bsdfPdf = cosineHemispherePdf(cosTheta);
            
            // MIS weight (power heuristic)
            float misWeight = powerHeuristic(lightPdf, bsdfPdf);
            
            // Get albedo (texture or constant)
            float3 albedo = getMaterialAlbedo(mat, hit.uv,
                reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
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
            
            // Safety check: skip if contribution is invalid
            if (!isnan(contrib.x) && !isnan(contrib.y) && !isnan(contrib.z) &&
                !isinf(contrib.x) && !isinf(contrib.y) && !isinf(contrib.z) &&
                contrib.x >= 0.0f && contrib.y >= 0.0f && contrib.z >= 0.0f) {
                // Store contribution for shadow test
                ray.shadowContribution = contrib;
                ray.savedDirection = ray.direction;
                ray.direction = toLight;
                ray.origin = offsetRayOrigin(hit.position, hit.normal, toLight);
                ray.tMin = 1e-5f;
                ray.tMax = dist - 1e-5f;
                ray.neeDone = 1;
                ray.stage = RayStage::Shadow;
                return;
            }
        }
    }
    
    // Reset neeDone for next bounce
    ray.neeDone = 0;
    
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
    
    // Update throughput: albedo / pi * cos(theta) / pdf
    // For cosine sampling: pdf = cos(theta) / pi, so this simplifies to albedo
    float3 albedo = getMaterialAlbedo(mat, hit.uv,
        reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
    ray.throughput = ray.throughput * albedo;
    
    ray.prevPdf = bsdfPdf;
    ray.prevWasDelta = false;
    
    // Russian Roulette
    if (!applyRussianRoulette(ray, p, false)) {
        return;
    }
    
    // Setup next bounce
    setupNextBounce(ray, hit, worldDir, p);
}

} // namespace internal
} // namespace wr
