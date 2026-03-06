#pragma once
#include "material_common.cuh"
#include "ggx.cuh"
#include "utils/fresnel.cuh"

namespace wr {
namespace internal {

// GGX Microfacet BSDF (Rough Glass) material shader
// Current: Simplified specular refraction (no microfacet sampling)
// TODO: Implement full GGX VNDF sampling for physically accurate rough glass
__device__ __forceinline__ void shadeGGXTransmission(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p
) {
    float3 wo = -ray.direction;
    float cosI = dot(wo, hit.normal);
    
    float etaI = 1.0f;
    float etaT = mat.ior;
    float3 n = hit.normal;
    
    // Determine if entering or exiting the dielectric
    if (cosI < 0.0f) {
        cosI = -cosI;
        n = -n;
        etaI = mat.ior;
        etaT = 1.0f;
    }
    
    // Fresnel reflectance
    float F = fresnelDielectric(cosI, etaI, etaT);
    
    // Sample reflection vs refraction based on Fresnel
    float r = rnd_dim(ray.seed, ray.rngDimension++);
    
    if (r < F) {
        // Fresnel reflection
        float3 reflected = ray.direction - n * (2.0f * dot(ray.direction, n));
        ray.direction = reflected;
        
        // Apply glass albedo (absorption coefficient)
        float3 albedo = getMaterialAlbedo(mat, hit.uv,
            reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
        
        ray.throughput = make_float3(
            ray.throughput.x * albedo.x,
            ray.throughput.y * albedo.y,
            ray.throughput.z * albedo.z
        );
    } else {
        // Attempt refraction
        float eta = etaI / etaT;
        float sinT2 = eta * eta * (1.0f - cosI * cosI);
        
        if (sinT2 >= 1.0f) {
            // Total internal reflection (no refraction possible)
            float3 reflected = ray.direction - n * (2.0f * dot(ray.direction, n));
            ray.direction = reflected;
            
            // Apply albedo for TIR
            float3 albedo = getMaterialAlbedo(mat, hit.uv,
                reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
            
            ray.throughput = make_float3(
                ray.throughput.x * albedo.x,
                ray.throughput.y * albedo.y,
                ray.throughput.z * albedo.z
            );
        } else {
            // Successful refraction
            float cosT = sqrtf(fmaxf(0.0f, 1.0f - sinT2));
            float3 refracted = eta * ray.direction + n * (eta * cosI - cosT);
            ray.direction = refracted;
            
            // Apply glass albedo (absorption coefficient)
            float3 albedo = getMaterialAlbedo(mat, hit.uv,
                reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
            
            ray.throughput = make_float3(
                ray.throughput.x * albedo.x,
                ray.throughput.y * albedo.y,
                ray.throughput.z * albedo.z
            );
        }
    }
    
    ray.prevWasDelta = true;
    
    // Russian Roulette (higher survival rate for delta materials)
    if (!applyRussianRoulette(ray, p, true)) {
        return;
    }
    
    // Setup next bounce
    setupNextBounce(ray, hit, ray.direction, p);
}

} // namespace internal
} // namespace wr
