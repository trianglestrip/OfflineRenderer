#pragma once
#include "material_common.cuh"
#include "utils/fresnel.cuh"

namespace wr {
namespace internal {

// Glass (Specular dielectric) material shader
// Implements physically-based reflection and refraction with Fresnel equations
// Supports nested dielectrics using ray.currentIOR
__device__ __forceinline__ void shadeGlass(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p
) {
    float3 wo = -ray.direction;
    
    // Determine if entering or exiting based on current medium
    // If currentIOR == 1.0, we're in air entering glass
    // If currentIOR == mat.ior, we're in glass exiting to air
    float etaI = ray.currentIOR;
    float etaT = (ray.currentIOR == 1.0f) ? mat.ior : 1.0f;
    
    // Use geometric normal to determine orientation
    float cosI = dot(wo, hit.geometricNormal);
    float3 n = hit.geometricNormal;
    
    // Flip normal if we're exiting (cosI < 0 means we're on the inside)
    if (cosI < 0.0f) {
        cosI = -cosI;
        n = -n;
    }
    
    // Fresnel reflectance
    float F = fresnelDielectric(cosI, etaI, etaT);
    
    // Sample reflection vs refraction based on Fresnel
    float r = rnd_dim(ray.seed, ray.rngDimension++);
    
    if (r < F) {
        // Fresnel reflection - stays in same medium
        float3 reflected = ray.direction - n * (2.0f * dot(ray.direction, n));
        ray.direction = reflected;
        
        // Apply glass albedo (absorption coefficient) for internal reflections
        float3 glassAlbedo = getMaterialAlbedo(mat, hit.uv,
            reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
        
        ray.throughput = make_float3(
            ray.throughput.x * glassAlbedo.x,
            ray.throughput.y * glassAlbedo.y,
            ray.throughput.z * glassAlbedo.z
        );
        
        // currentIOR stays the same for reflection
    } else {
        // Attempt refraction
        float eta = etaI / etaT;
        float sinT2 = eta * eta * (1.0f - cosI * cosI);
        
        if (sinT2 >= 1.0f) {
            // Total internal reflection (no refraction possible)
            float3 reflected = ray.direction - n * (2.0f * dot(ray.direction, n));
            ray.direction = reflected;
            
            // Apply albedo for TIR
            float3 glassAlbedo = getMaterialAlbedo(mat, hit.uv,
                reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
            
            ray.throughput = make_float3(
                ray.throughput.x * glassAlbedo.x,
                ray.throughput.y * glassAlbedo.y,
                ray.throughput.z * glassAlbedo.z
            );
            
            // currentIOR stays the same for TIR
        } else {
            // Successful refraction - switch medium
            float cosT = sqrtf(fmaxf(0.0f, 1.0f - sinT2));
            float3 refracted = eta * ray.direction + n * (eta * cosI - cosT);
            ray.direction = refracted;
            
            // Apply glass albedo (absorption coefficient)
            float3 glassAlbedo = getMaterialAlbedo(mat, hit.uv,
                reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
            
            ray.throughput = make_float3(
                ray.throughput.x * glassAlbedo.x,
                ray.throughput.y * glassAlbedo.y,
                ray.throughput.z * glassAlbedo.z
            );
            
            // Asymmetric scattering correction for refraction
            float correction = asymmetricScatteringCorrection(refracted, hit.normal, hit.geometricNormal);
            ray.throughput = make_float3(
                ray.throughput.x * correction,
                ray.throughput.y * correction,
                ray.throughput.z * correction
            );
            
            // Update current IOR - we've entered/exited the medium
            ray.currentIOR = etaT;
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
