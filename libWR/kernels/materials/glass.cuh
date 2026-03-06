#pragma once
#include "material_common.cuh"
#include "utils/fresnel.cuh"

namespace wr {
namespace internal {

// Glass (Specular dielectric) material shader
// Implements physically-based reflection and refraction with Fresnel equations
__device__ __forceinline__ void shadeGlass(
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
        
        // Throughput update for reflection:
        // BSDF returns: coeff * F / |cos(wo)|
        // Path tracer multiplies: bsdf_value * |cos(wo)| / pdf
        // PDF = F
        // Result: coeff * F / |cos| * |cos| / F = coeff
        // For ideal glass (coeff=1), throughput unchanged
        
        // Apply glass albedo (absorption coefficient)
        float3 glassAlbedo = getMaterialAlbedo(mat, hit.uv,
            reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
        
        ray.throughput = make_float3(
            ray.throughput.x * glassAlbedo.x,
            ray.throughput.y * glassAlbedo.y,
            ray.throughput.z * glassAlbedo.z
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
            float3 glassAlbedo = getMaterialAlbedo(mat, hit.uv,
                reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
            
            ray.throughput = make_float3(
                ray.throughput.x * glassAlbedo.x,
                ray.throughput.y * glassAlbedo.y,
                ray.throughput.z * glassAlbedo.z
            );
        } else {
            // Successful refraction
            float cosT = sqrtf(fmaxf(0.0f, 1.0f - sinT2));
            float3 refracted = eta * ray.direction + n * (eta * cosI - cosT);
            ray.direction = refracted;
            
            // Throughput update for refraction:
            // BSDF returns: coeff * (1-F) * squeeze / |cos(wi)|
            // Path tracer multiplies: bsdf_value * |cos(wi)| / pdf
            // PDF = (1-F) * squeeze
            // Result: coeff * (1-F) * squeeze / |cos| * |cos| / [(1-F) * squeeze] = coeff
            // For ideal glass (coeff=1), throughput unchanged
            //
            // Reference: libVLR materials.cu:894-898, path_tracing.cu:250-251
            // The squeeze factor appears in both BSDF and PDF, so they cancel out
            
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
