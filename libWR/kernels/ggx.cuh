#pragma once

#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace wr {
namespace internal {

// ============================================================================
// GGX Microfacet BRDF/BSDF
// Reference: Walter et al. 2007, "Microfacet Models for Refraction"
// ============================================================================

// GGX Normal Distribution Function (NDF)
__device__ __forceinline__ float ggxD(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float denom = NoH2 * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI * denom * denom);
}

// Smith G1 term (height-correlated masking-shadowing)
__device__ __forceinline__ float ggxG1(float NoV, float alpha) {
    float a2 = alpha * alpha;
    float NoV2 = NoV * NoV;
    return 2.0f * NoV / (NoV + sqrtf(a2 + (1.0f - a2) * NoV2));
}

// Smith G term (combined masking-shadowing)
__device__ __forceinline__ float ggxG(float NoV, float NoL, float alpha) {
    return ggxG1(NoV, alpha) * ggxG1(NoL, alpha);
}

// Fresnel-Schlick approximation
__device__ __forceinline__ float3 fresnelSchlick(float VoH, const float3& F0) {
    float f = powf(1.0f - VoH, 5.0f);
    return make_float3(
        F0.x + (1.0f - F0.x) * f,
        F0.y + (1.0f - F0.y) * f,
        F0.z + (1.0f - F0.z) * f
    );
}

// Calculate F0 from IOR (for dielectrics)
__device__ __forceinline__ float3 iorToF0(float ior) {
    float f0 = ((ior - 1.0f) / (ior + 1.0f));
    f0 = f0 * f0;
    return make_float3(f0, f0, f0);
}

// Mix F0 based on metallic parameter
__device__ __forceinline__ float3 mixF0(const float3& albedo, float metallic, float ior) {
    float3 dielectricF0 = iorToF0(ior);
    float3 metallicF0 = albedo;  // Metals use albedo as F0
    return make_float3(
        dielectricF0.x * (1.0f - metallic) + metallicF0.x * metallic,
        dielectricF0.y * (1.0f - metallic) + metallicF0.y * metallic,
        dielectricF0.z * (1.0f - metallic) + metallicF0.z * metallic
    );
}

// ============================================================================
// GGX VNDF Sampling (Visible Normal Distribution Function)
// Reference: Heitz 2018, "Sampling the GGX Distribution of Visible Normals"
// ============================================================================

__device__ __forceinline__ float3 sampleGGXVNDF(
    const float3& V,  // View direction in tangent space
    float alpha,
    float u1, float u2
) {
    // Section 3.2: transforming the view direction to the hemisphere configuration
    float3 Vh = normalize(make_float3(alpha * V.x, alpha * V.y, V.z));
    
    // Section 4.1: orthonormal basis
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : make_float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);
    
    // Section 4.2: parameterization of the projected area
    float r = sqrtf(u1);
    float phi = 2.0f * M_PI * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
    
    // Section 4.3: reprojection onto hemisphere
    float3 Nh = make_float3(
        t1 * T1.x + t2 * T2.x + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh.x,
        t1 * T1.y + t2 * T2.y + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh.y,
        t1 * T1.z + t2 * T2.z + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh.z
    );
    
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    return normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
}

// GGX VNDF PDF
__device__ __forceinline__ float ggxVNDFPdf(float NoV, float NoH, float VoH, float alpha) {
    float G1 = ggxG1(NoV, alpha);
    float D = ggxD(NoH, alpha);
    return G1 * VoH * D / NoV;
}

// ============================================================================
// GGX BRDF Evaluation
// ============================================================================

__device__ __forceinline__ float3 evaluateGGXReflection(
    const float3& wo,      // Outgoing direction (to viewer)
    const float3& wi,      // Incoming direction (to light)
    const float3& n,       // Surface normal
    const float3& albedo,
    float roughness,
    float metallic,
    float ior
) {
    float NoV = fmaxf(0.0f, dot(n, wo));
    float NoL = fmaxf(0.0f, dot(n, wi));
    
    if (NoV < 1e-5f || NoL < 1e-5f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float3 h = normalize(make_float3(wo.x + wi.x, wo.y + wi.y, wo.z + wi.z));
    float NoH = fmaxf(0.0f, dot(n, h));
    float VoH = fmaxf(0.0f, dot(wo, h));
    
    float alpha = roughness * roughness;
    
    // Cook-Torrance microfacet specular BRDF
    float D = ggxD(NoH, alpha);
    float G = ggxG(NoV, NoL, alpha);
    float3 F0 = mixF0(albedo, metallic, ior);
    float3 F = fresnelSchlick(VoH, F0);
    
    // Specular term: D * G * F / (4 * NoV * NoL)
    float3 specular = make_float3(
        D * G * F.x / (4.0f * NoV * NoL),
        D * G * F.y / (4.0f * NoV * NoL),
        D * G * F.z / (4.0f * NoV * NoL)
    );
    
    // Diffuse term (for non-metals)
    float3 kD = make_float3(
        (1.0f - F.x) * (1.0f - metallic),
        (1.0f - F.y) * (1.0f - metallic),
        (1.0f - F.z) * (1.0f - metallic)
    );
    
    float3 diffuse = make_float3(
        kD.x * albedo.x / M_PI,
        kD.y * albedo.y / M_PI,
        kD.z * albedo.z / M_PI
    );
    
    // Return BRDF only (caller will multiply by NoL if needed)
    return make_float3(
        diffuse.x + specular.x,
        diffuse.y + specular.y,
        diffuse.z + specular.z
    );
}

// ============================================================================
// GGX PDF Calculation (for MIS)
// ============================================================================

__device__ __forceinline__ float ggxReflectionPdf(
    const float3& wo,      // Outgoing direction (to viewer)
    const float3& wi,      // Incoming direction (to light)
    const float3& n,       // Surface normal
    float roughness
) {
    float NoV = fmaxf(0.0f, dot(n, wo));
    float NoL = fmaxf(0.0f, dot(n, wi));
    
    if (NoV < 1e-5f || NoL < 1e-5f) {
        return 0.0f;
    }
    
    float3 h = normalize(make_float3(wo.x + wi.x, wo.y + wi.y, wo.z + wi.z));
    float NoH = fmaxf(0.0f, dot(n, h));
    float VoH = fmaxf(0.0f, dot(wo, h));
    
    if (VoH < 1e-5f) {
        return 0.0f;
    }
    
    float alpha = roughness * roughness;
    float G1 = ggxG1(NoV, alpha);
    float D = ggxD(NoH, alpha);
    
    // PDF = G1 * |V·m| * D / |V.z| / (4 * |V·m|) = G1 * D / (4 * |V.z|)
    return G1 * D / (4.0f * NoV);
}

// ============================================================================
// GGX BRDF Sampling (VNDF - Visible Normal Distribution Function)
// Reference: libVLR implementation, based on Heitz 2018
// ============================================================================

__device__ __forceinline__ float3 sampleGGXReflection(
    const float3& wo,      // Outgoing direction (to viewer, world space)
    const float3& n,       // Surface normal
    const float3& tangent,
    const float3& bitangent,
    float roughness,
    float u0, float u1,
    float3& wi,            // Output: sampled incoming direction
    float& pdf
) {
    float alpha = roughness * roughness;
    
    // Transform wo to tangent space
    float3 v = make_float3(
        dot(wo, tangent),
        dot(wo, bitangent),
        dot(wo, n)
    );
    
    // Check if view direction is valid (must be in upper hemisphere)
    if (v.z <= 0.0f) {
        pdf = 0.0f;
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Clamp alpha to avoid numerical issues
    alpha = fmaxf(alpha, 0.001f);
    
    // Stretch view direction
    float3 sv = normalize(make_float3(alpha * v.x, alpha * v.y, v.z));
    
    // Build orthonormal basis
    float distIn2D = sqrtf(sv.x * sv.x + sv.y * sv.y);
    float3 T1, T2;
    if (distIn2D > 1e-6f) {
        float recDistIn2D = 1.0f / distIn2D;
        T1 = make_float3(sv.y * recDistIn2D, -sv.x * recDistIn2D, 0.0f);
        T2 = make_float3(T1.y * sv.z, -T1.x * sv.z, distIn2D);
    } else {
        T1 = make_float3(1.0f, 0.0f, 0.0f);
        T2 = make_float3(0.0f, 1.0f, 0.0f);
    }
    
    // Sample point with polar coordinates
    float a = 1.0f / (1.0f + sv.z);
    float r = sqrtf(u0);
    float phi = M_PI * ((u1 < a) ? u1 / a : 1.0f + (u1 - a) / (1.0f - a));
    float sinPhi = sinf(phi);
    float cosPhi = cosf(phi);
    float P1 = r * cosPhi;
    float P2 = r * sinPhi * ((u1 < a) ? 1.0f : sv.z);
    
    // Compute normal in stretched space
    float P1P1_P2P2 = P1 * P1 + P2 * P2;
    float sqrtTerm = sqrtf(fmaxf(0.0f, 1.0f - P1P1_P2P2));
    float3 mr = make_float3(
        P1 * T1.x + P2 * T2.x + sqrtTerm * sv.x,
        P1 * T1.y + P2 * T2.y + sqrtTerm * sv.y,
        P1 * T1.z + P2 * T2.z + sqrtTerm * sv.z
    );
    
    // Unstretch
    mr = normalize(make_float3(alpha * mr.x, alpha * mr.y, mr.z));
    
    // Transform back to world space
    float3 hLocal = mr;
    float3 h = make_float3(
        tangent.x * hLocal.x + bitangent.x * hLocal.y + n.x * hLocal.z,
        tangent.y * hLocal.x + bitangent.y * hLocal.y + n.y * hLocal.z,
        tangent.z * hLocal.x + bitangent.z * hLocal.y + n.z * hLocal.z
    );
    
    // Reflect wo around h to get wi
    float VoH = dot(wo, h);
    if (VoH <= 0.0f) {
        pdf = 0.0f;
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    wi = make_float3(
        2.0f * VoH * h.x - wo.x,
        2.0f * VoH * h.y - wo.y,
        2.0f * VoH * h.z - wo.z
    );
    
    // Check if wi is in the correct hemisphere
    float NoL = dot(n, wi);
    if (NoL <= 0.0f) {
        pdf = 0.0f;
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Calculate PDF using libVLR's formula
    float NoV = fmaxf(0.0f, dot(n, wo));
    float NoH = fmaxf(0.0f, dot(n, h));
    float G1 = ggxG1(NoV, alpha);
    float D = ggxD(NoH, alpha);
    
    // PDF_m = G1(v, m) * |v·m| * D(m) / |v.z|
    // PDF_wi = PDF_m / (4 * |v·m|)
    pdf = G1 * VoH * D / NoV / (4.0f * VoH);
    pdf = G1 * D / (4.0f * NoV);
    
    return wi;
}

} // namespace internal
} // namespace wr
