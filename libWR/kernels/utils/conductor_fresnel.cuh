#pragma once

#include <cuda_runtime.h>
#include "../vector_math.cuh"

namespace wr {
namespace internal {

// ============================================================================
// Conductor Fresnel (for metals)
// Uses complex IOR: eta (real part) and k (imaginary part)
// Reference: PBRT v3, Chapter 8.2.1
// ============================================================================

// Fresnel for conductors (metals)
// eta: real part of complex IOR
// k: imaginary part of complex IOR (absorption)
__device__ __forceinline__ float3 fresnelConductor(float cosTheta, const float3& eta, const float3& k) {
    cosTheta = fmaxf(-1.0f, fminf(1.0f, cosTheta));
    
    float3 eta2 = eta * eta;
    float3 k2 = k * k;
    float3 cosTheta2 = make_float3(cosTheta * cosTheta, cosTheta * cosTheta, cosTheta * cosTheta);
    
    float3 twoEtaCosTheta = make_float3(
        2.0f * eta.x * cosTheta,
        2.0f * eta.y * cosTheta,
        2.0f * eta.z * cosTheta
    );
    
    // tmp_f = eta^2 + k^2
    float3 tmp_f = eta2 + k2;
    
    // tmp = tmp_f * cos^2(theta)
    float3 tmp = tmp_f * cosTheta2;
    
    // R_parallel^2 = (tmp - 2*eta*cos(theta) + 1) / (tmp + 2*eta*cos(theta) + 1)
    float3 rParl2 = (tmp - twoEtaCosTheta + make_float3(1.0f, 1.0f, 1.0f)) / 
                    (tmp + twoEtaCosTheta + make_float3(1.0f, 1.0f, 1.0f));
    
    // R_perpendicular^2 = (tmp_f - 2*eta*cos(theta) + cos^2(theta)) / 
    //                     (tmp_f + 2*eta*cos(theta) + cos^2(theta))
    float3 rPerp2 = (tmp_f - twoEtaCosTheta + cosTheta2) / 
                    (tmp_f + twoEtaCosTheta + cosTheta2);
    
    // F = (R_parallel^2 + R_perpendicular^2) / 2
    return (rParl2 + rPerp2) * 0.5f;
}

// Simplified version using scalar eta and k (same for all channels)
__device__ __forceinline__ float fresnelConductor(float cosTheta, float eta, float k) {
    cosTheta = fmaxf(-1.0f, fminf(1.0f, cosTheta));
    
    float eta2 = eta * eta;
    float k2 = k * k;
    float cosTheta2 = cosTheta * cosTheta;
    
    float twoEtaCosTheta = 2.0f * eta * cosTheta;
    float tmp_f = eta2 + k2;
    float tmp = tmp_f * cosTheta2;
    
    float rParl2 = (tmp - twoEtaCosTheta + 1.0f) / (tmp + twoEtaCosTheta + 1.0f);
    float rPerp2 = (tmp_f - twoEtaCosTheta + cosTheta2) / (tmp_f + twoEtaCosTheta + cosTheta2);
    
    return (rParl2 + rPerp2) * 0.5f;
}

// Common metal IOR values (eta, k) for RGB
// Reference: PBRT v3, https://refractiveindex.info/

// Gold (Au)
__device__ __forceinline__ void getGoldIOR(float3& eta, float3& k) {
    // RGB wavelengths: ~650nm, ~530nm, ~460nm
    eta = make_float3(0.12481f, 0.468228f, 1.44476f);
    k = make_float3(3.32107f, 2.23761f, 1.69196f);
}

// Silver (Ag)
__device__ __forceinline__ void getSilverIOR(float3& eta, float3& k) {
    eta = make_float3(0.157099f, 0.144013f, 0.134847f);
    k = make_float3(3.82438f, 3.1451f, 2.27711f);
}

// Copper (Cu)
__device__ __forceinline__ void getCopperIOR(float3& eta, float3& k) {
    eta = make_float3(0.237698f, 0.734847f, 1.37062f);
    k = make_float3(3.44233f, 2.55751f, 2.23429f);
}

// Aluminum (Al)
__device__ __forceinline__ void getAluminumIOR(float3& eta, float3& k) {
    eta = make_float3(1.27579f, 0.940922f, 0.574879f);
    k = make_float3(7.30257f, 6.33458f, 5.16694f);
}

// Iron (Fe)
__device__ __forceinline__ void getIronIOR(float3& eta, float3& k) {
    eta = make_float3(2.71866f, 2.50954f, 2.22767f);
    k = make_float3(3.79528f, 3.40035f, 3.00114f);
}

// Get IOR for a specific metal type
enum class MetalType : uint32_t {
    Gold = 0,
    Silver = 1,
    Copper = 2,
    Aluminum = 3,
    Iron = 4,
    Custom = 5
};

__device__ __forceinline__ void getMetalIOR(MetalType type, float3& eta, float3& k) {
    switch (type) {
        case MetalType::Gold:
            getGoldIOR(eta, k);
            break;
        case MetalType::Silver:
            getSilverIOR(eta, k);
            break;
        case MetalType::Copper:
            getCopperIOR(eta, k);
            break;
        case MetalType::Aluminum:
            getAluminumIOR(eta, k);
            break;
        case MetalType::Iron:
            getIronIOR(eta, k);
            break;
        default:
            // Default to gold
            getGoldIOR(eta, k);
            break;
    }
}

} // namespace internal
} // namespace wr
