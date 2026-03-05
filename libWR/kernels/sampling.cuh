#pragma once

#include "internal/gpu_types.h"
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace wr {
namespace internal {

// ============================================================================
// Random Number Generation (PCG)
// ============================================================================

__device__ __forceinline__ uint32_t pcg_hash(uint32_t seed) {
    uint32_t state = seed * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float rnd(uint32_t& seed) {
    seed = pcg_hash(seed);
    return float(seed) / 4294967296.0f;
}

// ============================================================================
// Coordinate Frame (for BSDF sampling)
// ============================================================================

__device__ __forceinline__ void createCoordinateFrame(
    const float3& n,
    float3& tangent,
    float3& bitangent
) {
    float3 up = fabsf(n.y) < 0.999f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    tangent = normalize(cross(up, n));
    bitangent = cross(n, tangent);
}

__device__ __forceinline__ float3 toWorld(const float3& v, const float3& n, const float3& t, const float3& b) {
    return make_float3(
        t.x * v.x + b.x * v.y + n.x * v.z,
        t.y * v.x + b.y * v.y + n.y * v.z,
        t.z * v.x + b.z * v.y + n.z * v.z
    );
}

// ============================================================================
// Cosine-weighted Hemisphere Sampling
// ============================================================================

__device__ __forceinline__ float3 sampleCosineHemisphere(float u1, float u2, float& pdf) {
    float r = sqrtf(u1);
    float theta = 2.0f * M_PI * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    
    pdf = z / M_PI;  // cos(theta) / pi
    return make_float3(x, y, z);
}

__device__ __forceinline__ float cosineHemispherePdf(float cosTheta) {
    return fmaxf(0.0f, cosTheta / M_PI);
}

// ============================================================================
// Triangle Sampling (for NEE)
// ============================================================================

__device__ __forceinline__ float3 sampleTriangle(
    float u1, float u2,
    const float3& v0, const float3& v1, const float3& v2,
    float3& normal, float& area
) {
    // Uniform triangle sampling
    float su = sqrtf(u1);
    float b0 = 1.0f - su;
    float b1 = su * (1.0f - u2);
    float b2 = su * u2;
    
    float3 p = make_float3(
        v0.x * b0 + v1.x * b1 + v2.x * b2,
        v0.y * b0 + v1.y * b1 + v2.y * b2,
        v0.z * b0 + v1.z * b1 + v2.z * b2
    );
    
    // Calculate normal and area
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 crossProd = cross(e1, e2);
    area = 0.5f * length(crossProd);
    normal = normalize(crossProd);
    
    return p;
}

// ============================================================================
// Light Sampling (NEE)
// ============================================================================

__device__ __forceinline__ uint32_t sampleEmissiveTriangle(
    float u,
    const float* cdf,
    uint32_t numLights
) {
    // Binary search in CDF
    uint32_t left = 0;
    uint32_t right = numLights;
    
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        if (cdf[mid] < u) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return min(left, numLights - 1);
}

// ============================================================================
// Multiple Importance Sampling (MIS)
// ============================================================================

__device__ __forceinline__ float powerHeuristic(float pdf1, float pdf2) {
    float p1 = pdf1 * pdf1;
    float p2 = pdf2 * pdf2;
    return p1 / (p1 + p2);
}

__device__ __forceinline__ float balanceHeuristic(float pdf1, float pdf2) {
    return pdf1 / (pdf1 + pdf2);
}

// ============================================================================
// Russian Roulette
// ============================================================================

__device__ __forceinline__ bool russianRoulette(
    const float3& throughput,
    uint32_t depth,
    float rrStartDepth,
    float xi,
    float3& newThroughput
) {
    if (depth < rrStartDepth) {
        newThroughput = throughput;
        return true;
    }
    
    // Survival probability based on throughput luminance
    float luminance = 0.2126f * throughput.x + 0.7152f * throughput.y + 0.0722f * throughput.z;
    float q = fmaxf(0.05f, fminf(0.95f, luminance));
    
    if (xi < q) {
        newThroughput = make_float3(throughput.x / q, throughput.y / q, throughput.z / q);
        return true;
    }
    
    return false;
}

} // namespace internal
} // namespace wr
