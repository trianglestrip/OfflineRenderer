#pragma once

#include <cuda_runtime.h>
#include "internal/gpu_types.h"
#include "vector_math.cuh"
#include "sampling.cuh"

namespace wr {
namespace internal {

// Store a photon in the global photon map
__device__ inline void storePhoton(
    PhotonMapParams& photonMap,
    const float3& position,
    const float3& direction,
    const float3& power,
    uint32_t flags) {
    
    uint32_t idx = atomicAdd(&photonMap.numPhotons, 1);
    if (idx >= photonMap.maxPhotons) return;
    
    Photon p;
    p.position = position;
    p.direction = direction;
    p.power = power;
    p.flags = flags;
    
    // Use pointer arithmetic to avoid assignment operator issues
    Photon* photonArray = const_cast<Photon*>(photonMap.photons);
    photonArray[idx] = p;
}

// Store a caustic photon
__device__ inline void storeCausticPhoton(
    CausticPhotonMapParams& causticMap,
    const float3& position,
    const float3& direction,
    const float3& power) {
    
    uint32_t idx = atomicAdd(&causticMap.numPhotons, 1);
    if (idx >= causticMap.maxPhotons) return;
    
    Photon p;
    p.position = position;
    p.direction = direction;
    p.power = power;
    p.flags = PHOTON_CAUSTIC;
    
    Photon* photonArray = const_cast<Photon*>(causticMap.photons);
    photonArray[idx] = p;
}

// Estimate radiance from photon map at a point
__device__ inline float3 estimateRadiance(
    const PhotonMapParams& photonMap,
    const float3& position,
    const float3& normal,
    const float3& albedo) {
    
    if (photonMap.numPhotons == 0) return make_float3(0.0f, 0.0f, 0.0f);
    
    float3 accumulatedPower = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t photonCount = 0;
    
    // Simple linear search (can be optimized with KD-tree or hash grid)
    // For now, use a simple radius search
    float radiusSqr = photonMap.searchRadius * photonMap.searchRadius;
    
    for (uint32_t i = 0; i < photonMap.numPhotons && i < photonMap.maxPhotons; ++i) {
        const Photon& p = photonMap.photons[i];
        
        float3 diff = p.position - position;
        float distSqr = dot(diff, diff);
        
        if (distSqr < radiusSqr) {
            // Check if photon is on the same side of the surface
            if (dot(p.direction, normal) < 0.0f) {
                accumulatedPower = accumulatedPower + p.power;
                photonCount++;
                
                if (photonCount >= photonMap.maxPhotonsPerQuery) break;
            }
        }
    }
    
    if (photonCount == 0) return make_float3(0.0f, 0.0f, 0.0f);
    
    // Divide by area (pi * r^2) for density estimation
    const float kPi = 3.14159265f;
    float area = kPi * radiusSqr;
    float3 radiance = accumulatedPower / area;
    
    // Multiply by BRDF (Lambertian: albedo / pi)
    return radiance * albedo / kPi;
}

// Estimate caustic radiance
__device__ inline float3 estimateCausticRadiance(
    const CausticPhotonMapParams& causticMap,
    const float3& position,
    const float3& normal,
    const float3& albedo) {
    
    if (causticMap.numPhotons == 0) return make_float3(0.0f, 0.0f, 0.0f);
    
    float3 accumulatedPower = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t photonCount = 0;
    
    float radiusSqr = causticMap.searchRadius * causticMap.searchRadius;
    
    for (uint32_t i = 0; i < causticMap.numPhotons && i < causticMap.maxPhotons; ++i) {
        const Photon& p = causticMap.photons[i];
        
        float3 diff = p.position - position;
        float distSqr = dot(diff, diff);
        
        if (distSqr < radiusSqr) {
            if (dot(p.direction, normal) < 0.0f) {
                accumulatedPower = accumulatedPower + p.power;
                photonCount++;
                
                if (photonCount >= causticMap.maxPhotonsPerQuery) break;
            }
        }
    }
    
    if (photonCount == 0) return make_float3(0.0f, 0.0f, 0.0f);
    
    const float kPi = 3.14159265f;
    float area = kPi * radiusSqr;
    float3 radiance = accumulatedPower / area;
    
    return radiance * albedo / kPi;
}

} // namespace internal
} // namespace wr
