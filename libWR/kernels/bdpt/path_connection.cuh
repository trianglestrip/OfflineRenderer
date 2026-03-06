#pragma once
#include "bdpt_types.cuh"
#include "sampling.cuh"
#include "vector_math.cuh"

namespace wr {
namespace internal {

// Calculate MIS weight for connecting eye path vertex s and light path vertex t
__device__ __forceinline__ float calculateBDPTMISWeight(
    const EyePath& eyePath,
    const LightPath& lightPath,
    uint32_t s,  // Number of eye path vertices (excluding camera)
    uint32_t t   // Number of light path vertices (excluding light source)
) {
    // Placeholder: Full MIS weight calculation for BDPT
    // Reference: Veach thesis, Chapter 10
    
    // For now, return uniform weight (1.0)
    // Full implementation requires:
    // 1. Calculate all path probabilities (p_s, p_t for all strategies)
    // 2. Apply power heuristic or balance heuristic
    // 3. Handle delta vertices (infinite PDF)
    
    return 1.0f;
}

// Connect eye path vertex and light path vertex
__device__ __forceinline__ PathConnection connectPaths(
    const EyePath& eyePath,
    const LightPath& lightPath,
    uint32_t s,  // Eye path length
    uint32_t t,  // Light path length
    const LaunchParams* p
) {
    PathConnection result;
    result.valid = false;
    result.contribution = make_float3(0.0f, 0.0f, 0.0f);
    result.misWeight = 0.0f;
    
    if (s == 0 || t == 0 || s > eyePath.length || t > lightPath.length) {
        return result;
    }
    
    const PathVertex& eyeVertex = eyePath.vertices[s - 1];
    const PathVertex& lightVertex = lightPath.vertices[t - 1];
    
    // Calculate connection vector
    float3 connectionVec = lightVertex.position - eyeVertex.position;
    float distSq = dot(connectionVec, connectionVec);
    float dist = sqrtf(distSq);
    
    if (dist < 1e-6f) {
        return result;
    }
    
    float3 connectionDir = connectionVec / dist;
    
    // Check geometric term (both vertices must face the connection)
    float cosEye = dot(eyeVertex.normal, connectionDir);
    float cosLight = dot(lightVertex.normal, -connectionDir);
    
    if (cosEye <= 0.0f || cosLight <= 0.0f) {
        return result;
    }
    
    // TODO: Visibility test (shadow ray)
    // For now, assume visible
    
    // Calculate geometric term: G = cos(eye) * cos(light) / dist^2
    float G = (cosEye * cosLight) / distSq;
    
    // Calculate contribution
    result.contribution = make_float3(
        eyeVertex.throughput.x * lightVertex.throughput.x * G,
        eyeVertex.throughput.y * lightVertex.throughput.y * G,
        eyeVertex.throughput.z * lightVertex.throughput.z * G
    );
    
    // Calculate MIS weight
    result.misWeight = calculateBDPTMISWeight(eyePath, lightPath, s, t);
    
    result.valid = true;
    return result;
}

} // namespace internal
} // namespace wr
