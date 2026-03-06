#pragma once
#include "internal/gpu_types.h"
#include "bdpt_types.cuh"
#include "sampling.cuh"
#include "vector_math.cuh"

namespace wr {
namespace internal {

// Generate a light path starting from a random emissive triangle
__device__ __forceinline__ void generateLightPath(
    LightPath& lightPath,
    const LaunchParams* p,
    uint32_t& seed,
    uint32_t& rngDim
) {
    lightPath.length = 0;
    
    if (p->numEmissiveTriangles == 0) {
        return;
    }
    
    // Sample a random emissive triangle
    float xi = rnd_dim(seed, rngDim++);
    uint32_t triIdx = sampleEmissiveTriangle(xi, p->emissiveTriangleCDF, p->numEmissiveTriangles);
    
    // Get triangle vertices
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
    
    // Sample a point on the triangle
    float u1 = rnd_dim(seed, rngDim++);
    float u2 = rnd_dim(seed, rngDim++);
    if (u1 + u2 > 1.0f) {
        u1 = 1.0f - u1;
        u2 = 1.0f - u2;
    }
    float u0 = 1.0f - u1 - u2;
    
    float3 lightPos = u0 * v0 + u1 * v1 + u2 * v2;
    
    // Calculate geometric normal
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 geometricNormal = normalize(cross(e1, e2));
    float area = 0.5f * length(cross(e1, e2));
    
    // Sample a direction from the light (cosine-weighted hemisphere)
    float3 tangent, bitangent;
    createCoordinateFrame(geometricNormal, tangent, bitangent);
    
    float xi1 = rnd_dim(seed, rngDim++);
    float xi2 = rnd_dim(seed, rngDim++);
    float3 localDir = sampleCosineHemisphere(xi1, xi2);
    float3 lightDir = localDir.x * tangent + localDir.y * bitangent + localDir.z * geometricNormal;
    
    // Get material emission
    uint32_t matId = p->geometry.triangleMaterialIds[triIdx];
    const MaterialData& mat = p->materials[matId];
    float3 emission = make_float3(mat.emission.x, mat.emission.y, mat.emission.z);
    
    // Initial throughput: emission * area * cos(theta) / (pdf_area * pdf_direction)
    // pdf_area = 1 / (numTriangles * area)
    // pdf_direction = cos(theta) / PI
    float cosTheta = localDir.z;
    float pdfArea = 1.0f / (p->numEmissiveTriangles * area);
    float pdfDirection = cosTheta / kPi;
    
    // First vertex
    PathVertex& v0_light = lightPath.vertices[0];
    v0_light.position = lightPos;
    v0_light.normal = geometricNormal;
    v0_light.geometricNormal = geometricNormal;
    v0_light.throughput = emission * area * p->numEmissiveTriangles * kPi;
    v0_light.direction = lightDir;
    v0_light.pdfFwd = pdfArea * pdfDirection;
    v0_light.pdfRev = 0.0f;  // Cannot sample light from scene
    v0_light.materialId = matId;
    v0_light.depth = 0;
    v0_light.isDelta = false;
    
    lightPath.length = 1;
    lightPath.emission = emission;
    
    // TODO: Trace light path through scene (requires OptiX trace from arbitrary origin)
    // This is a placeholder - full implementation requires significant refactoring
}

} // namespace internal
} // namespace wr
