#include <cuda_runtime.h>
#include <optix.h>
#include "internal/gpu_types.h"
#include "vector_math.cuh"
#include "sampling.cuh"
#include "photon_mapping.cuh"
#include "utils/fresnel.cuh"

using namespace wr::internal;

// Launch parameters for photon tracing
struct PhotonTraceParams {
    OptixTraversableHandle traversable;
    
    // Scene data
    const float* vertices;
    const float* normals;
    const uint32_t* indices;
    const uint32_t* triangleMaterialIds;
    const float* uvs;
    
    // Light data
    const uint32_t* emissiveTriangles;
    const float* emissiveTriangleCDF;
    uint32_t numEmissiveTriangles;
    
    // Materials and textures
    const MaterialData* materials;
    uint32_t numMaterials;
    const void* textures;
    uint32_t numTextures;
    
    // Photon maps
    PhotonMapParams photonMap;
    CausticPhotonMapParams causticMap;
    
    // Settings
    uint32_t numPhotonsPerLight;
    uint32_t maxBounces;
    uint32_t seed;
};

extern "C" {
    __constant__ PhotonTraceParams params;
}

// Ray payload for photon tracing
struct PhotonRayPayload {
    float3 position;
    float3 normal;
    float2 uv;
    uint32_t materialId;
    uint32_t primIndex;
    float t;
    bool hit;
};

// OptiX programs
extern "C" __global__ void __raygen__photon_trace() {
    const uint32_t photonIdx = optixGetLaunchIndex().x;
    const uint32_t numPhotons = params.numPhotonsPerLight * params.numEmissiveTriangles;
    
    if (photonIdx >= numPhotons) return;
    
    // Initialize random seed
    uint32_t seed = params.seed + photonIdx;
    
    // Select light source using CDF
    float lightSample = randf(seed);
    uint32_t lightIdx = 0;
    if (params.numEmissiveTriangles > 1) {
        // Binary search on CDF
        uint32_t low = 0, high = params.numEmissiveTriangles - 1;
        while (low < high) {
            uint32_t mid = (low + high) >> 1;
            if (params.emissiveTriangleCDF[mid] < lightSample) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        lightIdx = low;
    }
    
    // Get triangle vertices
    uint32_t triIdx = params.emissiveTriangles[lightIdx];
    uint32_t i0 = params.indices[triIdx * 3 + 0];
    uint32_t i1 = params.indices[triIdx * 3 + 1];
    uint32_t i2 = params.indices[triIdx * 3 + 2];
    
    float3 v0 = make_float3(params.vertices[i0 * 3 + 0], params.vertices[i0 * 3 + 1], params.vertices[i0 * 3 + 2]);
    float3 v1 = make_float3(params.vertices[i1 * 3 + 0], params.vertices[i1 * 3 + 1], params.vertices[i1 * 3 + 2]);
    float3 v2 = make_float3(params.vertices[i2 * 3 + 0], params.vertices[i2 * 3 + 1], params.vertices[i2 * 3 + 2]);
    
    // Sample point on triangle
    float2 uv = sampleTriangle(rand2f(seed));
    float3 origin = v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y;
    
    // Get material
    uint32_t matId = params.triangleMaterialIds[triIdx];
    const MaterialData& mat = params.materials[matId];
    
    // Calculate triangle normal
    float3 normal = normalize(cross(v1 - v0, v2 - v0));
    
    // Sample direction from emission (cosine weighted for diffuse)
    float3 localDir = cosineSampleHemisphere(seed);
    float3 tangent, bitangent;
    createONB(normal, tangent, bitangent);
    float3 direction = toWorld(localDir, normal, tangent, bitangent);
    
    // Initial photon power
    float3 power = mat.emission / (float)numPhotons;
    
    // Trace photon
    uint32_t bounces = 0;
    bool isCausticPath = false;
    
    while (bounces < params.maxBounces) {
        PhotonRayPayload payload = {};
        payload.t = 1e20f;
        payload.hit = false;
        
        uint32_t u0, u1;
        packPointer(&payload, u0, u1);
        
        optixTrace(
            params.traversable,
            origin, direction,
            0.001f, 1e20f, 0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            u0, u1
        );
        
        if (!payload.hit) break;
        
        // Get material at hit point
        const MaterialData& hitMat = params.materials[payload.materialId];
        
        // Store photon based on surface type and path history
        if (hitMat.type == MaterialType::Lambertian) {
            // Store photon on diffuse surface
            uint32_t photonType = isCausticPath ? PHOTON_CAUSTIC : PHOTON_INDIRECT;
            
            if (isCausticPath) {
                // Store in caustic map
                storeCausticPhoton(params.causticMap, payload.position, -direction, power);
            } else {
                // Store in global photon map
                storePhoton(params.photonMap, payload.position, -direction, power, photonType);
            }
            
            // Absorb or reflect
            float3 albedo = hitMat.albedo;
            float maxAlbedo = fmaxf(fmaxf(albedo.x, albedo.y), albedo.z);
            
            if (randf(seed) > maxAlbedo) break; // Absorbed
            
            // Continue with Russian roulette
            power = power * albedo / maxAlbedo;
            
            // Sample new direction (cosine weighted)
            float3 hitNormal = payload.normal;
            float3 newLocalDir = cosineSampleHemisphere(seed);
            float3 hitTangent, hitBitangent;
            createONB(hitNormal, hitTangent, hitBitangent);
            direction = toWorld(newLocalDir, hitNormal, hitTangent, hitBitangent);
            origin = payload.position;
            
        } else if (hitMat.type == MaterialType::Glass || hitMat.type == MaterialType::GGXTransmission) {
            // Specular transmission - mark as caustic path
            isCausticPath = true;
            
            // Calculate refraction/reflection
            float3 normal = payload.normal;
            float cosI = dot(-direction, normal);
            float etaI = 1.0f;
            float etaT = hitMat.ior;
            
            if (cosI < 0.0f) {
                cosI = -cosI;
                normal = -normal;
                etaI = hitMat.ior;
                etaT = 1.0f;
            }
            
            float F = fresnelDielectric(cosI, etaI, etaT);
            
            if (randf(seed) < F) {
                // Reflect
                direction = direction - normal * (2.0f * dot(direction, normal));
            } else {
                // Refract
                float eta = etaI / etaT;
                float k = 1.0f - eta * eta * (1.0f - cosI * cosI);
                if (k >= 0.0f) {
                    direction = eta * direction + normal * (eta * cosI - sqrtf(k));
                } else {
                    direction = direction - normal * (2.0f * dot(direction, normal));
                }
            }
            
            origin = payload.position;
            
        } else if (hitMat.type == MaterialType::GGXReflection) {
            // Specular reflection - mark as caustic path
            isCausticPath = true;
            
            // Simple reflection for photons
            float3 normal = payload.normal;
            direction = direction - normal * (2.0f * dot(direction, normal));
            origin = payload.position;
        } else {
            // Other materials - terminate
            break;
        }
        
        bounces++;
    }
}

extern "C" __global__ void __closesthit__photon_trace() {
    PhotonRayPayload* payload = getPayload<PhotonRayPayload>();
    
    payload->hit = true;
    payload->t = optixGetRayTmax();
    
    // Get primitive index
    uint32_t primIdx = optixGetPrimitiveIndex();
    payload->primIndex = primIdx;
    
    // Get material ID
    payload->materialId = params.triangleMaterialIds[primIdx];
    
    // Get barycentric coordinates
    float2 bary = optixGetTriangleBarycentrics();
    
    // Get triangle vertices
    uint32_t i0 = params.indices[primIdx * 3 + 0];
    uint32_t i1 = params.indices[primIdx * 3 + 1];
    uint32_t i2 = params.indices[primIdx * 3 + 2];
    
    float3 v0 = make_float3(params.vertices[i0 * 3 + 0], params.vertices[i0 * 3 + 1], params.vertices[i0 * 3 + 2]);
    float3 v1 = make_float3(params.vertices[i1 * 3 + 0], params.vertices[i1 * 3 + 1], params.vertices[i1 * 3 + 2]);
    float3 v2 = make_float3(params.vertices[i2 * 3 + 0], params.vertices[i2 * 3 + 1], params.vertices[i2 * 3 + 2]);
    
    // Interpolate position
    float u = bary.x;
    float v = bary.y;
    float w = 1.0f - u - v;
    payload->position = v0 * w + v1 * u + v2 * v;
    
    // Calculate normal
    float3 normal = normalize(cross(v1 - v0, v2 - v0));
    if (dot(normal, optixGetWorldRayDirection()) > 0.0f) {
        normal = -normal;
    }
    payload->normal = normal;
    
    // UV coordinates (if available)
    if (params.uvs) {
        float2 uv0 = make_float2(params.uvs[i0 * 2 + 0], params.uvs[i0 * 2 + 1]);
        float2 uv1 = make_float2(params.uvs[i1 * 2 + 0], params.uvs[i1 * 2 + 1]);
        float2 uv2 = make_float2(params.uvs[i2 * 2 + 0], params.uvs[i2 * 2 + 1]);
        payload->uv = uv0 * w + uv1 * u + uv2 * v;
    } else {
        payload->uv = make_float2(u, v);
    }
}

extern "C" __global__ void __miss__photon_trace() {
    PhotonRayPayload* payload = getPayload<PhotonRayPayload>();
    payload->hit = false;
}
