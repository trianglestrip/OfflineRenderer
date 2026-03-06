#include <optix.h>
#include <cuda_runtime.h>
#include "internal/gpu_types.h"
#include "vector_math.cuh"
#include "sampling.cuh"
#include "materials.cuh"

using namespace wr::internal;

static constexpr float kPi = 3.14159265f;

extern "C" {
    __constant__ const LaunchParams* params;
}

// Atomic add for float3
__device__ inline void atomicAddFloat3(float3* address, float3 value) {
    atomicAdd(&address->x, value.x);
    atomicAdd(&address->y, value.y);
    atomicAdd(&address->z, value.z);
}

__device__ inline float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 8) / 16777216.0f;
}

extern "C" __global__ void __raygen__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    if (idx >= params->numActive) return;
    
    const uint32_t rayIndex = params->activeIndices[idx];
    RayState& ray = params->rayPool[rayIndex];
    
    if (ray.stage == RayStage::Terminated) return;
    
    // Shadow ray visibility test (NEE)
    if (ray.stage == RayStage::Shadow) {
        uint32_t hitFlag = 0;
        optixTrace(
            params->traversable,
            ray.origin,
            ray.direction,
            ray.tMin,
            ray.tMax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            hitFlag
        );
        if (hitFlag == 0) {
            // Miss: light visible, accumulate contribution
            ray.radiance = ray.radiance + ray.shadowContribution;
        }
        // Hit: light occluded, discard contribution
        ray.direction = ray.savedDirection;  // Restore for BSDF sampling
        ray.stage = RayStage::Shade;
        return;
    }
    
    if (ray.stage == RayStage::Trace) {
        if (ray.depth == 0) {
            uint32_t px = rayIndex % params->width;
            uint32_t py = rayIndex / params->width;
            
            uint32_t seed = (rayIndex * 1664525u + params->sampleIndex * 1013904223u) ^ 0x9e3779b9u;
            uint32_t jitterSeed = seed;  // Separate seed for jittering
            
            float r1 = randf(jitterSeed);
            float r2 = randf(jitterSeed);
            
            float ndcX = (2.0f * (px + r1) / params->width - 1.0f) * params->camera.aspect;
            float ndcY = 1.0f - 2.0f * (py + r2) / params->height;
            
            // Extract xyz from float4
            float3 camPos = make_float3(params->camera.position.x, params->camera.position.y, params->camera.position.z);
            float3 camForward = make_float3(params->camera.forward.x, params->camera.forward.y, params->camera.forward.z);
            float3 camRight = make_float3(params->camera.right.x, params->camera.right.y, params->camera.right.z);
            float3 camUp = make_float3(params->camera.up.x, params->camera.up.y, params->camera.up.z);
            
            float3 rayDir = camForward + 
                           camRight * ndcX * params->camera.tanHalfFovY +
                           camUp * ndcY * params->camera.tanHalfFovY;
            rayDir = normalize(rayDir);
            
            ray.origin = camPos;
            ray.direction = rayDir;
            ray.throughput = make_float3(1.0f, 1.0f, 1.0f);
            ray.radiance = make_float3(0.0f, 0.0f, 0.0f);
            ray.pixelIndex = rayIndex;
            ray.depth = 0;
            ray.stage = RayStage::Trace;
            ray.seed = seed;
            ray.tMin = 0.0001f;  // Camera ray can use smaller epsilon
            ray.tMax = 1e20f;
            ray.isFirstHit = true;
            ray.prevPdf = 1.0f;
            ray.prevWasDelta = true;  // Camera ray is delta
            ray.rngDimension = 0;  // Reset RNG dimension counter
            ray.neeDone = 0;
        }
        
        uint32_t hitFlag = 0;
        optixTrace(
            params->traversable,
            ray.origin,
            ray.direction,
            ray.tMin,
            ray.tMax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            hitFlag
        );
        
        if (hitFlag == 0) {
            float3 envRad = make_float3(params->environmentRadiance.x, params->environmentRadiance.y, params->environmentRadiance.z);
            ray.radiance = ray.radiance + ray.throughput * envRad;
            atomicAddFloat3(&params->accumBuffer[ray.pixelIndex], ray.radiance);
            ray.stage = RayStage::Terminated;
        } else {
            ray.stage = RayStage::Shade;
        }
    }
}

extern "C" __global__ void __closesthit__trace() {
    const uint32_t idx = optixGetLaunchIndex().x;
    if (idx >= params->numActive) return;
    
    const uint32_t rayIndex = params->activeIndices[idx];
    RayState& ray = params->rayPool[rayIndex];
    
    if (ray.stage == RayStage::Shadow) {
        optixSetPayload_0(1);
        return;
    }
    
    float t = optixGetRayTmax();
    float3 origin = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float3 hitPos = origin + t * direction;
    
    uint32_t primIdx = optixGetPrimitiveIndex();
    
    uint32_t i0 = params->geometry.indices[primIdx * 3 + 0];
    uint32_t i1 = params->geometry.indices[primIdx * 3 + 1];
    uint32_t i2 = params->geometry.indices[primIdx * 3 + 2];
    
    float3 v0 = make_float3(
        params->geometry.vertices[i0 * 3 + 0],
        params->geometry.vertices[i0 * 3 + 1],
        params->geometry.vertices[i0 * 3 + 2]
    );
    float3 v1 = make_float3(
        params->geometry.vertices[i1 * 3 + 0],
        params->geometry.vertices[i1 * 3 + 1],
        params->geometry.vertices[i1 * 3 + 2]
    );
    float3 v2 = make_float3(
        params->geometry.vertices[i2 * 3 + 0],
        params->geometry.vertices[i2 * 3 + 1],
        params->geometry.vertices[i2 * 3 + 2]
    );
    
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 geometricNormal = normalize(cross(e1, e2));
    // Flip normal to face the ray (if ray and normal are in the same direction, flip)
    float3 normal = dot(geometricNormal, direction) > 0.0f ? -geometricNormal : geometricNormal;
    
    HitInfo& hit = params->hitBuffer[rayIndex];
    hit.position = hitPos;
    hit.normal = normal;
    hit.materialId = params->geometry.triangleMaterialIds[primIdx];
    hit.primIndex = primIdx;
    
    // Interpolate UV from barycentrics
    float3 v0p = hitPos - v0;
    float d00 = dot(e1, e1);
    float d01 = dot(e1, e2);
    float d11 = dot(e2, e2);
    float d20 = dot(v0p, e1);
    float d21 = dot(v0p, e2);
    float denom = d00 * d11 - d01 * d01;
    float v_bary = (d11 * d20 - d01 * d21) / fmaxf(denom, 1e-8f);
    float w_bary = (d00 * d21 - d01 * d20) / fmaxf(denom, 1e-8f);
    float u_bary = 1.0f - v_bary - w_bary;
    
    if (params->geometry.uvs) {
        float uv0_u = params->geometry.uvs[i0 * 2 + 0], uv0_v = params->geometry.uvs[i0 * 2 + 1];
        float uv1_u = params->geometry.uvs[i1 * 2 + 0], uv1_v = params->geometry.uvs[i1 * 2 + 1];
        float uv2_u = params->geometry.uvs[i2 * 2 + 0], uv2_v = params->geometry.uvs[i2 * 2 + 1];
        hit.uv = make_float2(u_bary * uv0_u + v_bary * uv1_u + w_bary * uv2_u,
                             u_bary * uv0_v + v_bary * uv1_v + w_bary * uv2_v);
    } else {
        hit.uv = make_float2(0.0f, 0.0f);
    }
    
    // Record first hit albedo and normal for denoiser guide layers
    if (ray.isFirstHit) {
        uint32_t matId = hit.materialId;
        if (matId < params->numMaterials) {
            const MaterialData& mat = params->materials[matId];
            
            // Get albedo for denoiser guide (texture or constant)
            float3 albedo = getMaterialAlbedo(mat, hit.uv,
                reinterpret_cast<const cudaTextureObject_t*>(params->textures), params->numTextures);
            
            // Accumulate albedo and normal (will be averaged over samples)
            if (params->sampleIndex == 0) {
                params->albedoBuffer[ray.pixelIndex] = albedo;
                params->normalBuffer[ray.pixelIndex] = normal;
            } else {
                atomicAddFloat3(&params->albedoBuffer[ray.pixelIndex], albedo);
                atomicAddFloat3(&params->normalBuffer[ray.pixelIndex], normal);
            }
        }
        ray.isFirstHit = false;
    }
    
    // Handle emissive hit with MIS
    if (hit.materialId < params->numMaterials) {
        const MaterialData& mat = params->materials[hit.materialId];
        
        if (mat.type == MaterialType::Emissive) {
            float misWeight = 1.0f;
            
            // Apply MIS if not from camera or delta surface
            if (params->useNEE && ray.depth > 0 && !ray.prevWasDelta && params->numEmissiveTriangles > 0) {
                // Calculate light sampling PDF
                float3 e1 = v1 - v0;
                float3 e2 = v2 - v0;
                float area = 0.5f * length(cross(e1, e2));
                
                float distSq = t * t;
                float cosLight = fabsf(dot(normal, direction));
                
                // Correct PDF calculation: pdfArea * distSq / cosLight
                float pdfArea = 1.0f / (params->numEmissiveTriangles * area);
                float lightPdf = pdfArea * distSq / fmaxf(cosLight, 1e-8f);
                
                // MIS weight: power heuristic
                misWeight = powerHeuristic(ray.prevPdf, lightPdf);
            }
            
            float3 emission = make_float3(mat.emission.x, mat.emission.y, mat.emission.z);
            ray.radiance = ray.radiance + ray.throughput * emission * misWeight;
        }
    }
    
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__trace() {
    optixSetPayload_0(0);
}
