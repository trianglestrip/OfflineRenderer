#include <cuda_runtime.h>
#include "internal/gpu_types.h"
#include "vector_math.cuh"
#include "sampling.cuh"
#include "ggx.cuh"
#include "materials.cuh"
#include "utils/atomic_ops.cuh"
#include "utils/fresnel.cuh"
#include "utils/ray_offset.cuh"
#include "utils/normal_map.cuh"
#include "materials/emissive.cuh"
#include "materials/lambertian.cuh"
#include "materials/glass.cuh"
#include "materials/ggx_material.cuh"
#include "materials/ggx_transmission.cuh"

using namespace wr::internal;

extern "C" {
    __constant__ const LaunchParams* params_shade;
}

// Main shade kernel - dispatches to material-specific shaders
extern "C" __global__ void shade(const LaunchParams* p) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p->numActive) return;

    const uint32_t rayIndex = p->activeIndices[idx];
    RayState& ray = p->rayPool[rayIndex];

    if (ray.stage != RayStage::Shade) return;

    const HitInfo& hit = p->hitBuffer[rayIndex];
    
    if (hit.materialId >= p->numMaterials) {
        ray.stage = RayStage::Terminated;
        return;
    }
    
    const MaterialData& mat = p->materials[hit.materialId];
    
    // Debug: Track material hits for center pixel
    uint32_t px = ray.pixelIndex % p->width;
    uint32_t py = ray.pixelIndex / p->width;
    bool isCenter = (px == p->width / 2 && py == p->height / 2);
    
    if (isCenter && ray.depth <= 5) {
        const char* matName = "Unknown";
        if (mat.type == MaterialType::Lambertian) matName = "Lambertian";
        else if (mat.type == MaterialType::Glass) matName = "Glass";
        else if (mat.type == MaterialType::GGXReflection) matName = "GGX";
        else if (mat.type == MaterialType::Emissive) matName = "Emissive";
        printf("[Shade] depth=%u, material=%s, pos=(%.2f,%.2f,%.2f)\n",
               ray.depth, matName, hit.position.x, hit.position.y, hit.position.z);
    }
    
    // Apply normal map if available (modify hit info)
    HitInfo effectiveHit = hit;
    if (mat.normalTextureId > 0) {
        effectiveHit.normal = getEffectiveNormal(mat, hit,
            reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
    }
    
    // Dispatch to material-specific shaders
    switch (mat.type) {
        case MaterialType::Emissive:
            shadeEmissive(ray, effectiveHit, mat, p);
            break;
        case MaterialType::Lambertian:
            shadeLambertian(ray, effectiveHit, mat, p);
            break;
        case MaterialType::Glass:
            shadeGlass(ray, effectiveHit, mat, p);
            break;
        case MaterialType::GGXReflection:
            shadeGGX(ray, effectiveHit, mat, p);
            break;
        case MaterialType::GGXTransmission:
            shadeGGXTransmission(ray, effectiveHit, mat, p);
            break;
        default:
            terminateRay(ray, p);
            break;
    }
}
