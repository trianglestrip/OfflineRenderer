#include "../shared/wavefront_types.h"

namespace vlrm {
    RT_PIPELINE_LAUNCH_PARAMETERS WavefrontParams params;

    __device__ __forceinline__ float rnd_dim(uint32_t seed, uint32_t dimension) {
        uint32_t state = seed * 747796405u + 2891336453u + dimension * 1013904223u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return float((word >> 22u) ^ word) / 4294967296.0f;
    }

    __device__ __forceinline__ float3 sampleCosineHemisphere(float u1, float u2, float& pdf) {
        float phi = 2.0f * VLRM_M_PI * u1;
        float cosTheta = sqrtf(u2);
        float sinTheta = sqrtf(1.0f - u2);
        
        pdf = cosTheta / VLRM_M_PI;
        
        return make_float3(
            cosf(phi) * sinTheta,
            sinf(phi) * sinTheta,
            cosTheta
        );
    }

    __device__ __forceinline__ float3 transformToWorld(
        const float3& localDir,
        const float3& normal,
        const float3& tangent,
        const float3& bitangent
    ) {
        Vector3D result = asVector3D(localDir).x * asVector3D(tangent) + 
                          asVector3D(localDir).y * asVector3D(bitangent) + 
                          asVector3D(localDir).z * asVector3D(normal);
        return asOptiXType(result);
    }

    __device__ __forceinline__ void makeCoordinateSystem(
        const float3& normal,
        float3& tangent,
        float3& bitangent
    ) {
        float sign = normal.z >= 0 ? 1 : -1;
        const float a = -1 / (sign + normal.z);
        const float b = normal.x * normal.y * a;
        tangent = asOptiXType(Vector3D(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x));
        bitangent = asOptiXType(Vector3D(b, sign + normal.y * normal.y * a, -normal.y));
    }

    CUDA_DEVICE_KERNEL void shadeAndGenerateRays(
        uint32_t numRays
    ) {
        uint32_t rayIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (rayIndex >= numRays)
            return;
        
        RayState& ray = params.rayQueue[rayIndex];
        
        if (ray.stage != RayStage::Shade)
            return;
        
        PTReadOnlyPayload payload;
        payload.hitPosition = make_float3(0, 0, 0);
        payload.hitNormal = make_float3(0, 1, 0);
        payload.materialId = 0;
        payload.primitiveId = 0;
        payload.t = -1.0f;
        
        uint32_t p0 = reinterpret_cast<uint32_t>(&payload);
        optixTrace(
            params.traversable,
            ray.origin,
            ray.direction,
            ray.tMin,
            ray.tMax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0
        );
        
        if (payload.t < 0.0f) {
            atomicAdd(&params.accumBuffer[ray.pixelIndex].r, ray.radiance.r);
            atomicAdd(&params.accumBuffer[ray.pixelIndex].g, ray.radiance.g);
            atomicAdd(&params.accumBuffer[ray.pixelIndex].b, ray.radiance.b);
            ray.stage = RayStage::Terminated;
            return;
        }
        
        MaterialData mat = params.materials[payload.materialId];
        
        if (mat.type == MaterialType::Emissive) {
            RGB contrib = ray.throughput * mat.emission;
            atomicAdd(&params.accumBuffer[ray.pixelIndex].r, ray.radiance.r + contrib.r);
            atomicAdd(&params.accumBuffer[ray.pixelIndex].g, ray.radiance.g + contrib.g);
            atomicAdd(&params.accumBuffer[ray.pixelIndex].b, ray.radiance.b + contrib.b);
            ray.stage = RayStage::Terminated;
            return;
        }
        
        Point3D hitPos = asPoint3D(payload.hitPosition);
        Normal3D hitNormal = asNormal3D(payload.hitNormal);
        
        float3 tangent, bitangent;
        makeCoordinateSystem(asOptiXType(hitNormal), tangent, bitangent);
        
        float bsdfPdf;
        float3 localDir = sampleCosineHemisphere(
            rnd_dim(ray.seed, ray.rngDimension++),
            rnd_dim(ray.seed, ray.rngDimension++),
            bsdfPdf
        );
        
        float3 worldDir = transformToWorld(localDir, asOptiXType(hitNormal), tangent, bitangent);
        
        RGB bsdf = mat.albedo / VLRM_M_PI;
        float cosTheta = localDir.z;
        
        ray.throughput = ray.throughput * bsdf * (cosTheta / bsdfPdf);
        
        float3 offsetDir = dot(asVector3D(worldDir), asVector3D(asOptiXType(hitNormal))) > 0.0f ? 
            asOptiXType(hitNormal) : asOptiXType(-hitNormal);
        ray.origin = asOptiXType(Vector3D(hitPos.x, hitPos.y, hitPos.z) + asVector3D(offsetDir) * 1e-4f);
        ray.direction = worldDir;
        ray.tMin = 0.0f;
        ray.tMax = 1e30f;
        ray.depth++;
        
        if (ray.depth >= params.maxDepth) {
            atomicAdd(&params.accumBuffer[ray.pixelIndex].r, ray.radiance.r);
            atomicAdd(&params.accumBuffer[ray.pixelIndex].g, ray.radiance.g);
            atomicAdd(&params.accumBuffer[ray.pixelIndex].b, ray.radiance.b);
            ray.stage = RayStage::Terminated;
        } else {
            ray.stage = RayStage::Trace;
        }
    }
}
