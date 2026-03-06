#include "../shared/wavefront_types.h"

namespace vlrm {
    RT_PIPELINE_LAUNCH_PARAMETERS WavefrontParams params;

    CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTracingClosestHit)() {
        PTReadOnlyPayload* payload = reinterpret_cast<PTReadOnlyPayload*>(optixGetPayload_0());
        
        float3 hitPoint = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
        
        float3 geometricNormal;
        {
            uint32_t primIdx = optixGetPrimitiveIndex();
            uint3 indices = params.indices[primIdx];
            float3 v0 = params.vertices[indices.x];
            float3 v1 = params.vertices[indices.y];
            float3 v2 = params.vertices[indices.z];
            float3 e1 = v1 - v0;
            float3 e2 = v2 - v0;
            geometricNormal = normalize(cross(e1, e2));
        }
        
        float2 bc = optixGetTriangleBarycentrics();
        
        payload->hitPosition = hitPoint;
        payload->hitNormal = geometricNormal;
        payload->materialId = params.materialIndices[optixGetPrimitiveIndex()];
        payload->primitiveId = optixGetPrimitiveIndex();
        payload->t = optixGetRayTmax();
    }

    CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTracingMiss)() {
        PTReadOnlyPayload* payload = reinterpret_cast<PTReadOnlyPayload*>(optixGetPayload_0());
        payload->t = -1.0f;
    }
}
