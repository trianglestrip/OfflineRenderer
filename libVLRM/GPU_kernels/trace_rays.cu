#include "../shared/wavefront_types.h"

namespace vlrm {
    RT_PIPELINE_LAUNCH_PARAMETERS WavefrontParams params;

    CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTracingClosestHit)() {
        PTReadOnlyPayload* payload = reinterpret_cast<PTReadOnlyPayload*>(optixGetPayload_0());
        
        float3 hitPoint = asOptiXType(asVector3D(optixGetWorldRayOrigin()) + optixGetRayTmax() * asVector3D(optixGetWorldRayDirection()));
        
        float3 geometricNormal;
        {
            uint32_t primIdx = optixGetPrimitiveIndex();
            uint3 indices = params.indices[primIdx];
            float3 v0 = params.vertices[indices.x];
            float3 v1 = params.vertices[indices.y];
            float3 v2 = params.vertices[indices.z];
            Vector3D e1 = asVector3D(v1) - asVector3D(v0);
            Vector3D e2 = asVector3D(v2) - asVector3D(v0);
            geometricNormal = asOptiXType(normalize(cross(e1, e2)));
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
