#include "../shared/wavefront_types.h"

namespace vlrm {
    RT_PIPELINE_LAUNCH_PARAMETERS WavefrontParams params;

    CUDA_DEVICE_KERNEL void RT_CH_NAME(shadowClosestHit)() {
        ShadowPayload* payload = reinterpret_cast<ShadowPayload*>(optixGetPayload_0());
        payload->visibility = 0.0f;
    }

    CUDA_DEVICE_KERNEL void RT_MS_NAME(shadowMiss)() {
        ShadowPayload* payload = reinterpret_cast<ShadowPayload*>(optixGetPayload_0());
        payload->visibility = 1.0f;
    }
}
