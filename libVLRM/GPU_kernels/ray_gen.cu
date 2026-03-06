#include "../shared/wavefront_types.h"

namespace vlrm {
    RT_PIPELINE_LAUNCH_PARAMETERS WavefrontParams params;

    CUDA_DEVICE_KERNEL void generateCameraRays(
        uint32_t sampleIndex
    ) {
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= params.width || y >= params.height)
            return;
        
        uint32_t pixelIndex = y * params.width + x;
        
        uint32_t seed = (pixelIndex + params.frameIndex * params.width * params.height) * 1013904223u;
        seed = (seed ^ 61) ^ (seed >> 16);
        seed = seed + (seed << 3);
        seed = seed ^ (seed >> 4);
        seed = seed * 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        
        uint32_t rngDim = 0;
        auto rnd_dim = [&](uint32_t dim) -> float {
            uint32_t state = seed * 747796405u + 2891336453u + dim * 1013904223u;
            uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
            return float((word >> 22u) ^ word) / 4294967296.0f;
        };
        
        float u = (float(x) + rnd_dim(rngDim++)) / float(params.width);
        float v = (float(y) + rnd_dim(rngDim++)) / float(params.height);
        
        float aspectRatio = float(params.width) / float(params.height);
        float tanHalfFovY = tanf(params.fovY * 0.5f);
        
        float ndcX = (2.0f * u - 1.0f) * aspectRatio * tanHalfFovY;
        float ndcY = (1.0f - 2.0f * v) * tanHalfFovY;
        
        float3 rayDir = asOptiXType(normalize(
            ndcX * asVector3D(params.cameraRight) +
            ndcY * asVector3D(params.cameraUp) +
            asVector3D(params.cameraForward)
        ));
        
        uint32_t rayIndex = atomicAdd(params.rayQueueSize, 1);
        if (rayIndex >= params.maxRayQueueSize)
            return;
        
        RayState& ray = params.rayQueue[rayIndex];
        ray.origin = params.cameraPosition;
        ray.direction = rayDir;
        ray.throughput = RGB(1.0f, 1.0f, 1.0f);
        ray.radiance = RGB(0.0f, 0.0f, 0.0f);
        ray.seed = seed;
        ray.rngDimension = rngDim;
        ray.pixelIndex = pixelIndex;
        ray.depth = 0;
        ray.tMin = 0.0f;
        ray.tMax = 1e30f;
        ray.stage = RayStage::Trace;
    }
}
