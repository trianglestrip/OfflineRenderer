// Primary ray generation kernel
#include "wavefront_types.cuh"

namespace wpt {

struct CameraParams {
    float3 position;
    float3 target;
    float3 up;
    float fov;
    uint32_t width;
    uint32_t height;
};

__device__ float randf_jitter(uint32_t& seed) {
    seed = seed * 1103515245u + 12345u;
    return (float)(seed >> 16) / 65536.0f;
}

__global__ void generatePrimaryRays(
    RayState* rayPool,
    uint32_t* activeQueue,
    const CameraParams camera,
    uint32_t numPixels,
    uint32_t sampleIndex)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;

    uint32_t x = idx % camera.width;
    uint32_t y = idx / camera.width;

    // Use proper random jitter for anti-aliasing
    uint32_t jitterSeed = (idx * 1664525u + sampleIndex * 1013904223u) ^ 0x9e3779b9u;
    float jitterU = randf_jitter(jitterSeed) - 0.5f;
    float jitterV = randf_jitter(jitterSeed) - 0.5f;

    float aspect = (float)camera.width / (float)camera.height;
    float tanHalfFov = tanf(camera.fov * 0.5f);
    float u = (2.0f * (x + 0.5f + jitterU) / camera.width - 1.0f) * aspect * tanHalfFov;
    float v = (1.0f - 2.0f * (y + 0.5f + jitterV) / camera.height) * tanHalfFov;

    float3 forward = normalize(camera.target - camera.position);
    float3 right = normalize(cross(forward, camera.up));
    float3 up = cross(right, forward);
    float3 origin = camera.position;
    float3 direction = normalize(forward + u * right + v * up);

    RayState ray;
    ray.origin = origin;
    ray.direction = direction;
    ray.throughput = make_rgb(1, 1, 1);
    ray.radiance = make_rgb(0, 0, 0);
    ray.pixel_index = idx;
    ray.depth = 0;
    ray.material_id = 0;
    ray.stage = Stage_Intersect;
    // Better seed: hash pixel coordinates and sample index to break spatial correlation
    ray.seed = (idx * 1664525u + sampleIndex * 1013904223u) ^ 0x9e3779b9u;
    ray.tmin = 1e-5f;
    ray.tmax = 1e30f;

    rayPool[idx] = ray;
    activeQueue[idx] = idx;
}

extern "C" void wpt_launchGeneratePrimaryRays(
    RayState* rayPool,
    uint32_t* activeQueue,
    const void* camera,
    uint32_t numPixels,
    unsigned int gridSize,
    unsigned int blockSize,
    uint32_t sampleIndex)
{
    const CameraParams& cam = *reinterpret_cast<const CameraParams*>(camera);
    generatePrimaryRays<<<gridSize, blockSize>>>(rayPool, activeQueue, cam, numPixels, sampleIndex);
}

} // namespace wpt
