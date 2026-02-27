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

__device__ void computeCameraRay(
    const CameraParams& cam,
    uint32_t x, uint32_t y,
    float3* origin,
    float3* direction)
{
    *origin = cam.position;
    
    float3 forward = normalize(cam.target - cam.position);
    float3 right = normalize(cross(forward, cam.up));
    float3 up = cross(right, forward);
    
    float aspect = (float)cam.width / (float)cam.height;
    float tanHalfFov = tanf(cam.fov * 0.5f);
    
    float u = (2.0f * (x + 0.5f) / cam.width - 1.0f) * aspect * tanHalfFov;
    float v = (1.0f - 2.0f * (y + 0.5f) / cam.height) * tanHalfFov;
    
    *direction = normalize(forward + u * right + v * up);
}

__global__ void generatePrimaryRays(
    RayState* rayPool,
    uint32_t* activeQueue,
    const CameraParams camera,
    uint32_t numPixels)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;
    
    uint32_t x = idx % camera.width;
    uint32_t y = idx / camera.width;
    
    RayState ray;
    computeCameraRay(camera, x, y, &ray.origin, &ray.direction);
    
    ray.throughput = make_rgb(1, 1, 1);
    ray.radiance = make_rgb(0, 0, 0);
    ray.pixel_index = idx;
    ray.depth = 0;
    ray.material_id = 0;
    ray.stage = Stage_Intersect;
    ray.seed = idx * 0x9e3779b9u + 1;
    ray.tmin = 1e-5f;
    ray.tmax = 1e30f;
    
    rayPool[idx] = ray;
    activeQueue[idx] = idx;
}

} // namespace wpt
