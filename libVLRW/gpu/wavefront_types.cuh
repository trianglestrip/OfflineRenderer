#pragma once
#include <cuda_runtime.h>

namespace wpt {

inline __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __device__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
inline __device__ float3 operator*(float s, float3 a) { return make_float3(a.x * s, a.y * s, a.z * s); }
inline __device__ float3 operator/(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
inline __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline __device__ float length(float3 a) { return sqrtf(dot(a, a)); }
inline __device__ float3 normalize(float3 a) { float len = length(a); return (len > 0) ? (a / len) : make_float3(0, 0, 0); }

struct float3_rgb { float r, g, b; };
inline __device__ float3_rgb make_rgb(float r, float g, float b) { float3_rgb c; c.r = r; c.g = g; c.b = b; return c; }
inline __device__ float3_rgb operator+(float3_rgb a, float3_rgb b) { return make_rgb(a.r + b.r, a.g + b.g, a.b + b.b); }
inline __device__ float3_rgb operator*(float3_rgb a, float s) { return make_rgb(a.r * s, a.g * s, a.b * s); }
inline __device__ float3_rgb operator*(float3_rgb a, float3_rgb b) { return make_rgb(a.r * b.r, a.g * b.g, a.b * b.b); }
inline __device__ float3_rgb operator/(float3_rgb a, float s) { return make_rgb(a.r / s, a.g / s, a.b / s); }
inline __device__ void operator+=(float3_rgb& a, float3_rgb b) { a.r += b.r; a.g += b.g; a.b += b.b; }
inline __device__ float luminance(float3_rgb c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; }

enum MaterialKind : uint32_t { Material_Diffuse = 0, Material_GGX, Material_Dielectric, Material_Emissive, Material_Count };
enum RayStage : uint32_t { Stage_Intersect = 0, Stage_Shade, Stage_Shadow, Stage_Terminated };

struct RayState {
    float3 origin, direction;
    float3_rgb throughput, radiance;
    uint32_t pixel_index;
    uint32_t depth;
    uint32_t material_id;
    uint32_t stage;
    uint32_t seed;
    float tmin, tmax;
};

struct HitInfo {
    float3 position;
    float3 normal;
    uint32_t material_id;
};
struct MaterialData { float r, g, b; float er, eg, eb; };
struct PointLight { float3 position; float3_rgb intensity; };

struct AreaLight {
    float3 position;
    float3 normal;
    float3 tangent;
    float width;
    float height;
    float3_rgb emission;
    uint32_t doubleSided;  // 1 = both sides, 0 = single sided
};
struct IndexQueue { uint32_t* indices; uint32_t capacity; };
struct GlobalState {
    RayState* rayPool;
    HitInfo* hitBuffer;
    MaterialData* materials;
    PointLight* lights;
    uint32_t numLights;
    uint32_t* visibilityBuffer;  // [rayIdx * numLights + lightIdx]: 1=visible, 0=occluded
    AreaLight* areaLights;
    uint32_t numAreaLights;
    IndexQueue activeQueue, nextQueue, shadowQueue;
    float3_rgb* accumBuffer;
    uint32_t* queueCounters;
};

} // namespace wpt