#pragma once

#include "wr/types.h"
#include <cmath>

namespace wr {
namespace utils {

// Create sphere mesh (UV sphere)
void createSphere(std::vector<float>& vertices, std::vector<uint32_t>& indices, 
                  float cx, float cy, float cz, float radius, int segments = 32, int rings = 24);

// Normalize a 3D vector
template<typename T>
inline void normalize(Vec3T<T>& v) {
    T len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len > 0) {
        v.x /= len;
        v.y /= len;
        v.z /= len;
    }
}

// Calculate camera basis vectors (forward, right, up)
inline void calculateCameraBasis(const Camera& camera, float3& forward, float3& right, float3& up) {
    // Calculate forward vector
    forward = make_float3(
        camera.target.x - camera.position.x,
        camera.target.y - camera.position.y,
        camera.target.z - camera.position.z
    );
    float lenF = sqrtf(forward.x*forward.x + forward.y*forward.y + forward.z*forward.z);
    forward.x /= lenF; forward.y /= lenF; forward.z /= lenF;
    
    // Calculate right vector
    float3 upVec = make_float3(camera.up.x, camera.up.y, camera.up.z);
    right = make_float3(
        forward.y * upVec.z - forward.z * upVec.y,
        forward.z * upVec.x - forward.x * upVec.z,
        forward.x * upVec.y - forward.y * upVec.x
    );
    float lenR = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
    right.x /= lenR; right.y /= lenR; right.z /= lenR;
    
    // Calculate up vector
    up = make_float3(
        right.y * forward.z - right.z * forward.y,
        right.z * forward.x - right.x * forward.z,
        right.x * forward.y - right.y * forward.x
    );
}

} // namespace utils
} // namespace wr