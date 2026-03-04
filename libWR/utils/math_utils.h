#pragma once

#include "wr/wr.h"
#include <vector>
#include <cmath>

namespace wr {
namespace utils {

// Create sphere mesh (UV sphere)
void createSphere(std::vector<float>& vertices, std::vector<uint32_t>& indices, 
                  float cx, float cy, float cz, float radius, int segments = 32, int rings = 24);

// Normalize a 3D vector
inline void normalize(Vec3& v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len > 0) {
        v.x /= len;
        v.y /= len;
        v.z /= len;
    }
}

// Calculate camera basis vectors (forward, right, up)
inline void calculateCameraBasis(const Camera& camera, Vec3& forward, Vec3& right, Vec3& up) {
    // Calculate forward vector
    forward.x = camera.target.x - camera.position.x;
    forward.y = camera.target.y - camera.position.y;
    forward.z = camera.target.z - camera.position.z;
    float lenF = sqrtf(forward.x*forward.x + forward.y*forward.y + forward.z*forward.z);
    forward.x /= lenF; forward.y /= lenF; forward.z /= lenF;
    
    // Calculate right vector
    Vec3 upVec;
    upVec.x = camera.up.x;
    upVec.y = camera.up.y;
    upVec.z = camera.up.z;
    right.x = forward.y * upVec.z - forward.z * upVec.y;
    right.y = forward.z * upVec.x - forward.x * upVec.z;
    right.z = forward.x * upVec.y - forward.y * upVec.x;
    float lenR = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
    right.x /= lenR; right.y /= lenR; right.z /= lenR;
    
    // Calculate up vector
    up.x = right.y * forward.z - right.z * forward.y;
    up.y = right.z * forward.x - right.x * forward.z;
    up.z = right.x * forward.y - right.y * forward.x;
}

} // namespace utils
} // namespace wr