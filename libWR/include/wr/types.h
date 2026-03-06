#pragma once

#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"

namespace wr {

// Use GLM types as public API types
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;

// Camera configuration
struct Camera {
    Vec3 position{0.0f, 0.0f, 0.0f};
    Vec3 target{0.0f, 0.0f, -1.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovY{glm::radians(45.0f)};
    float aspect{1.0f};
    
    // Depth of Field (Thin Lens Model)
    float focalDistance{0.0f};  // 0 = infinite focus (no DOF)
    float lensRadius{0.0f};     // 0 = pinhole camera (no DOF)
    
    // Helper: Get view matrix
    Mat4 getViewMatrix() const {
        return glm::lookAt(position, target, up);
    }
    
    // Helper: Get projection matrix
    Mat4 getProjectionMatrix(float nearPlane = 0.1f, float farPlane = 100.0f) const {
        return glm::perspective(fovY, aspect, nearPlane, farPlane);
    }
    
    // Helper: Get forward direction
    Vec3 getForward() const {
        return glm::normalize(target - position);
    }
    
    // Helper: Get right direction
    Vec3 getRight() const {
        return glm::normalize(glm::cross(getForward(), up));
    }
    
    // Helper: Get up direction (orthogonalized)
    Vec3 getUp() const {
        return glm::normalize(glm::cross(getRight(), getForward()));
    }
};

// Photon mapping configuration
struct PhotonMapConfig {
    bool enabled = false;
    uint32_t numPhotons = 100000;        // Total photons to emit
    uint32_t maxBounces = 8;             // Max photon bounces
    float searchRadius = 0.05f;          // Search radius for k-NN
    uint32_t maxPhotonsPerQuery = 100;   // Max photons to gather per query
    uint32_t causticPhotons = 50000;     // Max caustic photons
    float causticSearchRadius = 0.02f;   // Smaller radius for sharper caustics
};

} // namespace wr
