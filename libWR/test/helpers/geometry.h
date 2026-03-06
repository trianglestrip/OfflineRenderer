#pragma once

#include <wr/types.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace test_helpers {

// Geometry generation utilities for testing

// Create sphere mesh (UV sphere)
void createSphere(std::vector<float>& vertices, 
                  std::vector<uint32_t>& indices, 
                  float cx, float cy, float cz, 
                  float radius, 
                  int segments = 32, 
                  int rings = 24);

// Create sphere mesh with normals (UV sphere)
void createSphere(std::vector<float>& vertices,
                  std::vector<float>& normals,
                  std::vector<uint32_t>& indices,
                  float cx, float cy, float cz,
                  float radius,
                  int segments = 32,
                  int rings = 24);

// Math constants (for convenience, from GLM)
constexpr float PI = glm::pi<float>();
constexpr float TWO_PI = glm::two_pi<float>();
constexpr float HALF_PI = glm::half_pi<float>();
constexpr float EPSILON = glm::epsilon<float>();

} // namespace test_helpers
