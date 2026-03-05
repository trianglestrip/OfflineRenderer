#include "geometry.h"
#include <cmath>

namespace test_helpers {

void createSphere(std::vector<float>& vertices, 
                  std::vector<uint32_t>& indices, 
                  float cx, float cy, float cz, 
                  float radius, 
                  int segments, 
                  int rings) {
    vertices.clear();
    indices.clear();
    
    // Generate vertices using standard UV sphere parametrization
    // latitude (phi): 0 at top pole, PI at bottom pole
    // longitude (theta): 0 to 2*PI around equator
    
    for (int lat = 0; lat <= rings; ++lat) {
        float phi = PI * float(lat) / float(rings);
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        
        for (int lon = 0; lon <= segments; ++lon) {
            float theta = TWO_PI * float(lon) / float(segments);
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);
            
            float x = cx + radius * sinPhi * cosTheta;
            float y = cy + radius * cosPhi;
            float z = cz + radius * sinPhi * sinTheta;
            
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    
    // Generate indices
    for (int lat = 0; lat < rings; ++lat) {
        for (int lon = 0; lon < segments; ++lon) {
            uint32_t first = lat * (segments + 1) + lon;
            uint32_t second = first + segments + 1;
            
            // Skip degenerate triangles at poles
            if (lat != 0) {
                indices.push_back(first);
                indices.push_back(second);
                indices.push_back(first + 1);
            }
            
            if (lat != rings - 1) {
                indices.push_back(first + 1);
                indices.push_back(second);
                indices.push_back(second + 1);
            }
        }
    }
}

} // namespace test_helpers
