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

void createSphere(std::vector<float>& vertices,
                  std::vector<float>& normals,
                  std::vector<uint32_t>& indices,
                  float cx, float cy, float cz,
                  float radius,
                  int segments,
                  int rings) {
    vertices.clear();
    normals.clear();
    indices.clear();
    
    // Generate vertices and normals using standard UV sphere parametrization
    for (int lat = 0; lat <= rings; ++lat) {
        float phi = PI * float(lat) / float(rings);
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        
        for (int lon = 0; lon <= segments; ++lon) {
            float theta = TWO_PI * float(lon) / float(segments);
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);
            
            // Vertex position
            float x = cx + radius * sinPhi * cosTheta;
            float y = cy + radius * cosPhi;
            float z = cz + radius * sinPhi * sinTheta;
            
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            
            // Normal (unit vector pointing outward from sphere center)
            float nx = sinPhi * cosTheta;
            float ny = cosPhi;
            float nz = sinPhi * sinTheta;
            
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);
        }
    }
    
    // Generate indices (same as before)
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

void createBox(std::vector<float>& vertices,
               std::vector<uint32_t>& indices,
               float cx, float cy, float cz,
               float size) {
    vertices.clear();
    indices.clear();
    
    float halfSize = size * 0.5f;
    
    // 8 vertices of the cube
    // 0: -x, -y, -z (left, bottom, back)
    // 1: +x, -y, -z (right, bottom, back)
    // 2: +x, +y, -z (right, top, back)
    // 3: -x, +y, -z (left, top, back)
    // 4: -x, -y, +z (left, bottom, front)
    // 5: +x, -y, +z (right, bottom, front)
    // 6: +x, +y, +z (right, top, front)
    // 7: -x, +y, +z (left, top, front)
    
    float verts[] = {
        // Front face (z = +halfSize, facing +z)
        cx - halfSize, cy - halfSize, cz + halfSize,  // 4
        cx + halfSize, cy - halfSize, cz + halfSize,  // 5
        cx + halfSize, cy + halfSize, cz + halfSize,  // 6
        cx - halfSize, cy + halfSize, cz + halfSize,  // 7
        
        // Back face (z = -halfSize, facing -z)
        cx + halfSize, cy - halfSize, cz - halfSize,  // 1
        cx - halfSize, cy - halfSize, cz - halfSize,  // 0
        cx - halfSize, cy + halfSize, cz - halfSize,  // 3
        cx + halfSize, cy + halfSize, cz - halfSize,  // 2
        
        // Left face (x = -halfSize, facing -x)
        cx - halfSize, cy - halfSize, cz - halfSize,  // 0
        cx - halfSize, cy - halfSize, cz + halfSize,  // 4
        cx - halfSize, cy + halfSize, cz + halfSize,  // 7
        cx - halfSize, cy + halfSize, cz - halfSize,  // 3
        
        // Right face (x = +halfSize, facing +x)
        cx + halfSize, cy - halfSize, cz + halfSize,  // 5
        cx + halfSize, cy - halfSize, cz - halfSize,  // 1
        cx + halfSize, cy + halfSize, cz - halfSize,  // 2
        cx + halfSize, cy + halfSize, cz + halfSize,  // 6
        
        // Top face (y = +halfSize, facing +y)
        cx - halfSize, cy + halfSize, cz + halfSize,  // 7
        cx + halfSize, cy + halfSize, cz + halfSize,  // 6
        cx + halfSize, cy + halfSize, cz - halfSize,  // 2
        cx - halfSize, cy + halfSize, cz - halfSize,  // 3
        
        // Bottom face (y = -halfSize, facing -y)
        cx - halfSize, cy - halfSize, cz - halfSize,  // 0
        cx + halfSize, cy - halfSize, cz - halfSize,  // 1
        cx + halfSize, cy - halfSize, cz + halfSize,  // 5
        cx - halfSize, cy - halfSize, cz + halfSize,  // 4
    };
    
    vertices.assign(verts, verts + 72);
    
    // Indices for each face
    // Note: OptiX calculates normal as normalize(cross(v1 - v0, v2 - v0))
    // For counter-clockwise vertices (from outside), this points outward
    uint32_t inds[] = {
        // Front face (facing +z)
        0, 2, 1, 0, 3, 2,
        // Back face (facing -z)
        4, 6, 5, 4, 7, 6,
        // Left face (facing -x)
        8, 10, 9, 8, 11, 10,
        // Right face (facing +x)
        12, 14, 13, 12, 15, 14,
        // Top face (facing +y)
        16, 18, 17, 16, 19, 18,
        // Bottom face (facing -y)
        20, 22, 21, 20, 23, 22
    };
    
    indices.assign(inds, inds + 36);
}

void createRotatedBox(std::vector<float>& vertices,
                      std::vector<uint32_t>& indices,
                      float cx, float cy, float cz,
                      float size,
                      float rotationY) {
    // First create an axis-aligned box at origin
    createBox(vertices, indices, 0.0f, 0.0f, 0.0f, size);
    
    // Apply Y-axis rotation and translation
    float cosY = std::cos(rotationY);
    float sinY = std::sin(rotationY);
    
    // Rotate each vertex around Y axis, then translate to (cx, cy, cz)
    for (size_t i = 0; i < vertices.size(); i += 3) {
        float x = vertices[i + 0];
        float y = vertices[i + 1];
        float z = vertices[i + 2];
        
        // Rotate around Y axis
        float newX = cosY * x - sinY * z;
        float newZ = sinY * x + cosY * z;
        
        // Translate to final position
        vertices[i + 0] = newX + cx;
        vertices[i + 1] = y + cy;
        vertices[i + 2] = newZ + cz;
    }
}

} // namespace test_helpers
