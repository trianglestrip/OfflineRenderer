#include <optixw/optixw.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace optixw;

// Save image as PPM
void savePPM(const char* filename, const RGB* image, uint32_t width, uint32_t height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    
    for (uint32_t i = 0; i < width * height; ++i) {
        float gamma = 1.0f / 2.2f;
        uint8_t r = static_cast<uint8_t>(std::pow(std::clamp(image[i].r, 0.0f, 1.0f), gamma) * 255);
        uint8_t g = static_cast<uint8_t>(std::pow(std::clamp(image[i].g, 0.0f, 1.0f), gamma) * 255);
        uint8_t b = static_cast<uint8_t>(std::pow(std::clamp(image[i].b, 0.0f, 1.0f), gamma) * 255);
        file.write(reinterpret_cast<const char*>(&r), 1);
        file.write(reinterpret_cast<const char*>(&g), 1);
        file.write(reinterpret_cast<const char*>(&b), 1);
    }
    
    std::cout << "Saved " << filename << std::endl;
}

// Create sphere mesh (UV sphere)
void createSphere(std::vector<float>& vertices, std::vector<uint32_t>& indices, 
                  float cx, float cy, float cz, float radius, int segments = 32, int rings = 24) {
    vertices.clear();
    indices.clear();
    
    // Generate vertices
    for (int ring = 0; ring <= rings; ++ring) {
        float phi = 3.14159f * ring / rings;
        for (int seg = 0; seg <= segments; ++seg) {
            float theta = 2.0f * 3.14159f * seg / segments;
            
            float x = cx + radius * sinf(phi) * cosf(theta);
            float y = cy + radius * cosf(phi);
            float z = cz + radius * sinf(phi) * sinf(theta);
            
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    
    // Generate indices
    for (int ring = 0; ring < rings; ++ring) {
        for (int seg = 0; seg < segments; ++seg) {
            int current = ring * (segments + 1) + seg;
            int next = current + segments + 1;
            
            indices.push_back(current);
            indices.push_back(next);
            indices.push_back(current + 1);
            
            indices.push_back(current + 1);
            indices.push_back(next);
            indices.push_back(next + 1);
        }
    }
}

// Build Cornell Box Variation scene (based on libVLR CornellBox_var.jpg)
// Features: checkerboard floor, red left wall, blue right wall, two glass spheres
void buildCornellBoxVar(Scene* scene) {
    // Create materials
    uint32_t whiteMat = scene->addLambertianMaterial(RGB(0.75f, 0.75f, 0.75f));
    uint32_t redMat = scene->addLambertianMaterial(RGB(0.75f, 0.25f, 0.25f));
    uint32_t blueMat = scene->addLambertianMaterial(RGB(0.25f, 0.25f, 0.75f));
    uint32_t lightMat = scene->addEmissiveMaterial(RGB(30.0f, 30.0f, 30.0f));
    uint32_t glassMat = scene->addGlassMaterial(RGB(0.999f, 0.999f, 0.999f), 2.42f);  // Diamond IOR
    
    // Room dimensions (larger than standard Cornell Box)
    const float L = -1.5f, R = 1.5f;  // Left, Right
    const float B = 0.0f, T = 3.0f;   // Bottom, Top
    const float N = -1.5f, F = 1.5f;  // Near, Far
    
    // Floor (white - will use checkerboard texture in future)
    {
        float verts[] = { L, B, F,  L, B, N,  R, B, N,  R, B, F };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), whiteMat);
    }
    
    // Ceiling (white)
    {
        float verts[] = { L, T, N,  L, T, F,  R, T, F,  R, T, N };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), whiteMat);
    }
    
    // Back wall (white)
    {
        float verts[] = { L, B, N,  R, B, N,  R, T, N,  L, T, N };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), whiteMat);
    }
    
    // Left wall (red)
    {
        float verts[] = { L, B, F,  L, B, N,  L, T, N,  L, T, F };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), redMat);
    }
    
    // Right wall (blue)
    {
        float verts[] = { R, B, N,  R, B, F,  R, T, F,  R, T, N };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), blueMat);
    }
    
    // Ceiling light (area light)
    {
        const float lx = 0.5f;
        const float lz = 0.5f;
        const float ly = T - 0.01f;
        float verts[] = { -lx, ly, -lz,  lx, ly, -lz,  lx, ly, lz,  -lx, ly, lz };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), lightMat);
    }
    
    // Glass sphere A (left, position from libVLR: translate(-0.7, 0, -0.7) * scale(0.5) * translate(0, 1, 0))
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        // Final position: (-0.7, 0.5, -0.7), radius: 0.5
        createSphere(verts, inds, -0.7f, 0.5f, -0.7f, 0.5f, 32, 24);
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    
    // Glass sphere B (right, position from libVLR: translate(0.7, 0, 0.7) * scale(0.5) * translate(0, 1, 0))
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        // Final position: (0.7, 0.5, 0.7), radius: 0.5
        createSphere(verts, inds, 0.7f, 0.5f, 0.7f, 0.5f, 32, 24);
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    
    std::cout << "[Test] Cornell Box Variation scene built" << std::endl;
}

int main() {
    try {
        std::cout << "=== libOptixW Cornell Box Variation Test ===" << std::endl;
        
        // Create context
        Context context;
        
        // Create scene
        Scene* scene = context.createScene();
        buildCornellBoxVar(scene);
        scene->finalize();
        
        // Create renderer
        Renderer* renderer = context.createRenderer();
        
        // Setup camera (from libVLR: position(0, 1.5, 6), looking at origin)
        Camera camera;
        camera.position = RGB(0.0f, 1.5f, 6.0f);
        camera.target = RGB(0.0f, 1.5f, 0.0f);
        camera.up = RGB(0.0f, 1.0f, 0.0f);
        camera.fovY = 40.0f * 3.14159f / 180.0f;
        camera.aspect = 1.0f;
        
        // Render
        const uint32_t width = 1024;
        const uint32_t height = 1024;
        const uint32_t spp = 16;  // More samples for better quality
        
        std::cout << "[Test] Rendering " << width << "x" << height << " @ " << spp << " spp..." << std::endl;
        
        std::vector<RGB> outputBuffer(width * height);
        renderer->render(scene, camera, outputBuffer.data(), width, height, spp);
        
        // Save to gallery folder
        savePPM("../../../gallery/cornell_box_var.ppm", outputBuffer.data(), width, height);
        
        std::cout << "[Test] Test completed successfully" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
