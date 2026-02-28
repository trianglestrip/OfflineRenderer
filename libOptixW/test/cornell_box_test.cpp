#include <optixw/optixw.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace optixw;

// Save image as PNG with gamma correction
void savePNG(const char* filename, const RGB* image, uint32_t width, uint32_t height) {
    std::vector<uint8_t> pixels(width * height * 3);
    
    // Debug: check first few pixels
    float maxVal = 0.0f;
    for (uint32_t i = 0; i < std::min(10u, width * height); ++i) {
        maxVal = std::max(maxVal, std::max(image[i].r, std::max(image[i].g, image[i].b)));
    }
    std::cout << "[Debug] First 10 pixels max value: " << maxVal << std::endl;
    
    for (uint32_t i = 0; i < width * height; ++i) {
        float gamma = 1.0f / 2.2f;
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::pow(std::clamp(image[i].r, 0.0f, 1.0f), gamma) * 255);
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::pow(std::clamp(image[i].g, 0.0f, 1.0f), gamma) * 255);
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::pow(std::clamp(image[i].b, 0.0f, 1.0f), gamma) * 255);
    }
    
    stbi_write_png(filename, width, height, 3, pixels.data(), width * 3);
    std::cout << "Saved " << filename << std::endl;
}

// Create sphere mesh (UV sphere)
void createSphere(std::vector<float>& vertices, std::vector<uint32_t>& indices, 
                  float cx, float cy, float cz, float radius, int segments = 16, int rings = 12) {
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

// Build Cornell Box scene with metal cube, lambertian cube, and glass sphere
void buildCornellBox(Scene* scene) {
    // Create materials
    uint32_t whiteMat = scene->addLambertianMaterial(RGB(0.75f, 0.75f, 0.75f));
    uint32_t redMat = scene->addLambertianMaterial(RGB(0.75f, 0.25f, 0.25f));
    uint32_t blueMat = scene->addLambertianMaterial(RGB(0.25f, 0.25f, 0.75f));
    uint32_t lightMat = scene->addEmissiveMaterial(RGB(10.0f, 10.0f, 10.0f));
    uint32_t metalMat = scene->addMetalMaterial(RGB(0.9f, 0.9f, 0.9f), 0.1f);  // Shiny metal
    uint32_t glassMat = scene->addGlassMaterial(RGB(1.0f, 1.0f, 1.0f), 1.5f);  // Glass (IOR=1.5)
    
    // Room dimensions
    const float L = -1.0f, R = 1.0f;  // Left, Right
    const float B = 0.0f, T = 2.0f;   // Bottom, Top
    const float N = -1.0f, F = 1.0f;  // Near, Far
    
    // Floor (white)
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
    
    // Light (ceiling, small quad)
    {
        const float lx = 0.25f;
        const float lz = 0.25f;
        const float ly = T - 0.01f;
        float verts[] = { -lx, ly, -lz,  lx, ly, -lz,  lx, ly, lz,  -lx, ly, lz };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), lightMat);
    }
    
    // Metal cube (left, shiny)
    {
        const float s = 0.25f, h = 0.5f;
        const float cx = -0.5f, cz = 0.3f;
        float verts[] = {
            // Top
            cx-s, h, cz-s,  cx+s, h, cz-s,  cx+s, h, cz+s,  cx-s, h, cz+s,
            // Bottom
            cx-s, B, cz-s,  cx+s, B, cz-s,  cx+s, B, cz+s,  cx-s, B, cz+s,
        };
        uint32_t inds[] = {
            0, 1, 2, 0, 2, 3,  // Top
            4, 5, 1, 4, 1, 0,  // Front
            5, 6, 2, 5, 2, 1,  // Right
            6, 7, 3, 6, 3, 2,  // Back
            7, 4, 0, 7, 0, 3   // Left
        };
        scene->addTriangleMesh(std::span(verts, 24), std::span(inds, 30), metalMat);
    }
    
    // Lambertian cube (right, white)
    {
        const float s = 0.25f, h = 0.5f;
        const float cx = 0.5f, cz = 0.3f;
        float verts[] = {
            // Top
            cx-s, h, cz-s,  cx+s, h, cz-s,  cx+s, h, cz+s,  cx-s, h, cz+s,
            // Bottom
            cx-s, B, cz-s,  cx+s, B, cz-s,  cx+s, B, cz+s,  cx-s, B, cz+s,
        };
        uint32_t inds[] = {
            0, 1, 2, 0, 2, 3,  // Top
            4, 5, 1, 4, 1, 0,  // Front
            5, 6, 2, 5, 2, 1,  // Right
            6, 7, 3, 6, 3, 2,  // Back
            7, 4, 0, 7, 0, 3   // Left
        };
        scene->addTriangleMesh(std::span(verts, 24), std::span(inds, 30), whiteMat);
    }
    
    // Glass sphere (center, transparent)
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        createSphere(verts, inds, 0.0f, 0.4f, -0.2f, 0.3f, 20, 15);
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    
    std::cout << "[Test] Cornell Box scene built" << std::endl;
}

int main() {
    try {
        std::cout << "=== libOptixW Cornell Box Test ===" << std::endl;
        
        // Create context
        Context context;
        
        // Create scene
        Scene* scene = context.createScene();
        buildCornellBox(scene);
        scene->finalize();
        
        // Create renderer
        Renderer* renderer = context.createRenderer();
        
        // Setup camera
        Camera camera;
        camera.position = RGB(0.0f, 1.0f, 2.5f);
        camera.target = RGB(0.0f, 1.0f, 0.0f);
        camera.up = RGB(0.0f, 1.0f, 0.0f);
        camera.fovY = 40.0f * 3.14159f / 180.0f;
        camera.aspect = 1.0f;
        
        // Render
        const uint32_t width = 512;
        const uint32_t height = 512;
        const uint32_t spp = 4;
        
        std::vector<RGB> image(width * height);
        
        std::cout << "[Test] Rendering " << width << "x" << height 
                  << " @ " << spp << " spp..." << std::endl;
        
        renderer->render(scene, camera, image.data(), width, height, spp, false);
        
        // Save image to gallery
        savePNG("../../../gallery/cornell_box.png", image.data(), width, height);
        
        std::cout << "[Test] Test completed successfully" << std::endl;
        
        delete renderer;
        delete scene;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
