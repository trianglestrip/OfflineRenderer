#include <optixw/optixw.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "render_config.h"

using namespace optixw;

// ACES Filmic Tone Mapping (industry standard for HDR to LDR conversion)
inline float acesToneMap(float x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

// Save image as PNG with tone mapping and gamma correction
void savePNG(const char* filename, const Vec3* image, uint32_t width, uint32_t height) {
    std::vector<uint8_t> pixels(width * height * 3);
    
    // Calculate luminance for each pixel and find a good exposure
    std::vector<float> luminances(width * height);
    for (uint32_t i = 0; i < width * height; ++i) {
        luminances[i] = 0.2126f * image[i].x + 0.7152f * image[i].y + 0.0722f * image[i].z;
    }
    
    // Sort to find 95th percentile (ignore extreme highlights)
    std::sort(luminances.begin(), luminances.end());
    float percentile95 = luminances[static_cast<size_t>(width * height * 0.95)];
    
    std::cout << "[Debug] 95th percentile luminance: " << percentile95 << std::endl;
    
    // Exposure based on 95th percentile - target it to map to ~1.0 before tone mapping
    float exposure = 1.0f;
    if (percentile95 > 0.001f) {
        exposure = 1.0f / percentile95;
    }
    std::cout << "[Debug] Exposure: " << exposure << std::endl;
    
    for (uint32_t i = 0; i < width * height; ++i) {
        // Apply exposure
        float r = image[i].x * exposure;
        float g = image[i].y * exposure;
        float b = image[i].z * exposure;
        
        // Apply ACES tone mapping
        r = acesToneMap(r);
        g = acesToneMap(g);
        b = acesToneMap(b);
        
        // Apply gamma correction
        float gamma = 1.0f / 2.2f;
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::pow(r, gamma) * 255);
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::pow(g, gamma) * 255);
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::pow(b, gamma) * 255);
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
    uint32_t whiteMat = scene->addLambertianMaterial(Vec3(0.75f, 0.75f, 0.75f));
    uint32_t redMat = scene->addLambertianMaterial(Vec3(0.75f, 0.25f, 0.25f));
    uint32_t blueMat = scene->addLambertianMaterial(Vec3(0.25f, 0.25f, 0.75f));
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(10.0f, 10.0f, 10.0f));
    (void)scene->addGlassMaterial(Vec3(1.0f, 1.0f, 1.0f), 1.5f);  // 5 materials total, match cornell_box_var
    
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

    std::cout << "[Test] Cornell Box scene built (box + light only)" << std::endl;
}

int main() {
    try {
        std::cout << "=== libOptixW Cornell Box Test with Taskflow Integration ===" << std::endl;

        Context context;
        Scene* scene = context.createScene();
        buildCornellBox(scene);
        scene->setEnvironmentRadiance(Vec3(0.1f, 0.1f, 0.1f));
        scene->finalize();

        Renderer* renderer = context.createRenderer();

        Camera camera;
        camera.position = Vec3(0.0f, 1.0f, 2.5f);
        camera.target = Vec3(0.0f, 1.0f, 0.0f);
        camera.up = Vec3(0.0f, 1.0f, 0.0f);
        camera.fovY = 40.0f * 3.14159f / 180.0f;
        camera.aspect = 1.0f;

        const ::RenderConfig cfg = render_config::load("cornell_box");
        const uint32_t width = cfg.width;
        const uint32_t height = cfg.height;
        const uint32_t spp = cfg.spp;
        
        std::vector<Vec3> image(width * height);
        
        std::cout << "[Test] Rendering " << width << "x" << height 
                  << " @ " << spp << " spp..." << std::endl;

        renderer->render(scene, camera, image.data(), width, height, spp, cfg.denoiser, cfg.denoiserBlend, cfg.enableTiling, cfg.tileWidth, cfg.tileHeight);
        
        const std::filesystem::path outputPath = render_config::resolveGalleryPath("cornell_box.png");
        savePNG(outputPath.string().c_str(), image.data(), width, height);
        
        std::cout << "[Test] Test completed successfully" << std::endl;
        
        delete renderer;
        delete scene;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}