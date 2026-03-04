#include <wr/wr.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "render_config.h"

using namespace wr;

// helper functions (savePNG, createSphere) copied/adapted from original test
void savePNG(const char* filename, const Vec3* image, uint32_t width, uint32_t height) {
    std::vector<uint8_t> pixels(width * height * 3);
    
    for (uint32_t i = 0; i < width * height; ++i) {
        float gamma = 1.0f / 2.2f;
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::pow(std::clamp(image[i].x, 0.0f, 1.0f), gamma) * 255);
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::pow(std::clamp(image[i].y, 0.0f, 1.0f), gamma) * 255);
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::pow(std::clamp(image[i].z, 0.0f, 1.0f), gamma) * 255);
    }
    
    stbi_write_png(filename, width, height, 3, pixels.data(), width * 3);
    std::cout << "Saved " << filename << std::endl;
}

void createSphere(std::vector<float>& vertices, std::vector<uint32_t>& indices, 
                  float cx, float cy, float cz, float radius, int segments = 32, int rings = 24) {
    vertices.clear();
    indices.clear();
    
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

void buildCornellBoxVar(Scene* scene) {
    uint32_t whiteMat = scene->addLambertianMaterial(Vec3(0.75f, 0.75f, 0.75f));
    uint32_t redMat = scene->addLambertianMaterial(Vec3(0.75f, 0.25f, 0.25f));
    uint32_t blueMat = scene->addLambertianMaterial(Vec3(0.25f, 0.25f, 0.75f));
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(40.0f, 40.0f, 40.0f));
    uint32_t glassMat = scene->addGlassMaterial(Vec3(0.999f, 0.999f, 0.999f), 1.5f);
    
    const float L = -1.5f, R = 1.5f;
    const float B = 0.0f, T = 3.0f;
    const float N = -1.5f, F = 1.5f;
    
    {
        float verts[] = { L, B, F,  L, B, N,  R, B, N,  R, B, F };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), whiteMat);
    }
    {
        float verts[] = { L, T, N,  L, T, F,  R, T, F,  R, T, N };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), whiteMat);
    }
    {
        float verts[] = { L, B, N,  R, B, N,  R, T, N,  L, T, N };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), whiteMat);
    }
    {
        float verts[] = { L, B, F,  L, B, N,  L, T, N,  L, T, F };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), redMat);
    }
    {
        float verts[] = { R, B, N,  R, B, F,  R, T, F,  R, T, N };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), blueMat);
    }
    {
        const float lx = 0.5f;
        const float lz = 0.5f;
        const float ly = T - 0.01f;
        float verts[] = { -lx, ly, -lz,  lx, ly, -lz,  lx, ly, lz,  -lx, ly, lz };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), lightMat);
    }
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        createSphere(verts, inds, -0.7f, 0.5f, -0.7f, 0.5f, 32, 24);
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        createSphere(verts, inds, 0.7f, 0.5f, 0.7f, 0.5f, 32, 24);
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    std::cout << "[Test] Cornell Box Variation scene built" << std::endl;
}

int main() {
    std::cout << "=== libWR Cornell Box Variation Test ===" << std::endl;
    
    RenderConfig config = render_config::load("cornell_box_var");
    
    wr::Context context;
    wr::Scene* scene = context.createScene();
    buildCornellBoxVar(scene);
    scene->setEnvironmentRadiance(Vec3(0.0f,0.0f,0.0f));
    scene->finalize();
    wr::Renderer* renderer = context.createRenderer();
    wr::Camera camera;
    camera.position = {0.0f,1.0f,3.0f};
    camera.target = {0.0f,1.0f,0.0f};
    camera.up = {0.0f,1.0f,0.0f};
    camera.fovY = 45.0f * 3.14159265f / 180.0f;
    camera.aspect = static_cast<float>(config.width) / config.height;
    
    std::vector<wr::Vec3> image(config.width * config.height);
    renderer->render(scene, camera, image.data(), config.width, config.height, config.spp);
    
    std::filesystem::path outputPath = render_config::resolveGalleryPath("wr_cornell.png");
    savePNG(outputPath.string().c_str(), image.data(), config.width, config.height);
    
    return 0;
}
