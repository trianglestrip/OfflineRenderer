#include <wr/wr.h>
#include "helpers/geometry.h"
#include "helpers/image.h"
#include "helpers/file.h"
#include "helpers/config.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

using namespace wr;
using namespace test_helpers;

void buildCornellBoxVar(Scene* scene) {
    uint32_t checkerTex = Texture2D::create(scene, resolveResourcePath("checkerboard_line.png").string());
    uint32_t floorMat = scene->addLambertianMaterial(Vec3(0.75f, 0.75f, 0.75f), checkerTex);
    
    uint32_t whiteMat = scene->addLambertianMaterial(Vec3(0.75f, 0.75f, 0.75f));
    uint32_t redMat = scene->addLambertianMaterial(Vec3(0.63f, 0.065f, 0.05f));  // Saturated red
    uint32_t blueMat = scene->addLambertianMaterial(Vec3(0.14f, 0.16f, 0.55f));  // Saturated blue
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(15.0f, 15.0f, 15.0f));
    uint32_t glassMat = scene->addGlassMaterial(Vec3(0.999f, 0.999f, 0.999f), 1.5f);
    
    // GGX materials for metal box
    uint32_t goldMat = scene->addGGXReflectionMaterial(
        Vec3(1.0f, 0.782f, 0.344f),  // Standard gold albedo (more accurate)
        0.2f,                         // Roughness (increased for better NEE coverage)
        1.0f                          // Metallic
    );
    
    const float L = -1.5f, R = 1.5f;
    const float B = 0.0f, T = 3.0f;
    const float N = -1.5f, F = 1.5f;
    
    {
        float verts[] = { L, B, F,  L, B, N,  R, B, N,  R, B, F };
        float uvs[] = { 0.0f, 4.0f,  0.0f, 0.0f,  4.0f, 0.0f,  4.0f, 4.0f };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), std::span(uvs, 8), floorMat);
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
    // Glass sphere on the right
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        createSphere(verts, inds, 0.6f, 0.5f, 0.0f, 0.5f, 64, 48);  // Higher resolution for glass
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    
    // Gold metal box on the left
    {
        float boxSize = 0.6f;
        float boxY = 0.3f;   // Raise box so it's visible (center at y=0.6)
        float boxX = -0.5f;  // Position on the left
        float boxZ = -0.3f;  // Slightly forward
        float verts[] = {
            // Front face
            boxX - boxSize*0.5f, boxY, boxZ - boxSize*0.5f,
            boxX + boxSize*0.5f, boxY, boxZ - boxSize*0.5f,
            boxX + boxSize*0.5f, boxY + boxSize, boxZ - boxSize*0.5f,
            boxX - boxSize*0.5f, boxY + boxSize, boxZ - boxSize*0.5f,
            // Back face
            boxX - boxSize*0.5f, boxY, boxZ + boxSize*0.5f,
            boxX + boxSize*0.5f, boxY, boxZ + boxSize*0.5f,
            boxX + boxSize*0.5f, boxY + boxSize, boxZ + boxSize*0.5f,
            boxX - boxSize*0.5f, boxY + boxSize, boxZ + boxSize*0.5f,
            // Left face
            boxX - boxSize*0.5f, boxY, boxZ - boxSize*0.5f,
            boxX - boxSize*0.5f, boxY, boxZ + boxSize*0.5f,
            boxX - boxSize*0.5f, boxY + boxSize, boxZ + boxSize*0.5f,
            boxX - boxSize*0.5f, boxY + boxSize, boxZ - boxSize*0.5f,
            // Right face
            boxX + boxSize*0.5f, boxY, boxZ - boxSize*0.5f,
            boxX + boxSize*0.5f, boxY, boxZ + boxSize*0.5f,
            boxX + boxSize*0.5f, boxY + boxSize, boxZ + boxSize*0.5f,
            boxX + boxSize*0.5f, boxY + boxSize, boxZ - boxSize*0.5f,
            // Top face
            boxX - boxSize*0.5f, boxY + boxSize, boxZ - boxSize*0.5f,
            boxX + boxSize*0.5f, boxY + boxSize, boxZ - boxSize*0.5f,
            boxX + boxSize*0.5f, boxY + boxSize, boxZ + boxSize*0.5f,
            boxX - boxSize*0.5f, boxY + boxSize, boxZ + boxSize*0.5f,
            // Bottom face
            boxX - boxSize*0.5f, boxY, boxZ - boxSize*0.5f,
            boxX + boxSize*0.5f, boxY, boxZ - boxSize*0.5f,
            boxX + boxSize*0.5f, boxY, boxZ + boxSize*0.5f,
            boxX - boxSize*0.5f, boxY, boxZ + boxSize*0.5f
        };
        uint32_t inds[] = {
            // Front face
            0, 1, 2, 0, 2, 3,
            // Back face
            4, 5, 6, 4, 6, 7,
            // Left face
            8, 9, 10, 8, 10, 11,
            // Right face
            12, 13, 14, 12, 14, 15,
            // Top face
            16, 17, 18, 16, 18, 19,
            // Bottom face
            20, 21, 22, 20, 22, 23
        };
        scene->addTriangleMesh(std::span(verts, 72), std::span(inds, 36), goldMat);  // Gold metal box
    }
    
    std::cout << "[Test] Cornell Box Variation scene built" << std::endl;
}

int main() {
    std::cout << "=== libWR Cornell Box Variation Test ===" << std::endl;
    
    RenderConfig config = config::load("cornell_box_var");
    
    wr::Context context;
    wr::Scene* scene = context.createScene();
    buildCornellBoxVar(scene);
    scene->setEnvironmentRadiance(Vec3(0.0f,0.0f,0.0f));
    scene->finalize();
    wr::Renderer* renderer = context.createRenderer();
    wr::Camera camera;
    camera.position = Vec3(0.0f, 1.5f, 4.5f);  // Higher and further back
    camera.target = Vec3(0.0f, 1.0f, 0.0f);    // Look at scene center (lower than before)
    camera.up = Vec3(0.0f, 1.0f, 0.0f);
    camera.fovY = glm::radians(45.0f);         // Wider FOV to see more of the ceiling
    camera.aspect = static_cast<float>(config.width) / config.height;
    
    std::vector<wr::Vec3> image(config.width * config.height);
    
    // Test with NEE disabled to diagnose diagonal black lines
    wr::RenderParams renderParams;
    renderParams.width = config.width;
    renderParams.height = config.height;
    renderParams.spp = config.spp;
    renderParams.denoiser.enabled = config.denoiser;
    renderParams.useNEE = true;
    renderParams.maxBounces = 12;  // Increased for better glass/metal quality
    renderParams.russianRouletteDepth = 5.0f;
    
    renderer->render(scene, camera, image.data(), renderParams);
    
    std::filesystem::path outputPath = resolveGalleryPath("wr_cornell.png");
    savePNG(outputPath.string().c_str(), image.data(), config.width, config.height);
    
    return 0;
}
