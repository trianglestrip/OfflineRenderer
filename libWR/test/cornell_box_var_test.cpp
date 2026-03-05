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
    uint32_t whiteMat = scene->addLambertianMaterial(Vec3(0.75f, 0.75f, 0.75f));
    uint32_t redMat = scene->addLambertianMaterial(Vec3(0.75f, 0.25f, 0.25f));
    uint32_t blueMat = scene->addLambertianMaterial(Vec3(0.25f, 0.25f, 0.75f));
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(40.0f, 40.0f, 40.0f));
    uint32_t glassMat = scene->addGlassMaterial(Vec3(0.999f, 0.999f, 0.999f), 1.5f);
    
    // Temporarily use Lambertian for debugging
    uint32_t roughMetalMat = scene->addLambertianMaterial(Vec3(1.0f, 0.85f, 0.3f));  // Gold color
    uint32_t smoothMetalMat = scene->addLambertianMaterial(Vec3(0.95f, 0.95f, 0.95f));  // Silver color
    
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
    // Temporarily remove spheres for debugging
    /*
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        createSphere(verts, inds, -0.7f, 0.5f, -0.7f, 0.5f, 32, 24);
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        createSphere(verts, inds, 0.7f, 0.5f, 0.7f, 0.4f, 32, 24);
        scene->addTriangleMesh(std::span(verts), std::span(inds), smoothMetalMat);  // Smooth silver sphere
    }
    */
    
    // Add rough gold box in the center
    {
        float boxSize = 0.2f;
        float boxY = 0.2f;
        float verts[] = {
            // Front face
            -boxSize, boxY, -boxSize,
            boxSize, boxY, -boxSize,
            boxSize, boxY + boxSize, -boxSize,
            -boxSize, boxY + boxSize, -boxSize,
            // Back face
            -boxSize, boxY, boxSize,
            boxSize, boxY, boxSize,
            boxSize, boxY + boxSize, boxSize,
            -boxSize, boxY + boxSize, boxSize,
            // Left face
            -boxSize, boxY, -boxSize,
            -boxSize, boxY, boxSize,
            -boxSize, boxY + boxSize, boxSize,
            -boxSize, boxY + boxSize, -boxSize,
            // Right face
            boxSize, boxY, -boxSize,
            boxSize, boxY, boxSize,
            boxSize, boxY + boxSize, boxSize,
            boxSize, boxY + boxSize, -boxSize,
            // Top face
            -boxSize, boxY + boxSize, -boxSize,
            boxSize, boxY + boxSize, -boxSize,
            boxSize, boxY + boxSize, boxSize,
            -boxSize, boxY + boxSize, boxSize,
            // Bottom face
            -boxSize, boxY, -boxSize,
            boxSize, boxY, -boxSize,
            boxSize, boxY, boxSize,
            -boxSize, boxY, boxSize
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
        scene->addTriangleMesh(std::span(verts, 72), std::span(inds, 36), roughMetalMat);  // Rough gold box
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
    camera.position = Vec3(0.0f, 1.0f, 3.0f);
    camera.target = Vec3(0.0f, 1.0f, 0.0f);
    camera.up = Vec3(0.0f, 1.0f, 0.0f);
    camera.fovY = glm::radians(45.0f);
    camera.aspect = static_cast<float>(config.width) / config.height;
    
    std::vector<wr::Vec3> image(config.width * config.height);
    
    // Test with NEE disabled to diagnose diagonal black lines
    wr::RenderParams renderParams;
    renderParams.width = config.width;
    renderParams.height = config.height;
    renderParams.spp = config.spp;
    renderParams.denoiser.enabled = config.denoiser;
    renderParams.useNEE = true;  // Re-enable NEE with fixed PDF
    
    renderer->render(scene, camera, image.data(), renderParams);
    
    std::filesystem::path outputPath = resolveGalleryPath("wr_cornell.png");
    savePNG(outputPath.string().c_str(), image.data(), config.width, config.height);
    
    return 0;
}
