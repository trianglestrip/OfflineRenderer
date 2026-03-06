#include <wr/wr.h>
#include "helpers/geometry.h"
#include "helpers/image.h"
#include "helpers/file.h"
#include "helpers/config.h"
#include <iostream>
#include <vector>
#include <filesystem>

using namespace wr;
using namespace test_helpers;

void buildRoughGlassScene(Scene* scene) {
    // Materials
    Vec3 white(0.73f, 0.73f, 0.73f);
    Vec3 red(0.63f, 0.065f, 0.05f);
    Vec3 blue(0.14f, 0.16f, 0.55f);
    
    uint32_t whiteMat = scene->addLambertianMaterial(white);
    uint32_t redMat = scene->addLambertianMaterial(red);
    uint32_t blueMat = scene->addLambertianMaterial(blue);
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(15.0f, 15.0f, 15.0f));
    
    // Glass materials with different roughness
    uint32_t idealGlass = scene->addGlassMaterial(Vec3(1.0f, 1.0f, 1.0f), 1.5f);
    uint32_t roughGlass1 = scene->addGGXTransmissionMaterial(Vec3(1.0f, 1.0f, 1.0f), 0.05f, 1.5f);
    uint32_t roughGlass2 = scene->addGGXTransmissionMaterial(Vec3(1.0f, 1.0f, 1.0f), 0.15f, 1.5f);
    uint32_t roughGlass3 = scene->addGGXTransmissionMaterial(Vec3(1.0f, 1.0f, 1.0f), 0.3f, 1.5f);
    
    // Cornell Box walls
    const float L = -1.5f, R = 1.5f;
    const float B = 0.0f, T = 3.0f;
    const float N = -1.5f, F = 1.5f;
    
    // Floor
    {
        float verts[] = { L, B, F,  L, B, N,  R, B, N,  R, B, F };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), whiteMat);
    }
    // Ceiling
    {
        float verts[] = { L, T, N,  L, T, F,  R, T, F,  R, T, N };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), whiteMat);
    }
    // Back wall
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
    // Area light
    {
        const float lx = 0.5f;
        const float lz = 0.5f;
        const float ly = T - 0.01f;
        float verts[] = { -lx, ly, -lz,  lx, ly, -lz,  lx, ly, lz,  -lx, ly, lz };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), lightMat);
    }
    
    // Four glass spheres with different roughness
    // Ideal glass (left-back)
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        test_helpers::createSphere(verts, inds, -0.6f, 0.35f, -0.4f, 0.35f, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), idealGlass);
    }
    // Slightly rough (right-back)
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        test_helpers::createSphere(verts, inds, 0.6f, 0.35f, -0.4f, 0.35f, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), roughGlass1);
    }
    // Medium rough (left-front)
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        test_helpers::createSphere(verts, inds, -0.6f, 0.35f, 0.4f, 0.35f, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), roughGlass2);
    }
    // Very rough (right-front)
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        test_helpers::createSphere(verts, inds, 0.6f, 0.35f, 0.4f, 0.35f, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), roughGlass3);
    }
    
    std::cout << "[Test] Rough glass scene built (4 spheres: roughness 0.0, 0.05, 0.15, 0.3)\n";
}

int main() {
    std::cout << "=== libWR Rough Glass Test ===" << std::endl;
    
    RenderConfig config = config::load("rough_glass");
    
    wr::Context context;
    wr::Scene* scene = context.createScene();
    
    buildRoughGlassScene(scene);
    scene->setEnvironmentRadiance(Vec3(0.0f, 0.0f, 0.0f));
    scene->finalize();
    
    wr::Renderer* renderer = context.createRenderer();
    
    wr::Camera camera;
    camera.position = Vec3(0.0f, 1.5f, 4.5f);
    camera.target = Vec3(0.0f, 1.0f, 0.0f);
    camera.up = Vec3(0.0f, 1.0f, 0.0f);
    camera.fovY = glm::radians(45.0f);
    camera.aspect = static_cast<float>(config.width) / config.height;
    
    std::vector<wr::Vec3> image(config.width * config.height);
    
    wr::RenderParams renderParams;
    renderParams.width = config.width;
    renderParams.height = config.height;
    renderParams.spp = config.spp;
    renderParams.denoiser.enabled = config.denoiser;
    renderParams.useNEE = true;
    renderParams.maxBounces = 12;
    renderParams.russianRouletteDepth = 5.0f;
    
    renderer->render(scene, camera, image.data(), renderParams);
    
    std::filesystem::path outputPath = resolveGalleryPath("wr_rough_glass.png");
    savePNG(outputPath.string().c_str(), image.data(), config.width, config.height);
    
    return 0;
}
