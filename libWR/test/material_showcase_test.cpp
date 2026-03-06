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

void buildMaterialShowcase(Scene* scene) {
    // Materials
    uint32_t whiteMat = scene->addLambertianMaterial(Vec3(0.8f, 0.8f, 0.8f));
    uint32_t redMat = scene->addLambertianMaterial(Vec3(0.63f, 0.065f, 0.05f));
    uint32_t blueMat = scene->addLambertianMaterial(Vec3(0.14f, 0.16f, 0.55f));
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(20.0f, 20.0f, 20.0f));
    
    // Row 1: Dielectrics (Glass)
    uint32_t idealGlass = scene->addGlassMaterial(Vec3(0.999f, 0.999f, 0.999f), 1.5f);
    uint32_t roughGlass1 = scene->addGGXTransmissionMaterial(Vec3(0.999f, 0.999f, 0.999f), 0.1f, 1.5f);
    uint32_t roughGlass2 = scene->addGGXTransmissionMaterial(Vec3(0.999f, 0.999f, 0.999f), 0.2f, 1.5f);
    uint32_t roughGlass3 = scene->addGGXTransmissionMaterial(Vec3(0.999f, 0.999f, 0.999f), 0.4f, 1.5f);
    
    // Row 2: Metals (GGX Reflection)
    uint32_t mirror = scene->addGGXReflectionMaterial(Vec3(0.95f, 0.95f, 0.95f), 0.0f, 1.0f);
    uint32_t roughMetal1 = scene->addGGXReflectionMaterial(Vec3(0.9f, 0.9f, 0.9f), 0.1f, 1.0f);
    uint32_t roughMetal2 = scene->addGGXReflectionMaterial(Vec3(0.9f, 0.9f, 0.9f), 0.2f, 1.0f);
    uint32_t roughMetal3 = scene->addGGXReflectionMaterial(Vec3(0.9f, 0.9f, 0.9f), 0.4f, 1.0f);
    
    // Row 3: Colored metals (Gold, Copper, etc.)
    uint32_t gold = scene->addGGXReflectionMaterial(Vec3(1.0f, 0.782f, 0.344f), 0.05f, 1.0f);
    uint32_t copper = scene->addGGXReflectionMaterial(Vec3(0.955f, 0.638f, 0.538f), 0.05f, 1.0f);
    uint32_t roughGold = scene->addGGXReflectionMaterial(Vec3(1.0f, 0.782f, 0.344f), 0.3f, 1.0f);
    uint32_t roughCopper = scene->addGGXReflectionMaterial(Vec3(0.955f, 0.638f, 0.538f), 0.3f, 1.0f);
    
    // Cornell Box walls
    const float L = -2.0f, R = 2.0f;
    const float B = 0.0f, T = 4.0f;
    const float N = -2.0f, F = 2.0f;
    
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
    
    // Ceiling light
    {
        const float lx = 0.8f;
        const float lz = 0.8f;
        const float ly = T - 0.01f;
        float verts[] = { -lx, ly, -lz,  lx, ly, -lz,  lx, ly, lz,  -lx, ly, lz };
        uint32_t inds[] = { 0, 1, 2, 0, 2, 3 };
        scene->addTriangleMesh(std::span(verts, 12), std::span(inds, 6), lightMat);
    }
    
    // Row 1: Dielectrics (Glass) - Back row, higher
    const float row1Y = 0.8f;
    const float row1Z = -1.0f;
    const float sphereRadius = 0.35f;
    const float spacing = 1.0f;
    
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, -1.5f, row1Y, row1Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), idealGlass);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, -0.5f, row1Y, row1Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), roughGlass1);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, 0.5f, row1Y, row1Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), roughGlass2);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, 1.5f, row1Y, row1Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), roughGlass3);
    }
    
    // Row 2: White metals - Middle row
    const float row2Y = 0.5f;
    const float row2Z = 0.0f;
    
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, -1.5f, row2Y, row2Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), mirror);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, -0.5f, row2Y, row2Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), roughMetal1);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, 0.5f, row2Y, row2Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), roughMetal2);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, 1.5f, row2Y, row2Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), roughMetal3);
    }
    
    // Row 3: Colored metals - Front row
    const float row3Y = 0.5f;
    const float row3Z = 1.0f;
    
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, -1.5f, row3Y, row3Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), gold);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, -0.5f, row3Y, row3Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), copper);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, 0.5f, row3Y, row3Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), roughGold);
    }
    {
        std::vector<float> verts, norms;
        std::vector<uint32_t> inds;
        createSphere(verts, norms, inds, 1.5f, row3Y, row3Z, sphereRadius, 48, 32);
        scene->addTriangleMesh(std::span(verts), std::span(inds), std::span(norms), std::span<const float>(), roughCopper);
    }
    
    std::cout << "[Test] Material showcase scene built (12 spheres in 3 rows)" << std::endl;
}

int main() {
    std::cout << "=== libWR Material Showcase Test ===" << std::endl;
    
    RenderConfig config = config::load("material_showcase");
    
    wr::Context context;
    wr::Scene* scene = context.createScene();
    buildMaterialShowcase(scene);
    scene->setEnvironmentRadiance(Vec3(0.0f, 0.0f, 0.0f));
    scene->finalize();
    
    wr::Renderer* renderer = context.createRenderer();
    
    wr::Camera camera;
    camera.position = Vec3(0.0f, 2.0f, 5.5f);
    camera.target = Vec3(0.0f, 1.5f, 0.0f);
    camera.up = Vec3(0.0f, 1.0f, 0.0f);
    camera.fovY = glm::radians(50.0f);
    camera.aspect = static_cast<float>(config.width) / config.height;
    camera.focalDistance = 5.5f;
    camera.lensRadius = 0.0f;
    
    std::vector<wr::Vec3> image(config.width * config.height);
    
    wr::RenderParams renderParams;
    renderParams.width = config.width;
    renderParams.height = config.height;
    renderParams.spp = config.spp;
    renderParams.maxBounces = 12;
    renderParams.useNEE = true;
    renderParams.denoiser.enabled = config.denoiser;
    renderParams.russianRouletteDepth = 5.0f;
    
    renderer->render(scene, camera, image.data(), renderParams);
    
    std::filesystem::path outputPath = resolveGalleryPath("wr_material_showcase.png");
    savePNG(outputPath.string().c_str(), image.data(), config.width, config.height);
    
    std::cout << "Saved " << outputPath << std::endl;
    
    return 0;
}
