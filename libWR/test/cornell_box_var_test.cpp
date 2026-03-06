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
    
    // VLR uses "sRGB Gamma" color space, which converts sRGB to linear
    // sRGB 0.75 -> linear ~0.522
    // sRGB 0.25 -> linear ~0.0508
    uint32_t whiteMat = scene->addLambertianMaterial(Vec3(0.522f, 0.522f, 0.522f));  // VLR white (sRGB 0.75 -> linear)
    uint32_t redMat = scene->addLambertianMaterial(Vec3(0.522f, 0.0508f, 0.0508f));  // VLR red (sRGB to linear)
    uint32_t blueMat = scene->addLambertianMaterial(Vec3(0.0508f, 0.0508f, 0.522f));  // VLR blue (sRGB to linear)
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(30.0f, 30.0f, 30.0f));
    uint32_t glassMat = scene->addGlassMaterial(Vec3(0.999f, 0.999f, 0.999f), 2.4f);  // Diamond IOR (matching VLR)
    
    // GGX materials for metal box (matching VLR's gold appearance)
    // Note: The gold color comes from the conductor Fresnel equation with physical eta/k values
    uint32_t goldMat = scene->addGGXReflectionMaterial(
        Vec3(1.0f, 0.782f, 0.344f),  // Standard gold albedo
        0.10f,                        // Low roughness for clear reflections
        1.0f                          // Metallic (triggers conductor Fresnel)
    );
    
    const float L = -1.5f, R = 1.5f;
    const float B = 0.0f, T = 3.0f;
    const float N = -1.5f, F = 1.5f;
    
    {
        float verts[] = { L, B, F,  L, B, N,  R, B, N,  R, B, F };
        float uvs[] = { 0.0f, 5.0f,  0.0f, 0.0f,  5.0f, 0.0f,  5.0f, 5.0f };  // VLR UV coordinates
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
        const float ly = 2.9f;  // VLR exact light position
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
    
    // Gold metal box on the left (matching CornellBox_var.jpg)
    // Rotated for more interesting reflections
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        // Position: x=-0.6 (left), y=0.5 (half height), z=0.0 (center)
        // Size: 1.0, Rotation: ~20 degrees around Y axis
        createRotatedBox(verts, inds, -0.6f, 0.5f, 0.0f, 1.0f, glm::radians(20.0f));
        scene->addTriangleMesh(std::span(verts), std::span(inds), goldMat);
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
    camera.position = Vec3(0.0f, 1.5f, 6.0f);  // Match VLR camera position
    camera.target = Vec3(0.0f, 1.5f, 0.0f);    // Look at center
    camera.up = Vec3(0.0f, 1.0f, 0.0f);
    camera.fovY = glm::radians(40.0f);         // Match VLR FOV (40 degrees)
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
