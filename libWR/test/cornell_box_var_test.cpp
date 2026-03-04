#include <wr/wr.h>
#include "utils.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

#include "render_config.h"

using namespace wr;
using namespace wr::utils;

void buildCornellBoxVar(Scene* scene) {
    uint32_t whiteMat = scene->addLambertianMaterial(Vec3(0.75f, 0.75f, 0.75f));
    uint32_t redMat = scene->addLambertianMaterial(Vec3(0.75f, 0.25f, 0.25f));
    uint32_t blueMat = scene->addLambertianMaterial(Vec3(0.25f, 0.25f, 0.75f));
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(40.0f, 40.0f, 40.0f));
    uint32_t glassMat = scene->addGlassMaterial(Vec3(0.999f, 0.999f, 0.999f), 1.5f);
    uint32_t metalMat = scene->addLambertianMaterial(Vec3(0.9f, 0.7f, 0.1f)); // Yellow metal material
    
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
        utils::createSphere(verts, inds, -0.7f, 0.5f, -0.7f, 0.5f, 32, 24);
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    {
        std::vector<float> verts;
        std::vector<uint32_t> inds;
        utils::createSphere(verts, inds, 0.7f, 0.5f, 0.7f, 0.5f, 32, 24);
        scene->addTriangleMesh(std::span(verts), std::span(inds), glassMat);
    }
    
    // Add yellow metal box in the center
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
        scene->addTriangleMesh(std::span(verts, 72), std::span(inds, 36), metalMat);
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
    renderer->render(scene, camera, image.data(), config.width, config.height, config.spp, config.denoiser);
    
    std::filesystem::path outputPath = utils::resolveGalleryPath("wr_cornell.png");
    utils::savePNG(outputPath.string().c_str(), image.data(), config.width, config.height);
    
    return 0;
}
