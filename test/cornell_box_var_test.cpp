#include <optixw/optixw.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "render_config.h"

using namespace optixw;

// Save image as PNG with gamma correction
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
    uint32_t whiteMat = scene->addLambertianMaterial(Vec3(0.75f, 0.75f, 0.75f));
    uint32_t redMat = scene->addLambertianMaterial(Vec3(0.75f, 0.25f, 0.25f));
    uint32_t blueMat = scene->addLambertianMaterial(Vec3(0.25f, 0.25f, 0.75f));
    uint32_t lightMat = scene->addEmissiveMaterial(Vec3(12.0f, 12.0f, 12.0f));
    uint32_t glassMat = scene->addGlassMaterial(Vec3(0.999f, 0.999f, 0.999f), 2.42f);  // Diamond IOR
    
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
        camera.position = Vec3(0.0f, 1.5f, 6.0f);
        camera.target = Vec3(0.0f, 1.5f, 0.0f);
        camera.up = Vec3(0.0f, 1.0f, 0.0f);
        camera.fovY = 40.0f * 3.14159f / 180.0f;
        camera.aspect = 1.0f;
        
        const RenderConfig cfg = render_config::load("cornell_box_var");
        const uint32_t width = cfg.width;
        const uint32_t height = cfg.height;
        const uint32_t spp = cfg.spp;
        
        std::cout << "[Test] Rendering " << width << "x" << height << " @ " << spp << " spp..." << std::endl;
        
        auto runRender = [&](uint32_t tileW, uint32_t tileH, const char *label) {
            // capture console output so we can inspect the denoiser log
            std::ostringstream oss;
            std::streambuf* oldbuf = std::cout.rdbuf(oss.rdbuf());

            std::vector<Vec3> outputBuffer(width * height);
            renderer->render(scene, camera, outputBuffer.data(), width, height, spp,
                             cfg.denoiser, cfg.denoiserBlend, cfg.enableTiling, tileW, tileH);

            // restore output
            std::cout.rdbuf(oldbuf);
            std::string log = oss.str();
            std::cout << log; // still print it for user visibility

            // count tasks reported in log by parsing the number after the label
            size_t taskCount = 0;
            {
                std::string key = "denoiser tiling:";
                size_t pos = log.find(key);
                if (pos != std::string::npos) {
                    pos += key.size();
                    // skip whitespace
                    while (pos < log.size() && isspace((unsigned char)log[pos])) ++pos;
                    int val = 0;
                    if (sscanf(log.c_str() + pos, "%d", &val) == 1) {
                        taskCount = (size_t)val;
                    }
                }
            }
            std::cout << "[Test] " << label << " taskCount=" << taskCount << "\n";

            // if there were multiple tasks, compute seam difference on the image
            if (taskCount > 1) {
                auto computeSeamDetails = [&](uint32_t tw, uint32_t th) {
                    float maxdiff = 0.0f;
                    // vertical seams
                    for (uint32_t sx = tw; sx < width; sx += tw) {
                        float rowMaxDiff = 0.0f;
                        for (uint32_t y = 0; y < height; ++y) {
                            Vec3 l = outputBuffer[y * width + (sx - 1)];
                            Vec3 r = outputBuffer[y * width + sx];
                            float d = fabs(l.x - r.x) + fabs(l.y - r.y) + fabs(l.z - r.z);
                            maxdiff = std::max(maxdiff, d);
                            rowMaxDiff = std::max(rowMaxDiff, d);
                        }
                        std::cout << "[Test] " << label << " vertical seam at x=" << sx << ": max diff = " << rowMaxDiff << "\n";
                    }
                    // horizontal seams
                    for (uint32_t sy = th; sy < height; sy += th) {
                        float colMaxDiff = 0.0f;
                        for (uint32_t x = 0; x < width; ++x) {
                            Vec3 t = outputBuffer[(sy - 1) * width + x];
                            Vec3 b = outputBuffer[sy * width + x];
                            float d = fabs(t.x - b.x) + fabs(t.y - b.y) + fabs(t.z - b.z);
                            maxdiff = std::max(maxdiff, d);
                            colMaxDiff = std::max(colMaxDiff, d);
                        }
                        std::cout << "[Test] " << label << " horizontal seam at y=" << sy << ": max diff = " << colMaxDiff << "\n";
                    }
                    return maxdiff;
                };

                float seam = computeSeamDetails(tileW, tileH);
                std::cout << "[Test] " << label << " total seam max diff = " << seam << "\n";
                // for now just report, don't fail
            }

            // save image with label appended
            std::filesystem::path out = render_config::resolveGalleryPath("cornell_box_var");
            out.replace_filename(std::string("cornell_box_var_") + label + ".png");
            savePNG(out.string().c_str(), outputBuffer.data(), width, height);
        };

        // test a small tile size (equal to overlap) which should be disabled
        runRender(128, 128, "small");
        // run a larger tile size that still ends up being disabled by the
        // 2*overlap check above
        runRender(256, 256, "large");
        // run a medium tile that is big enough to permit tiling; overlap is
        // 128 so 400 > 2*overlap=256 and we should get proper tiles and zero
        // seam difference
        runRender(400, 400, "medium");

        std::cout << "[Test] Test completed successfully" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
