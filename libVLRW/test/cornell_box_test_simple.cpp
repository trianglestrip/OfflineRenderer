// Simple single quad test
#include <vlrw/vlrw.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <span>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define CUDA_CHECK(call) \
    do { \
        CUresult res = call; \
        if (res != CUDA_SUCCESS) { \
            const char* errMsg; \
            cuGetErrorString(res, &errMsg); \
            std::stringstream ss; \
            ss << "CUDA call failed: " << #call << " - " << errMsg; \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

void buildSimpleQuad(vlrw::Scene* scene) {
    // Single quad facing camera at z=-1, covering x=[-0.5,0.5], y=[0.5,1.5]
    std::vector<float> verts = {
        -0.5f, 0.5f, -1.0f,
         0.5f, 0.5f, -1.0f,
         0.5f, 1.5f, -1.0f,
        -0.5f, 1.5f, -1.0f
    };
    std::vector<uint32_t> idx = { 0, 1, 2, 0, 2, 3 };
    scene->addTriangleMesh(std::span<const float>(verts), std::span<const uint32_t>(idx), 0);
    
    vlrw::MaterialDesc mat;
    mat.albedo = vlrw::RGB(0.8f, 0.2f, 0.2f);  // red
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.ior = 1.5f;
    mat.emission = vlrw::RGB::black();
    scene->setMaterial(0, mat);
    scene->finalize();
}

int main() {
    try {
        CUDA_CHECK(cuInit(0));
        CUdevice device;
        CUDA_CHECK(cuDeviceGet(&device, 0));
        char deviceName[256];
        CUDA_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), device));
        std::cout << "[SimpleTest] Using device: " << deviceName << std::endl;

        CUcontext cuContext;
        CUctxCreateParams params = {};
        params.execAffinityParams = nullptr;
        params.numExecAffinityParams = 0;
        params.cigParams = nullptr;
        CUDA_CHECK(cuCtxCreate(&cuContext, &params, 0, device));

        vlrw::Context* ctx = vlrw::Context::create(cuContext);
        vlrw::Scene* scene = ctx->createScene();
        vlrw::Renderer* renderer = ctx->createRenderer();

        std::cout << "[SimpleTest] Building single quad..." << std::endl;
        buildSimpleQuad(scene);

        const uint32_t width = 512;
        const uint32_t height = 512;
        std::vector<vlrw::RGB> image(width * height);

        vlrw::Camera camera;
        camera.position = vlrw::RGB(0.0f, 1.0f, 0.0f);
        camera.target = vlrw::RGB(0.0f, 1.0f, -1.0f);
        camera.up = vlrw::RGB(0, 1, 0);
        camera.fovY = 45.0f * 3.14159f / 180.0f;
        camera.aspect = (float)width / height;

        std::cout << "[SimpleTest] Rendering..." << std::endl;
        renderer->render(scene, camera, image.data(), width, height, 1);

        const float gamma = 1.0f / 2.2f;
        std::vector<unsigned char> pixels(width * height * 3);
        for (uint32_t i = 0; i < width * height; ++i) {
            float r = std::min(1.f, std::max(0.f, image[i][0]));
            float g = std::min(1.f, std::max(0.f, image[i][1]));
            float b = std::min(1.f, std::max(0.f, image[i][2]));
            r = std::pow(r, gamma);
            g = std::pow(g, gamma);
            b = std::pow(b, gamma);
            pixels[i * 3 + 0] = (unsigned char)(r * 255.0f);
            pixels[i * 3 + 1] = (unsigned char)(g * 255.0f);
            pixels[i * 3 + 2] = (unsigned char)(b * 255.0f);
        }

        stbi_write_png("simple_quad.png", width, height, 3, pixels.data(), width * 3);
        std::cout << "[SimpleTest] Rendered simple_quad.png" << std::endl;

        delete renderer;
        delete scene;
        delete ctx;
        CUDA_CHECK(cuCtxDestroy(cuContext));
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[SimpleTest] ERROR: " << e.what() << std::endl;
        return 1;
    }
}
