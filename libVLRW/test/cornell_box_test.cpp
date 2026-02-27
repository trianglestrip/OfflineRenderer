// Cornell Box test for libVLRW
#include <vlrw/vlrw.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <span>
#include <sstream>
#include <stdexcept>

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

void buildCornellBox(vlrw::Scene* scene) {
    // Floor
    std::vector<float> vertices = {
        -1.0f, 0.0f, -1.0f,
         1.0f, 0.0f, -1.0f,
         1.0f, 0.0f,  1.0f,
        -1.0f, 0.0f,  1.0f
    };
    std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};
    
    scene->addTriangleMesh(
        std::span<const float>(vertices),
        std::span<const uint32_t>(indices),
        0);
    
    vlrw::MaterialDesc mat;
    mat.albedo = vlrw::RGB(0.8f, 0.8f, 0.8f);
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.ior = 1.5f;
    mat.emission = vlrw::RGB::black();
    scene->setMaterial(0, mat);
    
    scene->finalize();
}

int main() {
    try {
        std::cout << "[Test] Initializing CUDA..." << std::endl;
        CUDA_CHECK(cuInit(0));
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    
    char deviceName[256];
    CUDA_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    std::cout << "[Test] Using device: " << deviceName << std::endl;
    
    CUcontext cuContext;
    CUctxCreateParams params = {};
    params.execAffinityParams = nullptr;
    params.numExecAffinityParams = 0;
    params.cigParams = nullptr;
    CUDA_CHECK(cuCtxCreate(&cuContext, &params, 0, device));
    
    std::cout << "[Test] Creating libVLRW context..." << std::endl;
    vlrw::Context* ctx = vlrw::Context::create(cuContext);
    std::cout << "[Test] Creating scene and renderer..." << std::endl;
    vlrw::Scene* scene = ctx->createScene();
    vlrw::Renderer* renderer = ctx->createRenderer();
    
    std::cout << "[Test] Building Cornell Box scene..." << std::endl;
    buildCornellBox(scene);
    std::cout << "[Test] Scene built successfully" << std::endl;
    
    const uint32_t width = 512;
    const uint32_t height = 512;
    std::vector<vlrw::RGB> image(width * height);
    
    vlrw::Camera camera;
    camera.position = vlrw::RGB(0, 1, 3);
    camera.target = vlrw::RGB(0, 1, 0);
    camera.up = vlrw::RGB(0, 1, 0);
    camera.fovY = 45.0f * 3.14159f / 180.0f;
    camera.aspect = (float)width / height;
    
    std::cout << "[Test] Starting render..." << std::endl;
    renderer->render(scene, camera, image.data(), width, height, 1);
    std::cout << "[Test] Render completed" << std::endl;
    
    std::vector<unsigned char> pixels(width * height * 3);
    for (uint32_t i = 0; i < width * height; ++i) {
        pixels[i * 3 + 0] = (unsigned char)(image[i][0] * 255.0f);
        pixels[i * 3 + 1] = (unsigned char)(image[i][1] * 255.0f);
        pixels[i * 3 + 2] = (unsigned char)(image[i][2] * 255.0f);
    }
    
    stbi_write_png("cornell_box.png", width, height, 3, pixels.data(), width * 3);
    std::cout << "Rendered cornell_box.png" << std::endl;
    
        delete renderer;
        delete scene;
        delete ctx;
        CUDA_CHECK(cuCtxDestroy(cuContext));
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[Test] ERROR: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[Test] ERROR: Unknown exception" << std::endl;
        return 1;
    }
}
