// Closed room with RGB three walls + 2 objects (libVLRW test)
#include <vlrw/vlrw.h>
#include <vlrw/config.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <span>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>

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

// Add a quad (4 vertices, 2 triangles); vertices in CCW when viewed from front
static void addQuad(vlrw::Scene* scene, const float* v0, const float* v1, const float* v2, const float* v3, uint32_t matId) {
    std::vector<float> verts = {
        v0[0], v0[1], v0[2], v1[0], v1[1], v1[2],
        v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]
    };
    std::vector<uint32_t> idx = { 0, 1, 2, 0, 2, 3 };
    scene->addTriangleMesh(std::span<const float>(verts), std::span<const uint32_t>(idx), matId);
}

// Add an axis-aligned box from (x0,y0,z0) to (x1,y1,z1), 8 vertices, 12 triangles
static void addBox(vlrw::Scene* scene, float x0, float y0, float z0, float x1, float y1, float z1, uint32_t matId) {
    std::vector<float> verts = {
        x0, y0, z0, x1, y0, z0, x1, y0, z1, x0, y0, z1,  // bottom (y0)
        x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1   // top (y1)
    };
    // 6 faces: bottom(0,1,2,3), top(4,5,6,7), -x(0,3,7,4), +x(1,2,6,5), -z(0,1,5,4), +z(2,3,7,6)
    std::vector<uint32_t> idx = {
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3
    };
    scene->addTriangleMesh(std::span<const float>(verts), std::span<const uint32_t>(idx), matId);
}

// Closed room: box [x=-1..1, y=0..2, z=-1..1]. RGB 3 walls: back=blue, left=green, right=red; floor/ceiling/front gray/white. + 2 boxes inside.
void buildClosedRoomRGB(vlrw::Scene* scene) {
    const float L = -1.0f, R = 1.0f;
    const float B = 0.0f, T = 2.0f;
    const float N = -1.0f, F = 1.0f;

    // Floor (y=B)
    float f0[] = { L, B, N }, f1[] = { R, B, N }, f2[] = { R, B, F }, f3[] = { L, B, F };
    addQuad(scene, f0, f1, f2, f3, 0);

    // Ceiling (y=T)
    float c0[] = { L, T, N }, c1[] = { L, T, F }, c2[] = { R, T, F }, c3[] = { R, T, N };
    addQuad(scene, c0, c1, c2, c3, 1);

    // Back wall (z=N) - Blue
    float bn0[] = { L, B, N }, bn1[] = { L, T, N }, bn2[] = { R, T, N }, bn3[] = { R, B, N };
    addQuad(scene, bn0, bn1, bn2, bn3, 2);

    // Front wall (z=F) - White
    float fr0[] = { R, B, F }, fr1[] = { R, T, F }, fr2[] = { L, T, F }, fr3[] = { L, B, F };
    addQuad(scene, fr0, fr1, fr2, fr3, 3);

    // Left wall (x=L) - Green
    float l0[] = { L, B, F }, l1[] = { L, T, F }, l2[] = { L, T, N }, l3[] = { L, B, N };
    addQuad(scene, l0, l1, l2, l3, 4);

    // Right wall (x=R) - Red
    float r0[] = { R, B, N }, r1[] = { R, T, N }, r2[] = { R, T, F }, r3[] = { R, B, F };
    addQuad(scene, r0, r1, r2, r3, 5);

    // Object 1: small box left-back
    addBox(scene, -0.5f, 0.0f, -0.5f, -0.2f, 0.6f, -0.2f, 6);

    // Object 2: small box right-front
    addBox(scene, 0.2f, 0.0f, 0.2f, 0.6f, 0.5f, 0.6f, 7);

    vlrw::MaterialDesc mat;
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.ior = 1.5f;
    mat.emission = vlrw::RGB::black();

    mat.albedo = vlrw::RGB(0.82f, 0.82f, 0.82f);
    scene->setMaterial(0, mat);  // floor
    scene->setMaterial(1, mat);  // ceiling
    mat.albedo = vlrw::RGB(0.2f, 0.25f, 0.85f);
    scene->setMaterial(2, mat);  // back - blue
    mat.albedo = vlrw::RGB(0.9f, 0.9f, 0.9f);
    scene->setMaterial(3, mat);  // front - white
    mat.albedo = vlrw::RGB(0.2f, 0.75f, 0.25f);
    scene->setMaterial(4, mat);  // left - green
    mat.albedo = vlrw::RGB(0.85f, 0.2f, 0.2f);
    scene->setMaterial(5, mat);  // right - red
    mat.albedo = vlrw::RGB(0.9f, 0.88f, 0.85f);
    scene->setMaterial(6, mat);  // object 1
    mat.albedo = vlrw::RGB(0.95f, 0.85f, 0.4f);
    scene->setMaterial(7, mat);  // object 2

    scene->finalize();
}

// Classic Cornell Box variant (similar to CornellBox_var.png)
// - Red left wall + Blue right wall
// - White floor/ceiling/back wall
// - Gold metallic box (left)
// - Reflective sphere (right)
void buildClassicCornellBox(vlrw::Scene* scene) {
    const float L = -1.0f, R = 1.0f;
    const float B = 0.0f, T = 2.0f;
    const float N = -1.0f, F = 1.0f;

    // Materials
    // 0: White floor
    vlrw::MaterialDesc white;
    white.albedo = vlrw::RGB(0.82f, 0.82f, 0.82f);
    white.roughness = 1.0f;
    white.metallic = 0.0f;
    white.ior = 1.5f;
    white.emission = vlrw::RGB::black();
    scene->setMaterial(0, white);
    scene->setMaterial(1, white);  // ceiling
    scene->setMaterial(2, white);  // back wall
    scene->setMaterial(3, white);  // front wall

    // 4: Red left wall
    vlrw::MaterialDesc red;
    red.albedo = vlrw::RGB(0.73f, 0.14f, 0.14f);
    red.roughness = 1.0f;
    red.metallic = 0.0f;
    red.ior = 1.5f;
    red.emission = vlrw::RGB::black();
    scene->setMaterial(4, red);

    // 5: Blue right wall
    vlrw::MaterialDesc blue;
    blue.albedo = vlrw::RGB(0.14f, 0.14f, 0.73f);
    blue.roughness = 1.0f;
    blue.metallic = 0.0f;
    blue.ior = 1.5f;
    blue.emission = vlrw::RGB::black();
    scene->setMaterial(5, blue);

    // 6: Gold metallic box
    vlrw::MaterialDesc gold;
    gold.albedo = vlrw::RGB(1.0f, 0.84f, 0.0f);
    gold.roughness = 0.2f;  // Shiny
    gold.metallic = 1.0f;   // Full metallic
    gold.ior = 1.5f;
    gold.emission = vlrw::RGB::black();
    scene->setMaterial(6, gold);

    // 7: Reflective sphere
    vlrw::MaterialDesc sphere;
    sphere.albedo = vlrw::RGB(0.9f, 0.9f, 0.9f);
    sphere.roughness = 0.05f;  // Very shiny
    sphere.metallic = 1.0f;
    sphere.ior = 1.5f;
    sphere.emission = vlrw::RGB::black();
    scene->setMaterial(7, sphere);

    // Floor
    float f0[] = { L, B, N }, f1[] = { R, B, N }, f2[] = { R, B, F }, f3[] = { L, B, F };
    addQuad(scene, f0, f1, f2, f3, 0);

    // Ceiling
    float c0[] = { L, T, N }, c1[] = { L, T, F }, c2[] = { R, T, F }, c3[] = { R, T, N };
    addQuad(scene, c0, c1, c2, c3, 1);

    // Back wall (白色)
    float bn0[] = { L, B, N }, bn1[] = { L, T, N }, bn2[] = { R, T, N }, bn3[] = { R, B, N };
    addQuad(scene, bn0, bn1, bn2, bn3, 2);

    // Front wall - REMOVED (open to camera)
    // 经典 Cornell Box 前面是开口的，相机从这里看进去

    // Left wall (RED)
    float l0[] = { L, B, F }, l1[] = { L, T, F }, l2[] = { L, T, N }, l3[] = { L, B, N };
    addQuad(scene, l0, l1, l2, l3, 4);

    // Right wall (BLUE)
    float r0[] = { R, B, N }, r1[] = { R, T, N }, r2[] = { R, T, F }, r3[] = { R, B, F };
    addQuad(scene, r0, r1, r2, r3, 5);

    // Gold box (left side)
    addBox(scene, -0.7f, 0.0f, -0.7f, -0.3f, 0.4f, -0.3f, 6);

    // Sphere (right side) - using cube as placeholder
    const float sphereR = 0.3f;
    addBox(scene, 
        0.5f - sphereR, sphereR - sphereR, 0.3f - sphereR,
        0.5f + sphereR, sphereR + sphereR, 0.3f + sphereR,
        7);

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

        // Load configuration
        std::cout << "[Test] Loading config.ini..." << std::endl;
        vlrw::Config config;
        if (!config.load("config.ini")) {
            std::cout << "[Test] Warning: config.ini not found, using defaults" << std::endl;
        }

        // Get render settings
        uint32_t width = config.getInt("Render.width", 512);
        uint32_t height = config.getInt("Render.height", 512);
        uint32_t spp = config.getInt("Render.spp", 4);
        uint32_t maxDepth = config.getInt("Render.maxDepth", 8);
        bool enableDenoiser = config.getBool("Render.denoiser", false);

        std::cout << "[Test] Render settings: " << width << "x" << height 
                  << ", spp=" << spp << ", maxDepth=" << maxDepth 
                  << ", denoiser=" << (enableDenoiser ? "ON" : "OFF") << std::endl;

        std::cout << "[Test] Creating libVLRW context..." << std::endl;
        vlrw::Context* ctx = vlrw::Context::create(cuContext);
        vlrw::Scene* scene = ctx->createScene();
        vlrw::Renderer* renderer = ctx->createRenderer();

        // Get scene type
        std::string sceneType = config.getString("Scene.type", "classic");
        std::cout << "[Test] Building scene: " << sceneType << std::endl;
        
        if (sceneType == "classic") {
            buildClassicCornellBox(scene);
        } else {
            buildClosedRoomRGB(scene);
        }

        // Add lights from config
        std::string lightType = config.getString("Light.type", "point");
        if (lightType == "point" || lightType == "both") {
            vlrw::PointLightDesc light;
            light.position = config.getRGB("Light.position", vlrw::RGB(0.0f, 1.8f, 0.0f));
            light.intensity = config.getRGB("Light.intensity", vlrw::RGB(20.0f, 20.0f, 20.0f));
            scene->addPointLight(light);
            std::cout << "[Test] Added point light at (" << light.position.r << ", "
                      << light.position.g << ", " << light.position.b << ")" << std::endl;
        }
        if (lightType == "area" || lightType == "both") {
            vlrw::AreaLightDesc areaLight;
            areaLight.position = config.getRGB("Light.areaPosition", vlrw::RGB(0.0f, 2.0f, 0.0f));
            areaLight.normal = config.getRGB("Light.areaNormal", vlrw::RGB(0.0f, -1.0f, 0.0f));
            areaLight.tangent = config.getRGB("Light.areaTangent", vlrw::RGB(1.0f, 0.0f, 0.0f));
            areaLight.width = config.getFloat("Light.areaWidth", 1.6f);
            areaLight.height = config.getFloat("Light.areaHeight", 1.6f);
            areaLight.emission = config.getRGB("Light.areaEmission", vlrw::RGB(15.0f, 15.0f, 15.0f));
            areaLight.doubleSided = config.getBool("Light.areaDoubleSided", false);
            scene->addAreaLight(areaLight);
            std::cout << "[Test] Added area light (ceiling) at (" << areaLight.position.r << ", "
                      << areaLight.position.g << ", " << areaLight.position.b << "), "
                      << areaLight.width << "x" << areaLight.height << std::endl;
        }

        std::cout << "[Test] Finalizing scene..." << std::endl;
        scene->finalize();
        std::cout << "[Test] Scene built successfully" << std::endl;

        std::vector<vlrw::RGB> image(width * height);

        // Camera from config (or use defaults for classic Cornell Box)
        vlrw::Camera camera;
        if (sceneType == "classic") {
            // Classic Cornell Box camera: from front, slightly above center, looking in
            camera.position = vlrw::RGB(0.0f, 1.0f, 2.5f);    // Outside front wall
            camera.target = vlrw::RGB(0.0f, 1.0f, 0.0f);      // Center of room
            camera.up = vlrw::RGB(0.0f, 1.0f, 0.0f);
            camera.fovY = 40.0f * 3.14159f / 180.0f;
        } else {
            camera.position = config.getRGB("Camera.position", vlrw::RGB(0.0f, 1.0f, 0.6f));
            camera.target = config.getRGB("Camera.target", vlrw::RGB(0.0f, 1.0f, -0.5f));
            camera.up = config.getRGB("Camera.up", vlrw::RGB(0.0f, 1.0f, 0.0f));
            camera.fovY = config.getFloat("Camera.fov", 60.0f) * 3.14159f / 180.0f;
        }
        camera.aspect = (float)width / height;

        // Debug mode
        std::string debugModeStr = config.getString("Render.debugMode", "default");
        int debugMode = 0;
        if (debugModeStr == "normal") {
            debugMode = 1;
            std::cout << "[Test] Debug mode: Normal visualization" << std::endl;
        }

        std::cout << "[Test] Starting render (spp=" << spp << ", denoiser=" << (enableDenoiser ? "ON" : "OFF") << ")..." << std::endl;
        renderer->render(scene, camera, image.data(), width, height, spp, enableDenoiser, debugMode);
        std::cout << "[Test] Render completed" << std::endl;

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
