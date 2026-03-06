#include <VLRM/VLRM.h>
#include <iostream>
#include <vector>
#include <fstream>

using namespace vlrm;

void savePPM(const char* filename, uint32_t width, uint32_t height, const RGB* pixels) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    
    for (uint32_t i = 0; i < width * height; ++i) {
        RGB color = pixels[i];
        color.r = std::pow(std::clamp(color.r, 0.0f, 1.0f), 1.0f / 2.2f);
        color.g = std::pow(std::clamp(color.g, 0.0f, 1.0f), 1.0f / 2.2f);
        color.b = std::pow(std::clamp(color.b, 0.0f, 1.0f), 1.0f / 2.2f);
        
        uint8_t r = uint8_t(color.r * 255.0f);
        uint8_t g = uint8_t(color.g * 255.0f);
        uint8_t b = uint8_t(color.b * 255.0f);
        
        file.write(reinterpret_cast<const char*>(&r), 1);
        file.write(reinterpret_cast<const char*>(&g), 1);
        file.write(reinterpret_cast<const char*>(&b), 1);
    }
}

int main() {
    try {
        Context context;
        context.initialize();
        
        Scene scene(&context);
        
        RGB white(0.73f, 0.73f, 0.73f);
        RGB red(0.65f, 0.05f, 0.05f);
        RGB green(0.12f, 0.45f, 0.15f);
        RGB light(17.0f, 12.0f, 4.0f);
        
        uint32_t whiteMat = 0;
        uint32_t redMat = 1;
        uint32_t greenMat = 2;
        uint32_t lightMat = 3;
        
        scene.addMaterial(white);
        scene.addMaterial(red);
        scene.addMaterial(green);
        scene.addMaterial(RGB::Zero(), light);
        
        std::vector<float> floorVerts = {
            -1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f
        };
        std::vector<uint32_t> floorIndices = { 0, 1, 2, 0, 2, 3 };
        scene.addTriangleMesh(floorVerts, floorIndices, whiteMat);
        
        std::vector<float> ceilVerts = {
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f, -1.0f
        };
        std::vector<uint32_t> ceilIndices = { 0, 1, 2, 0, 2, 3 };
        scene.addTriangleMesh(ceilVerts, ceilIndices, whiteMat);
        
        std::vector<float> backVerts = {
            -1.0f, -1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f, -1.0f, -1.0f
        };
        std::vector<uint32_t> backIndices = { 0, 1, 2, 0, 2, 3 };
        scene.addTriangleMesh(backVerts, backIndices, whiteMat);
        
        std::vector<float> leftVerts = {
            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f, -1.0f
        };
        std::vector<uint32_t> leftIndices = { 0, 1, 2, 0, 2, 3 };
        scene.addTriangleMesh(leftVerts, leftIndices, redMat);
        
        std::vector<float> rightVerts = {
             1.0f, -1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f, -1.0f,  1.0f
        };
        std::vector<uint32_t> rightIndices = { 0, 1, 2, 0, 2, 3 };
        scene.addTriangleMesh(rightVerts, rightIndices, greenMat);
        
        std::vector<float> lightVerts = {
            -0.25f,  0.99f, -0.25f,
            -0.25f,  0.99f,  0.25f,
             0.25f,  0.99f,  0.25f,
             0.25f,  0.99f, -0.25f
        };
        std::vector<uint32_t> lightIndices = { 0, 1, 2, 0, 2, 3 };
        scene.addTriangleMesh(lightVerts, lightIndices, lightMat);
        
        scene.buildAccelerationStructure();
        
        Renderer renderer(&context, &scene);
        
        Point3D cameraPos(0.0f, 0.0f, 3.0f);
        Vector3D cameraForward(0.0f, 0.0f, -1.0f);
        Vector3D cameraUp(0.0f, 1.0f, 0.0f);
        float fovY = 45.0f * (VLRM_M_PI / 180.0f);
        
        renderer.setCamera(cameraPos, cameraForward, cameraUp, fovY);
        
        uint32_t width = 512;
        uint32_t height = 512;
        uint32_t spp = 256;
        uint32_t maxDepth = 8;
        
        std::vector<RGB> outputBuffer(width * height);
        
        std::cout << "Rendering Cornell Box (" << width << "x" << height 
                  << ", " << spp << " spp)..." << std::endl;
        
        renderer.render(width, height, spp, maxDepth, outputBuffer.data());
        
        savePPM("cornell_box_vlrm.ppm", width, height, outputBuffer.data());
        
        std::cout << "Rendering complete. Output saved to cornell_box_vlrm.ppm" << std::endl;
        
        context.finalize();
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
