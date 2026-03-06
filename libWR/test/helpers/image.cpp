#include "image.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace test_helpers {

void savePNG(const char* filename, const wr::Vec3* image, uint32_t width, uint32_t height) {
    std::vector<uint8_t> pixels(width * height * 3);

    // Tone mapping parameters
    const float exposure = 1.0f;  // Exposure adjustment
    const float gamma = 1.0f / 2.2f;

    for (uint32_t i = 0; i < width * height; ++i) {
        // Apply exposure
        float r = image[i].x * exposure;
        float g = image[i].y * exposure;
        float b = image[i].z * exposure;
        
        // Reinhard tone mapping: x / (1 + x)
        r = r / (1.0f + r);
        g = g / (1.0f + g);
        b = b / (1.0f + b);
        
        // Clamp to [0, 1]
        if (r < 0.0f) r = 0.0f;
        if (r > 1.0f) r = 1.0f;
        if (g < 0.0f) g = 0.0f;
        if (g > 1.0f) g = 1.0f;
        if (b < 0.0f) b = 0.0f;
        if (b > 1.0f) b = 1.0f;
        
        // Gamma correction
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::pow(r, gamma) * 255);
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::pow(g, gamma) * 255);
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::pow(b, gamma) * 255);
    }

    stbi_write_png(filename, width, height, 3, pixels.data(), width * 3);
    std::cout << "Saved " << filename << std::endl;
}

} // namespace test_helpers