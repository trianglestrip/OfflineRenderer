#include "image.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace test_helpers {

// ACES Filmic Tone Mapping
// Reference: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
inline float ACESFilmic(float x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

void savePNG(const char* filename, const wr::Vec3* image, uint32_t width, uint32_t height) {
    std::vector<uint8_t> pixels(width * height * 3);

    // Tone mapping parameters
    const float exposure = 2.2f;  // Adjusted for ACES
    const float gamma = 1.0f / 2.2f;

    for (uint32_t i = 0; i < width * height; ++i) {
        // Apply exposure
        float r = image[i].x * exposure;
        float g = image[i].y * exposure;
        float b = image[i].z * exposure;
        
        // ACES Filmic tone mapping (better color preservation and contrast)
        r = ACESFilmic(r);
        g = ACESFilmic(g);
        b = ACESFilmic(b);
        
        // Gamma correction (ACES output is in linear space)
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::pow(r, gamma) * 255);
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::pow(g, gamma) * 255);
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::pow(b, gamma) * 255);
    }

    stbi_write_png(filename, width, height, 3, pixels.data(), width * 3);
    std::cout << "Saved " << filename << std::endl;
}

} // namespace test_helpers