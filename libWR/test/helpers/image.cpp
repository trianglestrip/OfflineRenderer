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

// Reinhard Tone Mapping (matching VLR)
// Formula: 1 - exp(-x)
inline float ReinhardToneMap(float x) {
    return 1.0f - std::exp(-x);
}

void savePNG(const char* filename, const wr::Vec3* image, uint32_t width, uint32_t height) {
    std::vector<uint8_t> pixels(width * height * 3);

    // Tone mapping parameters (matching VLR's Reinhard with brightnessCoeff=1.0)
    const float brightnessCoeff = 1.0f;  // VLR's brightness coefficient
    const float gamma = 1.0f / 2.2f;

    for (uint32_t i = 0; i < width * height; ++i) {
        // Apply brightness coefficient (VLR's exposure equivalent)
        float r = image[i].x * brightnessCoeff;
        float g = image[i].y * brightnessCoeff;
        float b = image[i].z * brightnessCoeff;
        
        // Clamp negative values
        r = std::max(r, 0.0f);
        g = std::max(g, 0.0f);
        b = std::max(b, 0.0f);
        
        // Reinhard tone mapping (matching VLR: 1 - exp(-x))
        r = ReinhardToneMap(r);
        g = ReinhardToneMap(g);
        b = ReinhardToneMap(b);
        
        // Gamma correction (sRGB gamma)
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::pow(r, gamma) * 255);
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::pow(g, gamma) * 255);
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::pow(b, gamma) * 255);
    }

    stbi_write_png(filename, width, height, 3, pixels.data(), width * 3);
    std::cout << "Saved " << filename << std::endl;
}

} // namespace test_helpers