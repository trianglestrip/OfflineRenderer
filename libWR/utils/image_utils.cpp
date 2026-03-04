#include "utils/image_utils.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace wr {
namespace utils {

void savePNG(const char* filename, const wr::Vec3* image, uint32_t width, uint32_t height) {
    std::vector<uint8_t> pixels(width * height * 3);
    
    for (uint32_t i = 0; i < width * height; ++i) {
        float gamma = 1.0f / 2.2f;
        float r = image[i].x;
        if (r < 0.0f) r = 0.0f;
        if (r > 1.0f) r = 1.0f;
        float g = image[i].y;
        if (g < 0.0f) g = 0.0f;
        if (g > 1.0f) g = 1.0f;
        float b = image[i].z;
        if (b < 0.0f) b = 0.0f;
        if (b > 1.0f) b = 1.0f;
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::pow(r, gamma) * 255);
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::pow(g, gamma) * 255);
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::pow(b, gamma) * 255);
    }
    
    stbi_write_png(filename, width, height, 3, pixels.data(), width * 3);
    std::cout << "Saved " << filename << std::endl;
}

} // namespace utils
} // namespace wr