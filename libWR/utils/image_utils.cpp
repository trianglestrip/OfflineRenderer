#include "utils/image_utils.h"
#include <vector>
#include <cmath>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace wr {
namespace utils {

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

} // namespace utils
} // namespace wr