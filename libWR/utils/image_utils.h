#pragma once

#include "wr/wr.h"
#include <vector>
#include <string>

namespace wr {
namespace utils {

// Save image as PNG with gamma correction
void savePNG(const char* filename, const wr::Vec3* image, uint32_t width, uint32_t height);

} // namespace utils
} // namespace wr