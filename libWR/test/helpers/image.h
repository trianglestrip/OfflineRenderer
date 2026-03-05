#pragma once

#include <wr/types.h>
#include <cstdint>

namespace test_helpers {

// Save image as PNG with gamma correction
void savePNG(const char* filename, const wr::Vec3* image, uint32_t width, uint32_t height);

} // namespace test_helpers
