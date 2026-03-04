#pragma once

#include <filesystem>
#include <string>

namespace wr {
namespace utils {

// Resolve gallery path for output images
std::filesystem::path resolveGalleryPath(const std::string& fileName);

} // namespace utils
} // namespace wr