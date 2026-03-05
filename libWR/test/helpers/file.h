#pragma once

#include <filesystem>
#include <string>

namespace test_helpers {

// Get the directory containing the current executable
std::filesystem::path getExecutableDirectory();

// Resolve gallery path for output images
std::filesystem::path resolveGalleryPath(const std::string& fileName);

} // namespace test_helpers
