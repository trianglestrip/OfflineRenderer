#include "utils/file_utils.h"
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace wr {
namespace utils {

std::filesystem::path getExecutableDirectory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    std::filesystem::path exePath(buffer);
    return exePath.parent_path();
#else
    char buffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        std::filesystem::path exePath(buffer);
        return exePath.parent_path();
    }
    return std::filesystem::current_path();
#endif
}

std::filesystem::path resolveGalleryPath(const std::string& fileName) {
    const std::vector<std::filesystem::path> galleryCandidates = {
        "gallery",
        "../gallery",
        "../../gallery",
        "../../../gallery"
    };

    for (const std::filesystem::path& dir : galleryCandidates) {
        if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
            return dir / fileName;
        }
    }

    std::filesystem::create_directories("gallery");
    return std::filesystem::path("gallery") / fileName;
}

} // namespace utils
} // namespace wr