#include "utils/file_utils.h"
#include <filesystem>

namespace wr {
namespace utils {

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