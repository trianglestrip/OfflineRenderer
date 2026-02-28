#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct RenderConfig {
    uint32_t width = 512;
    uint32_t height = 512;
    uint32_t spp = 4;
};

namespace render_config {

inline std::string trim(const std::string& input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }

    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }

    return input.substr(begin, end - begin);
}

inline std::vector<std::filesystem::path> candidatePaths() {
    return {
        "render_config.ini",
        "libOptixW/test/render_config.ini",
        "../test/render_config.ini",
        "../../../libOptixW/test/render_config.ini"
    };
}

inline std::filesystem::path resolveGalleryPath(const std::string& fileName) {
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

inline bool parseUInt32(const std::string& text, uint32_t* outValue) {
    try {
        unsigned long parsed = std::stoul(text);
        if (parsed > UINT32_MAX) {
            return false;
        }
        *outValue = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

inline RenderConfig load(const std::string& sectionName, const RenderConfig& defaults = {}) {
    RenderConfig config = defaults;
    std::filesystem::path foundPath;

    for (const std::filesystem::path& path : candidatePaths()) {
        if (std::filesystem::exists(path)) {
            foundPath = path;
            break;
        }
    }

    if (foundPath.empty()) {
        std::cout << "[Config] render_config.ini not found, using defaults: "
                  << config.width << "x" << config.height << " @ " << config.spp << " spp\n";
        return config;
    }

    std::ifstream file(foundPath);
    if (!file) {
        std::cout << "[Config] Failed to open " << foundPath.string()
                  << ", using defaults: " << config.width << "x" << config.height
                  << " @ " << config.spp << " spp\n";
        return config;
    }

    std::string currentSection;
    std::string line;
    while (std::getline(file, line)) {
        const std::string content = trim(line);
        if (content.empty() || content[0] == ';' || content[0] == '#') {
            continue;
        }

        if (content.front() == '[' && content.back() == ']') {
            currentSection = trim(content.substr(1, content.size() - 2));
            std::transform(currentSection.begin(), currentSection.end(), currentSection.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            continue;
        }

        const size_t eqPos = content.find('=');
        if (eqPos == std::string::npos) {
            continue;
        }

        std::string key = trim(content.substr(0, eqPos));
        std::string value = trim(content.substr(eqPos + 1));
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

        std::string targetSection = sectionName;
        std::transform(targetSection.begin(), targetSection.end(), targetSection.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

        if (!currentSection.empty() && currentSection != "default" && currentSection != targetSection) {
            continue;
        }

        uint32_t parsed = 0;
        if (!parseUInt32(value, &parsed)) {
            continue;
        }

        if (key == "width" && parsed > 0) {
            config.width = parsed;
        } else if (key == "height" && parsed > 0) {
            config.height = parsed;
        } else if (key == "spp" && parsed > 0) {
            config.spp = parsed;
        }
    }

    std::cout << "[Config] Loaded " << foundPath.string() << " for [" << sectionName << "]: "
              << config.width << "x" << config.height << " @ " << config.spp << " spp\n";
    return config;
}

} // namespace render_config
