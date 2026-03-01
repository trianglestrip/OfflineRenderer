// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include "path_utils.h"
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace optixw {
namespace utils {

std::filesystem::path findRuntimeFile(const char* subDir, const char* fileName) {
    // 候选路径列表
    std::vector<std::filesystem::path> candidates;
    
    // 候选 1: 相对于当前工作目录
    candidates.emplace_back(std::filesystem::path(subDir) / fileName);

#ifdef _WIN32
    // Windows: 获取可执行文件路径
    char exePath[MAX_PATH] = {};
    DWORD len = GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        // 候选 2: 相对于可执行文件目录
        candidates.emplace_back(exeDir / subDir / fileName);
        // 候选 3: 相对于可执行文件父目录
        candidates.emplace_back(exeDir.parent_path() / subDir / fileName);
    }
#else
    // Linux/Unix: 获取可执行文件路径
    char exePath[PATH_MAX] = {};
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len != -1) {
        exePath[len] = '\0';
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        candidates.emplace_back(exeDir / subDir / fileName);
        candidates.emplace_back(exeDir.parent_path() / subDir / fileName);
    }
#endif

    // 遍历候选路径，返回第一个存在的文件
    for (const std::filesystem::path& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    // 所有候选路径都不存在，抛出异常
    throw std::runtime_error(
        std::string("Failed to locate runtime file: ") + subDir + "/" + fileName);
}

std::filesystem::path findPTXPath(const char* fileName) {
    return findRuntimeFile("ptx", fileName);
}

std::filesystem::path findCubinPath(const char* fileName) {
    return findRuntimeFile("cubin", fileName);
}

std::vector<char> loadPTX(const char* filename) {
    const std::filesystem::path ptxPath = findPTXPath(filename);
    std::ifstream file(ptxPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            std::string("Failed to open PTX file: ") + ptxPath.string());
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 读取文件内容
    std::vector<char> data(size);
    file.read(data.data(), size);
    
    return data;
}

std::vector<char> loadCubin(const char* filename) {
    const std::filesystem::path cubinPath = findCubinPath(filename);
    std::ifstream file(cubinPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            std::string("Failed to open CUBIN file: ") + cubinPath.string());
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 读取文件内容
    std::vector<char> data(size);
    file.read(data.data(), size);
    
    return data;
}

} // namespace utils
} // namespace optixw
