// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace optixw {
namespace utils {

// 查找运行时文件（PTX、CUBIN 等）
// subDir: 子目录名（如 "ptx", "cubin"）
// fileName: 文件名
// 返回: 文件的完整路径
// 抛出: std::runtime_error 如果文件未找到
std::filesystem::path findRuntimeFile(const char* subDir, const char* fileName);

// 查找 PTX 文件路径
std::filesystem::path findPTXPath(const char* fileName);

// 查找 CUBIN 文件路径
std::filesystem::path findCubinPath(const char* fileName);

// 加载 PTX 文件内容
std::vector<char> loadPTX(const char* filename);

// 加载 CUBIN 文件内容
std::vector<char> loadCubin(const char* filename);

} // namespace utils
} // namespace optixw
