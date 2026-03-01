// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <optixw/core/task_scheduler.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>

namespace optixw {

// Texture2DData - 纹理数据结构（与 GPU 端共享）
struct Texture2DData {
    float4* pixels;
    uint32_t width;
    uint32_t height;
    uint32_t pad;
};

// TextureManager - CPU 层纹理管理器
// 职责：
//   - 加载纹理（支持 Taskflow 并行加载）
//   - 管理纹理数据
//   - 创建 CUDA 纹理对象
// 注意：
//   - 这是 CPU 层代码，不包含 GPU 计算逻辑
//   - 使用 Taskflow 并行加载多个纹理
class TextureManager {
public:
    explicit TextureManager(TaskScheduler* scheduler);
    ~TextureManager();

    // 加载单个纹理
    // path: 纹理文件路径
    // decodeSRGB: 是否解码 sRGB
    // 返回: 纹理 ID
    uint32_t loadTexture2D(const std::string& path, bool decodeSRGB = true);

    // 批量加载纹理（使用 Taskflow 并行）
    // paths: 纹理文件路径列表
    // decodeSRGB: 是否解码 sRGB
    // 返回: 纹理 ID 列表
    std::vector<uint32_t> loadTextures(const std::vector<std::string>& paths, bool decodeSRGB = true);

    // 上传纹理数据到 GPU
    void uploadToDevice();

    // 访问器
    CUdeviceptr getDeviceBuffer() const { return m_d_textures; }
    uint32_t getNumTextures() const { return static_cast<uint32_t>(m_textures.size()); }

    // 清空纹理数据
    void clear();

private:
    // Taskflow 调度器
    TaskScheduler* m_scheduler;

    // 主机端纹理数据
    struct TextureData {
        std::vector<float4> pixels;
        uint32_t width;
        uint32_t height;
        CUdeviceptr d_pixels;  // 设备端像素缓冲
    };
    std::vector<TextureData> m_textures;

    // 设备端纹理数组缓冲
    CUdeviceptr m_d_textures;

    // 内部方法
    void freeDeviceBuffers();

    // 禁止拷贝和赋值
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;
};

} // namespace optixw
