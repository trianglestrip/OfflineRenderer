// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cstdint>

namespace optixw {

// Denoiser - CPU 层 OptiX 降噪器管理
// 职责：
//   - 创建和管理 OptiX Denoiser
//   - 管理降噪所需的缓冲
//   - 提供降噪接口（支持分块降噪）
// 注意：
//   - 这是 CPU 层代码，不包含 GPU 计算逻辑
//   - 降噪器需要 OptiX 上下文
class Denoiser {
public:
    explicit Denoiser(OptixDeviceContext context);
    ~Denoiser();

    // 设置降噪器
    // width, height: 图像分辨率
    // useAlbedo: 是否使用反照率引导
    // useNormal: 是否使用法线引导
    void setup(uint32_t width, uint32_t height, bool useAlbedo = true, bool useNormal = true);

    // 执行降噪（整图）
    // inputColor: 输入颜色缓冲
    // inputAlbedo: 输入反照率缓冲（可选）
    // inputNormal: 输入法线缓冲（可选）
    // output: 输出缓冲
    // stream: CUDA 流
    void denoise(
        CUdeviceptr inputColor,
        CUdeviceptr inputAlbedo,
        CUdeviceptr inputNormal,
        CUdeviceptr output,
        CUstream stream = 0);

    // 执行分块降噪（用于大图像）
    // tileWidth, tileHeight: 分块大小
    // overlap: 分块重叠像素数
    void denoiseTiled(
        CUdeviceptr inputColor,
        CUdeviceptr inputAlbedo,
        CUdeviceptr inputNormal,
        CUdeviceptr output,
        uint32_t tileWidth,
        uint32_t tileHeight,
        uint32_t overlap,
        CUstream stream = 0);

    // 销毁降噪器
    void destroy();

    // 访问器
    bool isSetup() const { return m_setup; }
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    
    // 获取降噪器缓冲（供外部使用）
    CUdeviceptr getStateBuffer() const { return m_d_state; }
    CUdeviceptr getScratchBuffer() const { return m_d_scratch; }
    CUdeviceptr getIntensityBuffer() const { return m_d_intensity; }
    
    // ==================== 结构体版本的方法 ====================
    void setup(const DenoiserSetupParams& params);
    void denoise(const DenoiserParams& params);
    void denoiseTiled(const DenoiserTiledParams& params);

private:
    // OptiX 上下文
    OptixDeviceContext m_context;

    // OptiX Denoiser
    OptixDenoiser m_denoiser;

    // 降噪器缓冲
    CUdeviceptr m_d_state;
    CUdeviceptr m_d_scratch;
    CUdeviceptr m_d_intensity;
    size_t m_stateSize;
    size_t m_scratchSize;

    // 分块降噪临时缓冲
    CUdeviceptr m_d_tileBuffer;
    CUdeviceptr m_d_tileInputBuffer;
    CUdeviceptr m_d_tileAlbedoBuffer;
    CUdeviceptr m_d_tileNormalBuffer;

    // 降噪器参数
    uint32_t m_width;
    uint32_t m_height;
    bool m_useAlbedo;
    bool m_useNormal;
    bool m_setup;

    // 内部方法
    void allocateBuffers();
    void freeBuffers();

    // 禁止拷贝和赋值
    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;
};

} // namespace optixw
