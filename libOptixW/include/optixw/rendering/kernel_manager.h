// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

namespace optixw {

// KernelManager - CPU 层 CUDA 核管理器
// 职责：
//   - 加载和管理 CUDA 核模块（CUBIN 文件）
//   - 提供核函数句柄
//   - 提供核启动接口（封装 cuLaunchKernel）
// 注意：
//   - 这是 CPU 层代码，不包含 GPU 计算逻辑
//   - 使用 CUDA Driver API
class KernelManager {
public:
    KernelManager();
    ~KernelManager();

    // 加载所有 Wavefront 核
    void loadKernels();

    // 卸载所有核
    void unloadKernels();

    // 启动着色核（CPU 端调用，GPU 端执行）
    // params: 核参数指针
    // gridDim: 网格维度
    // blockDim: 块维度
    // stream: CUDA 流（0 表示默认流）
    void launchShade(void** params, dim3 gridDim, dim3 blockDim, CUstream stream = 0);

    // 启动压缩核
    void launchCompact(void** params, dim3 gridDim, dim3 blockDim, CUstream stream = 0);

    // 启动缩放核
    void launchScale(void** params, dim3 gridDim, dim3 blockDim, CUstream stream = 0);

    // 启动降噪合并核
    void launchMergeTile(void** params, dim3 gridDim, dim3 blockDim, CUstream stream = 0);

    // 启动归一化核
    void launchNormalizeAccum(void** params, dim3 gridDim, dim3 blockDim, CUstream stream = 0);

    // 获取核句柄（供高级用户使用）
    CUfunction getShadeKernel() const { return m_shadeKernel; }
    CUfunction getCompactKernel() const { return m_compactKernel; }
    CUfunction getScaleKernel() const { return m_scaleKernel; }
    CUfunction getMergeTileKernel() const { return m_mergeKernel; }
    CUfunction getNormalizeAccumKernel() const { return m_normalizeKernel; }

    // 检查核是否已加载
    bool isLoaded() const { return m_loaded; }

private:
    // 核模块
    CUmodule m_shadeModule;
    CUmodule m_compactModule;
    CUmodule m_scaleModule;
    CUmodule m_mergeModule;

    // 核函数句柄
    CUfunction m_shadeKernel;
    CUfunction m_compactKernel;
    CUfunction m_scaleKernel;
    CUfunction m_mergeKernel;
    CUfunction m_normalizeKernel;

    // 加载状态
    bool m_loaded;

    // 禁止拷贝和赋值
    KernelManager(const KernelManager&) = delete;
    KernelManager& operator=(const KernelManager&) = delete;
};

} // namespace optixw
