// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/rendering/kernel_manager.h>
#include "../utils/checks.h"
#include "../utils/path_utils.h"
#include <iostream>

namespace optixw {

KernelManager::KernelManager()
    : m_shadeModule(nullptr)
    , m_compactModule(nullptr)
    , m_scaleModule(nullptr)
    , m_mergeModule(nullptr)
    , m_shadeKernel(nullptr)
    , m_compactKernel(nullptr)
    , m_scaleKernel(nullptr)
    , m_mergeKernel(nullptr)
    , m_normalizeKernel(nullptr)
    , m_loaded(false)
{
}

KernelManager::~KernelManager() {
    unloadKernels();
}

void KernelManager::loadKernels() {
    if (m_loaded) {
        std::cout << "[KernelManager] Kernels already loaded, skipping" << std::endl;
        return;
    }

    std::cout << "[KernelManager] Loading Wavefront kernels..." << std::endl;

    // 加载着色核
    const std::string shadeCubin = utils::findCubinPath("shade.cubin").string();
    CU_CHECK(cuModuleLoad(&m_shadeModule, shadeCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&m_shadeKernel, m_shadeModule, "shade"));

    // 加载压缩核
    const std::string compactCubin = utils::findCubinPath("compact.cubin").string();
    CU_CHECK(cuModuleLoad(&m_compactModule, compactCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&m_compactKernel, m_compactModule, "compact"));

    // 加载缩放核
    const std::string scaleCubin = utils::findCubinPath("scale.cubin").string();
    CU_CHECK(cuModuleLoad(&m_scaleModule, scaleCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&m_scaleKernel, m_scaleModule, "scale_to_float4"));

    // 加载降噪合并核
    const std::string mergeCubin = utils::findCubinPath("denoise_merge.cubin").string();
    CU_CHECK(cuModuleLoad(&m_mergeModule, mergeCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&m_mergeKernel, m_mergeModule, "merge_tile"));
    CU_CHECK(cuModuleGetFunction(&m_normalizeKernel, m_mergeModule, "normalize_accum"));

    m_loaded = true;
    std::cout << "[KernelManager] Wavefront kernels loaded" << std::endl;
}

void KernelManager::unloadKernels() {
    if (!m_loaded) return;

    // 卸载模块
    if (m_shadeModule) {
        CU_CHECK(cuModuleUnload(m_shadeModule));
        m_shadeModule = nullptr;
    }
    if (m_compactModule) {
        CU_CHECK(cuModuleUnload(m_compactModule));
        m_compactModule = nullptr;
    }
    if (m_scaleModule) {
        CU_CHECK(cuModuleUnload(m_scaleModule));
        m_scaleModule = nullptr;
    }
    if (m_mergeModule) {
        CU_CHECK(cuModuleUnload(m_mergeModule));
        m_mergeModule = nullptr;
    }

    // 清空核句柄
    m_shadeKernel = nullptr;
    m_compactKernel = nullptr;
    m_scaleKernel = nullptr;
    m_mergeKernel = nullptr;
    m_normalizeKernel = nullptr;

    m_loaded = false;
    std::cout << "[KernelManager] Kernels unloaded" << std::endl;
}

void KernelManager::launchShade(void** params, dim3 gridDim, dim3 blockDim, CUstream stream) {
    CU_CHECK(cuLaunchKernel(
        m_shadeKernel,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,  // 共享内存大小
        stream,
        params,
        nullptr
    ));
}

void KernelManager::launchCompact(void** params, dim3 gridDim, dim3 blockDim, CUstream stream) {
    CU_CHECK(cuLaunchKernel(
        m_compactKernel,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,
        stream,
        params,
        nullptr
    ));
}

void KernelManager::launchScale(void** params, dim3 gridDim, dim3 blockDim, CUstream stream) {
    CU_CHECK(cuLaunchKernel(
        m_scaleKernel,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,
        stream,
        params,
        nullptr
    ));
}

void KernelManager::launchMergeTile(void** params, dim3 gridDim, dim3 blockDim, CUstream stream) {
    CU_CHECK(cuLaunchKernel(
        m_mergeKernel,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,
        stream,
        params,
        nullptr
    ));
}

void KernelManager::launchNormalizeAccum(void** params, dim3 gridDim, dim3 blockDim, CUstream stream) {
    CU_CHECK(cuLaunchKernel(
        m_normalizeKernel,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,
        stream,
        params,
        nullptr
    ));
}

} // namespace optixw
