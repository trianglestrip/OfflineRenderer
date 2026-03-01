// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/rendering/kernel_manager.h>
#include "../utils/checks.h"
#include "../utils/path_utils.h"
#include <iostream>

namespace optixw {

KernelManager::KernelManager()
    : m_modules({nullptr, nullptr, nullptr, nullptr})
    , m_functions({nullptr, nullptr, nullptr, nullptr, nullptr})
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
    CU_CHECK(cuModuleLoad(&m_modules.shadeModule, shadeCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&m_functions.shadeKernel, m_modules.shadeModule, "shade"));

    // 加载压缩核
    const std::string compactCubin = utils::findCubinPath("compact.cubin").string();
    CU_CHECK(cuModuleLoad(&m_modules.compactModule, compactCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&m_functions.compactKernel, m_modules.compactModule, "compact"));

    // 加载缩放核
    const std::string scaleCubin = utils::findCubinPath("scale.cubin").string();
    CU_CHECK(cuModuleLoad(&m_modules.scaleModule, scaleCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&m_functions.scaleKernel, m_modules.scaleModule, "scale_to_float4"));

    // 加载降噪合并核
    const std::string mergeCubin = utils::findCubinPath("denoise_merge.cubin").string();
    CU_CHECK(cuModuleLoad(&m_modules.mergeModule, mergeCubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&m_functions.mergeKernel, m_modules.mergeModule, "merge_tile"));
    CU_CHECK(cuModuleGetFunction(&m_functions.normalizeKernel, m_modules.mergeModule, "normalize_accum"));

    m_loaded = true;
    std::cout << "[KernelManager] Wavefront kernels loaded" << std::endl;
}

void KernelManager::unloadKernels() {
    if (!m_loaded) return;

    // 卸载模块
    if (m_modules.shadeModule) {
        CU_CHECK(cuModuleUnload(m_modules.shadeModule));
        m_modules.shadeModule = nullptr;
    }
    if (m_modules.compactModule) {
        CU_CHECK(cuModuleUnload(m_modules.compactModule));
        m_modules.compactModule = nullptr;
    }
    if (m_modules.scaleModule) {
        CU_CHECK(cuModuleUnload(m_modules.scaleModule));
        m_modules.scaleModule = nullptr;
    }
    if (m_modules.mergeModule) {
        CU_CHECK(cuModuleUnload(m_modules.mergeModule));
        m_modules.mergeModule = nullptr;
    }

    // 清空核句柄
    m_functions.shadeKernel = nullptr;
    m_functions.compactKernel = nullptr;
    m_functions.scaleKernel = nullptr;
    m_functions.mergeKernel = nullptr;
    m_functions.normalizeKernel = nullptr;

    m_loaded = false;
    std::cout << "[KernelManager] Kernels unloaded" << std::endl;
}

void KernelManager::launchShade(void** params, dim3 gridDim, dim3 blockDim, CUstream stream) {
    CU_CHECK(cuLaunchKernel(
        m_functions.shadeKernel,
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
        m_functions.compactKernel,
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
        m_functions.scaleKernel,
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
        m_functions.mergeKernel,
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
        m_functions.normalizeKernel,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,
        stream,
        params,
        nullptr
    ));
}

} // namespace optixw
