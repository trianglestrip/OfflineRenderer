// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/core/buffer_manager.h>
#include "../utils/checks.h"
#include <cstring>

namespace optixw {

BufferManager::BufferManager()
    : m_width(0)
    , m_height(0)
    , m_maxBounces(0)
    , m_numPixels(0)
    , m_maxRays(0)
    , m_d_outputBuffer(0)
    , m_d_rayPool(0)
    , m_d_activeIndices(0)
    , m_d_compactIndices(0)
    , m_d_accumBuffer(0)
    , m_d_hitBuffer(0)
    , m_d_compactCounter(0)
    , m_d_albedoBuffer(0)
    , m_d_normalBuffer(0)
    , m_d_launchParams(0)
    , m_d_materials(0)
    , m_d_texcoords(0)
    , m_d_textures(0)
    , m_d_environmentMap(0)
    , m_d_denoiserState(0)
    , m_d_denoiserScratch(0)
    , m_d_denoiserInput(0)
    , m_d_denoiserOutput(0)
    , m_d_tileBuffer(0)
    , m_d_tileInputBuffer(0)
    , m_d_tileAlbedoInput(0)
    , m_d_tileNormalInput(0)
    , m_d_denoiserIntensity(0)
    , m_d_denoiserAlbedoInput(0)
    , m_d_denoiserNormalInput(0)
    , m_d_mergeAccum(0)
    , m_d_mergeWeight(0)
{
}

BufferManager::~BufferManager() {
    free();
}

void BufferManager::allocate(uint32_t width, uint32_t height, uint32_t maxBounces) {
    // 先释放旧缓冲
    free();

    m_width = width;
    m_height = height;
    m_maxBounces = maxBounces;
    m_numPixels = width * height;
    m_maxRays = m_numPixels * (maxBounces + 1);

    // 分配渲染缓冲
    allocateBuffer(m_d_outputBuffer, m_numPixels * sizeof(float) * 4);  // RGBA
    allocateBuffer(m_d_rayPool, m_maxRays * 64);  // 每条光线约 64 字节（根据实际结构调整）
    allocateBuffer(m_d_activeIndices, m_maxRays * sizeof(uint32_t));
    allocateBuffer(m_d_compactIndices, m_maxRays * sizeof(uint32_t));
    allocateBuffer(m_d_accumBuffer, m_numPixels * sizeof(float) * 3);  // RGB 累加
    allocateBuffer(m_d_hitBuffer, m_maxRays * 64);  // 命中信息（根据实际结构调整）
    allocateBuffer(m_d_compactCounter, sizeof(uint32_t));

    // 分配降噪引导缓冲
    allocateBuffer(m_d_albedoBuffer, m_numPixels * sizeof(float) * 3);  // RGB
    allocateBuffer(m_d_normalBuffer, m_numPixels * sizeof(float) * 3);  // XYZ

    // 分配 Launch Params 缓冲
    allocateBuffer(m_d_launchParams, 4096);  // 足够大的缓冲（根据实际结构调整）
}

void BufferManager::resize(uint32_t width, uint32_t height) {
    // 重新分配缓冲
    allocate(width, height, m_maxBounces);
}

void BufferManager::free() {
    // 释放渲染缓冲
    freeBuffer(m_d_outputBuffer);
    freeBuffer(m_d_rayPool);
    freeBuffer(m_d_activeIndices);
    freeBuffer(m_d_compactIndices);
    freeBuffer(m_d_accumBuffer);
    freeBuffer(m_d_hitBuffer);
    freeBuffer(m_d_compactCounter);
    freeBuffer(m_d_albedoBuffer);
    freeBuffer(m_d_normalBuffer);
    freeBuffer(m_d_launchParams);

    // 注意：场景数据缓冲和降噪器缓冲由其他模块管理，这里不释放
    m_d_materials = 0;
    m_d_texcoords = 0;
    m_d_textures = 0;
    m_d_environmentMap = 0;
    m_d_denoiserState = 0;
    m_d_denoiserScratch = 0;
    m_d_denoiserInput = 0;
    m_d_denoiserOutput = 0;
    m_d_tileBuffer = 0;
    m_d_tileInputBuffer = 0;
    m_d_tileAlbedoInput = 0;
    m_d_tileNormalInput = 0;
    m_d_denoiserIntensity = 0;
    m_d_denoiserAlbedoInput = 0;
    m_d_denoiserNormalInput = 0;
    m_d_mergeAccum = 0;
    m_d_mergeWeight = 0;

    m_width = 0;
    m_height = 0;
    m_numPixels = 0;
    m_maxRays = 0;
}

void BufferManager::uploadData(CUdeviceptr dst, const void* src, size_t size) {
    CU_CHECK(cuMemcpyHtoD(dst, src, size));
}

void BufferManager::downloadData(void* dst, CUdeviceptr src, size_t size) {
    CU_CHECK(cuMemcpyDtoH(dst, src, size));
}

void BufferManager::clearBuffer(CUdeviceptr buffer, size_t size, int value) {
    CU_CHECK(cuMemsetD8(buffer, value, size));
}

void BufferManager::allocateBuffer(CUdeviceptr& buffer, size_t size) {
    if (size == 0) return;
    CU_CHECK(cuMemAlloc(&buffer, size));
}

void BufferManager::freeBuffer(CUdeviceptr& buffer) {
    if (buffer != 0) {
        CU_CHECK(cuMemFree(buffer));
        buffer = 0;
    }
}

} // namespace optixw
