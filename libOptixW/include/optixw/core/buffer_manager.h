// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <cuda.h>
#include <cstdint>
#include <vector>

namespace optixw {

// BufferManager - CPU 层 GPU 缓冲管理器
// 职责：
//   - 管理所有 GPU 端缓冲的分配、释放
//   - 提供 CPU/GPU 数据传输接口
//   - 管理缓冲生命周期
// 注意：
//   - 这是 CPU 层代码，不包含 GPU 计算逻辑
//   - 使用 CUDA Driver API (CUdeviceptr)
class BufferManager {
public:
    BufferManager();
    ~BufferManager();

    // 分配渲染所需的所有缓冲
    // width, height: 渲染分辨率
    // maxBounces: 最大光线弹射次数
    void allocate(uint32_t width, uint32_t height, uint32_t maxBounces);

    // 调整缓冲大小（重新分配）
    void resize(uint32_t width, uint32_t height);

    // 释放所有缓冲
    void free();

    // 数据传输接口
    void uploadData(CUdeviceptr dst, const void* src, size_t size);
    void downloadData(void* dst, CUdeviceptr src, size_t size);
    void clearBuffer(CUdeviceptr buffer, size_t size, int value = 0);

    // 访问器 - 渲染缓冲
    CUdeviceptr getOutputBuffer() const { return m_d_outputBuffer; }
    CUdeviceptr getRayPool() const { return m_d_rayPool; }
    CUdeviceptr getActiveIndices() const { return m_d_activeIndices; }
    CUdeviceptr getCompactIndices() const { return m_d_compactIndices; }
    CUdeviceptr getAccumBuffer() const { return m_d_accumBuffer; }
    CUdeviceptr getHitBuffer() const { return m_d_hitBuffer; }
    CUdeviceptr getCompactCounter() const { return m_d_compactCounter; }

    // 访问器 - 降噪引导缓冲
    CUdeviceptr getAlbedoBuffer() const { return m_d_albedoBuffer; }
    CUdeviceptr getNormalBuffer() const { return m_d_normalBuffer; }

    // 访问器 - 场景数据缓冲
    CUdeviceptr getMaterialsBuffer() const { return m_d_materials; }
    CUdeviceptr getTexcoordsBuffer() const { return m_d_texcoords; }
    CUdeviceptr getTexturesBuffer() const { return m_d_textures; }

    // 访问器 - Launch Params 缓冲
    CUdeviceptr getLaunchParamsBuffer() const { return m_d_launchParams; }

    // 访问器 - 降噪器缓冲
    CUdeviceptr getDenoiserState() const { return m_d_denoiserState; }
    CUdeviceptr getDenoiserScratch() const { return m_d_denoiserScratch; }
    CUdeviceptr getDenoiserInput() const { return m_d_denoiserInput; }
    CUdeviceptr getDenoiserOutput() const { return m_d_denoiserOutput; }
    CUdeviceptr getTileBuffer() const { return m_d_tileBuffer; }
    CUdeviceptr getTileInputBuffer() const { return m_d_tileInputBuffer; }
    CUdeviceptr getTileAlbedoInput() const { return m_d_tileAlbedoInput; }
    CUdeviceptr getTileNormalInput() const { return m_d_tileNormalInput; }
    CUdeviceptr getDenoiserIntensity() const { return m_d_denoiserIntensity; }
    CUdeviceptr getDenoiserAlbedoInput() const { return m_d_denoiserAlbedoInput; }
    CUdeviceptr getDenoiserNormalInput() const { return m_d_denoiserNormalInput; }
    CUdeviceptr getMergeAccum() const { return m_d_mergeAccum; }
    CUdeviceptr getMergeWeight() const { return m_d_mergeWeight; }

    // 访问器 - 环境贴图
    CUdeviceptr getEnvironmentMap() const { return m_d_environmentMap; }

    // 获取缓冲信息
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    uint32_t getNumPixels() const { return m_numPixels; }
    uint32_t getMaxRays() const { return m_maxRays; }

    // 设置场景数据缓冲（由 Scene 管理，这里只保存指针）
    void setMaterialsBuffer(CUdeviceptr buffer) { m_d_materials = buffer; }
    void setTexcoordsBuffer(CUdeviceptr buffer) { m_d_texcoords = buffer; }
    void setTexturesBuffer(CUdeviceptr buffer) { m_d_textures = buffer; }
    void setEnvironmentMap(CUdeviceptr buffer) { m_d_environmentMap = buffer; }

    // 设置降噪器缓冲（由 Denoiser 管理，这里只保存指针）
    void setDenoiserState(CUdeviceptr buffer) { m_d_denoiserState = buffer; }
    void setDenoiserScratch(CUdeviceptr buffer) { m_d_denoiserScratch = buffer; }
    void setDenoiserInput(CUdeviceptr buffer) { m_d_denoiserInput = buffer; }
    void setDenoiserOutput(CUdeviceptr buffer) { m_d_denoiserOutput = buffer; }
    void setTileBuffer(CUdeviceptr buffer) { m_d_tileBuffer = buffer; }
    void setTileInputBuffer(CUdeviceptr buffer) { m_d_tileInputBuffer = buffer; }
    void setTileAlbedoInput(CUdeviceptr buffer) { m_d_tileAlbedoInput = buffer; }
    void setTileNormalInput(CUdeviceptr buffer) { m_d_tileNormalInput = buffer; }
    void setDenoiserIntensity(CUdeviceptr buffer) { m_d_denoiserIntensity = buffer; }
    void setDenoiserAlbedoInput(CUdeviceptr buffer) { m_d_denoiserAlbedoInput = buffer; }
    void setDenoiserNormalInput(CUdeviceptr buffer) { m_d_denoiserNormalInput = buffer; }
    void setMergeAccum(CUdeviceptr buffer) { m_d_mergeAccum = buffer; }
    void setMergeWeight(CUdeviceptr buffer) { m_d_mergeWeight = buffer; }

private:
    // 渲染参数
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_maxBounces;
    uint32_t m_numPixels;
    uint32_t m_maxRays;

    // 渲染缓冲（由 BufferManager 管理）
    CUdeviceptr m_d_outputBuffer;      // 输出图像缓冲
    CUdeviceptr m_d_rayPool;           // 光线池
    CUdeviceptr m_d_activeIndices;     // 活跃光线索引
    CUdeviceptr m_d_compactIndices;    // 压缩后的光线索引
    CUdeviceptr m_d_accumBuffer;       // 累加缓冲
    CUdeviceptr m_d_hitBuffer;         // 命中信息缓冲
    CUdeviceptr m_d_compactCounter;    // 压缩计数器
    CUdeviceptr m_d_albedoBuffer;      // 反照率缓冲（降噪引导）
    CUdeviceptr m_d_normalBuffer;      // 法线缓冲（降噪引导）

    // Launch Params 缓冲
    CUdeviceptr m_d_launchParams;      // OptiX Launch Params

    // 场景数据缓冲（由 Scene 管理，这里只保存指针）
    CUdeviceptr m_d_materials;         // 材质数据
    CUdeviceptr m_d_texcoords;         // 纹理坐标
    CUdeviceptr m_d_textures;          // 纹理数据
    CUdeviceptr m_d_environmentMap;    // 环境贴图

    // 降噪器缓冲（由 Denoiser 管理，这里只保存指针）
    CUdeviceptr m_d_denoiserState;
    CUdeviceptr m_d_denoiserScratch;
    CUdeviceptr m_d_denoiserInput;
    CUdeviceptr m_d_denoiserOutput;
    CUdeviceptr m_d_tileBuffer;
    CUdeviceptr m_d_tileInputBuffer;
    CUdeviceptr m_d_tileAlbedoInput;
    CUdeviceptr m_d_tileNormalInput;
    CUdeviceptr m_d_denoiserIntensity;
    CUdeviceptr m_d_denoiserAlbedoInput;
    CUdeviceptr m_d_denoiserNormalInput;
    CUdeviceptr m_d_mergeAccum;
    CUdeviceptr m_d_mergeWeight;

    // 辅助函数
    void allocateBuffer(CUdeviceptr& buffer, size_t size);
    void freeBuffer(CUdeviceptr& buffer);

    // 禁止拷贝和赋值
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;
};

} // namespace optixw
