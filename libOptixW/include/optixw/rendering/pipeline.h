// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace optixw {

// Pipeline - CPU 层 OptiX 管线管理器
// 职责：
//   - 创建和管理 OptiX 管线
//   - 管理 SBT (Shader Binding Table)
//   - 管理程序组 (Program Groups)
//   - 提供 OptiX Launch 接口
// 注意：
//   - 这是 CPU 层代码，不包含 GPU 计算逻辑
//   - 管线创建需要 OptiX 上下文
class Pipeline {
public:
    explicit Pipeline(OptixDeviceContext context);
    ~Pipeline();

    // 创建 OptiX 管线
    void create();

    // 销毁管线
    void destroy();

    // 启动 OptiX 管线（CPU 端调用，GPU 端执行）
    // params: Launch Params 设备指针
    // width, height: 启动维度
    // stream: CUDA 流（0 表示默认流）
    void launch(CUdeviceptr params, uint32_t width, uint32_t height, CUstream stream = 0);

    // 访问器
    OptixPipeline getPipeline() const { return m_pipeline; }
    const OptixShaderBindingTable& getSBT() const { return m_sbt; }
    bool isCreated() const { return m_created; }

private:
    // OptiX 上下文
    OptixDeviceContext m_context;

    // OptiX 管线
    OptixPipeline m_pipeline;
    OptixModule m_traceModule;

    // 程序组
    OptixProgramGroup m_raygenPG;
    OptixProgramGroup m_missPG;
    OptixProgramGroup m_hitgroupPG;

    // SBT
    OptixShaderBindingTable m_sbt;
    CUdeviceptr m_sbtRaygenRecord;
    CUdeviceptr m_sbtMissRecord;
    CUdeviceptr m_sbtHitgroupRecord;

    // 创建状态
    bool m_created;

    // 内部方法
    void createModule();
    void createProgramGroups();
    void linkPipeline();
    void buildSBT();

    // 禁止拷贝和赋值
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
};

} // namespace optixw
