// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <optixw/types.h>
#include <cuda.h>
#include <vector>
#include <cstdint>

namespace optixw {

// 前向声明（这些类型在 types.h 中定义）
struct PointLightData;
struct AreaLightData;

// LightManager - CPU 层光源管理器
// 职责：
//   - 管理点光源、面光源
//   - 管理环境贴图
//   - 上传光源数据到 GPU
// 注意：
//   - 这是 CPU 层代码，不包含 GPU 计算逻辑
class LightManager {
public:
    LightManager();
    ~LightManager();

    // 添加点光源
    void addPointLight(const PointLightData& light);

    // 添加面光源
    void addAreaLight(const AreaLightData& light);

    // 设置环境光
    void setEnvironmentRadiance(const Vec3& radiance);

    // 设置环境贴图
    // pixels: 像素数据 (RGBA float4)
    // width, height: 尺寸
    // scale: 缩放系数
    void setEnvironmentMap(const float* pixels, uint32_t width, uint32_t height, float scale = 1.0f);

    // 上传光源数据到 GPU
    void uploadToDevice();

    // 访问器
    CUdeviceptr getPointLightsBuffer() const { return m_d_pointLights; }
    CUdeviceptr getAreaLightsBuffer() const { return m_d_areaLights; }
    CUdeviceptr getEnvironmentMapBuffer() const { return m_d_environmentMap; }
    uint32_t getNumPointLights() const { return static_cast<uint32_t>(m_pointLights.size()); }
    uint32_t getNumAreaLights() const { return static_cast<uint32_t>(m_areaLights.size()); }
    const Vec3& getEnvironmentRadiance() const { return m_environmentRadiance; }
    uint32_t getEnvironmentMapWidth() const { return m_envMapWidth; }
    uint32_t getEnvironmentMapHeight() const { return m_envMapHeight; }
    float getEnvironmentMapScale() const { return m_envMapScale; }

    // 清空光源数据
    void clear();

private:
    // 主机端光源数据
    std::vector<PointLightData> m_pointLights;
    std::vector<AreaLightData> m_areaLights;
    Vec3 m_environmentRadiance;

    // 环境贴图数据
    std::vector<float> m_environmentMapPixels;
    uint32_t m_envMapWidth;
    uint32_t m_envMapHeight;
    float m_envMapScale;

    // 设备端缓冲
    CUdeviceptr m_d_pointLights;
    CUdeviceptr m_d_areaLights;
    CUdeviceptr m_d_environmentMap;

    // 内部方法
    void freeDeviceBuffers();

    // 禁止拷贝和赋值
    LightManager(const LightManager&) = delete;
    LightManager& operator=(const LightManager&) = delete;
};

} // namespace optixw
