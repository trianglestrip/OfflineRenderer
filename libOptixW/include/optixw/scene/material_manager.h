// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <optixw/types.h>
#include <optixw/core/math_types.h>
#include <cuda.h>
#include <vector>
#include <cstdint>

namespace optixw {

// MaterialManager - CPU 层材质数据管理器
// 职责：
//   - 管理材质数据
//   - 上传材质数据到 GPU
//   - 提供材质访问接口
// 注意：
//   - 这是 CPU 层代码，不包含 GPU 计算逻辑
class MaterialManager {
public:
    MaterialManager();
    ~MaterialManager();

    // 添加 Matte 材质（漫反射）
    uint32_t addMatteMaterial(const Vec3& albedo);

    // 添加 UE4 材质（PBR）
    uint32_t addUE4Material(
        const Vec3& albedo,
        float roughness,
        float metallic);

    // 添加镜面反射材质
    uint32_t addSpecularReflectionMaterial(const Vec3& albedo);

    // 添加微表面反射材质
    uint32_t addMicrofacetReflectionMaterial(
        const Vec3& albedo,
        float roughness);

    // 添加发光材质
    uint32_t addDiffuseEmitterMaterial(const Vec3& emission);

    // 设置材质的反照率纹理
    void setMaterialAlbedoTexture(uint32_t materialId, uint32_t textureId);

    // 上传材质数据到 GPU
    void uploadToDevice();

    // 访问器
    CUdeviceptr getDeviceBuffer() const { return m_d_materials; }
    uint32_t getNumMaterials() const { return static_cast<uint32_t>(m_materials.size()); }
    const MaterialData& getMaterial(uint32_t id) const { return m_materials[id]; }

    // 清空材质数据
    void clear();

private:
    // 主机端材质数据
    std::vector<MaterialData> m_materials;

    // 设备端缓冲
    CUdeviceptr m_d_materials;

    // 内部方法
    uint32_t addMaterial(const MaterialData& mat);
    void freeDeviceBuffer();

    // 禁止拷贝和赋值
    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;
};

} // namespace optixw
