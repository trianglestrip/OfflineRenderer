// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/scene/material_manager.h>
#include <optixw/core/math_types.h>
#include "../utils/checks.h"
#include <cstring>
#include <cuda_runtime.h>

namespace optixw {

// 材质类型常量（与 wavefront_kernel_params.h 中的 MaterialTag 一致）
namespace {
    constexpr uint32_t kMatte = 0;
    constexpr uint32_t kLambertianScattering = 1;
    constexpr uint32_t kSpecularReflection = 2;
    constexpr uint32_t kSpecularScattering = 3;
    constexpr uint32_t kMicrofacetReflection = 4;
    constexpr uint32_t kMicrofacetScattering = 5;
    constexpr uint32_t kUE4 = 6;
    constexpr uint32_t kOldStyle = 7;
    constexpr uint32_t kDiffuseEmitter = 8;
    constexpr uint32_t kDirectionalEmitter = 9;
    constexpr uint32_t kPointEmitter = 10;
    constexpr uint32_t kMulti = 11;
    constexpr uint32_t kEnvironmentEmitter = 12;

    inline float3 toFloat3(const Vec3& v) {
        return make_float3(v.x, v.y, v.z);
    }
}

MaterialManager::MaterialManager()
    : m_d_materials(0)
{
}

MaterialManager::~MaterialManager() {
    clear();
}

uint32_t MaterialManager::addMatteMaterial(const Vec3& albedo) {
    MaterialData mat = {};
    mat.type = kMatte;
    mat.baseColor = toFloat3(albedo);
    mat.emission = make_float3(0.0f, 0.0f, 0.0f);
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.iorInt = 1.5f;
    mat.baseColorTextureId = 0xFFFFFFFF;
    return addMaterial(mat);
}

uint32_t MaterialManager::addUE4Material(const Vec3& albedo, float roughness, float metallic) {
    MaterialData mat = {};
    mat.type = kUE4;
    mat.baseColor = toFloat3(albedo);
    mat.emission = make_float3(0.0f, 0.0f, 0.0f);
    mat.roughness = roughness;
    mat.metallic = metallic;
    mat.iorInt = 1.5f;
    mat.baseColorTextureId = 0xFFFFFFFF;
    return addMaterial(mat);
}

uint32_t MaterialManager::addSpecularReflectionMaterial(const Vec3& albedo) {
    MaterialData mat = {};
    mat.type = kSpecularReflection;
    mat.baseColor = toFloat3(albedo);
    mat.emission = make_float3(0.0f, 0.0f, 0.0f);
    mat.roughness = 0.0f;
    mat.metallic = 1.0f;
    mat.iorInt = 1.5f;
    mat.baseColorTextureId = 0xFFFFFFFF;
    return addMaterial(mat);
}

uint32_t MaterialManager::addMicrofacetReflectionMaterial(const Vec3& albedo, float roughness) {
    MaterialData mat = {};
    mat.type = kMicrofacetReflection;
    mat.baseColor = toFloat3(albedo);
    mat.emission = make_float3(0.0f, 0.0f, 0.0f);
    mat.roughness = roughness;
    mat.metallic = 1.0f;
    mat.iorInt = 1.5f;
    mat.baseColorTextureId = 0xFFFFFFFF;
    return addMaterial(mat);
}

uint32_t MaterialManager::addDiffuseEmitterMaterial(const Vec3& emission) {
    MaterialData mat = {};
    mat.type = kDiffuseEmitter;
    mat.baseColor = make_float3(0.0f, 0.0f, 0.0f);
    mat.emission = toFloat3(emission);
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.iorInt = 1.0f;
    mat.baseColorTextureId = 0xFFFFFFFF;
    return addMaterial(mat);
}

void MaterialManager::setMaterialAlbedoTexture(uint32_t materialId, uint32_t textureId) {
    if (materialId >= m_materials.size()) {
        throw std::runtime_error("Invalid material ID");
    }
    m_materials[materialId].baseColorTextureId = textureId;
}

void MaterialManager::uploadToDevice() {
    if (m_materials.empty()) return;

    // 释放旧缓冲
    freeDeviceBuffer();

    // 分配新缓冲
    const size_t size = m_materials.size() * sizeof(MaterialData);
    CU_CHECK(cuMemAlloc(&m_d_materials, size));

    // 上传数据
    CU_CHECK(cuMemcpyHtoD(m_d_materials, m_materials.data(), size));
}

void MaterialManager::clear() {
    m_materials.clear();
    freeDeviceBuffer();
}

uint32_t MaterialManager::addMaterial(const MaterialData& mat) {
    const uint32_t id = static_cast<uint32_t>(m_materials.size());
    m_materials.push_back(mat);
    return id;
}

void MaterialManager::freeDeviceBuffer() {
    if (m_d_materials != 0) {
        CU_CHECK(cuMemFree(m_d_materials));
        m_d_materials = 0;
    }
}

} // namespace optixw
