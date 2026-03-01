// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/scene/light_manager.h>
#include "../utils/checks.h"
#include <iostream>

namespace optixw {

LightManager::LightManager()
    : m_environmentRadiance{0.0f, 0.0f, 0.0f}
    , m_envMapWidth(0)
    , m_envMapHeight(0)
    , m_envMapScale(1.0f)
    , m_d_pointLights(0)
    , m_d_areaLights(0)
    , m_d_environmentMap(0)
{
}

LightManager::~LightManager() {
    clear();
}

void LightManager::addPointLight(const PointLightData& light) {
    m_pointLights.push_back(light);
}

void LightManager::addAreaLight(const AreaLightData& light) {
    m_areaLights.push_back(light);
}

void LightManager::setEnvironmentRadiance(const Vec3& radiance) {
    m_environmentRadiance = radiance;
}

void LightManager::setEnvironmentMap(const float* pixels, uint32_t width, uint32_t height, float scale) {
    m_envMapWidth = width;
    m_envMapHeight = height;
    m_envMapScale = scale;

    // 复制像素数据（RGBA float4 格式）
    const size_t numPixels = static_cast<size_t>(width) * height;
    m_environmentMapPixels.resize(numPixels * 4);
    std::memcpy(m_environmentMapPixels.data(), pixels, numPixels * 4 * sizeof(float));

    std::cout << "[LightManager] 设置环境贴图: " << width << "x" << height << std::endl;
}

void LightManager::uploadToDevice() {
    std::cout << "[LightManager] 上传光源数据到 GPU..." << std::endl;

    // 上传点光源
    if (!m_pointLights.empty()) {
        const size_t size = m_pointLights.size() * sizeof(PointLightData);
        if (m_d_pointLights) {
            CU_CHECK(cuMemFree(m_d_pointLights));
        }
        CU_CHECK(cuMemAlloc(&m_d_pointLights, size));
        CU_CHECK(cuMemcpyHtoD(m_d_pointLights, m_pointLights.data(), size));
    }

    // 上传面光源
    if (!m_areaLights.empty()) {
        const size_t size = m_areaLights.size() * sizeof(AreaLightData);
        if (m_d_areaLights) {
            CU_CHECK(cuMemFree(m_d_areaLights));
        }
        CU_CHECK(cuMemAlloc(&m_d_areaLights, size));
        CU_CHECK(cuMemcpyHtoD(m_d_areaLights, m_areaLights.data(), size));
    }

    // 上传环境贴图
    if (!m_environmentMapPixels.empty()) {
        const size_t size = m_environmentMapPixels.size() * sizeof(float);
        if (m_d_environmentMap) {
            CU_CHECK(cuMemFree(m_d_environmentMap));
        }
        CU_CHECK(cuMemAlloc(&m_d_environmentMap, size));
        CU_CHECK(cuMemcpyHtoD(m_d_environmentMap, m_environmentMapPixels.data(), size));
    }

    std::cout << "[LightManager] Light data uploaded (point lights: " << m_pointLights.size() 
              << ", area lights: " << m_areaLights.size() << ")" << std::endl;
}

void LightManager::clear() {
    m_pointLights.clear();
    m_areaLights.clear();
    m_environmentMapPixels.clear();
    m_environmentRadiance = Vec3{0.0f, 0.0f, 0.0f};
    m_envMapWidth = 0;
    m_envMapHeight = 0;
    m_envMapScale = 1.0f;

    freeDeviceBuffers();
}

void LightManager::freeDeviceBuffers() {
    if (m_d_pointLights) {
        CU_CHECK(cuMemFree(m_d_pointLights));
        m_d_pointLights = 0;
    }
    if (m_d_areaLights) {
        CU_CHECK(cuMemFree(m_d_areaLights));
        m_d_areaLights = 0;
    }
    if (m_d_environmentMap) {
        CU_CHECK(cuMemFree(m_d_environmentMap));
        m_d_environmentMap = 0;
    }
}

} // namespace optixw
