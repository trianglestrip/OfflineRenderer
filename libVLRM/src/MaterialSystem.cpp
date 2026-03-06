#include "../include/VLRM/VLRM.h"
#include "../include/VLRM/common.h"
#include "../include/VLRM/basic_types.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <vector>
#include <memory>

namespace vlrm {

// 材质评估函数实现
namespace material_eval {

    // Lambertian漫反射材质
    RGB evaluateLambertian(
        const RGB& albedo,
        const Normal3D& normal,
        const Vector3D& incident_dir,
        const Vector3D& outgoing_dir
    ) {
        // Lambert余弦定律：散射系数 = albedo / π
        // 但因为蒙特卡洛积分中会抵消π项，所以简化为albedo * inv_pi
        VLRM_UNUSED(normal);
        VLRM_UNUSED(incident_dir);
        VLRM_UNUSED(outgoing_dir);
        
        return albedo * VLRM_INV_PI;
    }

    // 镜面反射材质
    Vector3D reflect(const Vector3D& incident, const Normal3D& normal) {
        return incident - 2.0f * dot(incident, normal) * normal;
    }

    RGB evaluateMirror(
        const RGB& reflectance,
        const Normal3D& normal,
        const Vector3D& incident_dir,
        const Vector3D& outgoing_dir
    ) {
        Vector3D reflected_dir = reflect(-incident_dir, normal);
        
        // 检查出射方向是否接近镜面反射方向
        float cos_theta = dot(normalize(outgoing_dir), normalize(reflected_dir));
        
        // 使用一个小的容差值来判断方向是否一致
        if (cos_theta > 0.999f) {
            return reflectance;
        } else {
            return RGB(0.0f, 0.0f, 0.0f);
        }
    }

    // 玻璃/电介质材质（折射+反射）
    RGB evaluateGlass(
        const RGB& transmission,
        const RGB& reflection,
        float eta_i,  // 入射介质折射率
        float eta_t,  // 透射介质折射率
        const Normal3D& normal,
        const Vector3D& incident_dir,  // 从表面指向光源的方向
        const Vector3D& outgoing_dir   // 从表面指向观察者的方向
    ) {
        VLRM_UNUSED(transmission);
        VLRM_UNUSED(reflection);
        VLRM_UNUSED(eta_i);
        VLRM_UNUSED(eta_t);
        VLRM_UNUSED(normal);
        VLRM_UNUSED(incident_dir);
        VLRM_UNUSED(outgoing_dir);
        
        // TODO: 实现菲涅尔反射和斯内尔折射
        // 这里先返回一个简单的近似值
        return RGB(0.2f, 0.8f, 0.9f);  // 浅蓝色表示玻璃材质
    }

    // 计算菲涅尔反射系数（Schlick近似）
    float fresnelSchlick(float cos_theta, float eta_i, float eta_t) {
        float r0 = (eta_i - eta_t) / (eta_i + eta_t);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
    }

    // 斯内尔定律计算折射方向
    Vector3D refract(const Vector3D& incident, const Normal3D& normal, float eta_ratio) {
        float cos_i = dot(-incident, normal);
        float sin_t2 = eta_ratio * eta_ratio * (1.0f - cos_i * cos_i);
        
        if (sin_t2 > 1.0f) {
            // 全内反射
            return Vector3D(0.0f, 0.0f, 0.0f);
        }
        
        float cos_t = sqrtf(1.0f - sin_t2);
        Vector3D refracted = eta_ratio * incident + (eta_ratio * cos_i - cos_t) * normal;
        return normalize(refracted);
    }

} // namespace material_eval

} // namespace vlrm