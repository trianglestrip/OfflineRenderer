# shade.cu 重构计划

## 🎯 目标

1. **模块化材质处理**：每种材质独立文件
2. **消除代码重复**：提取公共函数到 `.cuh` 头文件
3. **提高可维护性**：新材质只需添加新文件
4. **保持性能**：使用 `__device__ __forceinline__` 避免函数调用开销

---

## 📁 新的文件结构

```
libWR/kernels/
├── shade.cu                    # 主调度器（100行以内）
├── materials/
│   ├── material_common.cuh     # 公共材质函数
│   ├── lambertian.cuh          # Lambertian BSDF
│   ├── glass.cuh               # Glass BSDF (Specular)
│   ├── ggx.cuh                 # GGX Microfacet BRDF（已存在，移动）
│   └── emissive.cuh            # Emissive 材质
├── utils/
│   ├── atomic_ops.cuh          # atomicAddFloat3 等
│   ├── fresnel.cuh             # Fresnel 方程
│   └── ray_offset.cuh          # Self-intersection 避免
├── sampling.cuh                # 采样函数（已存在）
└── vector_math.cuh             # 向量数学（已存在）
```

---

## 🔧 重构步骤

### Step 1: 提取公共工具函数

#### `utils/atomic_ops.cuh`
```cuda
#pragma once
#include <cuda_runtime.h>

namespace wr::internal {

__device__ __forceinline__ void atomicAddFloat3(float3* address, float3 value) {
    atomicAdd(&address->x, value.x);
    atomicAdd(&address->y, value.y);
    atomicAdd(&address->z, value.z);
}

} // namespace wr::internal
```

#### `utils/fresnel.cuh`
```cuda
#pragma once
#include <cuda_runtime.h>

namespace wr::internal {

// Fresnel equations for dielectric materials
__device__ __forceinline__ float fresnelDielectric(float cosI, float etaI, float etaT) {
    float sinT2 = etaI / etaT * etaI / etaT * (1.0f - cosI * cosI);
    if (sinT2 > 1.0f) return 1.0f;  // Total internal reflection
    
    float cosT = sqrtf(1.0f - sinT2);
    float rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    float rp = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
    return (rs * rs + rp * rp) * 0.5f;
}

// Schlick approximation (faster, less accurate)
__device__ __forceinline__ float fresnelSchlick(float cosI, float F0) {
    float x = 1.0f - cosI;
    float x2 = x * x;
    return F0 + (1.0f - F0) * x2 * x2 * x;
}

// Fresnel for conductors (metals)
__device__ __forceinline__ float3 fresnelConductor(float cosI, float3 eta, float3 k) {
    float cosI2 = cosI * cosI;
    float sinI2 = 1.0f - cosI2;
    
    float3 eta2 = eta * eta;
    float3 k2 = k * k;
    
    float3 t0 = eta2 - k2 - make_float3(sinI2, sinI2, sinI2);
    float3 a2plusb2 = sqrt(t0 * t0 + 4.0f * eta2 * k2);
    float3 t1 = a2plusb2 + make_float3(cosI2, cosI2, cosI2);
    float3 a = sqrt(0.5f * (a2plusb2 + t0));
    float3 t2 = 2.0f * a * cosI;
    float3 Rs = (t1 - t2) / (t1 + t2);
    
    float3 t3 = cosI2 * a2plusb2 + make_float3(sinI2 * sinI2, sinI2 * sinI2, sinI2 * sinI2);
    float3 t4 = t2 * sinI2;
    float3 Rp = Rs * (t3 - t4) / (t3 + t4);
    
    return 0.5f * (Rp + Rs);
}

} // namespace wr::internal
```

#### `utils/ray_offset.cuh`
```cuda
#pragma once
#include <cuda_runtime.h>

namespace wr::internal {

// Offset ray origin to avoid self-intersection
__device__ __forceinline__ float3 offsetRayOrigin(
    float3 position,
    float3 normal,
    float3 direction
) {
    const float offset = 1e-4f;
    float3 offsetDir = dot(direction, normal) > 0.0f ? normal : -normal;
    return position + offsetDir * offset;
}

} // namespace wr::internal
```

---

### Step 2: 创建材质模块

#### `materials/material_common.cuh`
```cuda
#pragma once
#include "internal/gpu_types.h"
#include "sampling.cuh"
#include "utils/atomic_ops.cuh"
#include "utils/ray_offset.cuh"

namespace wr::internal {

// Common NEE (Next Event Estimation) implementation
// Used by Lambertian and GGX materials
__device__ __forceinline__ bool performNEE(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p,
    float3 albedo,
    float3 (*evalBSDF)(float3 wo, float3 wi, float3 normal, const MaterialData& mat, float3 albedo),
    float (*pdfBSDF)(float3 wo, float3 wi, float3 normal, const MaterialData& mat)
) {
    if (!p->useNEE || p->numEmissiveTriangles == 0 || ray.neeDone) {
        return false;
    }
    
    // Sample light source
    float lightU = rnd_dim(ray.seed, ray.rngDimension++);
    uint32_t lightIdx = sampleEmissiveTriangle(lightU, p->emissiveTriangleCDF, p->numEmissiveTriangles);
    uint32_t triIdx = p->emissiveTriangles[lightIdx];
    
    // ... (完整的 NEE 逻辑)
    
    return true;  // Shadow ray dispatched
}

} // namespace wr::internal
```

#### `materials/lambertian.cuh`
```cuda
#pragma once
#include "material_common.cuh"

namespace wr::internal {

__device__ __forceinline__ float3 evalLambertianBSDF(
    float3 wo,
    float3 wi,
    float3 normal,
    const MaterialData& mat,
    float3 albedo
) {
    float cosTheta = dot(normal, wi);
    if (cosTheta <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    return albedo * (cosTheta / kPi);
}

__device__ __forceinline__ float pdfLambertianBSDF(
    float3 wo,
    float3 wi,
    float3 normal,
    const MaterialData& mat
) {
    float cosTheta = dot(normal, wi);
    return cosTheta > 0.0f ? cosineHemispherePdf(cosTheta) : 0.0f;
}

__device__ __forceinline__ void shadeLambertian(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p
) {
    // Get albedo (texture or constant)
    float3 albedo = getMaterialAlbedo(mat, hit.uv,
        reinterpret_cast<const cudaTextureObject_t*>(p->textures), p->numTextures);
    
    // Try NEE first
    if (performNEE(ray, hit, mat, p, albedo, evalLambertianBSDF, pdfLambertianBSDF)) {
        return;  // Shadow ray dispatched
    }
    
    // Reset neeDone for next bounce
    ray.neeDone = 0;
    
    // BSDF sampling (indirect lighting)
    float3 tangent, bitangent;
    createCoordinateFrame(hit.normal, tangent, bitangent);
    
    float bsdfPdf;
    float3 localDir = sampleCosineHemisphere(
        rnd_dim(ray.seed, ray.rngDimension++), 
        rnd_dim(ray.seed, ray.rngDimension++), 
        bsdfPdf
    );
    float3 worldDir = toWorld(localDir, hit.normal, tangent, bitangent);
    
    // Update throughput: albedo / pi * cos(theta) / pdf
    // For cosine sampling: pdf = cos(theta) / pi, so this simplifies to albedo
    ray.throughput = ray.throughput * albedo;
    
    ray.prevPdf = bsdfPdf;
    ray.prevWasDelta = false;
    
    // Russian Roulette
    float3 newThroughput;
    if (!russianRoulette(ray.throughput, ray.depth, p->rrStartDepth, 
                         rnd_dim(ray.seed, ray.rngDimension++), newThroughput)) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
        return;
    }
    ray.throughput = newThroughput;
    
    // Setup next ray
    ray.direction = worldDir;
    ray.origin = offsetRayOrigin(hit.position, hit.normal, worldDir);
    ray.tMin = 0.0f;
    ray.tMax = 1e20f;
    ray.depth++;
    ray.stage = RayStage::Trace;
    
    if (ray.depth >= p->maxBounces) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
    }
}

} // namespace wr::internal
```

#### `materials/glass.cuh`
```cuda
#pragma once
#include "material_common.cuh"
#include "utils/fresnel.cuh"

namespace wr::internal {

__device__ __forceinline__ void shadeGlass(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p
) {
    float3 wo = -ray.direction;
    float cosI = dot(wo, hit.normal);
    
    float etaI = 1.0f;
    float etaT = mat.ior;
    float3 n = hit.normal;
    
    // Determine if entering or exiting
    if (cosI < 0.0f) {
        cosI = -cosI;
        n = -n;
        etaI = mat.ior;
        etaT = 1.0f;
    }
    
    // Fresnel reflectance
    float F = fresnelDielectric(cosI, etaI, etaT);
    
    // Sample reflection vs refraction
    float r = rnd_dim(ray.seed, ray.rngDimension++);
    
    if (r < F) {
        // Fresnel reflection
        ray.direction = ray.direction - n * (2.0f * dot(ray.direction, n));
        // Throughput unchanged (F / F = 1)
    } else {
        // Attempt refraction
        float eta = etaI / etaT;
        float sinT2 = eta * eta * (1.0f - cosI * cosI);
        
        if (sinT2 >= 1.0f) {
            // Total internal reflection
            ray.direction = ray.direction - n * (2.0f * dot(ray.direction, n));
        } else {
            // Successful refraction
            float cosT = sqrtf(1.0f - sinT2);
            ray.direction = eta * ray.direction + n * (eta * cosI - cosT);
            
            // Apply squeeze factor for radiance transport
            float squeezeFactor = (etaI * etaI) / (etaT * etaT);
            ray.throughput = ray.throughput * squeezeFactor;
        }
    }
    
    ray.prevWasDelta = true;
    
    // Russian Roulette (higher survival rate for delta materials)
    float3 newThroughput;
    if (!russianRoulette(ray.throughput, ray.depth, p->rrStartDepth, 
                         rnd_dim(ray.seed, ray.rngDimension++), newThroughput, true)) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
        return;
    }
    ray.throughput = newThroughput;
    
    // Setup next ray
    ray.origin = offsetRayOrigin(hit.position, hit.normal, ray.direction);
    ray.tMin = 0.0f;
    ray.tMax = 1e20f;
    ray.depth++;
    ray.stage = RayStage::Trace;
    
    if (ray.depth >= p->maxBounces) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
    }
}

} // namespace wr::internal
```

#### `materials/ggx_material.cuh`（从现有 `ggx.cuh` 分离）
```cuda
#pragma once
#include "material_common.cuh"
#include "ggx.cuh"  // GGX 数学函数

namespace wr::internal {

__device__ __forceinline__ void shadeGGX(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p
) {
    // ... GGX 材质逻辑（从 shade.cu 移动）
}

} // namespace wr::internal
```

---

### Step 3: 重写 `shade.cu` 为调度器

#### 新的 `shade.cu`（简洁版本）
```cuda
#include <cuda_runtime.h>
#include "internal/gpu_types.h"
#include "vector_math.cuh"
#include "sampling.cuh"
#include "utils/atomic_ops.cuh"
#include "materials/lambertian.cuh"
#include "materials/glass.cuh"
#include "materials/ggx_material.cuh"
#include "materials/emissive.cuh"

using namespace wr::internal;

extern "C" {
    __constant__ const LaunchParams* params_shade;
}

extern "C" __global__ void shade(const LaunchParams* p) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p->numActive) return;

    const uint32_t rayIndex = p->activeIndices[idx];
    RayState& ray = p->rayPool[rayIndex];

    if (ray.stage != RayStage::Shade) return;

    const HitInfo& hit = p->hitBuffer[rayIndex];
    
    if (hit.materialId >= p->numMaterials) {
        ray.stage = RayStage::Terminated;
        return;
    }
    
    const MaterialData& mat = p->materials[hit.materialId];
    
    // Dispatch to material-specific shaders
    switch (mat.type) {
        case MaterialType::Emissive:
            shadeEmissive(ray, hit, mat, p);
            break;
        case MaterialType::Lambertian:
            shadeLambertian(ray, hit, mat, p);
            break;
        case MaterialType::Glass:
            shadeGlass(ray, hit, mat, p);
            break;
        case MaterialType::GGXReflection:
            shadeGGX(ray, hit, mat, p);
            break;
        default:
            ray.stage = RayStage::Terminated;
            break;
    }
}
```

**优点**：
- 主 kernel 只有 ~50 行
- 材质逻辑完全解耦
- 添加新材质只需：
  1. 创建 `materials/new_material.cuh`
  2. 在 `shade.cu` 中添加一行 `case`

---

## 🔍 代码复用策略

### 1. **NEE 逻辑复用**

**问题**：Lambertian 和 GGX 的 NEE 代码 90% 相同。

**解决方案**：
```cuda
// materials/material_common.cuh
template<typename EvalBSDF, typename PdfBSDF>
__device__ __forceinline__ bool performNEE(
    RayState& ray,
    const HitInfo& hit,
    const MaterialData& mat,
    const LaunchParams* p,
    float3 albedo,
    EvalBSDF evalBSDF,
    PdfBSDF pdfBSDF
) {
    // 通用 NEE 逻辑
    // 使用函数对象或函数指针来调用材质特定的 BSDF
}
```

**调用**：
```cuda
// Lambertian
performNEE(ray, hit, mat, p, albedo, 
    [](float3 wo, float3 wi, float3 n, const MaterialData& m, float3 a) {
        return a * (dot(n, wi) / kPi);
    },
    [](float3 wo, float3 wi, float3 n, const MaterialData& m) {
        return cosineHemispherePdf(dot(n, wi));
    }
);
```

**注意**：CUDA 的 lambda 支持有限，可能需要使用函数指针或宏。

### 2. **Russian Roulette 复用**

**当前**：每个材质都调用 `russianRoulette`。

**优化**：在 `material_common.cuh` 中提供包装函数：
```cuda
__device__ __forceinline__ bool applyRussianRoulette(
    RayState& ray,
    const LaunchParams* p,
    bool isDelta = false
) {
    float3 newThroughput;
    if (!russianRoulette(ray.throughput, ray.depth, p->rrStartDepth, 
                         rnd_dim(ray.seed, ray.rngDimension++), newThroughput, isDelta)) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
        return false;
    }
    ray.throughput = newThroughput;
    return true;
}
```

### 3. **Ray Setup 复用**

**当前**：每个材质都有相同的 ray setup 代码。

**优化**：
```cuda
__device__ __forceinline__ void setupNextBounce(
    RayState& ray,
    const HitInfo& hit,
    float3 newDirection,
    const LaunchParams* p
) {
    ray.direction = newDirection;
    ray.origin = offsetRayOrigin(hit.position, hit.normal, newDirection);
    ray.tMin = 0.0f;
    ray.tMax = 1e20f;
    ray.depth++;
    ray.stage = RayStage::Trace;
    
    if (ray.depth >= p->maxBounces) {
        atomicAddFloat3(&p->accumBuffer[ray.pixelIndex], ray.radiance);
        ray.stage = RayStage::Terminated;
    }
}
```

---

## 📊 重构前后对比

| 维度 | 重构前 | 重构后 |
|---|---|---|
| `shade.cu` 行数 | 472 | ~50 |
| 材质文件数 | 1 | 5 |
| 代码重复 | 高（3处 atomicAddFloat3） | 无 |
| 添加新材质 | 修改大文件 | 创建新 `.cuh` |
| 可测试性 | 低 | 高（可单独测试） |
| 编译时间 | 慢（大文件） | 快（头文件） |

---

## ⚠️ 注意事项

### 1. **性能考虑**

**问题**：函数调用可能增加寄存器压力。

**解决方案**：
- 所有材质函数使用 `__device__ __forceinline__`
- 编译器会完全内联，性能无损失
- 使用 `--ptxas-options=-v` 检查寄存器使用

### 2. **CUDA 编译限制**

**问题**：`.cuh` 头文件不能单独编译。

**解决方案**：
- 所有 `.cuh` 都是 header-only
- 只有 `shade.cu` 会被编译为 CUBIN
- CMakeLists.txt 无需修改

### 3. **调试友好性**

**优点**：
- 每个材质独立文件，容易定位问题
- 可以单独注释某个材质的 `#include`
- 调试输出更清晰

---

## 🚀 实施计划

### Phase 1: 提取公共工具（30分钟）
1. 创建 `utils/atomic_ops.cuh`
2. 创建 `utils/fresnel.cuh`
3. 创建 `utils/ray_offset.cuh`
4. 更新 `shade.cu` 和 `trace.cu` 的 include

### Phase 2: 提取材质模块（1小时）
1. 创建 `materials/material_common.cuh`
2. 创建 `materials/lambertian.cuh`
3. 创建 `materials/glass.cuh`
4. 移动 `ggx.cuh` 到 `materials/` 并创建 `ggx_material.cuh`
5. 创建 `materials/emissive.cuh`

### Phase 3: 重写 shade.cu（30分钟）
1. 简化为调度器
2. 测试编译
3. 验证渲染结果一致

### Phase 4: 清理和优化（30分钟）
1. 删除重复代码
2. 统一命名规范
3. 添加文档注释

**总计**：约 2.5 小时

---

## 🎓 参考实现

### libVLR 的材质架构
```
libVLR/GPU_kernels/
├── materials.cu          # 材质定义（类似我们的 materials/*.cuh）
├── path_tracing.cu       # 路径追踪主循环（类似我们的 shade.cu）
└── bsdfs/
    ├── diffuse.cuh
    ├── specular.cuh
    └── microfacet.cuh
```

### PBRT v3 的材质架构
```
src/materials/
├── matte.cpp             # Lambertian
├── glass.cpp             # Dielectric
├── metal.cpp             # Conductor
└── plastic.cpp           # Layered
```

---

## 结论

**推荐重构**：是的，应该拆分 `shade.cu`！

**优点**：
- ✅ 代码组织清晰
- ✅ 易于维护和扩展
- ✅ 消除重复代码
- ✅ 性能无损（内联）

**缺点**：
- ⚠️ 需要一次性重构（约 2.5 小时）
- ⚠️ 可能引入临时 bug（需要仔细测试）

**建议**：
1. 先完成当前的 Glass 材质修复
2. 在一个独立的 branch 中进行重构
3. 重构后运行完整的回归测试
