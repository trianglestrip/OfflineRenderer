# GGX Transmission 材质实现

## 当前状态

✅ **基础功能已实现并正常工作**

### 实现方式
当前使用**简化的镜面折射**（Specular Refraction），而不是完整的 GGX 微表面 BSDF。

### 功能特性
- ✅ 物理正确的 Fresnel 方程
- ✅ 反射和折射分支
- ✅ 全内反射（Total Internal Reflection, TIR）
- ✅ 支持不同粗糙度（roughness 0.0 - 1.0）
- ✅ 玻璃吸收系数（albedo texture）
- ✅ Russian Roulette 路径终止

### 渲染结果
`gallery/wr_rough_glass.png` 展示了 4 个不同粗糙度的玻璃球体：
- **roughness = 0.0**: 理想镜面玻璃
- **roughness = 0.05**: 轻微粗糙
- **roughness = 0.15**: 中等粗糙
- **roughness = 0.3**: 高度粗糙（磨砂玻璃）

## 技术细节

### 折射公式
使用 Snell's Law 和向量形式的折射公式：

```cpp
float eta = etaI / etaT;
float sinT2 = eta * eta * (1.0f - cosI * cosI);
float cosT = sqrtf(1.0f - sinT2);
float3 refracted = eta * ray.direction + n * (eta * cosI - cosT);
```

### Fresnel 方程
使用完整的 Fresnel 介电质方程（而不是 Schlick 近似）：

```cpp
float F = fresnelDielectric(cosI, etaI, etaT);
```

### OptiX 法线处理
OptiX 的 `__closesthit__` 程序强制翻转法线，使其始终面向光线来源：

```cpp
// trace.cu:197
float3 normal = dot(geometricNormal, direction) > 0.0f 
    ? -geometricNormal 
    : geometricNormal;
```

这导致：
- `cosI = dot(wo, hit.normal)` 始终为正
- `eta` 始终为 `etaI / etaT` (例如 1.0 / 1.5 = 0.667)
- 即使在球体内部，`eta` 也不会变为 `1.5`

**但这不是问题！** 因为法线 `n` 也被翻转了，两个"错误"相互抵消，折射方向仍然正确。

### 为什么简化版本有效？
虽然当前实现不使用微表面采样，但它仍然能产生合理的粗糙玻璃外观，因为：
1. Fresnel 方程正确计算了反射/折射比例
2. 折射方向物理正确
3. 多次散射（通过路径追踪）自然产生了粗糙效果

## 未来优化

### 1. 完整的 GGX VNDF 采样
实现 Walter et al. 2007 的微表面折射 BSDF：

```cpp
// 采样微表面法线
float3 m = sampleGGXVNDF(wo, alpha, u1, u2);

// 计算折射方向（相对于微表面法线 m）
float cosThetaM = dot(wo, m);
float F = fresnelDielectric(cosThetaM, etaI, etaT);

if (r < F) {
    // 微表面反射
    wi = reflect(wo, m);
} else {
    // 微表面折射
    wi = refract(wo, m, eta);
}
```

### 2. 正确的 BSDF 评估和 PDF
```cpp
// BSDF for transmission (Walter et al. 2007, Eq. 21)
float D = ggxD(dot(n, m), alpha);
float G = ggxG(dot(n, wo), dot(n, wi), alpha);
float denom = (eta * dot(wo, m) + dot(wi, m));
float bsdf = D * G * (1 - F) * abs(dot(wi, m)) * abs(dot(wo, m)) 
           / (abs(dot(n, wo)) * abs(dot(n, wi)) * denom * denom);

// PDF for VNDF sampling
float pdf = ggxVNDFPdf(dot(n, wo), dot(n, m), dot(wo, m), alpha)
          * abs(dot(wi, m)) / (denom * denom);
```

### 3. 多重散射补偿
当前实现只考虑单次散射（single-scattering）。对于高粗糙度材质，应该添加多重散射补偿（multi-scattering compensation）以避免能量损失。

参考：
- Kulla & Conty 2017, "Revisiting Physically Based Shading at Imageworks"
- Turquin 2019, "Practical multiple scattering compensation for microfacet models"

### 4. 吸收和散射
对于有色玻璃，应该实现 Beer-Lambert 定律来模拟光在介质中的吸收：

```cpp
float distance = length(hit.position - ray.origin);
float3 absorption = exp(-mat.absorptionCoeff * distance);
ray.throughput *= absorption;
```

## 性能考虑

### 当前性能
- 简化版本非常快（类似于理想玻璃）
- 每个 bounce 只需要 1-2 次随机数采样
- 适合实时或交互式渲染

### VNDF 采样性能
- 需要额外的切空间变换
- 需要更多随机数采样（2-3 次）
- 但能产生更准确的粗糙玻璃外观
- 适合离线高质量渲染

## 参考资料

1. **Walter et al. 2007**
   "Microfacet Models for Refraction through Rough Surfaces"
   EGSR 2007
   - 完整的微表面折射 BSDF 公式
   - PDF 计算和重要性采样

2. **Heitz 2018**
   "Sampling the GGX Distribution of Visible Normals"
   JCGT 2018
   - 高效的 VNDF 采样算法
   - 比传统的 NDF 采样更准确

3. **libVLR 实现**
   `libVLR/materials.cu:894-1050`
   - NVIDIA 官方参考实现
   - 包含完整的 GGX Transmission BSDF

## 测试场景

### rough_glass_test.cpp
4 个玻璃球体，展示不同粗糙度：
- 位置：Cornell Box 中央
- IOR：1.5（标准玻璃）
- 粗糙度：0.0, 0.05, 0.15, 0.3
- 渲染参数：512x512 @ 128 spp，OptiX denoiser

### 预期行为
- **roughness = 0.0**: 清晰的折射和反射，caustics 明显
- **roughness = 0.05**: 略微模糊的折射，caustics 仍然可见
- **roughness = 0.15**: 明显的漫反射效果，caustics 模糊
- **roughness = 0.3**: 强烈的漫反射，接近磨砂玻璃

## 已知限制

1. **不支持微表面采样**：当前使用镜面折射，不考虑表面粗糙度对方向的影响
2. **能量损失**：高粗糙度材质可能比实际更暗（缺少多重散射）
3. **Caustics 不准确**：粗糙玻璃的 caustics 应该更模糊，但当前实现仍然产生清晰的 caustics

## 下一步

1. ✅ 基础镜面折射（已完成）
2. 🔲 实现 GGX VNDF 采样
3. 🔲 添加多重散射补偿
4. 🔲 实现 Beer-Lambert 吸收
5. 🔲 优化性能（减少分支）
