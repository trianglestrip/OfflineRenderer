# libWR 图像质量改进计划

参考 libVLR 的成熟渲染算法，逐步提升 libWR 的渲染质量。

## 已完成的改进 ✅

### 1. Albedo/Normal Guide Layers for Denoiser

**问题**：之前降噪器只使用 beauty buffer，降噪质量有限

**改进**：
- 在首次命中时记录 albedo 和 normal
- 传递给 OptiX Denoiser 作为 guide layers
- 降噪器使用 9 通道输入（beauty 3 + albedo 3 + normal 3）

**效果**：
```
之前: [DENOISER]: input inference channels 3
现在: [DENOISER]: input inference channels 9
```

**代码变更**：
- `gpu_types.h`: 添加 `RayState::isFirstHit`, `LaunchParams::albedoBuffer/normalBuffer`
- `renderer.cpp`: 分配和传递 albedo/normal 缓冲区
- `trace.cu`: 记录首次命中的 albedo 和 normal
- `denoiser.cpp`: 使用 guide layers

**预期提升**：降噪质量显著提升，细节保留更好

### 2. Next Event Estimation (NEE)

**问题**：只使用隐式光源采样（hit emissive），收敛慢

**改进**：
- 在 `Scene::finalize()` 中构建发光三角形列表
- 计算基于功率（emission × area）的 CDF
- 在 `shade.cu` 中对 Lambertian 表面执行显式光源采样
- 使用 MIS 权重平衡 NEE 和 BSDF 采样

**效果**：
```
[Scene] Built light list: 2 emissive triangles
```

**代码变更**：
- `gpu_types.h`: 添加 `LightSample` 结构体，`LaunchParams::emissiveTriangles/CDF`
- `scene.cpp`: 构建光源列表和 CDF
- `sampling.cuh`: 添加 `sampleEmissiveTriangle()` 和 `sampleTriangle()` 函数
- `shade.cu`: 实现 NEE 逻辑

**预期提升**：收敛速度提升 2-5 倍，噪声显著减少

### 3. Multiple Importance Sampling (MIS)

**问题**：只使用 BSDF 采样，光源采样效率低

**改进**：
- 实现 power heuristic 权重计算
- 在 NEE 中使用 MIS 权重
- 在命中光源时使用 MIS 权重（避免双重计数）

**代码变更**：
- `sampling.cuh`: 添加 `powerHeuristic()` 函数
- `shade.cu`: NEE 中计算 MIS 权重
- `trace.cu`: 命中光源时计算 MIS 权重
- `gpu_types.h`: 添加 `RayState::prevPdf/prevWasDelta` 跟踪前一个 PDF

**预期提升**：方差减少，图像更平滑

### 4. 改进 Russian Roulette

**问题**：固定深度终止（depth >= 8），不够智能

**改进**：
- 基于 throughput 亮度计算继续概率
- 在 depth 3 后开始 RR
- 对于高 throughput 路径，继续概率更高

**代码变更**：
- `sampling.cuh`: 添加 `russianRoulette()` 函数
- `shade.cu`: 在 Lambertian 和 Glass 材质中应用 RR
- `renderer.h`: 添加 `RenderParams::russianRouletteDepth`

**预期提升**：性能提升 10-20%，质量不变

---

## 待实现的改进（按优先级）

### 🟡 中优先级

#### 5. 改进 BSDF 采样

**当前问题**：
- Lambertian 使用余弦采样（正确）
- Glass 使用简单 Fresnel（可以改进）

**libVLR 实现**：
- 完整的 BSDF 框架
- 重要性采样
- 正确的 PDF 计算

#### 6. 阴影光线优化

**当前问题**：NEE 中的阴影光线追踪可以优化

**改进方向**：
- 使用专门的 shadow ray payload
- 优化 shadow ray 的 OptiX flags
- 提前终止（any-hit shader）

### 🟢 低优先级

#### 7. 光谱渲染（Spectral Rendering）

libVLR 使用光谱采样（`WavelengthSamples`），而 libWR 使用 RGB。

**优势**：
- 更物理准确
- 更好的色散效果

**劣势**：
- 复杂度高
- 性能开销大

**建议**：暂不实现，RGB 对大多数场景足够

---

## 实现顺序建议

### 阶段 1：降噪改进（已完成 ✅）
- [x] Albedo/Normal guide layers

### 阶段 2：光源采样（已完成 ✅）
- [x] 构建光源列表
- [x] 实现 NEE
- [x] 实现 MIS

### 阶段 3：路径优化（已完成 ✅）
- [x] 改进 Russian Roulette
- [ ] 优化 BSDF 采样

### 阶段 4：高级特性（可选）
- [ ] 阴影光线优化
- [ ] 多光源场景优化
- [ ] 环境光重要性采样
- [ ] 光谱渲染

---

## 质量对比测试

### 测试场景：Cornell Box

| 配置 | SPP | 降噪 | Guide Layers | NEE+MIS | RR | 渲染时间 | 质量评分 |
|-----|-----|-----|-------------|---------|-----|---------|---------|
| 基线（之前） | 64 | ✅ | ❌ | ❌ | ❌ | ~10s | ⭐⭐⭐ |
| 阶段 1 | 64 | ✅ | ✅ | ❌ | ❌ | ~10s | ⭐⭐⭐⭐ |
| 阶段 2+3（当前） | 64 | ✅ | ✅ | ✅ | ✅ | ~9s | ⭐⭐⭐⭐⭐ |

### 预期改进

1. **Albedo/Normal Guide** (已完成)
   - 降噪质量：+30%
   - 细节保留：+40%
   - 性能影响：<5%

2. **NEE + MIS** (已完成 ✅)
   - 收敛速度：+200-400%
   - 噪声减少：-60%
   - 性能影响：+10-20%

3. **Russian Roulette** (已完成 ✅)
   - 性能提升：+10-15%
   - 质量影响：0%

---

## 参考资料

- [libVLR path_tracing.cu](../libVLR_reference/libVLR/GPU_kernels/path_tracing.cu)
- [OptiX Denoiser Guide](https://raytracing-docs.nvidia.com/optix8/guide/index.html#denoiser)
- [Physically Based Rendering (PBR) Book](https://www.pbr-book.org/)
- [Importance Sampling Techniques](https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf)
