# libOptixW - OptiX Wavefront Path Tracer

## 项目目标

创建一个基于 libVLR 的 **Wavefront 架构**路径追踪器，保留 libVLR 的所有功能（材质、场景、光源等），仅将渲染循环改为 Wavefront 模式。

## 设计原则

### 保留 libVLR
- ✅ 完整的材质系统（Matte, GGX, Dielectric, Emissive 等）
- ✅ 完整的着色器节点系统（纹理、法线贴图、bump mapping）
- ✅ 完整的场景管理（TriangleMesh, Instance, Transform）
- ✅ 完整的光源系统（Point, Area, Infinite Sphere）
- ✅ 光谱渲染（SampledSpectrum）
- ✅ 相机系统（Pinhole, ThinLens, Equirectangular）

### 改为 Wavefront
- 🔄 **渲染循环**：从 Megakernel 改为 Wavefront
- 🔄 **光线队列管理**：按 Stage 分组处理
- 🔄 **GPU 内核**：分离为多个专用 kernel
  - `ray_gen.cu`: 生成主光线
  - `trace.cu`: OptiX 光线追踪
  - `shade.cu`: 材质评估和 BRDF
  - `nee.cu`: Next Event Estimation
  - `compact.cu`: 队列压缩

### 架构对比

| 特性 | libVLR (Megakernel) | libOptixW (Wavefront) |
|------|---------------------|----------------------|
| 材质系统 | ✅ 完整 | ✅ **复用 libVLR** |
| 场景管理 | ✅ 完整 | ✅ **复用 libVLR** |
| 光源系统 | ✅ 完整 | ✅ **复用 libVLR** |
| 渲染循环 | Megakernel | **Wavefront** |
| GPU 利用率 | 中等（发散执行） | **高（SIMD 友好）** |
| 内存占用 | 中等 | **略高（队列）** |

## 开发计划

### Phase 1: 基础 Wavefront 渲染器（2 周）
- [ ] 创建 libOptixW 目录结构
- [ ] 实现基础 Wavefront 调度器
- [ ] 实现 ray_gen kernel
- [ ] 实现 trace kernel（OptiX）
- [ ] 实现 shade kernel（仅 Matte 材质）
- [ ] Cornell Box 测试

### Phase 2: 集成 libVLR 材质（2 周）
- [ ] 集成 libVLR 的材质评估代码
- [ ] 支持所有材质类型（Matte, GGX, Dielectric）
- [ ] 支持着色器节点（纹理、法线贴图）
- [ ] 材质测试场景

### Phase 3: 完整功能（4 周）
- [ ] 集成所有光源类型
- [ ] 集成相机系统
- [ ] 支持复杂场景（Rungholt, San Miguel）
- [ ] 性能优化
- [ ] 与 libVLR 的性能对比

### Phase 4: 高级功能（可选）
- [ ] Bidirectional Path Tracing (BDPT)
- [ ] Light Vertex Cache (LVC)
- [ ] Photon Mapping
- [ ] ReSTIR

## 技术栈

- **OptiX**: 8.0+ (光线追踪)
- **CUDA**: 13.1+ (GPU 计算)
- **C++**: 20 (现代 C++ 特性)
- **CMake**: 3.26+ (构建系统)

## 目录结构

```
libOptixW/
├── include/
│   └── optixw/
│       ├── optixw.h          # 公共 API
│       └── types.h           # 类型定义
├── src/
│   ├── context.cpp           # OptiX 上下文管理
│   ├── scene.cpp             # 场景管理（复用 libVLR）
│   ├── renderer.cpp          # Wavefront 渲染器
│   └── scheduler.cpp         # Wavefront 调度器
├── kernels/
│   ├── ray_gen.cu            # 主光线生成
│   ├── trace.cu              # OptiX 光线追踪
│   ├── shade.cu              # 材质评估（复用 libVLR）
│   ├── nee.cu                # 直接光照采样
│   └── compact.cu            # 队列压缩
└── test/
    └── cornell_box_test.cpp  # 测试程序
```

## 与 libVLRW 的区别

| 特性 | libVLRW (已废弃) | libOptixW (新方案) |
|------|------------------|-------------------|
| 材质系统 | ❌ 简化（仅 Lambertian） | ✅ **完整（复用 libVLR）** |
| 场景管理 | ❌ 简化 | ✅ **完整（复用 libVLR）** |
| 光谱渲染 | ❌ RGB | ✅ **SampledSpectrum** |
| 代码复用 | ❌ 从零开始 | ✅ **最大化复用 libVLR** |
| 开发时间 | 长（需实现所有功能） | **短（只改渲染循环）** |
| 维护成本 | 高（两套代码） | **低（共享代码）** |

## 参考

- [libVLR](../libVLR_reference): 原始 Megakernel 实现
- [Megakernels Considered Harmful](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus): Wavefront 论文
- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix8/guide/index.html)
