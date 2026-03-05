# libWR - Wavefront Path Tracer

libWR 是一个基于 OptiX 8.0 和 CUDA 13.1 的 Wavefront 架构路径追踪渲染器，旨在替代旧的 libVLR（基于 OptiX 7 的 Megakernel 架构）。

## 快速开始

### 编译和运行

```batch
# 编译并运行测试（推荐）
.\run_test.bat

# 只编译库
.\build_libWR.bat

# 查看渲染结果
start gallery\wr_cornell.png
```

**输出示例**：
```
[Scene] Built light list: 2 emissive triangles
[Renderer] Sample 64/64
[OptiX][4][DENOISER]: input inference channels 9
[Denoiser] Denoising completed
Saved gallery\wr_cornell.png
```

详见 [BUILD_SCRIPTS.md](BUILD_SCRIPTS.md)

---

## 当前状态

### ✅ 已实现功能

#### 核心架构
- **Wavefront 渲染架构**：将渲染过程分解为多个阶段（Trace, Shade, Compact），提高 GPU 利用率
- **OptiX 8.0 集成**：使用 OptiX 进行硬件加速光线追踪
- **CUDA 13.1 支持**：支持 Turing (compute_75) 及更新架构的 GPU
- **C++20 现代 API**：使用 `std::span` 实现零拷贝数据传递
- **GPU 严格隔离**：公共 API 无 CUDA/OptiX 依赖，使用 opaque types

#### 场景管理
- **三角网格**：支持任意三角网格的添加和管理
- **加速结构**：自动构建 OptiX GAS (Geometry Acceleration Structure)
- **材质系统**：
  - Lambertian（漫反射）材质
  - Emissive（自发光）材质
  - Glass（玻璃/折射）材质，支持 IOR 参数
  - **GGX 微表面材质**（PBR）：支持 roughness 和 metallic 参数
- **环境光照**：支持均匀环境光
- **光源管理**：自动构建发光三角形列表和重要性采样 CDF

#### 高级渲染技术（参考 libVLR）
- **Next Event Estimation (NEE)**：显式光源采样，收敛速度提升 2-5 倍
- **Multiple Importance Sampling (MIS)**：Power heuristic 权重，减少方差
- **Russian Roulette**：基于 throughput 的动态路径终止，性能提升 10-20%
- **Denoiser Guide Layers**：使用 Albedo/Normal 提升降噪质量（9 通道输入）

#### 渲染功能
- **路径追踪**：基于蒙特卡洛的无偏路径追踪
- **多重采样**：可配置 SPP (Samples Per Pixel)
- **OptiX AI Denoiser**：集成 AI 降噪器，支持 guide layers
- **相机系统**：透视相机，支持 FOV 和 Aspect Ratio 配置

#### 第三方库集成
- **GLM (OpenGL Mathematics)**：用于公共 API 的数学类型（Vec3, Mat4 等）
- **STB Image Write**：PNG 图像输出

### 🔄 待改进功能（参考 libVLR）

#### 1. 阴影光线追踪 ⚠️ **（最高优先级）**
**当前问题**：NEE 未检查遮挡，光源会穿透墙壁（物理不正确）

**需要实现**：
- 在 NEE 中追踪阴影光线
- 使用 `RayStage::Shadow` 检查可见性
- 只在未遮挡时添加光源贡献

---

#### 2. 高级材质系统
libVLR 支持的材质类型：
- **Microfacet BRDF/BSDF (GGX)**：粗糙表面的物理正确反射/折射
- **Fresnel-blended Lambertian BSDF**：菲涅尔混合的漫反射
- **UE4/Frostbite 风格 BRDF**：
  - 支持 Base Color + Roughness/Metallic 参数化
  - 或 Diffuse + Specular + Glossiness 参数化
- **各向异性 BRDF**：支持沿切线方向的各向异性反射
- **Mixed BSDF**：多材质混合

**libWR 当前状态**：
- ✅ Lambertian（漫反射）
- ✅ Emissive（自发光）
- ✅ Glass（折射）
- ✅ **GGX 微表面 BRDF**（支持 roughness 和 metallic）

**待实现**：
- ❌ GGX 微表面 BSDF（粗糙玻璃）
- ❌ Fresnel-blended Lambertian
- ❌ 各向异性 BRDF
- ❌ Mixed BSDF

#### 3. Shader Node 系统
libVLR 的核心特性之一：
- **程序化纹理节点**：
  - Image2DTexture（2D 纹理采样）
  - GeometryShader（几何信息节点）
  - Tangent（切线节点，支持多种切线类型）
  - Float/Vector/Spectrum 常量节点
- **节点连接系统**：通过 Plug 系统灵活连接节点
- **运行时材质构建**：通过节点图动态定义材质行为

**libWR 当前状态**：材质参数硬编码，不支持纹理和节点系统

#### 4. 纹理和贴图
libVLR 支持：
- **2D 纹理**：支持多种图像格式（PNG, EXR 等）
- **法线贴图**（Normal Map）
- **高度贴图**（Height Map / Bump Mapping）
- **Alpha 贴图**：透明度控制
- **纹理过滤**：Nearest/Linear/Trilinear 过滤
- **颜色空间管理**：支持 sRGB、Rec709 等色彩空间

**libWR 当前状态**：完全不支持纹理

#### 5. 光源类型
libVLR 支持：
- **面光源**（Area Light）：多边形发光面
- **点光源**（Point Light）
- **环境贴图**（Image-Based Lighting, IBL）：支持 HDR 环境贴图

**libWR 当前状态**：
- ✅ 面光源（通过 Emissive 材质实现）
- ✅ 均匀环境光
- ❌ 点光源
- ❌ HDR 环境贴图

#### 6. 相机类型
libVLR 支持：
- **透视相机**：支持景深效果（薄透镜模型）
- **环境相机**（Equirectangular Camera）：360° 全景渲染

**libWR 当前状态**：
- ✅ 透视相机（无景深）
- ❌ 景深效果
- ❌ 环境相机

#### 6. 高级渲染算法
libVLR 支持：
- **Path Tracing with MIS**：路径追踪 + 多重重要性采样
- **Light Tracing**：从光源开始的路径追踪
- **LVC-BPT** (Light Vertex Cache Bidirectional Path Tracing)：双向路径追踪
- **非对称散射处理**：正确处理由 Shading Normal 引起的非对称散射

**libWR 当前状态**：
- ✅ Path Tracing with NEE + MIS（参考 libVLR 实现）
- ✅ 基于重要性的 Russian Roulette
- ❌ 光源追踪
- ❌ 双向路径追踪

#### 7. 高级渲染算法
libVLR 支持：
- **Light Tracing**：从光源开始的路径追踪
- **LVC-BPT** (Light Vertex Cache Bidirectional Path Tracing)：双向路径追踪
- **非对称散射处理**：正确处理由 Shading Normal 引起的非对称散射

**libWR 当前状态**：仅支持单向路径追踪

---

#### 8. 光谱渲染
libVLR 的独特功能：
- **全光谱渲染**：蒙特卡洛光谱采样
- **RGB-Spectrum 转换**：使用 Meng-Simon 方法进行 RGB 到光谱的转换
- **色散焦散**（Dispersive Caustics）：彩虹色的焦散效果

**libWR 当前状态**：仅支持 RGB 渲染

#### 9. 场景图和变换
libVLR 支持：
- **场景图**（Scene Graph）：树形层级结构
- **InternalNode**：支持嵌套变换
- **几何实例化**（Geometry Instancing）：高效复用几何体
- **静态/动态变换**：支持缩放、旋转、平移

**libWR 当前状态**：扁平场景结构，不支持变换和实例化

#### 10. 图像和颜色管理
libVLR 支持：
- **多种图像格式**：PNG, JPG, EXR, HDR 等
- **色彩空间**：sRGB, Rec709, Linear 等
- **Gamma 校正**：自动处理 Gamma 编码/解码
- **OpenEXR 输出**：支持高动态范围输出

**libWR 当前状态**：仅支持 PNG 输出（LDR）

#### 11. 调试和诊断
libVLR 支持：
- **调试渲染模式**：可视化法线、深度、材质 ID 等
- **Probe Pixel**：检查特定像素的渲染信息
- **日志系统**：详细的 OptiX 日志输出

**libWR 当前状态**：基础日志输出

#### 12. 性能优化
libVLR 特性：
- **异步渲染**：支持流式渲染
- **OpenGL 互操作**：直接渲染到 OpenGL 纹理
- **可调用程序深度**：支持复杂的材质递归调用

**libWR 当前状态**：同步渲染，无 OpenGL 互操作

#### 13. API 设计
libVLR 提供：
- **C API**：跨语言兼容
- **C++ Wrapper (vlrcpp.h)**：使用 `std::shared_ptr` 自动管理对象生命周期
- **错误处理**：完善的错误码系统

**libWR 当前状态**：
- ✅ C++ API
- ❌ C API
- ❌ 智能指针管理

## 功能对比表

| 功能类别 | libVLR | libWR |
|---------|--------|-------|
| **材质** | 8+ 种（含 GGX, UE4, 各向异性） | 3 种（Lambertian, Emissive, Glass） |
| **纹理** | ✅ 完整支持（2D, Normal, Alpha） | ❌ 不支持 |
| **Shader Node** | ✅ 完整节点系统 | ❌ 不支持 |
| **光源** | 面光源、点光源、IBL | 面光源、均匀环境光 |
| **相机** | 透视（含景深）、环境相机 | 透视（无景深） |
| **渲染算法** | PT+MIS, LT, LVC-BPT | 基础 PT |
| **光谱渲染** | ✅ 全光谱 + RGB | ❌ 仅 RGB |
| **场景图** | ✅ 树形结构 + 实例化 | ❌ 扁平结构 |
| **图像格式** | PNG, EXR, HDR 等 | PNG |
| **降噪** | ✅ OptiX Denoiser | ✅ OptiX Denoiser |
| **API** | C + C++ Wrapper | C++ |

## 开发优先级建议

### 高优先级（核心功能）
1. **纹理系统**：实现 2D 纹理采样，支持 PNG/EXR 加载
2. **MIS（多重重要性采样）**：显著提升收敛速度
3. **GGX 材质**：支持粗糙金属和塑料材质
4. **法线贴图**：提升视觉细节

### 中优先级（质量提升）
5. **HDR 环境贴图**：IBL 照明
6. **景深效果**：薄透镜模型
7. **场景图和变换**：支持物体变换和实例化
8. **EXR 输出**：高动态范围输出

### 低优先级（高级功能）
9. **Shader Node 系统**：程序化材质
10. **双向路径追踪**：处理复杂光照场景
11. **光谱渲染**：色散效果
12. **C API**：跨语言支持

## 技术架构对比

### libVLR (Megakernel)
- **单一大内核**：所有渲染逻辑在一个 OptiX kernel 中
- **优点**：代码简洁，易于调试
- **缺点**：GPU 利用率低，分支发散严重

### libWR (Wavefront)
- **多阶段内核**：Trace → Shade → Compact 分离
- **优点**：GPU 利用率高，减少分支发散
- **缺点**：需要更多内存和同步

## 测试场景

当前 libWR 可以成功渲染：
- ✅ Cornell Box 变体（包含玻璃球和金属盒）
- ✅ 面光源照明
- ✅ 漫反射和镜面反射
- ✅ 折射和菲涅尔效应

## 构建要求

- **CUDA Toolkit**: 13.1+
- **OptiX SDK**: 8.0.0
- **GPU**: NVIDIA Turing (compute_75) 或更新
- **编译器**: Visual Studio 2022
- **C++ 标准**: C++20

## 项目文档

- [BUILD_SCRIPTS.md](BUILD_SCRIPTS.md) - 构建脚本使用指南
- [libWR/STRUCTURE.md](libWR/STRUCTURE.md) - 项目结构和设计原则
- [libWR/API_DESIGN.md](libWR/API_DESIGN.md) - API 设计哲学
- [libWR/QUALITY_IMPROVEMENTS.md](libWR/QUALITY_IMPROVEMENTS.md) - 渲染质量改进计划
- [libWR/CHANGELOG.md](libWR/CHANGELOG.md) - 详细更新日志
- [libWR/GLM_USAGE.md](libWR/GLM_USAGE.md) - GLM 库使用指南

## 参考资料

- [OptiX 8.0 Programming Guide](https://raytracing-docs.nvidia.com/optix8/guide/index.html)
- [Wavefront Path Tracing 论文](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus)
- [libVLR 原始实现](https://github.com/shocker-0x15/VLR)
- [PBR Book](https://www.pbr-book.org/)

---

**当前版本**：v0.3.0 - GGX Microfacet BRDF (2026-03-05)
