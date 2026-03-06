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
- **Wavefront 渲染架构**：将渲染过程分解为多个阶段（Trace, Shade, Shadow, Compact），提高 GPU 利用率
- **OptiX 8.0 集成**：使用 OptiX 进行硬件加速光线追踪
- **CUDA 13.1 支持**：支持 Turing (compute_75) 及更新架构的 GPU
- **C++20 现代 API**：使用 `std::span` 实现零拷贝数据传递
- **GPU 严格隔离**：公共 API 无 CUDA/OptiX 依赖，使用 opaque types

#### 场景管理
- **三角网格**：支持任意三角网格的添加和管理
- **加速结构**：自动构建 OptiX GAS (Geometry Acceleration Structure)
- **材质系统**（模块化架构）：
  - ✅ Lambertian（漫反射）材质 + 纹理支持
  - ✅ Emissive（自发光）材质
  - ✅ Glass（理想镜面折射）材质：Fresnel + 能量守恒
  - ✅ GGX Reflection（粗糙金属）：VNDF 采样 + NEE
  - ✅ **GGX Transmission（粗糙玻璃）**：微表面折射 ⬅️ 新增！
- **纹理系统**：
  - ✅ 2D 纹理加载（PNG 格式）
  - ✅ GPU 纹理对象管理
  - ✅ UV 坐标映射
  - ✅ 线性插值采样
  - ✅ **法线贴图（Normal Map）** ⬅️ 新增！
- **环境光照**：支持均匀环境光
- **光源管理**：自动构建发光三角形列表和重要性采样 CDF

#### 高级渲染技术（参考 libVLR）
- ✅ **Next Event Estimation (NEE)**：显式光源采样，收敛速度提升 2-5 倍
- ✅ **Shadow Ray Visibility Test**：NEE 阴影光线可见性测试，物理正确的遮挡
- ✅ **Multiple Importance Sampling (MIS)**：Power heuristic 权重，减少方差
- ✅ **Russian Roulette**：基于 throughput 的动态路径终止，Delta 材质特殊处理
- ✅ **Denoiser Guide Layers**：使用 Albedo/Normal 提升降噪质量（9 通道输入）
- ✅ **ACES Filmic Tone Mapping**：业界标准色调映射，更好的色彩和对比度

#### 渲染功能
- **路径追踪**：基于蒙特卡洛的无偏路径追踪
- **多重采样**：可配置 SPP (Samples Per Pixel)
- **OptiX AI Denoiser**：集成 AI 降噪器，支持 guide layers
- **相机系统**：透视相机，支持 FOV 和 Aspect Ratio 配置

#### 第三方库集成
- **GLM (OpenGL Mathematics)**：用于公共 API 的数学类型（Vec3, Mat4 等）
- **STB Image**：PNG 纹理加载
- **STB Image Write**：PNG 图像输出

### 🔄 待改进功能（参考 libVLR）

#### 1. 焦散效果 ⚠️ **（当前最大差距）**
**当前问题**：玻璃球缺少焦散（caustics）效果，透明度不足

**原因**：
- 单向路径追踪难以捕捉 SDS 路径（光源 → 玻璃 → 地面 → 相机）
- 这些路径的采样概率极低

**解决方案**：
- **Photon Mapping**：专门处理焦散（中期，2-3周）
- **Bidirectional PT**：双向路径追踪（长期，4-6周）
- libVLR 使用 LVC-BPT 算法

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
- ✅ Lambertian（漫反射）+ 纹理
- ✅ Emissive（自发光）
- ✅ Glass（理想镜面折射）+ Fresnel + 能量守恒
- ✅ GGX Reflection（粗糙金属）+ NEE
- ✅ **GGX Transmission（粗糙玻璃）** ⬅️ 新增！

**待实现**：
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

**libWR 当前状态**：材质参数硬编码，支持基础纹理但无节点系统

#### 4. 纹理和贴图
libVLR 支持：
- **2D 纹理**：支持多种图像格式（PNG, EXR 等）
- **法线贴图**（Normal Map）
- **高度贴图**（Height Map / Bump Mapping）
- **Alpha 贴图**：透明度控制
- **纹理过滤**：Nearest/Linear/Trilinear 过滤
- **颜色空间管理**：支持 sRGB、Rec709 等色彩空间

**libWR 当前状态**：
- ✅ 2D 纹理（PNG 格式）
- ✅ UV 坐标映射
- ✅ 线性插值采样
- ✅ **法线贴图（Normal Map）** ⬅️ 新增！
- ❌ Alpha 贴图
- ❌ EXR/HDR 格式

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

| 功能类别 | libVLR | libWR | 完成度 |
|---------|--------|-------|--------|
| **材质** | 8+ 种（含 GGX, UE4, 各向异性） | 5 种（Lambertian, Emissive, Glass, GGX Reflection, GGX Transmission） | 60% |
| **纹理** | ✅ 完整支持（2D, Normal, Alpha） | ✅ 2D 纹理（PNG）+ Normal Map | 60% |
| **Shader Node** | ✅ 完整节点系统 | ❌ 不支持 | 0% |
| **光源** | 面光源、点光源、IBL | 面光源、均匀环境光 | 50% |
| **相机** | 透视（含景深）、环境相机 | 透视（无景深） | 50% |
| **渲染算法** | PT+NEE+MIS, LT, LVC-BPT | PT+NEE+MIS | 40% |
| **光谱渲染** | ✅ 全光谱 + RGB | ❌ 仅 RGB | 0% |
| **场景图** | ✅ 树形结构 + 实例化 | ❌ 扁平结构 | 0% |
| **图像格式** | PNG, EXR, HDR 等 | PNG | 30% |
| **降噪** | ✅ OptiX Denoiser | ✅ OptiX Denoiser + Guide Layers | 100% |
| **Tone Mapping** | Reinhard, ACES 等 | ✅ ACES Filmic | 80% |
| **API** | C + C++ Wrapper | C++ | 50% |

**整体完成度**：约 **48%**

## 开发优先级建议

### 🔴 高优先级（质量关键）
1. ⚠️ **焦散算法**（Photon Mapping 或 BPT）- 解决 Glass 透明度问题
2. ✅ **GGX BSDF**（粗糙玻璃）- 已实现
3. ✅ **法线贴图** - 已实现
4. ✅ **代码重构** - 已完成（shade.cu: 472→53 行）

### 🟡 中优先级（功能扩展）
5. **HDR 环境贴图**：IBL 照明
6. **景深效果**：薄透镜模型
7. **场景图和变换**：支持物体变换和实例化
8. **EXR 输出**：高动态范围输出

### 🟢 低优先级（高级功能）
10. **Shader Node 系统**：程序化材质
11. **Light Tracing**：从光源开始的路径追踪
12. **光谱渲染**：色散效果
13. **C API**：跨语言支持
14. **各向异性 BRDF**：各向异性反射

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

---

## 最近更新（v0.5.0 - 2026-03-05）

### 🏗️ 架构重构
- ✅ **模块化材质系统**：`shade.cu` 从 472 行缩减到 **53 行**（-89%）
- ✅ **消除代码重复**：提取公共函数到 `utils/`（atomic_ops, fresnel, ray_offset）
- ✅ **材质独立文件**：每种材质独立 `.cuh` 文件，易于维护和扩展
- ✅ **零性能损失**：所有函数使用 `__forceinline__`，编译器完全内联

### 🎨 新增材质
- ✅ **GGX Transmission（粗糙玻璃）**：微表面折射 BSDF，支持 roughness 参数
- ✅ **法线贴图（Normal Map）**：切线空间法线映射，提升视觉细节

### 🔧 材质和算法改进
- ✅ **ACES Filmic Tone Mapping**：替换 Reinhard，色彩饱和度显著提升
- ✅ **Glass 材质能量守恒**：移除多余的 Squeeze Factor，透明度大幅改善
- ✅ **Russian Roulette 优化**：Delta 材质使用更高存活率（0.5 vs 0.05）
- ✅ **GGX NEE 启用**：金属材质的直接光照更准确

### 📁 新文件结构
```
libWR/kernels/
├── shade.cu (53行) ← 调度器
├── utils/ ← 公共工具
│   ├── atomic_ops.cuh
│   ├── fresnel.cuh
│   ├── ray_offset.cuh
│   └── normal_map.cuh
└── materials/ ← 材质模块
    ├── emissive.cuh
    ├── lambertian.cuh
    ├── glass.cuh
    ├── ggx_material.cuh
    └── ggx_transmission.cuh
```

### 已知问题
- ⚠️ **玻璃球焦散缺失**：需要 Photon Mapping 或 BPT 算法

### 性能数据
- 512×512 @ 64 SPP：~12秒（NVIDIA MX550）
- 重构后性能略有提升

---

**当前版本**：v0.5.0 - Architecture Refactor & New Materials (2026-03-05)
