# OfflineRenderer

本项目包含多个基于 NVIDIA OptiX 的 GPU 光线追踪渲染器实现。

## 技术架构对比

| 特性 | libOptixW (Wavefront) | libVLR (Megakernel) |
|---|---|---|
| **架构** | Wavefront Path Tracing | Megakernel |
| **OptiX 版本** | 8.0+ | 7.x |
| **CUDA 版本** | 13.1+ | 11.8 |
| **GPU 要求** | compute_75+ (Turing+) | compute_52+ (Maxwell+) |
| **渲染模式** | RGB | 全光谱 / RGB |
| **API 风格** | 现代 C++20 | C API + C++ 封装 |
| **状态** | ✅ 核心功能完成 | ✅ 完整实现 |
| **性能** | 高（Wavefront 优化） | 中（Megakernel 发散） |

## libOptixW 特性

### 已实现 ✅
* **渲染架构**:
  - Wavefront Path Tracing（光线排序，减少发散）
  - Next Event Estimation (NEE) 直接光照
  - 多重采样抗锯齿 (SPP)
* **材质系统**:
  - Lambertian BRDF（理想漫反射）
  - Emissive 材质（自发光）
* **光源**:
  - 点光源
  - 面光源（矩形）
* **加速结构**:
  - OptiX GAS（几何加速结构）
  - 单层场景（无实例化）
* **输出**:
  - PNG 图像（通过 stb_image_write）
  - Gamma 校正（2.2）

### 计划中 🔲
* **材质**: Specular BRDF、GGX 微表面、混合材质
* **光源**: 环境光照（IBL）、方向光
* **纹理**: 2D 纹理映射、法线贴图
* **优化**: 队列压缩、流式多重采样、多 GPU
* **后处理**: OptiX Denoiser 降噪

## libVLR 特性（参考）

* 全光谱渲染（蒙特卡洛光谱采样）
* RGB->光谱转换（Meng-Simon 方法）
* 完整的 BSDF 系统（Lambert、Specular、GGX、UE4 风格等）
* 着色器节点系统
* 法线贴图 / 高度贴图 / Alpha 纹理
* 多种光线传输算法（Path Tracing、LVC-BPT）
* 几何体实例化和场景图
* 透视相机（景深）、环境相机

## 快速开始

### 1. 环境要求

| 工具 | 版本要求 |
|---|---|
| Visual Studio | 2022 (17.14+) |
| CMake | 3.26+ |
| CUDA Toolkit | 13.1+ |
| OptiX SDK | 8.0+ |
| GPU | compute_75+ (Turing/Ampere/Ada) |

### 2. 配置和编译

```powershell
# 配置项目
cmake -B build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" `
  -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"

# 编译 libOptixW 库
cmake --build build --config Release --target libOptixW

# 编译并运行 Cornell Box 测试
cmake --build build --config Release --target cornell_box_test
.\build\bin\Release\cornell_box_test.exe
```

### 3. 查看结果

测试成功后会在项目根目录生成 `cornell_box.png`：

<img src="cornell_box.png" width="512px" alt="Cornell Box">

## libOptixW API 示例

完整示例请参考 `libOptixW/test/cornell_box_test.cpp`。

```cpp
#include <optixw/optixw.h>

// 1. 初始化 OptiX 上下文
optixw::Context context(0);  // 使用 GPU 0

// 2. 创建场景
optixw::Scene* scene = context.createScene();

// 3. 添加材质
uint32_t whiteMat = scene->addLambertianMaterial({0.75f, 0.75f, 0.75f});
uint32_t lightMat = scene->addEmissiveMaterial({10.0f, 10.0f, 10.0f});

// 4. 添加几何体（使用 std::span 零拷贝）
float vertices[] = { -1.0f, 0.0f, -1.0f,  1.0f, 0.0f, -1.0f,  0.0f, 2.0f, 0.0f };
uint32_t indices[] = { 0, 1, 2 };
scene->addTriangleMesh(std::span(vertices, 9), std::span(indices, 3), whiteMat);

// 5. 添加光源
scene->addPointLight({{0.0f, 1.8f, 0.0f}, {5.0f, 5.0f, 5.0f}});

// 6. 构建加速结构
scene->finalize();

// 7. 设置相机
optixw::Camera camera = {
    .position = {0.0f, 1.0f, 3.0f},
    .target = {0.0f, 1.0f, 0.0f},
    .up = {0.0f, 1.0f, 0.0f},
    .fovY = 45.0f * M_PI / 180.0f,
    .aspect = 1.0f
};

// 8. 渲染
optixw::Renderer* renderer = context.createRenderer();
std::vector<optixw::RGB> image(1024 * 1024);
renderer->render(scene, camera, image.data(), 1024, 1024, 64);  // 64 SPP
```

## 项目结构

```
OfflineRenderer/
├── libOptixW/              # 新 Wavefront 渲染器（主要开发）
│   ├── include/optixw/     # 公共 API 头文件
│   │   ├── optixw.h        # Context, Scene, Renderer
│   │   └── types.h         # MaterialData, RGB 等
│   ├── src/                # CPU 端实现
│   │   ├── context.cpp     # OptiX 初始化
│   │   ├── scene.cpp       # 场景和加速结构管理
│   │   ├── renderer.cpp    # 渲染循环和 Pipeline
│   │   └── scheduler.cpp   # Wavefront 调度器
│   ├── kernels/            # GPU 内核
│   │   ├── ray_gen.cu      # 主光线生成
│   │   ├── trace.cu        # OptiX 光线追踪（OptiX IR）
│   │   ├── shade.cu        # 材质评估和 NEE
│   │   └── compact.cu      # 队列压缩
│   └── test/               # 测试程序
│       └── cornell_box_test.cpp
├── libVLR/                 # 旧 Megakernel 渲染器（参考）
├── HostProgram/            # 演示程序
├── cmake/                  # CMake 工具脚本
└── .cursor/rules/          # 编译配置文档
    ├── libvlrw-build.mdc   # 详细编译指南
    ├── build-env.mdc       # 环境配置
    └── cmake-build-warnings.mdc
```

## 已验证的运行环境

### libOptixW（当前开发环境）
* Windows 11 (26200) & Visual Studio 2022 (17.14.36202.13)
* NVIDIA GeForce MX550 (Turing, compute_75)
* CUDA Toolkit 13.1.115
* OptiX SDK 8.0.0
* NVIDIA 驱动 581.95
* CMake 3.31.0

### libVLR（原始实现）
* Windows 10 (21H2) & Visual Studio 2022 (17.2.4)
* Core i9-9900K, 32GB, RTX 3080 10GB
* NVIDIA 驱动 516.40
* CUDA 11.8
* OptiX 7.x

## 依赖库

### libOptixW
* **必需**: CUDA Toolkit 13.1+、OptiX SDK 8.0+、GPU compute_75+
* **内置**: stb_image_write、Taskflow（可选）

### libVLR
* **必需**: CUDA 11.8、OptiX 7.x、GPU compute_52+

### HostProgram
* OpenEXR 3.1、assimp 5.0、GLFW、gl3w（后三者已作为 submodule 包含）

## 编译问题排查

如遇到编译错误，请参考 `.cursor/rules/libvlrw-build.mdc`，其中包含：
- 常见编译错误及解决方案
- CUDA 版本兼容性问题
- OptiX IR vs PTX 编译配置
- 文件编码问题处理

## 待办事项

### libOptixW 下一步开发
- [ ] 添加更多材质类型（Specular、GGX 微表面）
- [ ] 实现环境光照（IBL）
- [ ] 添加纹理支持
- [ ] 性能优化（队列压缩、流式多重采样）
- [ ] 降噪器集成（OptiX Denoiser）
- [ ] 多 GPU 支持

### libVLR 改进
- [ ] 迁移到 CUDA 13+ 和 OptiX 8+
- [ ] Python 绑定
- [ ] 简单的场景编辑器

## 注意事项
* libVLR 的场景文件引用的模型和纹理资产**未**包含在本仓库中
* libVLR 与 libOptixW 使用不同的 CUDA/OptiX 版本，建议分别编译
* 推荐使用 libOptixW 进行新开发


## 参考文献
[Davidovi&#269;2014] "Progressive Light Transport Simulation on the GPU: Survey and Improvements"\
[Kajiya1986] "THE RENDERING EQUATION"\
[Karis2013] "Real Shading in Unreal Engine 4"\
[Lagarde2014] "Moving Frostbite to Physically Based Rendering 3.0"\
[Meng2015] "Physically Meaningful Rendering using Tristimulus Colours"\
[Veach1997] "ROBUST MONTE CARLO METHODS FOR LIGHT TRANSPORT SIMULATION"

----
2022 [@Shocker_0x15](https://twitter.com/Shocker_0x15)\
2026 libOptixW 实现
