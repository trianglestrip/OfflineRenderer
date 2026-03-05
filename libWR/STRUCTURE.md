# libWR 项目结构说明

## 设计理念

libWR 采用**方案 B** 的完整重构设计，遵循以下原则：

1. **降低代码耦合度** - 各模块独立，职责清晰
2. **配置参数就近原则** - 每个类的配置结构体定义在对应的头文件中
3. **公共类型集中管理** - 基础类型放在 `types.h`，使用 GLM 数学库
4. **内部实现隐藏** - GPU 内部类型和实现细节不对外暴露
5. **第三方库统一管理** - 所有第三方库放在 `external/` 目录

## 目录结构

```
libWR/
├── include/wr/              # 公共 API 头文件（完全不依赖 GPU）
│   ├── wr.h                # 主头文件（聚合所有公共 API）
│   ├── types.h             # 基础类型（Vec3=glm::vec3, Camera）
│   ├── context.h           # Context 类 + ContextConfig
│   ├── scene.h             # Scene 类 + SceneBuildConfig（使用不透明类型）
│   └── renderer.h          # Renderer 类 + RendererConfig, RenderParams, DenoiserConfig
│
├── src/
│   └── internal/           # GPU 相关实现（依赖 CUDA/OptiX）
│       ├── gpu_types.h     # GPU 内部类型（MaterialData, RayState, LaunchParams）
│       ├── cuda_utils.h    # CUDA/OptiX 错误检查宏
│       ├── denoiser.h      # Denoiser 内部接口
│       ├── context.cpp     # Context 实现 + OptiX 初始化
│       ├── scene.cpp       # Scene 实现 + 加速结构构建
│       ├── renderer.cpp    # Renderer 实现 + Wavefront 渲染循环
│       ├── pipeline.cpp    # OptiX Pipeline 管理
│       └── denoiser.cpp    # OptiX Denoiser 实现
│
├── kernels/                # CUDA/OptiX 内核
│   ├── trace.cu            # OptiX 光线追踪内核（编译为 PTX）
│   ├── shade.cu            # Shading 内核（编译为 CUBIN）
│   ├── compact.cu          # 队列压缩内核（编译为 CUBIN）
│   └── vector_math.cuh     # GPU 数学工具
│
├── external/               # 第三方库（header-only）
│   ├── glm/               # GLM 数学库 v1.0.1
│   │   ├── glm.hpp        # 主头文件
│   │   ├── gtc/           # 稳定扩展（matrix_transform, type_ptr 等）
│   │   ├── gtx/           # 实验性扩展
│   │   └── LICENSE.txt    # MIT 许可证
│   └── stb/               # STB 图像库
│       └── stb_image_write.h
│
└── test/                   # 测试代码
    ├── cornell_box_var_test.cpp
    ├── render_config.ini
    ├── CMakeLists.txt
    └── helpers/            # 测试辅助工具（不是库的一部分）
        ├── geometry.h/cpp  # 几何生成（createSphere）
        ├── image.h/cpp     # 图像保存（savePNG）
        ├── file.h/cpp      # 文件路径工具
        └── config.h        # INI 配置加载器
```

## 公共 API 设计

### 1. 基础类型 (`types.h`)

```cpp
#include <glm/glm.hpp>

namespace wr {
    // 使用 GLM 类型
    using Vec3 = glm::vec3;
    using Vec4 = glm::vec4;
    using Mat3 = glm::mat3;
    using Mat4 = glm::mat4;
    
    struct Camera {
        Vec3 position, target, up;
        float fovY, aspect;
        
        // 辅助方法
        Vec3 getForward() const;
        Vec3 getRight() const;
        Vec3 getUp() const;
        Mat4 getViewMatrix() const;
        Mat4 getProjectionMatrix(...) const;
    };
}
```

### 2. Context 类 (`context.h`)

```cpp
struct ContextConfig {
    int deviceId = 0;
    int logLevel = 4;
    bool enableValidation = false;
};

class Context {
    Context(const ContextConfig& config = {});
    Scene* createScene();
    Renderer* createRenderer();
    int getDeviceId() const;
};
```

### 3. Scene 类 (`scene.h`)

```cpp
struct SceneBuildConfig {
    bool allowUpdate = false;
    bool allowCompaction = true;
    bool preferFastTrace = true;
};

// 不透明类型（隐藏 GPU 实现细节）
using DevicePtr = unsigned long long;
using TraversableHandle = unsigned long long;

class Scene {
    uint32_t addLambertianMaterial(const Vec3& albedo);
    uint32_t addEmissiveMaterial(const Vec3& emission);
    uint32_t addGlassMaterial(const Vec3& albedo, float ior);
    void addTriangleMesh(std::span<const float> vertices, ...);
    void finalize(const SceneBuildConfig& config = {});
    
    // 内部访问器（使用不透明类型）
    DevicePtr getVerticesBuffer() const;
    TraversableHandle getGASHandle() const;
};
```

### 4. Renderer 类 (`renderer.h`)

```cpp
struct RendererConfig {
    uint32_t maxBounces = 8;
    bool useNEE = true;
    float russianRouletteDepth = 3;
};

struct DenoiserConfig {
    bool enabled = false;
    bool useAlbedo = false;
    bool useNormal = false;
    float hdrIntensity = 1.0f;
};

struct RenderParams {
    uint32_t width, height;
    uint32_t spp = 1;
    DenoiserConfig denoiser;
};

class Renderer {
    Renderer(const RendererConfig& config = {});
    void render(Scene* scene, const Camera& camera, Vec3* output, const RenderParams& params);
};
```

## 第三方库

### GLM (OpenGL Mathematics)

- **版本**: 1.0.1
- **许可证**: MIT
- **类型**: Header-only
- **用途**: 向量、矩阵、变换等数学运算
- **位置**: `external/glm/`
- **文档**: 参见 `GLM_USAGE.md`

**为什么选择 GLM**:
- ✅ 与 GLSL 语法一致
- ✅ 功能完整（向量、矩阵、四元数）
- ✅ 高性能（SIMD 优化）
- ✅ 图形学领域标准库

### STB (Sean Barrett's libraries)

- **版本**: Latest
- **许可证**: Public Domain / MIT
- **类型**: Header-only
- **用途**: 图像保存（PNG）
- **位置**: `external/stb/`

## 内部实现

### GPU 类型 (`src/internal/gpu_types.h`)

不对外暴露的 GPU 内部类型：

- `MaterialData` - GPU 材质数据（使用 `float3`）
- `RayState` - 光线状态
- `HitInfo` - 命中信息
- `LaunchParams` - OptiX 启动参数
- `CompactParams` - 队列压缩参数

### CUDA 工具 (`src/internal/cuda_utils.h`)

内部错误检查宏：

- `OPTIX_CHECK(call)` - OptiX API 错误检查
- `CUDA_CHECK(call)` - CUDA Runtime API 错误检查
- `CU_CHECK(call)` - CUDA Driver API 错误检查

## 使用示例

```cpp
#include <wr/wr.h>
#include <glm/glm.hpp>
#include <vector>

using namespace wr;

int main() {
    // 1. 创建上下文
    ContextConfig ctxConfig;
    ctxConfig.deviceId = 0;
    Context context(ctxConfig);
    
    // 2. 创建场景
    Scene* scene = context.createScene();
    uint32_t mat = scene->addLambertianMaterial(Vec3(0.8f, 0.8f, 0.8f));
    
    // 手动创建几何体（或使用 test/helpers 中的工具）
    std::vector<float> verts = {
        -1, -1, 0,  1, -1, 0,  1, 1, 0,  -1, 1, 0
    };
    std::vector<uint32_t> inds = {0, 1, 2, 0, 2, 3};
    scene->addTriangleMesh(std::span(verts), std::span(inds), mat);
    
    SceneBuildConfig sceneConfig;
    sceneConfig.preferFastTrace = true;
    scene->finalize(sceneConfig);
    
    // 3. 创建渲染器
    RendererConfig rendererConfig;
    rendererConfig.maxBounces = 8;
    Renderer* renderer = context.createRenderer();
    
    // 4. 设置相机（使用 GLM）
    Camera camera;
    camera.position = Vec3(0, 0, 5);
    camera.target = Vec3(0, 0, 0);
    camera.up = Vec3(0, 1, 0);
    camera.fovY = glm::radians(45.0f);  // 使用 GLM 的角度转换
    camera.aspect = 1.0f;
    
    // 5. 渲染
    RenderParams params;
    params.width = 512;
    params.height = 512;
    params.spp = 64;
    params.denoiser.enabled = true;
    
    std::vector<Vec3> image(params.width * params.height);
    renderer->render(scene, camera, image.data(), params);
    
    // 6. 保存图像（用户自己处理，或使用 test/helpers）
    // 参见 test/helpers/image.h 中的 savePNG()
    
    return 0;
}
```

## 使用 GLM 的示例

```cpp
// 向量运算（直接使用 GLM）
Vec3 a(1, 2, 3);
Vec3 b(4, 5, 6);

Vec3 sum = a + b;
Vec3 normalized = glm::normalize(a);
float dot = glm::dot(a, b);
Vec3 cross = glm::cross(a, b);
float len = glm::length(a);
Vec3 reflected = glm::reflect(a, b);

// 矩阵变换
Mat4 view = glm::lookAt(Vec3(0,0,5), Vec3(0,0,0), Vec3(0,1,0));
Mat4 proj = glm::perspective(glm::radians(45.0f), 16.0f/9.0f, 0.1f, 100.0f);

// 常量（从 utils::math）
float pi = wr::utils::PI;
```

## 编译

```powershell
# 配置
cmake -B build -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" `
  -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"

# 编译
cmake --build build --config Release --target wr_cornell_box_var_test

# 运行
.\build\bin\Release\wr_cornell_box_var_test.exe
```

## 优势

### 1. 低耦合与 GPU 隔离（参考 libVLR 设计）

- **公共 API 完全不依赖 GPU** - `include/wr/` 中的头文件不包含任何 CUDA/OptiX 头文件
- **使用不透明类型** - `DevicePtr` 和 `TraversableHandle` 隐藏 GPU 实现细节
- **GPU 实现严格隔离** - 所有依赖 CUDA/OptiX 的代码都在 `src/internal/`
- **工具函数移到测试层** - 几何生成、图像保存等辅助功能不是渲染器核心
- **各模块独立** - 易于维护、测试和移植

### 2. 配置灵活

- 每个类都有对应的配置结构体
- 配置参数就近定义，易于理解和修改

### 3. 类型安全

- 公共 API 使用 `Vec3`（GLM 类型）
- 内部使用 `float3`（CUDA 类型）
- 类型转换在实现层完成

### 4. 使用成熟库

- **GLM** - 图形学标准数学库，功能完整
- **STB** - 轻量级图像库，单头文件

### 5. 易于扩展

- 添加新功能只需修改对应模块
- 不影响其他模块和公共 API
- 第三方库统一管理

## include 和 src 对应关系

| include/wr/ | src/ | GPU 依赖 | 说明 |
|-------------|------|---------|------|
| context.h | internal/context.cpp | ✅ | Context 实现 + OptiX 初始化 |
| scene.h | internal/scene.cpp | ✅ | Scene 实现 + 加速结构构建 |
| renderer.h | internal/renderer.cpp | ✅ | Renderer 实现 + Wavefront 渲染循环 |
| - | internal/pipeline.cpp | ✅ | OptiX Pipeline 管理 |
| - | internal/denoiser.cpp | ✅ | OptiX Denoiser 实现 |
| types.h | - | ❌ | 纯类型定义（GLM 别名） |
| wr.h | - | ❌ | 聚合头文件 |

**设计原则**:
- `include/wr/` - 公共 API，完全不依赖 GPU 头文件，使用不透明类型
- `src/internal/` - 所有实现都依赖 CUDA/OptiX，严格隔离
- `test/helpers/` - 测试辅助工具（几何生成、图像保存、配置加载），不是库的一部分
- `external/` - 第三方库（GLM, STB），header-only

## 迁移指南

## 与 libVLR 的设计对比

| 特性 | libVLR | libWR |
|-----|--------|-------|
| 公共 API 依赖 GPU | ❌ 不依赖 | ✅ 完全不依赖 |
| 数学库 | 自定义 Vector3D (1503 行) | GLM 别名 |
| 工具函数位置 | `utils/` 在库根目录（内部） | `test/helpers/`（测试层） |
| 不透明类型 | 使用 C API 指针 | `DevicePtr`, `TraversableHandle` |
| 图像保存 | `image.cpp` 在库中（46KB） | 移到 `test/helpers/` |
| 配置加载 | 无 | 移到 `test/helpers/` |

**设计理念一致**：
- ✅ 公共 API 不暴露 GPU 实现
- ✅ 工具函数不是核心 API 的一部分
- ✅ 使用成熟的第三方库（GLM）而不是重复造轮子

## 注意事项

1. **公共 API 完全不依赖 GPU** - `include/wr/` 中没有任何 CUDA/OptiX 头文件
2. **使用不透明类型** - `DevicePtr` 和 `TraversableHandle` 隐藏 GPU 实现
3. **辅助工具在测试层** - 几何生成、图像保存等功能在 `test/helpers/`，不是库的一部分
4. **使用 GLM 进行向量/矩阵运算** - 不要重复实现，直接使用 GLM 的功能
5. **配置参数都有默认值** - 可以使用默认构造
6. **Scene 必须调用 `finalize()`** - 在添加几何体后
7. **第三方库只保留必要文件** - GLM 只保留头文件和许可证，删除文档和测试

## 文件大小对比

| 库 | 完整下载 | 只保留头文件 | 节省 |
|----|---------|-------------|------|
| GLM | ~10 MB | ~2 MB | 80% |
| STB | ~1 MB | ~100 KB | 90% |

**当前 external/ 目录大小**: ~2.1 MB（仅头文件）
