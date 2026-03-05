# libWR API 设计说明

## 设计理念（参考 libVLR）

libWR 的 API 设计遵循以下原则，参考了 libVLR 的成熟设计：

### 1. GPU 完全隔离

**libVLR 的做法**:
- 公共 API (`include/vlr/`) 使用 C API 风格
- 内部实现使用 CUDA/OptiX，但不暴露给用户
- `utils/` 目录在库根目录，是内部工具，不对外暴露

**libWR 的做法**:
- 公共 API (`include/wr/`) 完全不包含 CUDA/OptiX 头文件
- 使用不透明类型 (`DevicePtr`, `TraversableHandle`) 隐藏 GPU 实现
- 所有实现都在 `src/internal/`，严格隔离

### 2. 职责单一

**不在库中提供的功能**:
- ❌ 几何生成工具（`createSphere` 等）
- ❌ 图像保存功能（`savePNG` 等）
- ❌ 配置文件加载（INI/JSON 等）
- ❌ 文件路径处理

**原因**:
- 这些是**应用层**的职责，不是渲染器核心功能
- 用户可能有自己的偏好（PNG vs EXR，INI vs JSON）
- 减少库的依赖和体积

### 3. 使用成熟的第三方库

**libVLR**:
- 自定义 `Vector3DTemplate`（1503 行）
- 包含完整的向量/矩阵/四元数实现

**libWR**:
- 使用 GLM（图形学标准库）
- 只提供类型别名（`using Vec3 = glm::vec3`）
- 减少维护负担，提高可靠性

## 公共 API 结构

```cpp
namespace wr {

// types.h - 基础类型（GLM 别名）
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;

struct Camera {
    Vec3 position, target, up;
    float fovY, aspect;
    // 辅助方法...
};

// scene.h - 不透明类型（隐藏 GPU 实现）
using DevicePtr = unsigned long long;
using TraversableHandle = unsigned long long;

// context.h
struct ContextConfig { int deviceId; int logLevel; bool enableValidation; };
class Context { /* ... */ };

// scene.h
struct SceneBuildConfig { bool allowUpdate; bool allowCompaction; bool preferFastTrace; };
class Scene { /* ... */ };

// renderer.h
struct RendererConfig { uint32_t maxBounces; bool useNEE; float russianRouletteDepth; };
struct DenoiserConfig { bool enabled; bool useAlbedo; bool useNormal; float hdrIntensity; };
struct RenderParams { uint32_t width, height, spp; DenoiserConfig denoiser; };
class Renderer { /* ... */ };

} // namespace wr
```

## 不透明类型设计

### 为什么使用不透明类型？

```cpp
// ❌ 错误：暴露 GPU 类型
#include <optix.h>
#include <cuda.h>

class Scene {
    CUdeviceptr getVerticesBuffer() const;  // 用户需要包含 cuda.h
    OptixTraversableHandle getGASHandle() const;  // 用户需要包含 optix.h
};
```

```cpp
// ✅ 正确：使用不透明类型
using DevicePtr = unsigned long long;
using TraversableHandle = unsigned long long;

class Scene {
    DevicePtr getVerticesBuffer() const;  // 不需要 GPU 头文件
    TraversableHandle getGASHandle() const;  // 不需要 GPU 头文件
};
```

### 内部实现中的转换

```cpp
// src/internal/scene.cpp
DevicePtr Scene::getVerticesBuffer() const {
    return static_cast<DevicePtr>(m_impl->d_vertices);  // CUdeviceptr -> DevicePtr
}

// src/internal/renderer.cpp
CUdeviceptr verts = static_cast<CUdeviceptr>(scene->getVerticesBuffer());  // DevicePtr -> CUdeviceptr
```

## 测试辅助工具 (test/helpers/)

这些工具**不是库的一部分**，只是为了方便测试：

### geometry.h/cpp
```cpp
namespace test_helpers {
    void createSphere(std::vector<float>& vertices, 
                      std::vector<uint32_t>& indices, 
                      float cx, float cy, float cz, 
                      float radius, 
                      int segments = 32, 
                      int rings = 24);
}
```

### image.h/cpp
```cpp
namespace test_helpers {
    void savePNG(const char* filename, const wr::Vec3* image, uint32_t width, uint32_t height);
}
```

### config.h (header-only)
```cpp
namespace test_helpers {
    struct RenderConfig {
        uint32_t width, height, spp;
        bool denoiser;
    };
    
    namespace config {
        RenderConfig load(const std::string& sectionName);
    }
}
```

## 用户如何使用

### 最小示例（不使用 helpers）

```cpp
#include <wr/wr.h>
#include <glm/glm.hpp>
#include <vector>

int main() {
    wr::Context context;
    wr::Scene* scene = context.createScene();
    
    // 手动创建几何体
    std::vector<float> verts = {-1,-1,0, 1,-1,0, 1,1,0, -1,1,0};
    std::vector<uint32_t> inds = {0,1,2, 0,2,3};
    uint32_t mat = scene->addLambertianMaterial(wr::Vec3(0.8f));
    scene->addTriangleMesh(std::span(verts), std::span(inds), mat);
    scene->finalize();
    
    wr::Renderer* renderer = context.createRenderer();
    wr::Camera camera;
    camera.position = wr::Vec3(0, 0, 5);
    camera.target = wr::Vec3(0, 0, 0);
    camera.up = wr::Vec3(0, 1, 0);
    camera.fovY = glm::radians(45.0f);
    camera.aspect = 1.0f;
    
    wr::RenderParams params;
    params.width = 512;
    params.height = 512;
    params.spp = 64;
    
    std::vector<wr::Vec3> image(params.width * params.height);
    renderer->render(scene, camera, image.data(), params);
    
    // 用户自己处理输出（PNG/EXR/JPG/...）
    return 0;
}
```

### 使用 helpers 的示例

```cpp
#include <wr/wr.h>
#include "helpers/geometry.h"
#include "helpers/image.h"
#include "helpers/config.h"

using namespace test_helpers;

int main() {
    // 使用配置文件
    RenderConfig config = config::load("cornell_box_var");
    
    wr::Context context;
    wr::Scene* scene = context.createScene();
    
    // 使用几何生成工具
    std::vector<float> verts;
    std::vector<uint32_t> inds;
    createSphere(verts, inds, 0, 0, 0, 1.0f);
    
    uint32_t mat = scene->addLambertianMaterial(wr::Vec3(0.8f));
    scene->addTriangleMesh(std::span(verts), std::span(inds), mat);
    scene->finalize();
    
    wr::Renderer* renderer = context.createRenderer();
    // ... 渲染 ...
    
    // 使用图像保存工具
    savePNG("output.png", image.data(), config.width, config.height);
    
    return 0;
}
```

## 优势总结

### 对比 libVLR

| 特性 | libVLR | libWR |
|-----|--------|-------|
| 公共 API 依赖 GPU | ❌ | ✅ 完全不依赖 |
| 数学库 | 自定义（1503 行） | GLM 别名 |
| 工具函数 | `utils/` 内部 | `test/helpers/` |
| 图像保存 | 库内（46KB） | 测试层 |
| 不透明类型 | C API 指针 | `DevicePtr`, `TraversableHandle` |

### 核心优势

1. **API 纯粹** - 只关注渲染核心功能
2. **零 GPU 依赖** - 公共头文件可以在任何环境中包含
3. **灵活性高** - 用户自由选择 I/O 和配置方式
4. **易于维护** - 使用 GLM 而不是自定义数学库
5. **符合最佳实践** - 参考 libVLR 的成熟设计

## 文件大小对比

| 组件 | libVLR | libWR |
|-----|--------|-------|
| 公共 API | 5 个头文件 | 5 个头文件 |
| 数学类型 | `basic_types.h` (1503 行) | `types.h` (50 行，GLM 别名) |
| 图像处理 | `image.cpp` (46KB) | 移到 `test/helpers/` (1KB) |
| 工具函数 | `utils/` (400KB+) | 移到 `test/helpers/` (10KB) |

**libWR 更轻量**，专注于核心渲染功能。
