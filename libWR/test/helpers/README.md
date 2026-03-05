# Test Helpers

这些是测试辅助工具，**不是 libWR 库的一部分**。

## 设计理念

参考 libVLR 的设计，渲染库应该专注于核心渲染功能，而不包含：
- 几何生成工具
- 图像保存功能
- 配置文件加载
- 文件路径处理

这些功能属于**应用层**，不应该在渲染库的公共 API 中。

## 文件说明

### geometry.h/cpp
- `createSphere()` - UV 球体网格生成
- 数学常量（PI, TWO_PI 等）

### image.h/cpp
- `savePNG()` - 保存图像为 PNG 格式（带 gamma 校正）
- 依赖 STB 库

### file.h/cpp
- `getExecutableDirectory()` - 获取可执行文件目录
- `resolveGalleryPath()` - 解析输出图像路径

### config.h
- `RenderConfig` - 渲染配置结构体
- `config::load()` - 从 INI 文件加载配置
- Header-only

## 使用方式

```cpp
#include <wr/wr.h>
#include "helpers/geometry.h"
#include "helpers/image.h"
#include "helpers/config.h"

using namespace test_helpers;

int main() {
    // 1. 加载配置
    RenderConfig config = config::load("cornell_box_var");
    
    // 2. 创建几何体
    std::vector<float> verts;
    std::vector<uint32_t> inds;
    createSphere(verts, inds, 0, 0, 0, 1.0f);
    
    // 3. 渲染
    std::vector<wr::Vec3> image(config.width * config.height);
    renderer->render(scene, camera, image.data(), config.width, config.height, config.spp, config.denoiser);
    
    // 4. 保存图像
    savePNG("output.png", image.data(), config.width, config.height);
    
    return 0;
}
```

## 为什么不在库中？

1. **职责单一** - libWR 只负责渲染，不负责 I/O 和配置
2. **灵活性** - 用户可以选择自己的图像格式（PNG/EXR/JPG）和配置方式（INI/JSON/YAML）
3. **减少依赖** - 库不依赖 STB 等第三方 I/O 库
4. **参考 libVLR** - libVLR 也没有在公共 API 中提供这些工具

## 如果需要在自己的项目中使用

可以直接复制这些文件到你的项目中，或者：
- 使用 OpenEXR 保存高动态范围图像
- 使用 JSON/YAML 库加载配置
- 使用 Assimp 加载 3D 模型
