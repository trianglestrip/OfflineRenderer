# GLM 使用指南

libWR 使用 [GLM (OpenGL Mathematics)](https://github.com/g-truc/glm) 作为数学库。

## 为什么选择 GLM？

- ✅ **Header-only** - 无需编译，直接包含即可
- ✅ **与 GLSL 一致** - API 与 GPU 着色器语言相同
- ✅ **功能完整** - 向量、矩阵、四元数、变换等
- ✅ **高性能** - 支持 SIMD 优化
- ✅ **广泛使用** - 图形学领域标准库
- ✅ **MIT 许可证** - 商业友好

## 类型映射

libWR 使用 GLM 类型作为公共 API：

```cpp
namespace wr {
    using Vec3 = glm::vec3;  // 3D 向量
    using Vec4 = glm::vec4;  // 4D 向量
    using Mat3 = glm::mat3;  // 3x3 矩阵
    using Mat4 = glm::mat4;  // 4x4 矩阵
}
```

## 常用操作

### 向量创建

```cpp
#include <wr/types.h>

// 构造函数
Vec3 v1(1.0f, 2.0f, 3.0f);
Vec3 v2(5.0f);  // (5, 5, 5)
Vec3 v3;        // (0, 0, 0)

// 访问分量
float x = v1.x;  // 或 v1[0]
float y = v1.y;  // 或 v1[1]
float z = v1.z;  // 或 v1[2]
```

### 向量运算

```cpp
Vec3 a(1, 2, 3);
Vec3 b(4, 5, 6);

// 基本运算
Vec3 sum = a + b;
Vec3 diff = a - b;
Vec3 scaled = a * 2.0f;
Vec3 divided = a / 2.0f;

// 点积
float dot = glm::dot(a, b);

// 叉积
Vec3 cross = glm::cross(a, b);

// 长度
float len = glm::length(a);

// 归一化
Vec3 normalized = glm::normalize(a);

// 距离
float dist = glm::distance(a, b);

// 插值
Vec3 lerp = glm::mix(a, b, 0.5f);  // 50% 插值

// 反射
Vec3 reflected = glm::reflect(incident, normal);

// 折射
Vec3 refracted = glm::refract(incident, normal, 1.5f);

// 限制范围
Vec3 clamped = glm::clamp(a, 0.0f, 1.0f);
```

### 矩阵操作

```cpp
// 单位矩阵
Mat4 identity = glm::mat4(1.0f);

// 平移
Mat4 translation = glm::translate(identity, Vec3(1, 2, 3));

// 旋转
Mat4 rotation = glm::rotate(identity, glm::radians(45.0f), Vec3(0, 1, 0));

// 缩放
Mat4 scale = glm::scale(identity, Vec3(2, 2, 2));

// 组合变换
Mat4 transform = translation * rotation * scale;

// 矩阵乘向量
Vec4 transformed = transform * Vec4(1, 0, 0, 1);
```

### 相机变换

```cpp
// 视图矩阵 (lookAt)
Vec3 eye(0, 0, 5);
Vec3 center(0, 0, 0);
Vec3 up(0, 1, 0);
Mat4 view = glm::lookAt(eye, center, up);

// 透视投影
float fov = glm::radians(45.0f);
float aspect = 16.0f / 9.0f;
float near = 0.1f;
float far = 100.0f;
Mat4 proj = glm::perspective(fov, aspect, near, far);

// 正交投影
Mat4 ortho = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.1f, 100.0f);
```

### 角度转换

```cpp
// 度 → 弧度
float rad = glm::radians(45.0f);

// 弧度 → 度
float deg = glm::degrees(3.14159f);
```

### 常用常量

```cpp
#include <wr/utils/math.h>

using namespace wr::utils;

float pi = PI;          // 3.14159...
float twoPi = TWO_PI;   // 6.28318...
float halfPi = HALF_PI; // 1.57079...
float eps = EPSILON;    // 很小的数
```

## libWR Camera 辅助方法

Camera 结构体提供了便利方法：

```cpp
Camera camera;
camera.position = Vec3(0, 1, 3);
camera.target = Vec3(0, 0, 0);
camera.up = Vec3(0, 1, 0);
camera.fovY = glm::radians(45.0f);
camera.aspect = 16.0f / 9.0f;

// 获取方向向量
Vec3 forward = camera.getForward();
Vec3 right = camera.getRight();
Vec3 up = camera.getUp();

// 获取矩阵
Mat4 view = camera.getViewMatrix();
Mat4 proj = camera.getProjectionMatrix();
```

## 几何生成工具

libWR 提供了一些几何生成函数：

```cpp
#include <wr/utils/math.h>

std::vector<float> vertices;
std::vector<uint32_t> indices;

// 生成球体
wr::utils::createSphere(
    vertices, indices,
    0.0f, 0.0f, 0.0f,  // 中心 (cx, cy, cz)
    1.0f,               // 半径
    32,                 // 经线分段数
    24                  // 纬线分段数
);

// 添加到场景
scene->addTriangleMesh(
    std::span<const float>(vertices),
    std::span<const uint32_t>(indices),
    materialId
);
```

## 性能提示

1. **避免不必要的归一化**
   ```cpp
   // 不好
   Vec3 dir = glm::normalize(target - position);
   dir = glm::normalize(dir);  // 重复归一化
   
   // 好
   Vec3 dir = glm::normalize(target - position);
   ```

2. **使用引用传递大对象**
   ```cpp
   // 不好
   Vec3 transform(Mat4 matrix, Vec3 point) {
       return matrix * Vec4(point, 1.0f);
   }
   
   // 好
   Vec3 transform(const Mat4& matrix, const Vec3& point) {
       return matrix * Vec4(point, 1.0f);
   }
   ```

3. **预计算常量**
   ```cpp
   // 不好（循环中重复计算）
   for (int i = 0; i < 1000; ++i) {
       float angle = glm::radians(45.0f) * i;
   }
   
   // 好
   const float angleStep = glm::radians(45.0f);
   for (int i = 0; i < 1000; ++i) {
       float angle = angleStep * i;
   }
   ```

## 完整示例

```cpp
#include <wr/wr.h>
#include <wr/utils/math.h>

using namespace wr;

int main() {
    // 创建场景
    Context context;
    Scene* scene = context.createScene();
    
    // 添加材质
    uint32_t mat = scene->addLambertianMaterial(Vec3(0.8f, 0.8f, 0.8f));
    
    // 生成球体
    std::vector<float> verts;
    std::vector<uint32_t> inds;
    utils::createSphere(verts, inds, 0, 0, 0, 1.0f);
    scene->addTriangleMesh(std::span(verts), std::span(inds), mat);
    
    scene->finalize();
    
    // 设置相机
    Camera camera;
    camera.position = Vec3(0, 0, 5);
    camera.target = Vec3(0, 0, 0);
    camera.up = Vec3(0, 1, 0);
    camera.fovY = glm::radians(45.0f);
    camera.aspect = 1.0f;
    
    // 渲染
    Renderer* renderer = context.createRenderer();
    RenderParams params;
    params.width = 512;
    params.height = 512;
    params.spp = 64;
    
    std::vector<Vec3> image(params.width * params.height);
    renderer->render(scene, camera, image.data(), params);
    
    return 0;
}
```

## 参考资料

- [GLM 官方文档](https://github.com/g-truc/glm/blob/master/manual.md)
- [GLM API 参考](https://glm.g-truc.net/0.9.9/api/index.html)
- [GLSL 规范](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)
