# libVLRW 更新日志

## 2026-02-28 - 配置文件系统与点光源支持

### 新增功能

#### 1. INI 配置文件系统
- **文件**: `config.ini` (项目根目录)
- **实现**: `libVLRW/include/vlrw/config.h`
- **功能**:
  - 支持 `[Section]` 分组
  - 支持注释（`#` 和 `;`）
  - 类型安全的 getter：`getInt()`, `getFloat()`, `getRGB()`, `getBool()`
  - 默认值支持

**配置项**:
```ini
[Render]
spp = 4              # 每像素采样数（默认4）
maxDepth = 8         # 最大光线反弹次数
width = 512          # 图像宽度
height = 512         # 图像高度

[Camera]
position = 0.0, 1.0, 0.6    # 相机位置
target = 0.0, 1.0, -0.5     # 观察目标
up = 0.0, 1.0, 0.0          # 上向量
fov = 60.0                  # 视场角（度）

[Light]
position = 0.0, 1.8, 0.0    # 点光源位置
intensity = 20.0, 20.0, 20.0 # 光源强度
```

#### 2. 点光源支持（Next Event Estimation）
- **API**: `Scene::addPointLight(const PointLightDesc&)`
- **GPU 数据结构**: `wpt::PointLight` (position, intensity)
- **渲染器集成**:
  - 自动上传点光源数据到 GPU
  - 在 `shadeStage` kernel 中实现 NEE
  - 支持多个点光源（循环采样）
  - 使用平方反比衰减

**NEE 实现细节**:
```cuda
// 对每个点光源
for (uint32_t li = 0; li < g->numLights; ++li) {
    float3 toLight = light.position - hitPosition;
    float distSq = dot(toLight, toLight);
    float3 L = normalize(toLight);
    
    // Lambert BRDF
    float3_rgb brdf = albedo / PI;
    
    // 光照贡献（平方反比衰减）
    float3_rgb Li = light.intensity / distSq;
    float NdotL = dot(N, L);
    
    ray.radiance += ray.throughput * brdf * Li * NdotL;
}
```

**注意**: 当前版本**未实现阴影光线追踪**，会产生光线穿透（light leaking）。

#### 3. 测试场景更新
- **文件**: `libVLRW/test/cornell_box_test.cpp`
- **改进**:
  - 从 `config.ini` 加载所有参数
  - 自动添加配置的点光源
  - 支持多采样（spp > 1）
  - 更灵活的场景配置

### 渲染效果改进

**当前效果** (spp=4, 点光源在天花板):
- ✅ 点光源照明可见（白色物体被照亮）
- ✅ 颜色渗透正确（蓝/绿/红墙的间接光照）
- ✅ 多次反弹工作正常
- ⚠️ 噪点较多（spp=4 不足）
- ⚠️ 光线穿透（无阴影光线）

**与之前对比**:
| 特性 | 之前 (spp=1, 无点光源) | 现在 (spp=4, 点光源+NEE) |
|------|----------------------|------------------------|
| 采样数 | 1 | 4 |
| 光源 | 仅环境光 (0.05) | 点光源 + 微弱环境光 (0.01) |
| 直接光照 | ❌ | ✅ (NEE) |
| 噪点 | 极高 | 中等 |
| 亮度 | 过暗 | 正常 |

### 技术细节

#### GlobalState 扩展
```cuda
struct GlobalState {
    // ... 原有字段 ...
    PointLight* lights;      // 新增：点光源数组
    uint32_t numLights;      // 新增：光源数量
};
```

#### 渲染器内存管理
- 新增 `d_lights` 设备内存分配
- 自动上传 `PointLightDesc` → `wpt::PointLight`
- 在渲染结束时释放

#### 环境光调整
- 从 `0.05` 降低到 `0.01`
- 原因：点光源成为主要光源，环境光仅作为填充

### 已知限制

1. **无阴影光线追踪**
   - 点光源直接光照不检查遮挡
   - 会产生光线穿透墙壁的现象
   - 需要实现 shadow ray tracing

2. **噪点**
   - spp=4 仍不足以完全消除噪点
   - 建议 spp ≥ 16 用于生产

3. **性能**
   - 每个着色点对所有光源循环
   - 多光源场景可能较慢
   - 未来可考虑光源重要性采样

### 下一步建议

**优先级 1 - 阴影光线**:
```cuda
// 在 NEE 中添加
bool visible = traceShadowRay(position, toLight, dist);
if (visible) {
    ray.radiance += directLight;
}
```

**优先级 2 - 提高 spp**:
- 修改 `config.ini` 中的 `spp = 16` 或更高
- 考虑渐进式渲染（累积多帧）

**优先级 3 - 面光源**:
- 实现 area light 采样
- 更真实的 Cornell Box 天花板光源

### 文件变更

**新增**:
- `config.ini` - 渲染配置文件
- `libVLRW/include/vlrw/config.h` - 配置解析器
- `CHANGELOG.md` - 本文件

**修改**:
- `libVLRW/gpu/wavefront_types.cuh` - 添加 `PointLight` 和 `GlobalState.lights`
- `libVLRW/gpu/wavefront_kernel.cu` - 实现 NEE
- `libVLRW/src/scene.h` - 添加 `getLights()`
- `libVLRW/src/renderer.cpp` - 上传点光源数据
- `libVLRW/test/cornell_box_test.cpp` - 使用配置文件
- `.gitignore` - 忽略 `VLR_ref/` 和 `libVLR_reference/`

### 使用方法

```bash
# 1. 编辑配置文件
notepad config.ini

# 2. 编译
cmake --build build --config Release --target cornell_box_test

# 3. 复制配置文件到输出目录
copy config.ini build\bin\Release\

# 4. 运行
cd build\bin\Release
.\cornell_box_test.exe

# 5. 查看结果
cornell_box.png
```

### 性能数据

**测试环境**: NVIDIA GeForce MX550 (Turing, compute_75)

| 配置 | 渲染时间 | 图像质量 |
|------|---------|---------|
| 512×512, spp=1 | ~1s | 噪点极高 |
| 512×512, spp=4 | ~4s | 噪点中等 |
| 512×512, spp=16 (预估) | ~16s | 噪点较低 |

---

**提交者**: AI Assistant  
**日期**: 2026-02-28  
**相关 Issue**: 用户请求 "弄一个配置文件.ini spp 默认为4，测试里房间里一个点光源，方便测试多次反射"
