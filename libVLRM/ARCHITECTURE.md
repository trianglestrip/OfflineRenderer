# libVLRM 架构设计

## 概述

libVLRM 是一个遵循 libVLR 代码规范的 Wavefront 架构路径追踪器。它结合了：
- libVLR 的严格代码规范和宏定义系统
- Wavefront 架构的高效 GPU 利用率
- 现代 C++20 特性（std::span）

## 架构对比

### Megakernel (libVLR)

```
┌─────────────────────────────────────┐
│   单个 OptiX RayGen Kernel          │
│                                     │
│   while (depth < maxDepth) {        │
│       trace()                       │
│       shade()                       │
│       if (hit_light) break;         │
│   }                                 │
└─────────────────────────────────────┘
```

**优点**：
- 代码简单直观
- 调试容易

**缺点**：
- 线程分化严重（不同像素路径长度不同）
- GPU 利用率低

### Wavefront (libVLRM)

```
┌──────────────┐
│  RayGen      │  生成主光线
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Trace       │  OptiX 追踪
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Shade       │  材质评估 + BSDF 采样
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Shadow      │  NEE 阴影光线（可选）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Compact     │  压缩活跃光线
└──────┬───────┘
       │
       └─────► 重复 Trace-Shade-Shadow-Compact
```

**优点**：
- 消除线程分化
- GPU 利用率高
- 易于并行化

**缺点**：
- 代码复杂度增加
- 需要队列管理

## 数据流

### RayState 队列

```cpp
struct RayState {
    float3 origin;          // 光线起点
    float3 direction;       // 光线方向
    RGB throughput;         // 累积吞吐量
    RGB radiance;           // 累积辐射度
    uint32_t seed;          // RNG seed
    uint32_t rngDimension;  // RNG 维度（防止相关性）
    uint32_t pixelIndex;    // 像素索引
    uint32_t depth;         // 路径深度
    RayStage stage;         // 当前阶段
};
```

### 阶段转换

```
RayGen:
  stage = Terminated → stage = Trace
  
Trace:
  stage = Trace → stage = Shade (hit)
  stage = Trace → stage = Terminated (miss)
  
Shade:
  stage = Shade → stage = Trace (continue)
  stage = Shade → stage = Shadow (NEE)
  stage = Shade → stage = Terminated (max depth / hit light)
  
Shadow:
  stage = Shadow → stage = Trace (after NEE)
  
Compact:
  过滤掉 stage == Terminated 的光线
```

## 内核职责

### ray_gen.cu (CUDA Kernel)

**职责**：
- 为每个像素生成主光线
- 初始化 RayState
- 写入 rayQueue

**关键代码**：

```cpp
CUDA_DEVICE_KERNEL void generateCameraRays(uint32_t sampleIndex) {
    uint32_t pixelIndex = y * params.width + x;
    
    // 初始化 RNG
    ray.seed = (pixelIndex + params.frameIndex * ...) * 1013904223u;
    ray.rngDimension = 0;  // ⚠️ 重要
    
    // 生成光线
    ray.origin = params.cameraPosition;
    ray.direction = computeRayDirection(...);
    ray.throughput = RGB(1, 1, 1);
    ray.radiance = RGB(0, 0, 0);
    ray.stage = RayStage::Trace;
}
```

### trace_rays.cu (OptiX Kernel)

**职责**：
- OptiX ClosestHit 程序
- 记录交点信息到 Payload
- OptiX Miss 程序

**关键代码**：

```cpp
CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTracingClosestHit)() {
    PTReadOnlyPayload* payload = ...;
    
    payload->hitPosition = optixGetWorldRayOrigin() + ...;
    payload->hitNormal = computeGeometricNormal();
    payload->materialId = params.materialIndices[primIdx];
    payload->t = optixGetRayTmax();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(pathTracingMiss)() {
    payload->t = -1.0f;  // 标记 miss
}
```

### wavefront_kernel.cu (CUDA Kernel)

**职责**：
- 材质评估
- BSDF 采样
- 生成次级光线
- NEE（可选）

**关键代码**：

```cpp
CUDA_DEVICE_KERNEL void shadeAndGenerateRays(uint32_t numRays) {
    RayState& ray = params.rayQueue[rayIndex];
    
    // 1. Trace
    optixTrace(..., &payload);
    
    // 2. 处理 miss
    if (payload.t < 0.0f) {
        accumulate(ray.radiance);
        ray.stage = Terminated;
        return;
    }
    
    // 3. 处理 hit light
    if (mat.type == Emissive) {
        accumulate(ray.radiance + ray.throughput * mat.emission);
        ray.stage = Terminated;
        return;
    }
    
    // 4. BSDF 采样
    float3 localDir = sampleCosineHemisphere(
        rnd_dim(ray.seed, ray.rngDimension++),  // ⚠️ 维度递增
        rnd_dim(ray.seed, ray.rngDimension++),
        bsdfPdf
    );
    
    // 5. 更新 throughput
    ray.throughput = ray.throughput * bsdf * (cosTheta / bsdfPdf);
    
    // 6. 生成次级光线
    ray.origin = hitPos + normal * epsilon;
    ray.direction = worldDir;
    ray.depth++;
    ray.stage = (ray.depth < maxDepth) ? Trace : Terminated;
}
```

### compact.cu (CUDA Kernel)

**职责**：
- 压缩光线队列
- 移除已终止的光线

**关键代码**：

```cpp
CUDA_DEVICE_KERNEL void compactRayQueue(
    RayState* inputQueue,
    uint32_t inputSize,
    RayState* outputQueue,
    uint32_t* outputSize
) {
    RayState& ray = inputQueue[rayIndex];
    
    if (ray.stage != RayStage::Terminated) {
        uint32_t outIndex = atomicAdd(outputSize, 1);
        outputQueue[outIndex] = ray;
    }
}
```

## 内存布局

### GPU 内存分配

```cpp
// 光线队列（双缓冲）
RayState* rayQueueA;  // width * height * spp
RayState* rayQueueB;
uint32_t* rayQueueSize;

// 累积缓冲区
RGB* accumBuffer;     // width * height

// 几何数据
float3* vertices;     // numVertices
uint3* indices;       // numTriangles
uint32_t* materialIndices;  // numTriangles

// 材质数据
MaterialData* materials;  // numMaterials

// 发光三角形索引（用于 NEE）
uint32_t* emissiveTriangles;  // numEmissiveTriangles
```

### 对齐要求

```cpp
// 所有 GPU 结构体必须 16 字节对齐
struct alignas(16) RayState { ... };
struct alignas(16) MaterialData { ... };
```

## 随机数生成

### 维度装饰的 PCG Hash

```cpp
__device__ __forceinline__ float rnd_dim(uint32_t seed, uint32_t dimension) {
    uint32_t state = seed * 747796405u + 2891336453u + dimension * 1013904223u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967296.0f;
}
```

### 使用模式

```cpp
// ✅ 正确：每次调用递增维度
float u1 = rnd_dim(ray.seed, ray.rngDimension++);
float u2 = rnd_dim(ray.seed, ray.rngDimension++);

// ❌ 错误：重复使用同一维度
float u1 = rnd_dim(ray.seed, 0);
float u2 = rnd_dim(ray.seed, 0);  // 会产生屏幕空间相关性！
```

## OptiX Pipeline

### Module 加载

```cpp
// CUDA 内核（PTX）
OptixModule rayGenModule;
OptixModule shadeModule;
OptixModule compactModule;

// OptiX 内核（PTX）
OptixModule traceModule;
OptixModule shadowModule;
```

### Program Group

```cpp
OptixProgramGroup raygenPG;
OptixProgramGroup missPG;
OptixProgramGroup hitgroupPG;
OptixProgramGroup shadowMissPG;
OptixProgramGroup shadowHitgroupPG;
```

### Pipeline

```cpp
OptixPipeline pipeline;
OptixShaderBindingTable sbt;
```

## 材质系统

### MaterialType

```cpp
enum class MaterialType : uint32_t {
    Lambertian = 0,  // 漫反射
    Mirror,          // 镜面反射
    Glass,           // 玻璃（折射 + 反射）
    Emissive         // 发光材质
};
```

### MaterialData

```cpp
struct MaterialData {
    MaterialType type;
    RGB albedo;      // 反射率
    RGB emission;    // 发射（仅 Emissive）
    float ior;       // 折射率（仅 Glass）
    float roughness; // 粗糙度（未来扩展）
};
```

## 渲染循环

### 主循环伪代码

```cpp
void Renderer::render(...) {
    // 1. 生成主光线
    launchRayGen(width, height, spp);
    
    // 2. Wavefront 循环
    while (rayQueueSize > 0) {
        // 2.1 追踪光线
        launchTrace(rayQueue, rayQueueSize);
        
        // 2.2 着色和生成次级光线
        launchShade(rayQueue, rayQueueSize);
        
        // 2.3 NEE 阴影光线（可选）
        if (useNEE) {
            launchShadow(rayQueue, rayQueueSize);
        }
        
        // 2.4 压缩队列
        compactQueue(rayQueue, &rayQueueSize);
    }
    
    // 3. 输出结果
    copyAccumBufferToOutput(accumBuffer, outputBuffer);
}
```

## 调试技巧

### 法线可视化

```cpp
// 在 wavefront_kernel.cu 中
if (ray.depth == 0) {
    float3 normalColor = payload.hitNormal * 0.5f + make_float3(0.5f);
    atomicAdd(&params.accumBuffer[ray.pixelIndex].r, normalColor.x);
    atomicAdd(&params.accumBuffer[ray.pixelIndex].g, normalColor.y);
    atomicAdd(&params.accumBuffer[ray.pixelIndex].b, normalColor.z);
    ray.stage = RayStage::Terminated;
    return;
}
```

### 禁用 NEE

```cpp
params.useNEE = false;  // 在 Renderer::render() 中
```

### 降低 SPP

```cpp
uint32_t spp = 16;  // 快速测试
```

## 性能考虑

### 队列大小

```cpp
// 初始队列大小：width * height * spp
// 最大队列大小：width * height * spp * 2（考虑 NEE）
```

### 内存带宽

- 使用 `float3` 而非 `float4` 节省带宽（但注意对齐）
- 累积缓冲区使用原子操作（考虑使用 shared memory 优化）

### 线程配置

```cpp
// RayGen: 2D grid
dim3 blockSize(16, 16);
dim3 gridSize((width + 15) / 16, (height + 15) / 16);

// Shade/Compact: 1D grid
dim3 blockSize(256);
dim3 gridSize((numRays + 255) / 256);
```

## 扩展计划

### 短期

- [ ] 实现完整的 Context/Scene/Renderer
- [ ] 实现 OptiX Pipeline 加载
- [ ] 实现基础材质（Lambertian）
- [ ] Cornell Box 测试通过

### 中期

- [ ] 实现 NEE（Next Event Estimation）
- [ ] 实现 Mirror 和 Glass 材质
- [ ] 实现 Russian Roulette
- [ ] 性能优化

### 长期

- [ ] 实现 GGX 微表面模型
- [ ] 实现纹理系统
- [ ] 实现环境光照
- [ ] 实现双向路径追踪

## 与其他库的关系

```
libVLR_reference (参考)
    │
    ├─ 代码规范 ───────┐
    │                  │
    └─ 宏定义系统 ─────┤
                       │
                       ▼
                   libVLRM (本项目)
                       │
                       ├─ Wavefront 架构 ← libWR (灵感)
                       │
                       └─ 现代 C++ API
```

## 参考资源

- [libVLR](../libVLR_reference/): 原始 Megakernel 实现
- [libWR](../libWR/): 简化版 Wavefront 实现
- [Megakernels Considered Harmful](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus)
- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix8/guide/index.html)
