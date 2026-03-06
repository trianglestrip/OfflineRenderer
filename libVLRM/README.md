# libVLRM - Wavefront Path Tracer

**libVLRM** (VLR Modified) 是一个严格遵循 libVLR 代码规范的 Wavefront 架构路径追踪器。

## 设计目标

- ✅ 完全遵循 libVLR 的代码风格和命名规范
- ✅ 使用 Wavefront 架构而非 Megakernel
- ✅ 保持与 libVLR 相同的宏定义和文件组织结构
- ✅ 支持 CUDA 13.1+ 和 OptiX 8.0+

## 项目结构

```
libVLRM/
├── include/VLRM/           # 公共 API 头文件
│   ├── VLRM.h              # 主入口
│   ├── common.h            # 平台宏、CUDA_DEVICE_FUNCTION、Assert 等
│   └── basic_types.h       # Vector3D、Point3D、Normal3D、RGB 等
├── shared/                 # 内部共享头文件（Host + Device）
│   ├── common_internal.h   # 内部平台宏、VLR_Device/VLR_Host
│   ├── basic_types_internal.h  # OptiX 类型转换
│   └── wavefront_types.h   # Wavefront 特定类型（RayState、Params 等）
├── GPU_kernels/            # CUDA/OptiX 内核
│   ├── ray_gen.cu          # 主光线生成（CUDA）
│   ├── wavefront_kernel.cu # Shade kernel（CUDA）
│   ├── compact.cu          # 队列压缩（CUDA）
│   ├── trace_rays.cu       # 主光线追踪（OptiX）
│   └── shadow_trace.cu     # 阴影光线追踪（OptiX）
├── utils/                  # OptiX/CUDA 工具（从 libVLR 复制）
│   ├── optix_util.h
│   ├── optixu_on_cudau.h
│   └── cuda_util.h
└── test/                   # 测试程序
    └── cornell_box_test.cpp

```

## 命名规范（遵循 libVLR）

| 项目 | 规范 | 示例 |
|------|------|------|
| 命名空间 | `vlrm` | `namespace vlrm { ... }` |
| 宏前缀 | `VLRM_` | `VLRM_M_PI`, `VLRM_Device` |
| 设备函数 | `CUDA_DEVICE_FUNCTION CUDA_INLINE` | `CUDA_DEVICE_FUNCTION float dot(...)` |
| OptiX 入口 | `CUDA_DEVICE_KERNEL void RT_RG_NAME(...)` | `RT_RG_NAME(pathTracing)` |
| 文件名 | snake_case | `wavefront_kernel.cu`, `common_internal.h` |
| 类名 | PascalCase | `Context`, `Scene`, `Renderer` |
| 函数名 | camelCase | `initialize()`, `buildAccelerationStructure()` |
| 成员变量 | m_ 前缀 + camelCase | `m_context`, `m_traversable` |

## 宏定义（与 libVLR 一致）

### 平台宏

```cpp
#if defined(__CUDA_ARCH__)
#   define VLRM_Device
#else
#   define VLRM_Host
#endif
```

### CUDA_DEVICE_FUNCTION

```cpp
#if defined(VLRM_Host)
#   define CUDA_DEVICE_FUNCTION inline
#else
#   define CUDA_DEVICE_FUNCTION __device__ __forceinline__
#endif
```

### OptiX 入口宏（继承自 optix_util.h）

```cpp
#define RT_CALLABLE_PROGRAM extern "C" __device__
#define RT_RG_NAME(name) __raygen__ ## name
#define RT_MS_NAME(name) __miss__ ## name
#define RT_CH_NAME(name) __closesthit__ ## name
#define RT_AH_NAME(name) __anyhit__ ## name
```

## Wavefront 架构

### 渲染流程

```
1. RayGen (CUDA)
   ↓
2. Trace (OptiX)
   ↓
3. Shade (CUDA)
   ↓
4. Shadow (OptiX, 可选 NEE)
   ↓
5. Compact (CUDA)
   ↓
重复 2-5 直到所有光线终止
```

### 队列结构

```cpp
struct RayState {
    float3 origin;
    float3 direction;
    RGB throughput;
    RGB radiance;
    uint32_t seed;
    uint32_t rngDimension;  // 防止屏幕空间相关性
    uint32_t pixelIndex;
    uint32_t depth;
    RayStage stage;
};
```

## 编译

### 配置

```powershell
cmake -B build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" `
  -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"
```

### 编译

```powershell
# 编译 libVLRM 库
cmake --build build --config Release --target libvlrm

# 编译测试程序
cmake --build build --config Release --target cornell_box_vlrm_test

# 运行测试
.\build\bin\Release\cornell_box_vlrm_test.exe
```

## 与 libVLR 的主要区别

| 特性 | libVLR | libVLRM |
|------|--------|---------|
| 架构 | Megakernel | Wavefront |
| 命名空间 | `vlr` | `vlrm` |
| 宏前缀 | `VLR_` | `VLRM_` |
| 光谱渲染 | 支持 | 仅 RGB |
| 材质系统 | 完整 BSDF | 简化版 |
| 相机系统 | 多种相机类型 | 简单透视相机 |
| OptiX 版本 | 7.x | 8.0+ |
| CUDA 版本 | 11.8 (compute_52) | 13.1+ (compute_75) |

## 下一步开发

- [ ] 实现 Context、Scene、Renderer 的 C++ 类
- [ ] 实现 OptiX Pipeline 和 Module 加载
- [ ] 实现完整的材质评估（Lambertian、Mirror、Glass）
- [ ] 实现 NEE（Next Event Estimation）
- [ ] 实现 Russian Roulette
- [ ] 性能优化和调试

## 参考

- libVLR: 原始 Megakernel 渲染器
- libWR: 第一个 Wavefront 实现（简化版）
- libVLRM: 遵循 libVLR 规范的 Wavefront 实现（本项目）
