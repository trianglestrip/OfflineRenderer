# 双向路径追踪（BDPT）实现计划

## 当前状态

### 已完成
- ✅ **Shading Normal**: 插值顶点法线，提升渲染质量
- ✅ **算法优化**: 
  - Firefly 抑制（亮度限制）
  - 改进的 Russian Roulette（深度自适应）
  - MIS 辅助函数（Balance/Power Heuristic）
- ✅ **非对称散射修正**: 处理 Shading Normal 与 Geometric Normal 不一致
- ✅ **BDPT 框架**: 基础数据结构和骨架代码

### 框架结构

```
libWR/kernels/bdpt/
├── bdpt_types.cuh          # PathVertex, LightPath, EyePath 数据结构
├── light_tracing.cuh       # 从光源生成路径
├── path_connection.cuh     # 连接眼睛和光源路径
└── bdpt_kernel.cu          # BDPT 主内核（未集成）
```

## 实现挑战

### 1. 架构限制

**当前架构**: Wavefront Path Tracing
- 所有光线在同一阶段并行处理
- 使用 OptiX 的 `optixTrace` 从固定的 raygen 程序发起
- 状态机驱动（Trace -> Shade -> Trace ...）

**BDPT 需求**:
- 需要从**任意表面点**发起光线追踪（Light Tracing）
- OptiX 的 `optixTrace` 只能在 raygen/closesthit/miss 程序中调用
- 不能在 CUDA kernel 中直接调用 `optixTrace`

### 2. 解决方案

#### 方案 A: 混合渲染器（推荐）
- 保留现有 Wavefront PT 作为主渲染器
- 添加独立的 BDPT 模式（新的 Pipeline 和 SBT）
- 用户可选择渲染模式

**优点**:
- 不破坏现有架构
- PT 和 BDPT 可以共存
- 实现相对简单

**缺点**:
- 需要维护两套代码
- 内存开销增加

#### 方案 B: 统一 Wavefront BDPT
- 重构 Wavefront 架构，支持双向路径
- 使用两个 ray pool（eye rays + light rays）
- 在 closesthit 中记录 path vertices

**优点**:
- 统一架构
- 可以动态混合 PT 和 BDPT 策略

**缺点**:
- 需要大规模重构
- 复杂度高
- 调试困难

### 3. 当前实现（骨架）

已创建的文件提供了：
- `PathVertex`: 路径顶点数据结构
- `LightPath` / `EyePath`: 完整路径表示
- `generateLightPath()`: 从光源采样起点（未完成追踪）
- `connectPaths()`: 连接两条路径（未完成 MIS 权重）
- `bdpt_kernel.cu`: BDPT 主循环骨架

**未完成**:
- Light Tracing 的实际光线追踪
- 完整的 MIS 权重计算
- 可见性测试（Shadow Ray）
- 与现有渲染器的集成

## 实现建议

### 短期（当前 libWR）
1. **专注于单向 Path Tracing 的质量提升**:
   - ✅ Shading Normal（已完成）
   - ✅ 非对称散射修正（已完成）
   - ✅ Firefly 抑制（已完成）
   - 🔲 GGX VNDF 采样（提升粗糙表面质量）
   - 🔲 HDR 环境贴图（IBL 照明）
   - 🔲 Normal Mapping（细节增强）

2. **保留 BDPT 框架作为未来扩展**:
   - 当前骨架代码作为参考
   - 不集成到主渲染循环
   - 等待架构重构时机

### 长期（libWR 2.0）
1. **重构为统一 Wavefront BDPT**:
   - 支持多种路径采样策略
   - 动态 MIS 权重
   - Light Vertex Cache（LVC-BPT）

2. **高级特性**:
   - Photon Mapping
   - Vertex Connection and Merging (VCM)
   - Gradient-Domain Path Tracing

## 参考资源

- [Veach Thesis (1997)](http://graphics.stanford.edu/papers/veach_thesis/): BDPT 原始论文
- [PBRT Book](https://www.pbr-book.org/): Chapter 16 - Light Transport III: Bidirectional Methods
- [libVLR Implementation](https://github.com/shocker-0x15/VLR): 参考实现（支持 LVC-BPT）
