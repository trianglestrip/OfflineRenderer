# 已知问题：斜向黑线渲染 Artifact

## 问题描述

当前 libWR 渲染器在渲染 Cornell Box 场景时会出现规律的斜向黑线条纹（diagonal black lines）。这些黑线：

- 从左上到右下
- 间距规律
- 遍布整个图像表面
- 即使在 1 SPP 时也存在（作为噪点的一部分）
- 随着 SPP 增加会收敛但不会完全消失

##调试历程

### 尝试的修复方案

1. **原子操作修复（部分有效）**
   - 为 `accumBuffer` 的所有写操作添加了 `atomicAddFloat3`
   - 为 `albedoBuffer` 和 `normalBuffer` 的累加也添加了原子操作
   - 文件：`libWR/kernels/trace.cu`, `libWR/kernels/shade.cu`

2. **Trace Stage 逻辑修复（部分有效）**
   - 修正了 `__raygen__trace` 中的条件逻辑
   - 确保 `optixTrace` 只在 `ray.stage == RayStage::Trace` 时执行
   - 文件：`libWR/kernels/trace.cu`

3. **球体网格生成修复（无效）**
   - 尝试修复 `createSphere` 函数以避免退化三角形
   - 修改了极点处理逻辑
   - 结果：黑线依然存在，且在只有平面和盒子时不出现
   - 文件：`libWR/test/helpers/geometry.cpp`

### 排除的原因

- ❌ 不是球体网格问题（平面+盒子正常，平面+球体有黑线）
- ❌ 不是 GGX 材质问题（Lambertian 也有黑线）
- ❌ 不是多 sample 累积问题（1 SPP 时就存在）
- ❌ 不是简单的 race condition（已添加原子操作）

### 可能的原因（待进一步调查）

1. **OptiX Launch 维度问题**
   - 黑线间距似乎与某些内核参数相关
   - 可能与 OptiX 的线程调度有关

2. **Wavefront 架构实现细节**
   - 可能与 ray compaction 逻辑有关
   - 可能与 active indices 的 ping-pong 缓冲有关

3. **像素索引计算问题**
   - `rayIndex % width` 和 `rayIndex / width` 的计算可能在某些情况下不正确
   - 可能与光线重用或索引映射有关

## 当前状态

- ✅ 已添加原子操作保护
- ✅ 已修正 trace stage 逻辑
- ✅ Cornell Box 基本可以渲染（虽然有黑线）
- ⚠️ 黑线问题尚未完全解决

## 下一步建议

1. **参考其他 Wavefront 实现**
   - 对比 `mchenwang/WavefrontPathTracer` 的实现
   - 检查 OptiX launch 参数和线程调度

2. **添加调试输出**
   - 在 CUDA 内核中输出关键变量
   - 验证 `rayIndex` 和 `pixelIndex` 的对应关系

3. **简化测试场景**
   - 创建最小重现场景（如单个三角形）
   - 逐步增加复杂度以定位问题

4. **考虑架构重构**
   - 如果问题持续存在，可能需要重新审视 Wavefront 实现
   - 考虑参考 libVLR 的 Megakernel 架构

## 相关文件

- `libWR/kernels/trace.cu` - OptiX raygen/closest hit
- `libWR/kernels/shade.cu` - Shading kernel  
- `libWR/kernels/compact.cu` - Ray compaction
- `libWR/src/internal/renderer.cpp` - 主渲染循环
- `libWR/test/helpers/geometry.cpp` - 几何生成
