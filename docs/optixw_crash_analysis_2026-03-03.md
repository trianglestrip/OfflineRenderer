# libOptixW 崩溃分析 (2026-03-03)

## 当前状态

**崩溃现象**: 运行 `cornell_box_var_test.exe` 时在首帧内崩溃
- 退出码: `-1073741819` (0xC0000005 ACCESS_VIOLATION)
- 崩溃位置: 打印 `[Renderer] Sample 1/16` 之后
- 具体阶段: `renderSample()` 内第一次 `optixLaunch` 后的 `cudaDeviceSynchronize()` 调用

## 今日修复的问题

### 1. trace.cu 编译错误

**错误信息**:
```
D:\gitProject\OfflineRenderer\libOptixW\kernels\trace.cu(54): error : identifier "seed" is undefined
```

**原因**: 
- 虽然 `seed` 在同一个 if 块内声明和使用，但 nvcc 的某些优化或预处理导致作用域问题
- 变量被多次赋值（`seed = seed * ...`），可能触发编译器的作用域分析错误

**修复方案**:
将所有临时变量改为 `const`，使用不同的变量名避免重新赋值：

```cpp
// 修复前（编译失败）
uint32_t seed = (rayIndex * 1664525u + params.sampleIndex * 1013904223u) ^ 0x9e3779b9u;
float r1 = (float)(seed >> 8) / 16777216.0f;
seed = seed * 1664525u + 1013904223u;  // 重新赋值
float r2 = (float)(seed >> 8) / 16777216.0f;

// 修复后（编译成功）
const uint32_t initSeed = (rayIndex * 1664525u + params.sampleIndex * 1013904223u) ^ 0x9e3779b9u;
const float r1 = (float)(initSeed >> 8) / 16777216.0f;
const uint32_t nextSeed = initSeed * 1664525u + 1013904223u;
const float r2 = (float)(nextSeed >> 8) / 16777216.0f;
```

### 2. CMake nvcc include 路径缺失

**问题**: nvcc 编译时缺少 CUDA include 目录（特别是 `cccl`）

**修复**: 在 `libOptixW/CMakeLists.txt` 中动态构建 include 列表：

```cmake
# 修复前
COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
    -ptx
    -o ${PTX_FILE}
    ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
    -I${OptiX_INSTALL_DIR}/include
    --gpu-architecture=compute_75

# 修复后
set(NVCC_INCLUDES
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
    -I${OptiX_INSTALL_DIR}/include
)
foreach(inc_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
    list(APPEND NVCC_INCLUDES -I${inc_dir})
endforeach()

COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
    -ptx
    -o ${PTX_FILE}
    ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
    ${NVCC_INCLUDES}
    --gpu-architecture=compute_75
```

## 已验证的正确性

### 1. LaunchParams 大小一致性 ✅

创建了 `test/check_sizes.cpp` 工具验证：

```
sizeof(LaunchParams) = 200
sizeof(GeometryBuffers) = 32
sizeof(RenderBufferDataRW) = 56
sizeof(CameraData) = 56
sizeof(EnvironmentMappingData) = 32
sizeof(RayState) = 148
sizeof(HitInfo) = 48
sizeof(MaterialData) = 140
```

PTX 中声明: `.const .align 8 .b8 params[200]`  
**结论**: 完全匹配 ✅

### 2. Pipeline 和 SBT 创建成功 ✅

程序输出显示：
```
[Renderer] Creating OptiX pipeline...
[OptiX][4][COMPILER]: Info: Pipeline statistics
	module(s)                            :     1
	entry function(s)                    :     3
	trace call(s)                        :     1
[Renderer] SBT built
[Renderer] Wavefront kernels loaded
[Renderer] Pipeline created successfully
```

### 3. 缓冲区分配成功 ✅

```
[Renderer] allocateBuffers: malloc rayPool 38797312 (37 MB)
[Renderer] allocateBuffers: malloc hitBuffer 12582912 (12 MB)
[Renderer] allocateBuffers: malloc accumBuffer 3145728 (3 MB)
[Renderer] Allocated buffers: 262144 pixels, 262144 rays
```

所有 CUDA 内存分配均成功，无错误。

### 4. GAS 构建成功 ✅

```
[GeometryManager] 构建 GAS: 3084 个三角形
[GeometryManager] GAS 构建完成
[Renderer] GAS handle: 30134081034
```

## 当前崩溃分析

### 崩溃发生的精确位置

根据输出日志，崩溃发生在：
1. 打印 `[Renderer] Sample 1/16` 之后
2. `renderSample()` 函数内部
3. 第一次 `optixLaunch` 调用之后
4. `cudaDeviceSynchronize()` 调用期间或之后

### 已排除的可能原因

- ❌ Shade 内核逻辑错误（使用占位 Shade 仍崩溃）
- ❌ cout 重定向问题
- ❌ 工作目录问题
- ❌ 缓冲区空指针
- ❌ LaunchParams 大小不匹配
- ❌ Pipeline/SBT 配置错误
- ❌ 内存分配失败

### 可能的原因

#### 1. OptiX trace 内核内部错误 (最可能)

**症状**: 崩溃发生在 GPU 同步时，说明 GPU 内核执行时发生了错误

**可能的具体原因**:
- `__raygen__trace` 中的内存访问越界
- `__closesthit__trace` 中的非法指针解引用
- `params` 常量内存访问错误
- `optixTrace` 参数错误（如 tMin/tMax、ray flags）

**需要检查的代码**:
```cpp
// trace.cu:91-104
optixTrace(
    params.traversable,      // GAS handle 已验证非空
    ray.origin,              // 需要验证是否为合法值
    ray.direction,           // 需要验证是否归一化
    ray.tMin,                // 0.001f
    ray.tMax,                // 1e20f
    0.0f,                    // rayTime
    OptixVisibilityMask(255),
    OPTIX_RAY_FLAG_NONE,
    0,                       // SBT offset
    1,                       // SBT stride
    0,                       // missSBTIndex
    hitFlag                  // payload
);
```

#### 2. 异步 GPU 错误延迟报告

**症状**: CUDA 异步操作的错误可能延迟到下一次同步时才报告

**可能的具体原因**:
- 之前的 `cudaMemcpy` 操作实际失败（但未检查返回值）
- GAS 构建时的内存损坏
- 缓冲区分配时的对齐问题

#### 3. GPU 驱动/OptiX 兼容性问题

**环境**:
- GPU: NVIDIA GeForce MX550 (Turing, compute_75)
- 驱动: 581.95
- OptiX: 8.0.0
- CUDA: 13.1

**可能的问题**:
- 驱动 581.95 与 OptiX 8.0.0 的已知 bug
- compute_75 (Turing) 与 OptiX 8.0 的兼容性问题
- MX550 (移动 GPU) 的特殊限制

## 建议的调试步骤

### 1. Visual Studio 调试器 (最高优先级)

**步骤**:
1. 打开 `D:\gitProject\OfflineRenderer\build\OfflineRenderer.sln`
2. 右键 `cornell_box_var_test` → 设为启动项目
3. 按 F5 启动调试
4. 崩溃时观察：
   - 调用栈 (Call Stack)
   - 异常详情 (Exception Details)
   - CUDA 错误码
   - 局部变量值

**期望结果**:
- 如果调用栈显示在 `optix*.dll` 或 `nvcuda.dll` 内部，说明是驱动/OptiX 内部错误
- 如果调用栈显示在 `trace.cu` 的某一行，说明是内核代码错误
- 如果显示在 `cudaMemcpy` 或其他 host 代码，说明是之前的异步错误

### 2. 添加详细的 CUDA 错误检查

在 `renderer_loop.cpp` 中每个 CUDA 调用后添加同步和错误检查：

```cpp
CUDA_CHECK(cudaMemcpy(...));
CUDA_CHECK(cudaDeviceSynchronize());  // 立即同步，暴露异步错误

OPTIX_CHECK(optixLaunch(...));
CUDA_CHECK(cudaDeviceSynchronize());  // 检查 OptiX launch 是否成功
```

### 3. 简化测试场景

创建一个最小的测试场景：
- 单个三角形
- 单个像素 (1x1)
- 单个 sample
- 无材质、无光源

逐步增加复杂度，定位触发崩溃的最小配置。

### 4. 使用 Nsight Compute 或 cuda-gdb

**Nsight Compute**:
```bash
ncu --target-processes all cornell_box_var_test.exe
```

**cuda-gdb** (Linux):
```bash
cuda-gdb cornell_box_var_test
(cuda-gdb) run
(cuda-gdb) bt  # 崩溃时查看调用栈
```

### 5. 驱动/硬件测试

- 更新 NVIDIA 驱动到最新版本 (R600+)
- 尝试 OptiX 9.1 (需要驱动 R590+)
- 在不同机器上测试（如果可能）

## 代码修改记录

### 修改的文件

1. `libOptixW/kernels/trace.cu`
   - 修复 seed 作用域问题
   - 使用 const 变量避免重新赋值

2. `libOptixW/CMakeLists.txt`
   - 修复 nvcc include 路径
   - 动态添加所有 CUDA include 目录

3. `test/check_sizes.cpp` (新增)
   - 验证结构体大小一致性

4. `test/CMakeLists.txt`
   - 添加 check_sizes 目标

5. `libOptixW/src/renderer.cpp`
   - 添加调试输出 `[Renderer] render() entered`

6. `test/cornell_box_var_test.cpp`
   - 移除 cout 重定向（临时，用于调试）
   - 添加详细的调试输出

### 待回滚的临时修改

- `renderer.cpp` 中的 `[Renderer] render() entered` 输出
- `cornell_box_var_test.cpp` 中的调试输出和 cout 重定向移除

## 下一步行动

**必须**:
1. 使用 Visual Studio 调试器运行 `cornell_box_var_test`，获取崩溃调用栈

**可选**:
2. 添加详细的 CUDA 错误检查和同步
3. 创建最小测试场景
4. 使用 Nsight Compute 分析 GPU 执行
5. 更新驱动或在其他机器上测试

**注意**: 由于 AI 无法直接操作 Visual Studio 调试器，需要用户手动执行步骤 1。
