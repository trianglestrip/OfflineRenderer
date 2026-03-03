# optixw_test 崩溃说明

## 现象

- 运行 `optixw_test.exe` 或 `cornell_box_var_test.exe box` 时在首帧内崩溃，退出码 `-1073741819` (ACCESS_VIOLATION)。
- 崩溃发生在打印 `[Renderer] Sample 1/64`（或 1/16）之后、第一次 `cudaDeviceSynchronize()` 与 trace 完成之间，即 **OptiX trace 内核** 在 GPU 上执行时或同步时触发。

## 已排除

- 与 Shade 新逻辑无关（占位 Shade 仍崩溃）。
- 与 cout 重定向、分辨率(64x64)、工作目录(项目根 vs Release) 无关。
- 几何指针非空（gasHandle、d_vertices、d_indices、d_triangleMaterialIds 已校验）。

## 定位结论

1. **崩溃点**：`renderSample()` 内第一次 `cudaDeviceSynchronize()`（在 `optixLaunch` 之后），即 trace 内核执行/同步阶段。
2. **场景相关**：仅“box”场景（6 面+灯，12 三角形）在两种 exe 下均崩溃；增加冗余几何(14/24 三角形)仍崩溃。
3. **cornell_box_var_test**（var 场景，3084 三角形）在对话早期曾完整跑通并出图；同环境后续运行有时也会崩溃，可能与驱动/状态有关。

## 建议下一步

1. **用 Visual Studio 调试**：F5 运行 `optixw_test`，崩溃时查看调用栈与故障地址，区分是 host 端 (vector/cudaMemcpy) 还是 CUDA/OptiX 驱动内部。
2. **用 Nsight Compute / Nsight Systems**：抓一次 GPU 执行与同步，看是否报错在 trace 内核或 compact/shade。
3. **驱动与 OptiX**：本机驱动 581.95、OptiX 8.0.0；可尝试更新驱动或在不同机器上复现，排除驱动/小 GAS 兼容问题。

## 当前代码状态

- `renderer_loop.cpp`：保留 wavefront/launch 缓冲区的非空校验。
- `kernels/shade.cu`：在「无 shadow ray 直接 Terminated」分支中立即累加 `ray.radiance` 到 `accumBuffer`，避免纯黑图。
- `kernels/trace.cu`：raygen 开头对已 `Terminated` 的射线直接 return，不再调用 optixTrace。
- 测试：`cornell_box_var_test` 支持 `small`（256x256）、`box`（box 场景）；崩溃与分辨率无关（256x256 仍崩）。
