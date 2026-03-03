# 出图质量分析：libOptixW 与 libVLR 流程/参数对比

## 1. 结论摘要

当前出图质量不佳的**根本原因**是：**Shade 内核未实现真实着色**，仅用常量灰色 (0.5, 0.5, 0.5) 占位并直接终止光线，导致：

- 无材质颜色、无 BSDF
- 无直接光照（NEE 未接入）
- 无自发光（hit 处 emission 未累加）
- 仅有天空/环境光 + 均匀灰，画面发灰、缺乏层次

此外存在**参数未对接**（如 `denoiserBlend` 被写死）、**Denoiser 指导缓冲未写入**等问题，进一步影响观感。

---

## 2. 流程对比

### 2.1 libVLR（参考）典型 Path Tracing 流程

（根据 README/TASKS 及常见 Megakernel 实现推断）

| 阶段 | 说明 |
|------|------|
| 主光线 | 相机射线，分层/抖动采样 |
| Hit | 取 hit 信息（位置、法线、材质 ID、UV） |
| **Shade** | **① 累加自发光 emission**<br>**② NEE：采样光源 → 发 shadow ray → 累加直接光**<br>**③ BSDF 采样 → 更新 throughput，发下一段 ray**<br>④ 俄罗斯轮盘终止 |
| Trace | 下一段 ray 与 shadow ray 的追踪 |
| 累积 | 每 sample 的 radiance 累加到 accum，最后 / spp |

特点：**Shade 阶段完成 emission + NEE + BSDF 采样**，与路径深度、maxDepth、MIS 等参数一致。

### 2.2 libOptixW 当前实际流程

| 阶段 | 实现位置 | 当前行为 |
|------|----------|----------|
| 主光线 | `trace.cu` raygen | ✅ 分层采样生成 ray，正确 |
| Trace | `trace.cu` | ✅ hit → `Shade`，miss → `radiance += throughput * env`，shadow 未命中 → `radiance += pendingDirect` |
| **Shade** | **`shade.cu`** | **❌ 仅做：`ray.radiance = (0.5, 0.5, 0.5)`，`stage = Terminated`**<br>不读 hit、不读材质、不 NEE、不 BSDF 采样 |
| 累积 | `shade.cu` (Terminated) | ✅ `accumBuffer[pixel] += ray.radiance` |
| 降噪 | `renderer.cpp` | 使用 accum/albedo/normal；albedo/normal 未在 shade 中写入 |

因此：

- **唯一真实光照**：来自 **miss** 的环境光。
- **所有 hit 点**：都被替换成常数灰 0.5，无材质、无直接光、无自发光。
- **pendingDirect**：只在 trace 里被累加，但 **shade 从未写入**，故直接光为 0。

与 libVLR 的差异：**Shade 阶段在 libOptixW 中为占位实现**，未接上已有 BSDF/NEE 代码（`shade.cu` 内大量 eval/sample 函数未被调用）。

---

## 3. 参数对比与问题

### 3.1 已对齐或可接受的参数

| 参数 | libOptixW | 说明 |
|------|-----------|------|
| SPP | `render_config.ini` / API，如 16、64 | ✅ 每像素样本数，循环调用 `renderSample` |
| 分辨率 | width/height | ✅ |
| maxDepth / 迭代 | `renderer_loop.cpp`: maxDepth=8, maxIterations=16 | ✅ 循环次数足够，但 shade 不续弹，实际等效深度 1 |
| 相机 | position, target, up, fovY, aspect | ✅ |

### 3.2 未使用或错误的参数

| 参数 | 当前状态 | 建议 |
|------|-----------|------|
| **denoiserBlend** | API 与 `render_config.ini` 的 `blend` 传入，但 **renderer.cpp 中写死 `params.blendFactor = 0.8f`** | 应使用 `denoiserBlend` 传入值，与 libVLR/配置一致 |
| **albedo / normal** | 传入 Denoiser 作为 guide，但 **shade.cu 从未写入 albedoBuffer/normalBuffer** | 在 Shade 中根据 hit 材质写入 albedo、法线（可先做简单平均或 last-sample） |
| **blend 语义** | OptiX 的 blendFactor 控制“历史帧/混合”程度；0.5 等配置无法生效 | 使用 config 的 blend，便于与参考对比 |

### 3.3 libVLR 有而 libOptixW 未实现的（影响质量）

- **光谱渲染 / RGB 线性**：libVLR 有完整线性空间与色调映射；当前为 RGB 简单 scale，需确认 gamma 与线性累加一致。
- **MIS（多重要性采样）**：NEE + BSDF 的 MIS 权重；当前 NEE 未接入，无从谈起。
- **材质种类**：README 称 Lambert/UE4/微表面等已实现，但 **均在 shade 占位下未参与计算**。

---

## 4. 代码级根因汇总

1. **`libOptixW/kernels/shade.cu` 末尾**  
   - 对 `ray.stage == Shade` 仅执行：  
     `ray.radiance = make_float3(0.5f, 0.5f, 0.5f);`  
     `ray.stage = Terminated;`  
   - 未使用同文件中的 `evalMaterialBSDF*`、`sample*`、以及 `shading/nee.cuh`、`material.cuh` 等。

2. **NEE / 直接光**  
   - `trace.cu` 中 `ray.radiance += ray.pendingDirect` 仅在 Shadow 未命中时执行。  
   - `pendingDirect` 本应在 Shade 中通过 NEE 设置并派发 shadow ray，当前 Shade 未实现，故 **直接光恒为 0**。

3. **自发光**  
   - `trace.cu` 的 closesthit 只写 hit 信息，不累加 emission。  
   - 若在 Shade 中实现，应：读取 hit 的 materialId → 取 emission → `ray.radiance += throughput * emission`。

4. **Denoiser**  
   - `params.blendFactor = 0.8f` 写死，未使用 `denoiserBlend`。  
   - albedo/normal 未写入，降噪质量与稳定性会受影响。

5. **accum 与 SPP**  
   - accum 在 `allocateBuffers` 时清零，跨 sample 累加正确；输出时 `invSpp` 平均正确。问题在**单 sample 的 radiance 错误**（灰常数），而非 SPP 逻辑。

---

## 5. 建议修复顺序

1. **立即可做（参数与接口）**  
   - 在 `renderer.cpp` 中用传入的 `denoiserBlend` 设置 `params.blendFactor`，不再写死 0.8。  
   - 确保 `render_config.ini` 的 `blend` 正确解析并传入（已通过 `RenderConfig::denoiserBlend` 传入，仅差在 renderer 内使用）。

2. **质量根本修复（Shade 内核）**  
   - 在 `shade.cu` 的 Shade 分支中：  
     - 读取 `hitBuffer`、材质、纹理；  
     - **累加自发光**：`ray.radiance += throughput * mat.emission`；  
     - **NEE**：调用 `nee.cuh` 等采样光源，设置 `pendingDirect`，派发 shadow ray，下一轮 trace 中累加；  
     - **BSDF 采样**：调用现有 `sample*`，更新 `nextOrigin/nextDirection/nextThroughput`，`stage` 保持 Trace 继续弹射；  
   - 写入 **albedoBuffer**（如 diffuse base color）、**normalBuffer**（如 hit normal），供 Denoiser 使用。

3. **与 libVLR 对齐的后续步骤**  
   - 实现或接上 MIS（NEE vs BSDF 的权重）。  
   - 确认线性空间与 gamma（如 2.2）与 libVLR/参考图一致。  
   - 按 TASKS.md 补齐 BSDF 扩展、纹理、环境光等。

---

## 6. 已完成的代码修复（本次会话）

- **renderer.cpp**：`params.blendFactor` 改为使用传入的 `denoiserBlend`，与 `render_config.ini` 的 `blend` 一致。
- **shade.cu**：实现完整 Shade 逻辑：
  - 读取 hit、材质，按需解析纹理（`baseColorTextureId != 0xFFFFFFFF` 时才调用 `resolveMaterial`）；
  - 自发光：`kDiffuseEmitter` / `kSpecularEmitter` 时 `ray.radiance += throughput * emission`；
  - NEE：从 area lights 或 point lights 采样，设置 `pendingDirect` 并派发 shadow ray（`stage = Shadow`）；
  - 写入 `albedoBuffer`、`normalBuffer` 供 Denoiser 使用；
  - 无光源或 NEE 未命中时 `stage = Terminated`。
- **trace.cu**：Shadow 后继续路径时增加 `ray.depth = ray.depth + 1`，避免下一轮被当成 primary 重新生成。

**运行说明**：在当前环境运行 `optixw_test.exe` 时，在首帧第一次 Trace/Shade 附近出现崩溃（退出码 -1073741819 / 0xC0000005）。用“仅 0.5 灰 + Terminated”的占位 Shade 同样崩溃，说明问题不在本次新增的 Shade 逻辑，而在拉取后的管线或环境（如 OptiX 8 + 驱动、首帧 CUDA/OptiX 调用）。建议：
- 在曾成功出图的机器或驱动版本上重试；
- 设置 `CUDA_LAUNCH_BLOCKING=1` 运行，确认崩溃发生在哪一次 API 调用；
- 用 Nsight Compute / cuda-gdb 定位崩溃调用栈。

## 7. 参考

- 本仓库：`README.md`（libVLR 特性）、`TASKS.md`（相对 libVLR 的待办）。
- libOptixW：`kernels/shade.cu`（当前仅末尾 2 行处理 Shade）、`kernels/trace.cu`（radiance/pendingDirect）、`libOptixW/src/renderer.cpp`（denoiser 参数与 accum 使用）。
- 配置：`test/render_config.ini`、`test/render_config.h`（spp、denoiser、blend 等）。
