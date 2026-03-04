# test 出图全黑问题分析（拉取最新 + 与 VLR 对比）

## 1. 拉取最新后的变更摘要

本次拉取涉及：

- **libOptixW/kernels/shade.cu**：完整 Shade 逻辑（emission、NEE、albedo/normal 写入）
- **libOptixW/kernels/trace.cu**：Shadow 后路径 depth 递增、Terminated 提前 return
- **libOptixW/src/renderer.cpp**：denoiser blend 使用传入的 `denoiserBlend`
- **libOptixW/src/renderer_loop.cpp**：wavefront 循环与参数传递
- **libOptixW/src/scene/material_manager.cpp**：材质类型常量与接口
- **test**：cornell_box_test 精简、cornell_box_var_test 支持 box/small 及 tiling 测试
- **docs**：新增 `image_quality_analysis.md`、`optixw_test_crash_notes.md`

## 2. 与 libVLR 的流程对比

| 阶段       | libVLR（参考）                    | libOptixW 当前实现                         |
|------------|-----------------------------------|--------------------------------------------|
| 主光线     | 相机射线、分层/抖动采样           | trace raygen 分层采样，正确                 |
| Hit        | 位置、法线、材质 ID、UV           | closesthit 写入 hitBuffer，正确             |
| **Shade**  | ① emission ② NEE ③ BSDF 采样     | 已实现 emission + NEE；未接 BSDF 续弹       |
| Trace      | 下一段 ray + shadow ray           | hit→Shade，miss→radiance+=env，shadow 未命中→radiance+=pendingDirect |
| 累积       | 每 sample 累加后 / spp            | accum 在 Shade 的 Terminated 分支累加，/spp 在输出阶段 |

当前与 VLR 的主要差异：

- **Shade**：已做 emission 与 NEE（area/point lights），但**没有 BSDF 采样续弹**，路径深度实际为 1（仅 primary + 0/1 次 NEE）。
- **环境光**：若场景未设置 `setEnvironmentRadiance()`，miss 的 `radiance += throughput * env` 恒为 0。

## 3. 出图全黑的可能原因

### 3.1 环境光为 0（已修复）

- **原因**：`cornell_box_var_test` 在 var 场景下未调用 `scene->setEnvironmentRadiance()`，默认环境为 (0,0,0)。
- **结果**：所有未命中几何的射线（天空）贡献为 0，画面中“天空”和仅靠环境照亮的区域全黑。
- **修复**：在 var 分支中增加 `scene->setEnvironmentRadiance(Vec3(0.08f, 0.08f, 0.08f))`，与 cornell_box 的 0.1 类似，保证 miss 有非零贡献。

### 3.2 NEE（直接光）未生效

- **原因**：若 `numAreaLights == 0` 或 `areaLights` 在设备端未正确传递，Shade 中不会设置 `pendingDirect`，`haveShadowRay` 恒为 false。
- **结果**：除直接看到光源的像素外，墙面/地面等仅依赖 NEE 的像素为黑。
- **排查**：运行后查看控制台 `[Renderer] area lights: N`，N 应为正（Cornell 顶灯会为每个三角形注册 area light，至少为 2）。若为 0，需检查 `Scene::addTriangleMesh` 中 emissive 材质注册 area light 及 `LightManager::uploadToDevice()` 的调用时机。

### 3.3 Terminated 时 radiance 恒为 0

- **原因**：Shade 中只有 `ray.radiance.x/y/z > 0` 时才写入 `accumBuffer`。若所有 Terminated 的射线 radiance 都为 0（无环境光、无 NEE、且非 emissive hit），则 accum 全为 0。
- **结果**：整幅图黑。
- **与 VLR 对比**：VLR 的 Shade 会累加 emission + NEE + 后续弹射；当前实现只累加 emission 与 NEE（通过 shadow 在 trace 里加 pendingDirect），若两者都未贡献则必然全黑。

### 3.4 崩溃导致未写入输出（见 optixw_test_crash_notes.md）

- 若在首帧 trace 阶段崩溃（如 0xC0000005），则从未完成渲染，gallery 可能为旧图或空/黑。
- 文档结论：崩溃与本次 Shade 逻辑无关，多与 trace 内核或驱动/OptiX 环境有关，建议用 VS 调试或 Nsight 定位。

## 4. 已做的代码修改

1. **test/cornell_box_var_test.cpp**  
   - 在 var 场景分支中增加 `scene->setEnvironmentRadiance(Vec3(0.08f, 0.08f, 0.08f))`，在 `finalize()` 之前调用。

2. **libOptixW/src/renderer.cpp**  
   - 渲染前打印 `area lights` 与 `point lights` 数量，便于确认光源是否注册并传入。

## 5. 建议验证步骤

1. 重新编译并运行 var 测试（无参数或 `small`）：  
   - 若控制台出现 `[Renderer] area lights: 2`（或更多），且 `[Debug] Accum buffer max` 明显大于 0，则环境光 + NEE 已生效，画面不应再全黑。
2. 若仍全黑：  
   - 确认 `[Debug] Accum buffer max` 与中心像素数值；若仍为 0，则问题在 Shade/Trace 的累加或指针传递。  
   - 可在 Shade 的 Terminated 分支临时写死 `ray.radiance = make_float3(1,0,0)` 验证 accum 是否被写入。
3. 若出现崩溃：按 `docs/optixw_test_crash_notes.md` 用调试器/Nsight 定位 trace 阶段。

## 6. 参考文档

- `docs/image_quality_analysis.md`：Shade 占位 vs 完整实现、参数与 Denoiser 对比。
- `docs/optixw_test_crash_notes.md`：崩溃现象与当前代码状态说明。
