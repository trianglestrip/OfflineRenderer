# accumBuffer 累加逻辑分析报告

## 1. 数据流概览

```
ray_gen (primary) → trace_rays → [miss → accumBuffer] 或 [hit → shadeStage]
                                    ↓
                              shadeStage → [terminate → accumBuffer] 或 [bounce → nextQueue]
                                                                              ↓
                                                                    trace_rays (下一轮)
```

## 2. Miss 分支 (trace_rays.cu)

**位置**: `__raygen__trace` 中 `hitFlag == 0` 分支

```cpp
// Line 53-58
} else {
    wpt::float3_rgb bg = wpt::make_rgb(0.0f, 0.0f, 0.0f);  // black background
    ray.radiance = ray.radiance + ray.throughput * bg;
    if (tracePlp.accumBuffer) {
        tracePlp.accumBuffer[ray.pixel_index] = tracePlp.accumBuffer[ray.pixel_index] + ray.radiance;
    }
    ray.stage = wpt::Stage_Terminated;
}
```

### 验证 1: Miss 射线是否正确累加？

**✅ 正确**

- `ray.radiance` 在累加前已包含路径上所有已收集的贡献（环境光、多次 bounce 的 ambient 等）
- `ray.radiance += throughput * bg` 加入当前 miss 的环境贡献
- `accumBuffer[pi] += ray.radiance` 将完整路径 radiance 写入对应像素

### 累加链验证：Hit → Bounce → Miss

**✅ 正确**

1. **Primary 命中** → shadeStage: `ray.radiance += throughput * albedo * ambient`
2. **Bounce** → 更新 `throughput`，`radiance` 保持不变
3. **Bounce 射线 miss** → `ray.radiance += throughput * bg`，此时 `radiance` 已包含：
   - 第一次 hit 的 ambient 贡献
   - 当前 miss 的 background 贡献（throughput 已含 bounce 的 BRDF 权重）

## 3. Terminate 分支 (wavefront_kernel.cu)

**位置**: `shadeStage` 中 depth>=8 或 Russian Roulette 终止

```cpp
// Line 46-60
ray.radiance = ray.radiance + ray.throughput * albedo * ambient;  // 先累加当前 hit 贡献

if (ray.depth >= 8) {
    g->accumBuffer[pi] += ray.radiance;
    ray.stage = Stage_Terminated;
    return;
}
if (randf(ray.seed) > survivalProb) {
    g->accumBuffer[pi] += ray.radiance;
    ray.stage = Stage_Terminated;
    return;
}
```

### 验证 2: Terminate 射线是否正确累加？

**✅ 正确**

- 第 48 行**先**执行 `ray.radiance += throughput * albedo * ambient`
- 终止分支使用的 `ray.radiance` 已包含当前 hit 的 ambient
- 写入 `accumBuffer` 的是完整路径 radiance

## 4. 是否存在重复累加？

**✅ 无重复**

每条路径只会进入**一个**终止路径：

| 终止方式 | 发生位置 | 是否再次 trace | 是否再次 shade |
|---------|---------|----------------|----------------|
| Miss    | trace_rays | 否 (Stage_Terminated) | 否 (不进入 shade) |
| depth>=8 | shadeStage | 否 | 否 |
| RR 终止 | shadeStage | 否 | 否 |
| Bounce  | shadeStage | 是 (下一轮 trace) | 可能 (若再次 hit) |

- Miss 的射线：`stage=Terminated`，不会进入 shade，也不会进入下一轮 trace
- Terminate 的射线：不写入 nextQueue，不会进入下一轮 trace
- 因此每条路径在结束时只累加一次

## 5. pixel_index 映射是否正确？

**✅ 正确**

| 阶段 | pixel_index 来源 | 说明 |
|-----|------------------|------|
| ray_gen | `ray.pixel_index = idx` | idx = 0..numPixels-1，与像素一一对应 |
| Bounce | 保持不变 | `ray` 复用，`pixel_index` 不修改 |
| trace_rays | `ray.pixel_index` | 来自 rayPool，始终为初始像素 |
| shadeStage | `pi = ray.pixel_index` | 同上 |

- `accumBuffer` 大小为 `numPixels`，索引 0..numPixels-1
- `ray.pixel_index` 在整条路径上保持不变，映射正确

## 6. 潜在问题与建议

### 6.1 trace_rays 中的写法

当前：
```cpp
tracePlp.accumBuffer[ray.pixel_index] = tracePlp.accumBuffer[ray.pixel_index] + ray.radiance;
```

建议改为与 wavefront_kernel 一致的 `+=`，便于阅读和维护：
```cpp
tracePlp.accumBuffer[ray.pixel_index] += ray.radiance;
```

### 6.2 多路径 (spp>1) 时的并发

当前实现为每像素单路径（numPixels 条 primary ray），不存在同一像素多路径并发写 `accumBuffer` 的情况。

若未来支持 spp>1（每像素多条路径），需要对 `accumBuffer[pi] += ...` 使用原子加，避免写冲突。

## 7. 结论

| 检查项 | 结果 |
|-------|------|
| Miss 射线累加 | ✅ 正确 |
| Terminate 射线累加 | ✅ 正确 |
| 重复累加 | ✅ 无 |
| pixel_index 映射 | ✅ 正确 |
| Hit→Bounce→Miss 累加链 | ✅ 正确 |

**accumBuffer 的累加逻辑整体正确，无需修改核心逻辑。**
