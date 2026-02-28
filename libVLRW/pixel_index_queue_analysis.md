# 像素索引与队列管理检查报告

## 1. ray_gen.cu - pixel_index 计算

**当前实现**:
```cpp
uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
uint32_t x = idx % camera.width;
uint32_t y = idx / camera.width;
ray.pixel_index = idx;
rayPool[idx] = ray;
activeQueue[idx] = idx;
```

**验证**: ✅ **正确**
- `idx = y * width + x` 等价于 `x = idx % width`, `y = idx / width`
- 行优先 (row-major)，与输出 buffer 布局一致
- 主光线阶段：`activeQueue[i] = i`，队列索引与 rayPool 索引 1:1 对应

---

## 2. trace_rays.cu - rayIndex 获取

**当前实现**:
```cpp
const uint32_t idx = optixGetLaunchIndex().x;  // 0..numActive-1
const uint32_t rayIndex = tracePlp.activeQueue[idx];
wpt::RayState& ray = tracePlp.rayPool[rayIndex];
```

**验证**: ✅ **正确**
- `idx` 为 OptiX 启动索引 (0..numActive-1)
- `activeQueue[idx]` 存储的是 rayPool 索引
- 主光线：rayIndex = idx；弹跳光线：rayIndex 来自 shade 写入的 nextQueue
- `hitBuffer[rayIndex]`、`accumBuffer[ray.pixel_index]` 使用一致

**closesthit 中**:
```cpp
const uint32_t rayIndex = tracePlp.activeQueue[launchIdx];
```
- `optixGetLaunchIndex().x` 与 raygen 中的 idx 对应同一条光线 ✅

---

## 3. renderer.cpp - active/next 队列交换

**当前逻辑**:
```cpp
d_activeCur = d_activeIndices0;
d_nextCur = d_activeIndices1;

for (depth...) {
    // trace 使用 d_activeCur
    plp.activeQueue = d_activeCur;
    optixLaunch(..., numActive, ...);

    // shade 使用 hostG.activeQueue = d_activeCur, nextQueue = d_nextCur
    hostG.activeQueue.indices = d_activeCur;
    hostG.nextQueue.indices = d_nextCur;
    wpt_launchShadeStage(...);

    numActive = nextCounters[0];
    std::swap(d_activeCur, d_nextCur);  // 下一轮 trace 使用 shade 写入的 next
}
```

**验证**: ✅ **正确**
- 每轮：trace(active) → shade(读 active，写 next) → swap
- 下一轮 trace 使用上一轮 shade 写入的 next 作为新的 active

---

## 4. queueCounters 初始化与使用

**初始化**:
```cpp
uint32_t zeroCounters[2] = { 0, 0 };
CUDA_CHECK(cuMemcpyHtoD(d_queueCounters, zeroCounters, sizeof(zeroCounters)));
```
- 每轮 depth 开始前清零 ✅

**shade 使用**:
```cpp
uint32_t nextIdx = atomicAdd(&g->queueCounters[0], 1u);
g->nextQueue.indices[nextIdx] = rayIdx;
```
- 原子递增，无竞争 ✅
- `nextIdx` 最大为 numActive-1 < numPixels，不越界 ✅

**host 读取**:
```cpp
numActive = nextCounters[0];
```
- 正确获取下一轮活跃光线数量 ✅

---

## 5. 潜在问题与建议

### 5.1 accumBuffer 写入（非原子，但当前无竞争）

**trace_rays.cu (miss)**:
```cpp
tracePlp.accumBuffer[ray.pixel_index] = tracePlp.accumBuffer[ray.pixel_index] + ray.radiance;
```

**wavefront_kernel.cu (terminate)**:
```cpp
g->accumBuffer[pi] += ray.radiance;
```

**分析**: 每轮 depth 中，每条光线对应唯一 pixel_index，无并发写同一像素。当前实现安全。若未来引入多路径/多采样，需改为 `atomicAdd`。

### 5.2 调试代码建议移除

`trace_rays.cu` 中存在针对中心像素的 printf，建议移除：
```cpp
if (rayIndex == 131328) { printf(...); }
```

### 5.3 spp 未使用

`renderer::render` 接收 `spp` 参数但未使用，当前为 1 spp，会导致明显噪点。条纹更可能来自其他原因（见下）。

---

## 6. 条纹/噪点可能来源（非索引问题）

若索引与队列逻辑正确，条纹/噪点可能来自：

1. **1 spp**：单采样路径追踪本身噪声大
2. **随机数质量**：`ray.seed` 与 LCG 可能产生可见模式
3. **浮点精度**：`1e-4f * N` 等偏移可能引入 artifacts
4. **材质/光照**：albedo、ambient 等设置

---

## 结论

**像素索引与队列管理逻辑正确**，未发现：
- 索引越界
- 队列混乱
- 像素映射错误

若仍有条纹，建议排查：spp、随机数、浮点与材质设置。
