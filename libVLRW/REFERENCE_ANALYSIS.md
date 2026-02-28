# libVLR 参考实现分析

本文档分析 libVLR_reference 中的材质系统和 OptiX 集成，为 libVLRW 的开发提供参考。

---

## 一、材质系统架构

### 1.1 材质类层次结构

```cpp
// 基类
class SurfaceMaterial : public Queryable {
    uint32_t m_matIndex;  // 材质索引
    
    // 核心接口
    virtual void setupMaterialDescriptor(CUstream stream) const = 0;
    virtual bool isEmitting() const { return false; }
};
```

### 1.2 支持的材质类型

| 材质类型 | 类名 | 用途 |
|---------|------|------|
| **漫反射** | `MatteSurfaceMaterial` | 理想漫反射（Lambert） |
| **镜面反射** | `SpecularReflectionSurfaceMaterial` | 完美镜面（金属/电介质） |
| **镜面散射** | `SpecularScatteringSurfaceMaterial` | 折射+反射（玻璃） |
| **微表面反射** | `MicrofacetReflectionSurfaceMaterial` | GGX 粗糙金属 |
| **微表面散射** | `MicrofacetScatteringSurfaceMaterial` | GGX 粗糙玻璃 |
| **Lambert 散射** | `LambertianScatteringSurfaceMaterial` | 带 Fresnel 的漫反射 |
| **UE4 材质** | `UE4SurfaceMaterial` | BaseColor + Roughness + Metallic |
| **旧式材质** | `OldStyleSurfaceMaterial` | Diffuse + Specular + Glossiness |
| **漫射发光** | `DiffuseEmitterSurfaceMaterial` | 面光源 |
| **定向发光** | `DirectionalEmitterSurfaceMaterial` | 聚光灯 |
| **点光源** | `PointEmitterSurfaceMaterial` | 点光源 |
| **环境光** | `EnvironmentEmitterSurfaceMaterial` | HDR 环境贴图 |
| **多材质** | `MultiSurfaceMaterial` | 材质混合 |

### 1.3 材质描述符（GPU 端）

```cpp
// shared/shared.h
struct SurfaceMaterialDescriptor {
    int32_t progSetupBSDF;           // BSDF 初始化程序 ID
    uint32_t bsdfProcedureSetIndex;  // BSDF 函数集索引
    int32_t progSetupEDF;            // EDF 初始化程序 ID
    uint32_t edfProcedureSetIndex;   // EDF 函数集索引
    uint32_t data[28];               // 材质参数数据（112 字节）
};
```

**关键设计**：
- 使用 **Callable Programs** 实现多态（OptiX 的动态调度）
- 材质数据直接嵌入描述符（避免额外内存访问）
- BSDF 和 EDF 分离（双向散射 + 发光）

### 1.4 材质参数示例

#### Matte 材质
```cpp
struct MatteSurfaceMaterial {
    ShaderNodePlug nodeAlbedo;   // 纹理节点（可选）
    TripletSpectrum immAlbedo;   // 立即值颜色
};
```

#### UE4 材质
```cpp
struct UE4SurfaceMaterial {
    ShaderNodePlug nodeBaseColor;
    ShaderNodePlug nodeOcclusionRoughnessMetallic;
    TripletSpectrum immBaseColor;
    float immOcclusion;
    float immRoughness;
    float immMetallic;
};
```

**设计要点**：
- 每个参数支持 **纹理节点** 或 **立即值**
- 纹理节点通过 `ShaderNodePlug` 引用（32 位索引）
- 立即值直接存储在材质数据中

---

## 二、BSDF 系统

### 2.1 BSDF 接口（GPU）

```cpp
template <TransportMode transportMode>
class BSDF {
    const SurfaceMaterialDescriptor& m_matDesc;
    const SurfacePoint& m_surfPt;
    const WavelengthSamples& m_wls;
    
public:
    // 核心方法
    SampledSpectrum getBaseColor() const;
    bool matches(DirectionType flags) const;
    
    // 重要性采样
    SampledSpectrum sample(
        const BSDFQuery& query,
        const BSDFSample& sample,
        BSDFQueryResult* result) const;
    
    // 评估 BSDF 值
    SampledSpectrum evaluate(
        const BSDFQuery& query,
        const Vector3D& dirIn) const;
    
    // 评估 PDF
    float evaluatePDF(
        const BSDFQuery& query,
        const Vector3D& dirIn) const;
};
```

### 2.2 Callable Programs 架构

```cpp
// Host 端注册
struct BSDFProcedureSet {
    int32_t progGetBaseColor;
    int32_t progMatches;
    int32_t progSampleInternal;
    int32_t progSampleWithRevInternal;
    int32_t progEvaluateInternal;
    int32_t progEvaluateWithRevInternal;
    int32_t progEvaluatePDFInternal;
    int32_t progEvaluatePDFWithRevInternal;
    int32_t progWeightInternal;
};

// GPU 端调用
optixDirectCall<SampledSpectrum>(
    progSampleInternal,
    &query, &sample, result, &matData
);
```

**优点**：
- 避免巨大的 switch-case
- 支持动态材质加载
- 每种材质独立编译

### 2.3 Fresnel 实现

#### 导体（金属）
```cpp
class FresnelConductor {
    SampledSpectrum m_eta;  // 折射率
    SampledSpectrum m_k;    // 消光系数
    
    SampledSpectrum evaluate(float cosEnter) const {
        // 复数 Fresnel 公式
        float cosEnter2 = cosEnter * cosEnter;
        SampledSpectrum _2EtaCosEnter = 2.0f * m_eta * cosEnter;
        SampledSpectrum tmp_f = m_eta * m_eta + m_k * m_k;
        SampledSpectrum tmp = tmp_f * cosEnter2;
        SampledSpectrum Rparl2 = (tmp - _2EtaCosEnter + 1) / (tmp + _2EtaCosEnter + 1);
        SampledSpectrum Rperp2 = (tmp_f - _2EtaCosEnter + cosEnter2) / (tmp_f + _2EtaCosEnter + cosEnter2);
        return (Rparl2 + Rperp2) / 2.0f;
    }
};
```

#### 电介质（玻璃）
```cpp
class FresnelDielectric {
    SampledSpectrum m_etaExt;  // 外部折射率
    SampledSpectrum m_etaInt;  // 内部折射率
    
    SampledSpectrum evaluate(float cosEnter) const {
        // 处理全反射
        SampledSpectrum sinExit = eEnter / eExit * sqrt(1 - cosEnter^2);
        if (sinExit >= 1.0f) return 1.0f;  // 全反射
        
        // Fresnel 公式
        float Rparl = ((etaExit * cosEnter) - (etaEnter * cosExit)) / 
                      ((etaExit * cosEnter) + (etaEnter * cosExit));
        float Rperp = ((etaEnter * cosEnter) - (etaExit * cosExit)) / 
                      ((etaEnter * cosEnter) + (etaExit * cosExit));
        return (Rparl^2 + Rperp^2) / 2;
    }
};
```

### 2.4 GGX 微表面模型

```cpp
class GGXMicrofacetDistribution {
    float m_alpha_gx;  // X 方向粗糙度
    float m_alpha_gy;  // Y 方向粗糙度（各向异性）
    float m_cosRt, m_sinRt;  // 旋转角度
    
    // 法线分布函数 D(m)
    float evaluate(const Normal3D &m) {
        float temp = pow2(mr.x / m_alpha_gx) + pow2(mr.y / m_alpha_gy) + pow2(mr.z);
        return 1.0f / (PI * m_alpha_gx * m_alpha_gy * pow2(temp));
    }
    
    // Smith 遮蔽函数 G1(v, m)
    float evaluateSmithG1(const Vector3D &v, const Normal3D &m) {
        float alpha_g2_tanTheta2 = (pow2(vr.x * m_alpha_gx) + pow2(vr.y * m_alpha_gy)) / pow2(vr.z);
        float Lambda = (-1 + sqrt(1 + alpha_g2_tanTheta2)) / 2;
        return 1 / (1 + Lambda);
    }
    
    // 高度相关的 Smith G(v1, v2, m)
    float evaluateHeightCorrelatedSmithG(...) {
        return 1 / (1 + Lambda1 + Lambda2);
    }
    
    // 重要性采样（VNDF）
    float sample(const Vector3D &v, float u0, float u1, Normal3D* m, float* normalPDF);
};
```

**关键技术**：
- **VNDF 采样**（Visible Normal Distribution Function）：根据视角采样微表面法线
- **高度相关 Smith G**：考虑遮蔽和阴影的相关性
- **各向异性支持**：不同方向的粗糙度

---

## 三、OptiX 集成

### 3.1 OptiX 7+ 架构

```cpp
// 初始化流程
1. optixInit()
2. optixDeviceContextCreate(cuContext, &optixContext)
3. optixModuleCreate(ptxCode, &module)
4. optixProgramGroupCreate(moduleDesc, &programGroup)
5. optixPipelineCreate(programGroups, &pipeline)
6. optixSbtRecordPackHeader(programGroup, &sbtRecord)
7. optixLaunch(pipeline, stream, params, sbt, width, height, depth)
```

### 3.2 Module 组织

```cpp
enum OptiXModule {
    OptiXModule_LightTransport,  // path_tracing.cu
    OptiXModule_ShaderNode,      // shader_nodes.cu
    OptiXModule_Material,        // materials.cu
    OptiXModule_Triangle,        // triangle.cu
    OptiXModule_Point,           // point.cu
    OptiXModule_InfiniteSphere,  // infinite_sphere.cu
    OptiXModule_Camera,          // cameras.cu
};
```

**每个 Module 编译为独立的 PTX**：
- 减少编译时间
- 支持动态加载
- 模块化设计

### 3.3 Pipeline 配置

```cpp
OptixPipelineCompileOptions pipelineCompileOptions = {};
pipelineCompileOptions.usesMotionBlur = false;
pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
pipelineCompileOptions.numPayloadValues = 16;  // Payload 寄存器数量
pipelineCompileOptions.numAttributeValues = 2; // 属性寄存器数量
pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
pipelineCompileOptions.pipelineLaunchParamsVariableName = "plp";  // Launch Params 变量名
```

### 3.4 Shader Binding Table (SBT)

```cpp
// SBT 结构
struct SBT {
    RayGenRecord*    raygenRecord;
    MissRecord*      missRecords;
    HitGroupRecord*  hitgroupRecords;
    CallableRecord*  callableRecords;
};

// HitGroup Record 示例
struct HitGroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    GeometryInstanceData data;  // 自定义数据
};
```

### 3.5 Ray Tracing 内核

#### Ray Generation
```cpp
CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTracing)() {
    uint2 launchIndex = optixGetLaunchIndex();
    
    // 1. 生成主光线
    Camera camera(...);
    camera.sample(..., &rayOrg, &rayDir);
    
    // 2. 路径追踪循环
    for (int bounce = 0; bounce < MaxPathLength; ++bounce) {
        optixTrace(
            plp.topGroup,
            rayOrg, rayDir,
            tmin, tmax,
            rayTime,
            visibilityMask,
            rayFlags,
            rayType,
            &payload
        );
        
        if (payload.terminate) break;
        
        // 更新光线
        rayOrg = payload.nextOrigin;
        rayDir = payload.nextDirection;
    }
    
    // 3. 累积结果
    plp.accumBuffer[launchIndex].add(wls, contribution);
}
```

#### Closest Hit
```cpp
CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTracingIteration)() {
    // 1. 获取交点信息
    HitPointParameter hp = HitPointParameter::get();
    SurfacePoint surfPt = calcSurfacePoint(hp);
    
    // 2. 评估材质
    BSDF bsdf(matDesc, surfPt, wls);
    EDF edf(matDesc, surfPt, wls);
    
    // 3. 隐式光源采样（直接击中光源）
    SampledSpectrum Le = edf.evaluateEmittance();
    contribution += alpha * Le * MISWeight;
    
    // 4. Next Event Estimation（显式光源采样）
    if (bsdf.hasNonDelta()) {
        SurfaceLight light = sampleLight(...);
        Vector3D shadowRay = light.position - surfPt.position;
        
        // 发射阴影光线
        bool visible = traceOcclusion(shadowRay);
        if (visible) {
            contribution += alpha * bsdf.evaluate(...) * light.Le / lightPDF;
        }
    }
    
    // 5. BSDF 采样（下一次弹射）
    BSDFQueryResult result;
    SampledSpectrum fs = bsdf.sample(query, sample, &result);
    alpha *= fs * abs(cos(result.dirLocal)) / result.dirPDF;
    
    payload.nextOrigin = surfPt.position;
    payload.nextDirection = result.dirWorld;
}
```

### 3.6 Payload 设计

```cpp
// 只读 Payload（不会被修改）
struct PTReadOnlyPayload {
    WavelengthSamples wls;
    float initImportance;
    float prevDirPDF;
    DirectionType prevSampledType;
    uint32_t pathLength;
    bool maxLengthTerminate;
};

// 只写 Payload（由 CH/MS 写入）
struct PTWriteOnlyPayload {
    Point3D nextOrigin;
    Vector3D nextDirection;
    float dirPDF;
    DirectionType sampledType;
    bool terminate;
};

// 读写 Payload（双向通信）
struct PTReadWritePayload {
    KernelRNG rng;
    SampledSpectrum alpha;
    SampledSpectrum contribution;
};
```

**优化技巧**：
- 分离只读/只写/读写 Payload，减少寄存器压力
- 使用 `optixGetPayload_N()` 访问 Payload 寄存器
- 避免在 Payload 中传递大型结构体

---

## 四、关键技术点

### 4.1 Multiple Importance Sampling (MIS)

```cpp
// Power Heuristic (β = 2)
float MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);
```

**用途**：
- 结合 BSDF 采样和光源采样
- 减少方差（特别是对于镜面材质）

### 4.2 Russian Roulette

```cpp
float continueProb = min(alpha.importance() / initImportance, 1.0f);
if (rng.getFloat0cTo1o() >= continueProb) return;
alpha /= continueProb;  // 无偏估计
```

**用途**：
- 提前终止低贡献路径
- 保持无偏性

### 4.3 光源重要性采样

```cpp
// 1. 选择 Instance
float instProb = sampleLightInstance(&inst);

// 2. 选择 GeometryInstance
float geomInstProb = sampleGeometryInstance(inst, &geomInst);

// 3. 选择三角形
float triProb = sampleTriangle(geomInst, &tri);

// 4. 在三角形上采样点
float2 uv = sampleTriangleUniform(u0, u1);
Point3D lightPos = interpolate(tri, uv);

// 总 PDF
float lightPDF = instProb * geomInstProb * triProb * areaPDF;
```

### 4.4 纹理系统

```cpp
struct Image2DTextureShaderNode {
    CUtexObject texture;         // CUDA 纹理对象
    DataFormat dataFormat;       // RGB8, RGBA16F, BC7, etc.
    SpectrumType spectrumType;   // Reflectance, LightSource, IOR
    ColorSpace colorSpace;       // Rec709, sRGB, XYZ
    BumpType bumpType;           // NormalMap, HeightMap
    ShaderNodePlug nodeTexCoord; // UV 坐标节点
    uint32_t width, height;
};
```

**支持的格式**：
- 未压缩：RGB8, RGBA8, RGBA16F, RGBA32F
- 压缩：BC1-BC7（DDS）
- 特殊：法线贴图、高度图

---

## 五、对 libVLRW 的启示

### 5.1 简化方向

| libVLR 特性 | libVLRW 实现 | 理由 |
|------------|-------------|------|
| Callable Programs | **直接函数调用** | Wavefront 架构下材质类型已知 |
| 光谱渲染 | **RGB 渲染** | 简化实现，满足大部分需求 |
| 复杂材质系统 | **基础材质** | 先实现 Matte + Emitter |
| 纹理节点系统 | **立即值参数** | 第一阶段不支持纹理 |
| 多种相机类型 | **透视相机** | 聚焦核心功能 |

### 5.2 保留的核心设计

✅ **必须保留**：
1. **SurfaceMaterialDescriptor** 结构（材质数据传递）
2. **BSDF/EDF 分离**（清晰的职责划分）
3. **Fresnel 计算**（物理正确性）
4. **GGX 微表面模型**（工业标准）
5. **MIS + Russian Roulette**（方差减少）

### 5.3 Wavefront 架构适配

```cpp
// libVLR: Megakernel (递归)
while (pathLength < MaxLength) {
    optixTrace(...);  // 递归调用
    if (terminate) break;
}

// libVLRW: Wavefront (迭代)
for (int bounce = 0; bounce < MaxBounces; ++bounce) {
    // 1. 生成光线
    generateRays<<<>>>(activeRays, rayQueue);
    
    // 2. 追踪光线
    optixLaunch(traceRays, rayQueue);
    
    // 3. 着色
    shadeHits<<<>>>(hitQueue, materialQueue);
    
    // 4. 压缩队列
    compactQueue<<<>>>(activeRays);
}
```

### 5.4 材质系统实现建议

#### 阶段 1：基础材质
```cpp
enum class MaterialType : uint8_t {
    Matte,           // 漫反射
    DiffuseEmitter,  // 面光源
};

struct MaterialDescriptor {
    MaterialType type;
    float3 albedo;      // Matte: 反照率
    float3 emittance;   // Emitter: 发光强度
    float emitScale;    // Emitter: 缩放系数
};
```

#### 阶段 2：物理材质
```cpp
enum class MaterialType : uint8_t {
    Matte,
    SpecularReflection,   // 镜面
    MicrofacetReflection, // 粗糙金属
    DiffuseEmitter,
};

struct MaterialDescriptor {
    MaterialType type;
    float3 baseColor;
    float3 eta;        // 折射率
    float3 k;          // 消光系数
    float roughness;   // 粗糙度
    float metallic;    // 金属度
    // ...
};
```

---

## 六、参考代码位置

### 材质系统
- **材质定义**：`libVLR_reference/libVLR/materials.h`
- **材质实现**：`libVLR_reference/libVLR/GPU_kernels/materials.cu`
- **共享类型**：`libVLR_reference/libVLR/shared/shared.h`

### OptiX 集成
- **上下文管理**：`libVLR_reference/libVLR/context.h`
- **工具函数**：`libVLR_reference/libVLR/utils/optix_util.cpp`
- **路径追踪**：`libVLR_reference/libVLR/GPU_kernels/path_tracing.cu`

### BSDF 实现
- **Fresnel**：`libVLR_reference/libVLR/GPU_kernels/materials.cu` (L13-L130)
- **GGX**：`libVLR_reference/libVLR/GPU_kernels/materials.cu` (L133-L230)
- **材质评估**：`libVLR_reference/libVLR/GPU_kernels/materials.cu` (L300+)

---

## 七、下一步行动

### 立即实施
1. ✅ 创建 `libVLRW/gpu/materials.cuh`（Fresnel + GGX）
2. ✅ 实现 `MatteBSDF::evaluate()` 和 `sample()`
3. ✅ 实现 `DiffuseEmitterEDF::evaluate()`
4. ✅ 在 `wavefront_kernel.cu` 中调用材质评估

### 短期目标
1. 🔲 添加 `SpecularReflectionBSDF`（完美镜面）
2. 🔲 添加 `MicrofacetReflectionBSDF`（粗糙金属）
3. 🔲 实现 Next Event Estimation（NEE）
4. 🔲 实现 Multiple Importance Sampling（MIS）

### 长期目标
1. 🔲 纹理系统（UV 映射 + 采样）
2. 🔲 法线贴图支持
3. 🔲 更多材质类型（玻璃、塑料、布料）
4. 🔲 材质分层系统（Layered Material）

---

**总结**：libVLR 提供了一个工业级的材质系统参考实现，其核心设计（BSDF/EDF 分离、Callable Programs、物理正确的 Fresnel/GGX）值得学习。libVLRW 应该保留其物理正确性，但简化实现复杂度，优先实现核心功能。
