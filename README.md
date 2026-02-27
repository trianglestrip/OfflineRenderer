# VLR: 

![VLR](README_TOP.jpg)\
IBL 图像: [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html)

VLR 是一个基于 NVIDIA OptiX 7 的 GPU 蒙特卡洛光线追踪渲染器。

## 特性
* 基于 NVIDIA OptiX 7 的 GPU 渲染器
* 全光谱渲染（蒙特卡洛光谱采样）\
  （对于 RGB 资源，使用 Meng-Simon 方法进行 RGB->光谱转换 \[Meng2015\]）
* RGB 渲染（默认构建模式）
* BSDFs
    * 理想漫反射 (Lambert) BRDF
    * 理想镜面反射 BRDF/BSDF
    * 微表面 (GGX) BRDF/BSDF
    * 菲涅尔混合朗伯 BSDF
    * 类 UE4 或 Frostbite 风格的 BRDF \[Karis2013, Lagarde2014\]\
      参数可以使用 UE4 风格（基础颜色、粗糙度/金属度）或传统风格（漫反射、高光、光泽度）来指定。
    * 混合 BSDF
* 着色器节点系统
* 凹凸贴图（法线贴图 / 高度贴图）
* Alpha 纹理
* 光源类型
    * 面积（多边形）光源
    * 点光源
    * 基于图像的环境光
* 相机类型
    * 具有景深效果的透视相机（薄透镜模型）
    * 环境（等距柱状投影）相机
* 几何体实例化
* 光线传输算法
    * 路径追踪 \[Kajiya1986\] 结合 MIS
    * 光线追踪
    * 光顶点缓存双向路径追踪 (LVC-BPT) \[Davidovi&#269;2014\]
* 正确处理由着色法线引起的非对称散射 \[Veach1997\]

## 组件
* libVLR - 基于 OptiX 的渲染器库\
  提供 C 语言 API。
* vlrcpp.h - C++ 单文件封装\
  通过 std::shared_ptr 自动管理对象生命周期。
* HostProgram - 演示 VLR 使用方法的示例程序

## API
使用 VLRCpp（C++ 封装）的代码示例

```cpp
using namespace vlr;

ContextRef context = Context::create(cuContext, enableLogging, maxCallableDepth);

// 通过定义网格和材质来构建场景

SceneRef scene = context->createScene();

TriangleMeshSurfaceNodeRef mesh = context->createTriangleMeshSurfaceNode("My Mesh 1");
{
    Vertex vertices[] = {
        Vertex{ Point3D(-1.5f,  0.0f, -1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 5.0f) },
        // ...
    };
    // ...
    mesh->setVertices(vertices, lengthof(vertices));

    {
        Image2DRef imgAlbedo = loadImage2D(context, "checkerboard.png", "Reflectance", "Rec709(D65) sRGB Gamma");
        Image2DRef imgNormalAlpha = loadImage2D(context, "normal_alpha.png", "NA", "Rec709(D65)");

        ShaderNodeRef nodeAlbedo = context->createShaderNode("Image2DTexture");
        nodeAlbedo->set("image", imgAlbedo);
        nodeAlbedo->set("min filter", "Nearest");
        nodeAlbedo->set("mag filter", "Nearest");

        ShaderNodeRef nodeNormalAlpha = context->createShaderNode("Image2DTexture");
        nodeNormalAlpha->set("image", imgNormalAlpha);

        // 可以通过连接着色器节点来灵活定义材质
        SurfaceMaterialRef mat = context->createSurfaceMaterial("Matte");
        mat->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));

        ShaderNodeRef nodeTangent = context->createShaderNode("Tangent");
        nodeTangent->set("tangent type", "Radial Y");

        uint32_t matGroup[] = { 0, 1, 2, 0, 2, 3 };
        mesh->addMaterialGroup(matGroup, lengthof(matGroup), mat, 
                               nodeNormalAlpha->getPlug(VLRShaderNodePlugType_Normal3D, 0), // 法线贴图
                               nodeTangent->getPlug(VLRShaderNodePlugType_Vector3D, 0), // 切线
                               nodeNormalAlpha->getPlug(VLRShaderNodePlugType_Alpha, 0)); // Alpha 贴图
    }

    // ...
}

// 可以通过变换构建场景图
InternalNodeRef transformNode = context->createInternalNode("trf A");
transformNode->setTransform(context->createStaticTransform(scale(2.0f)));
transformNode->addChild(mesh);
scene->addChild(transformNode);

// 设置相机
CameraRef camera = context->createCamera("Perspective");
camera->set("position", Point3D(0, 1.5f, 6.0f));
camera->set("aspect", (float)renderTargetSizeX / renderTargetSizeY);
camera->set("sensitivity", 1.0f);
camera->set("fovy", 40 * M_PI / 180);
camera->set("lens radius", 0.0f);

// 设置输出缓冲区（也可以绑定 OpenGL 缓冲区）
context->bindOutputBuffer(1024, 1024, 0);

// 开始渲染场景！
context->setScene(scene);
context->render(cuStream, camera, enableDenoiser, 1, firstFrame, &numAccumFrames);
```

## 待办事项
- [ ] 使渲染过程真正异步化。
- [ ] Python 绑定
- [ ] 简单的场景编辑器
- [ ] 使用 NVRTC 在运行时编译着色器节点，以消除可调用程序的开销。

## 已验证的运行环境
目前已在以下环境中确认程序可以正常运行。

* Windows 10 (21H2) & Visual Studio 2022 (17.2.4)
* Core i9-9900K, 32GB, RTX 3080 10GB
* NVIDIA 驱动 516.40（注意：510-512 版本存在若干 OptiX 相关问题。）

运行本程序需要以下库：

* libVLR
    * CUDA 12.5
    * OptiX 8.0.0（需要 Maxwell 或更新架构的 NVIDIA GPU）
* Host Program
    * OpenEXR 3.1
    * assimp 5.0

## 注意事项
项目中包含一些加载模型数据和纹理的场景文件，但这些资产并**未**包含在本仓库中。

## 参考文献
[Davidovi&#269;2014] "Progressive Light Transport Simulation on the GPU: Survey and Improvements"\
[Kajiya1986] "THE RENDERING EQUATION"\
[Karis2013] "Real Shading in Unreal Engine 4"\
[Lagarde2014] "Moving Frostbite to Physically Based Rendering 3.0"\
[Meng2015] "Physically Meaningful Rendering using Tristimulus Colours"\
[Veach1997] "ROBUST MONTE CARLO METHODS FOR LIGHT TRANSPORT SIMULATION"

## 画廊
<img src = "gallery/CornellBox_var.jpg" width = "512px" alt = "CornellBox_var.jpg"><br>
经典 Cornell Box 场景的变体。左侧盒子具有各向异性 BRDF，沿其局部 Y 轴具有环形切线（沿切线方向更光滑，沿副切线方向更粗糙）。
<br><br>
<img src = "gallery/UE4LikeBRDF.jpg" width = "512px" alt = "UE4LikeBRDF.jpg"><br>
一个具有类 UE4 或 Frostbite 3.0 风格 BRDF 的物体（纹理从 Substance Painter 导出），由面积光源和环境光照明。

模型: Substance Painter\
IBL 图像: [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html)
<br><br>
<img src = "gallery/dispersive_caustics_closeup.jpg" width = "512px" alt = "dispersive_caustics_closeup.jpg"><br>
由方向性面积光源照射 Stanford Bunny 模型产生的焦散效果。\
渲染器在此处使用了光谱渲染。

模型: [Stanford Bunny](http://graphics.stanford.edu/data/3Dscanrep/)
<br><br>
<img src = "gallery/Rungholt_view1.jpg" width = "768px" alt = "Rungholt_view1.jpg"><br>
<img src = "gallery/Rungholt_view2.jpg" width = "768px" alt = "Rungholt_view2.jpg"><br>
由室外环境光照明的 Rungholt 模型。

模型: Rungholt，来自 Morgan McGuire 的 [Computer Graphics Archive](https://casual-effects.com/data)\
IBL 图像 1: [Direct HDR Capture of the Sun and Sky](https://vgl.ict.usc.edu/Data/SkyProbes/)\
IBL 图像 2: [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html)

----
2022 [@Shocker_0x15](https://twitter.com/Shocker_0x15)
