# libVLRM 快速开始指南

## 当前状态

✅ **项目框架已完成**
- 目录结构
- 头文件和类型定义
- 内核骨架
- CMake 配置
- 测试程序骨架

⚠️ **实现状态：骨架阶段**
- API 函数标记为 `VLRMAssert_NotImplemented()`
- 需要实现完整的渲染逻辑

## 编译

### 1. 配置项目（如果还没有）

```powershell
cmake -B build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" `
  -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"
```

### 2. 编译 libVLRM

使用快捷脚本：

```powershell
.\build_libVLRM.bat
```

或手动编译：

```powershell
cmake --build build --config Release --target libvlrm --clean-first
```

### 3. 编译测试程序

```powershell
cmake --build build --config Release --target cornell_box_vlrm_test
```

### 4. 运行测试（当实现完成后）

```powershell
.\run_vlrm_test.bat
```

## 下一步开发任务

### 阶段 1：基础设施（优先级：高）

1. **实现 Context**
   - [ ] OptiX 上下文初始化
   - [ ] CUDA 上下文创建
   - [ ] 设备查询和选择

2. **实现 Scene**
   - [ ] 几何数据管理（vertices、indices）
   - [ ] 材质数据管理
   - [ ] OptiX GAS 构建
   - [ ] 发光三角形索引

3. **实现 Renderer**
   - [ ] OptiX Pipeline 创建
   - [ ] Module 加载（PTX）
   - [ ] SBT（Shader Binding Table）设置
   - [ ] GPU 内存分配（rayQueue、accumBuffer）

### 阶段 2：渲染循环（优先级：高）

4. **实现 Wavefront 循环**
   - [ ] RayGen 启动
   - [ ] Trace 启动（OptiX）
   - [ ] Shade 启动（CUDA）
   - [ ] Compact 启动（CUDA）
   - [ ] 循环终止条件

5. **实现材质评估**
   - [ ] Lambertian BSDF
   - [ ] 余弦加权半球采样
   - [ ] Throughput 更新

### 阶段 3：高级特性（优先级：中）

6. **实现 NEE**
   - [ ] 光源采样
   - [ ] 阴影光线追踪
   - [ ] MIS 权重计算

7. **实现其他材质**
   - [ ] Mirror（镜面反射）
   - [ ] Glass（折射 + 反射）

8. **实现 Russian Roulette**
   - [ ] 路径终止概率
   - [ ] Throughput 补偿

### 阶段 4：优化和调试（优先级：低）

9. **性能优化**
   - [ ] 内存访问优化
   - [ ] 队列大小调优
   - [ ] Shared memory 使用

10. **调试工具**
    - [ ] 法线可视化
    - [ ] 深度可视化
    - [ ] 性能统计

## 开发工作流

### 修改内核代码后

```powershell
# 1. 重新编译 libVLRM（强制重新编译 CUDA 内核）
.\build_libVLRM.bat

# 2. 重新编译测试程序
cmake --build build --config Release --target cornell_box_vlrm_test

# 3. 运行测试
.\build\bin\Release\cornell_box_vlrm_test.exe
```

### 只修改测试代码后

```powershell
# 1. 重新编译测试程序
cmake --build build --config Release --target cornell_box_vlrm_test

# 2. 运行测试
.\build\bin\Release\cornell_box_vlrm_test.exe
```

### 修改头文件后

```powershell
# 强制重新编译整个库
.\build_libVLRM.bat
```

## 代码规范检查清单

在提交代码前，请检查：

- [ ] 所有宏使用 `VLRM_` 前缀
- [ ] 所有类型在 `vlrm` 命名空间
- [ ] 设备函数使用 `CUDA_DEVICE_FUNCTION CUDA_INLINE`
- [ ] OptiX 入口使用 `CUDA_DEVICE_KERNEL void RT_*_NAME(...)`
- [ ] 文件名使用 snake_case
- [ ] 类名使用 PascalCase
- [ ] 函数名使用 camelCase
- [ ] 成员变量使用 m_ 前缀
- [ ] 缩进使用 4 空格
- [ ] 头文件使用 `#pragma once`
- [ ] 随机数使用 `rnd_dim(seed, dimension++)` 模式

## 参考文档

- `README.md` - 项目概述
- `ARCHITECTURE.md` - 架构设计详解
- `VLR_COMPLIANCE_CHECKLIST.md` - 规范合规性检查
- `.cursor/rules/libvlrm-conventions.mdc` - 代码规范（AI 参考）
- `.cursor/rules/rendering-debug.mdc` - 渲染调试指南

## 常见问题

### Q: 为什么要严格遵循 libVLR 规范？

A: 
1. 代码一致性和可维护性
2. 便于参考 libVLR 的实现
3. 便于未来合并或对比
4. 保持专业的代码质量

### Q: 可以使用 libVLR 的代码吗？

A: 
- ✅ 可以参考 libVLR 的算法和逻辑
- ✅ 可以复制工具函数（utils/）
- ⚠️ 需要适配 Wavefront 架构
- ⚠️ 需要更新宏前缀（VLR_ → VLRM_）

### Q: 如何调试渲染问题？

A: 参考 `.cursor/rules/rendering-debug.mdc`，包括：
- 法线可视化
- 禁用 NEE
- 降低 SPP
- 检查 NaN/Inf

### Q: 编译错误怎么办？

A: 参考 `.cursor/rules/libvlrw-build.mdc` 和 `.cursor/rules/cmake-build-warnings.mdc`

## 贡献指南

### 代码提交前

1. 运行编译测试
2. 检查代码规范
3. 更新相关文档
4. 添加必要的注释

### 提交信息格式

```
[libVLRM] 简短描述

详细说明：
- 修改了什么
- 为什么修改
- 影响范围
```

## 联系方式

如有问题，请参考：
- libVLR_reference 源代码
- OptiX Programming Guide
- CUDA C++ Programming Guide
- 项目 README 和 ARCHITECTURE 文档
