# libWR 构建脚本使用指南

本项目提供了两个便捷的批处理脚本，用于快速编译和测试 libWR 库。

## 脚本列表

### 1. `build_libWR.bat` - 编译 libWR 库

编译 libWR 静态库，不包含测试程序。

**用法**：
```batch
# 编译 Release 版本（默认）
.\build_libWR.bat

# 编译 Debug 版本
.\build_libWR.bat Debug
```

**输出**：
- `build\bin\Release\libWR.lib` (或 `build\bin\Debug\libWR.lib`)
- `build\bin\Release\*.ptx` (OptiX 内核)
- `build\bin\Release\*.cubin` (CUDA 内核)

**适用场景**：
- 只需要编译库，不需要运行测试
- 检查编译错误
- 集成到其他项目

---

### 2. `run_test.bat` - 编译并运行测试

编译 libWR 库和测试程序，然后自动运行 Cornell Box 测试。

**用法**：
```batch
# 运行 Release 版本测试（默认）
.\run_test.bat

# 运行 Debug 版本测试
.\run_test.bat Debug
```

**注意**：脚本会自动切换到 `build\bin\Release` 目录运行测试，因为 PTX/CUBIN 内核文件必须与可执行文件在同一目录。

**输出**：
- 编译输出（同 `build_libWR.bat`）
- `build\bin\Release\wr_cornell_box_var_test.exe`
- `gallery\wr_cornell.png` (渲染结果)

**测试配置**：
- 从 `libWR\test\render_config.ini` 读取配置
- 默认：512x512 @ 64 spp，启用降噪

**适用场景**：
- 完整的编译和测试流程
- 验证渲染质量
- 快速迭代开发

---

## 脚本特性

### 自动 CMake 配置

如果 `build` 目录不存在，脚本会自动运行 CMake 配置：

```batch
cmake -B build ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" ^
    -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"
```

### 错误处理

- 如果 CMake 配置失败，脚本会立即退出
- 如果编译失败，脚本会显示错误信息并退出
- 如果测试运行失败，`run_test.bat` 会报告错误

### 环境要求

- Visual Studio 2022 (17.14+)
- CMake 3.26+
- CUDA Toolkit 13.1+
- OptiX SDK 8.0+
- NVIDIA GPU (compute_75+)

详见 [.cursor/rules/build-env.mdc](../.cursor/rules/build-env.mdc)

---

## 常见用法

### 日常开发流程

```batch
# 1. 修改代码
# 2. 快速测试
.\run_test.bat

# 3. 查看渲染结果
start gallery\wr_cornell.png
```

### 清理重新编译

```batch
# 删除 build 目录
Remove-Item -Path build -Recurse -Force

# 重新配置和编译
.\build_libWR.bat
```

### 编译 Debug 版本调试

```batch
# 编译 Debug 版本
.\build_libWR.bat Debug

# 运行 Debug 测试
.\run_test.bat Debug
```

---

## 手动编译（高级用法）

如果需要更细粒度的控制，可以直接使用 CMake 命令：

```powershell
# 配置
cmake -B build `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" `
    -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"

# 只编译 libWR 库
cmake --build build --config Release --target libWR

# 只编译测试程序（会自动编译 libWR）
cmake --build build --config Release --target wr_cornell_box_var_test

# 编译整个项目
cmake --build build --config Release

# 运行测试（必须在 build\bin\Release 目录）
cd build\bin\Release
.\wr_cornell_box_var_test.exe
cd ..\..\..
```

**重要**：测试程序必须在 `build\bin\Release` 目录运行，因为 PTX/CUBIN 内核文件与可执行文件在同一目录。

---

## 故障排除

### 错误：找不到 CUDA Toolkit

确保 CUDA 13.1 已安装在默认路径：
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
```

如果安装在其他路径，修改脚本中的 `-T` 参数。

### 错误：找不到 OptiX SDK

确保 OptiX SDK 8.0 已安装在默认路径：
```
C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0
```

如果安装在其他路径，修改脚本中的 `-DOptiX_INSTALL_DIR` 参数。

### 警告：pwsh.exe 不是内部或外部命令

这是无害的警告，来自 CMake 内部脚本。可以安全忽略。

详见 [.cursor/rules/cmake-build-warnings.mdc](../.cursor/rules/cmake-build-warnings.mdc)

---

## 性能提示

- **首次编译**：需要编译所有 CUDA/OptiX 内核，耗时约 60-90 秒
- **增量编译**：只重新编译修改的文件，耗时约 10-20 秒
- **测试运行**：512x512 @ 64 spp，耗时约 8-10 秒（MX550）

---

## 相关文档

- [libWR 编译配置指南](.cursor/rules/libvlrw-build.mdc)
- [环境配置](.cursor/rules/build-env.mdc)
- [项目结构](libWR/STRUCTURE.md)
- [质量改进计划](libWR/QUALITY_IMPROVEMENTS.md)
