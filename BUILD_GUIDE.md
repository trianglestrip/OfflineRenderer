# libWR 编译指南

## 快速开始

### 修改了 CUDA 内核（.cu / .cuh 文件）

```batch
build_libWR.bat
```

这会：
- 强制重新编译 libWR 库（包括所有 CUDA 内核）
- 使用 `--clean-first` 确保 CUBIN/PTX 文件被重新生成

### 只修改了测试代码（.cpp 文件）

```batch
run_test.bat
```

这会：
- 只重新编译测试程序
- 自动运行测试
- 输出结果到 `gallery\wr_cornell.png`

## 详细说明

### 文件类型与编译策略

| 修改的文件类型 | 需要执行的命令 | 原因 |
|---------------|---------------|------|
| `libWR/kernels/*.cu` | `build_libWR.bat` | CUDA 内核需要重新编译为 CUBIN/PTX |
| `libWR/kernels/*.cuh` | `build_libWR.bat` | 头文件修改可能不会触发自动重新编译 |
| `libWR/src/*.cpp` | `build_libWR.bat` | CPU 端代码修改 |
| `libWR/include/*.h` | `build_libWR.bat` | 公共头文件修改 |
| `libWR/test/*.cpp` | `run_test.bat` | 只需重新编译测试程序 |

### 为什么修改 CUDA 内核后需要 `--clean-first`？

CUDA 内核编译流程：
1. `.cu` 文件 → NVCC 编译 → `.cubin` 或 `.ptx` 文件
2. `.cubin/.ptx` 文件被嵌入到 `libWR.lib` 中
3. 运行时从库中加载内核

**问题**：CMake 的依赖检测可能无法正确追踪 `.cuh` 头文件的修改，导致内核不会重新编译。

**解决方案**：使用 `--clean-first` 强制重新编译所有目标。

## 完整编译流程

### 首次编译

```batch
REM 配置 CMake
cmake -B build ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" ^
    -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"

REM 编译 libWR
cmake --build build --config Release --target libWR

REM 编译测试程序
cmake --build build --config Release --target wr_cornell_box_var_test

REM 运行测试
cd build\bin\Release
.\wr_cornell_box_var_test.exe
cd ..\..\..
```

### 日常开发

**修改了内核代码**：
```batch
build_libWR.bat
run_test.bat
```

**只修改了测试代码**：
```batch
run_test.bat
```

## 配置选项

### Debug vs Release

```batch
REM Debug 编译（包含调试符号）
build_libWR.bat Debug
run_test.bat Debug

REM Release 编译（优化，默认）
build_libWR.bat Release
run_test.bat Release
```

### 渲染配置

编辑 `build\bin\Release\render_config.ini`：

```ini
[cornell_box_var]
width = 512
height = 512
spp = 256           # Samples per pixel
denoiser = off      # OptiX denoiser (on/off)
```

## 常见问题

### Q: 修改了 `.cu` 文件但渲染结果没变？

**A**: 必须先运行 `build_libWR.bat` 重新编译库，再运行 `run_test.bat`。

### Q: 编译很慢怎么办？

**A**: 
- 如果只修改了测试代码，直接用 `run_test.bat`，不要用 `build_libWR.bat`
- Debug 编译比 Release 快，但运行慢
- 降低 SPP 可以加快渲染速度（用于快速测试）

### Q: 出现 "pwsh.exe 不是内部或外部命令" 警告？

**A**: 这是无害的警告，来自 CMake 内部脚本，不影响编译结果。可以忽略。

### Q: 如何清理所有编译产物？

**A**: 
```batch
rmdir /s /q build
```
然后重新运行 `build_libWR.bat` 或 `run_test.bat`（会自动配置 CMake）。

## 输出文件

编译成功后：

```
build/
├── bin/Release/
│   ├── libWR.lib                    # 静态库
│   ├── wr_cornell_box_var_test.exe  # 测试程序
│   ├── trace.ptx                    # OptiX 光线追踪内核
│   └── shade.cubin                  # CUDA 着色内核
└── libWR/Release/
    └── libWR.lib                    # 库文件副本

gallery/
└── wr_cornell.png                   # 渲染输出
```

## 性能提示

- **首次编译**：约 20-30 秒（包括 CUDA 内核编译）
- **增量编译**（只改测试代码）：约 5-10 秒
- **渲染时间**（512x512 @ 256 SPP）：约 12-15 秒（MX550）

## 相关文档

- `.cursor/rules/libvlrw-build.mdc` - 详细的编译配置和错误解决方案
- `.cursor/rules/rendering-debug.mdc` - 渲染调试指南
- `README.md` - 项目总体说明
