# OptiX 驱动兼容性问题

## 问题诊断

**症状**: `cornell_box_test.exe` 崩溃（exit code -1073740791），崩溃发生在 `optixInit()` 调用时。

**根本原因**: **OptiX SDK 版本与 NVIDIA 驱动版本不匹配**

## 当前环境

| 组件 | 当前版本 | 要求 | 状态 |
|------|---------|------|------|
| NVIDIA 驱动 | **581.95** | - | ✅ 已安装 |
| OptiX SDK | **9.1.0** | R590+ 驱动 | ❌ **不兼容** |
| GPU | MX550 (Turing) | compute_75 | ✅ 支持 |

## 解决方案

### 方案 1: 升级 NVIDIA 驱动（推荐）

**升级到 R590+ 驱动以支持 OptiX 9.1**

1. 下载最新驱动:
   - [NVIDIA 驱动下载页面](https://www.nvidia.com/Download/index.aspx)
   - 选择: GeForce MX550, Windows 10/11 64-bit
   - 推荐: Game Ready Driver 或 Studio Driver (R590+)

2. 安装驱动后重启电脑

3. 验证驱动版本:
   ```powershell
   nvidia-smi
   ```

### 方案 2: 降级到 OptiX 8.0（快速方案）

**使用 OptiX 8.0（支持 R545+ 驱动，当前 581.95 满足）**

1. 下载 OptiX 8.0.0:
   - [OptiX Legacy Downloads](https://developer.nvidia.com/designworks/optix/downloads/legacy)
   - 选择: OptiX SDK 8.0.0

2. 安装到默认路径:
   ```
   C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0
   ```

3. 重新配置 CMake:
   ```powershell
   Remove-Item -Path build -Recurse -Force
   
   cmake -B build `
     -G "Visual Studio 17 2022" `
     -A x64 `
     -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" `
     -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"
   ```

4. 重新编译:
   ```powershell
   cmake --build build --config Release --target cornell_box_test
   ```

## OptiX 版本对照表

| OptiX 版本 | 最低驱动 | 支持 GPU | 发布日期 |
|-----------|---------|---------|---------|
| 9.1.0 | R590+ | Turing+ | 2025-12 |
| 9.0.0 | R570+ | Turing+ | 2025-06 |
| 8.0.0 | R545+ | Maxwell+ | 2023-11 |
| 7.7.0 | R515+ | Maxwell+ | 2023-03 |

**本机驱动 581.95 兼容**: OptiX 8.0.0 ✅

## 推荐方案

**短期（快速出图）**: 使用 OptiX 8.0.0
- 优点: 无需升级驱动，立即可用
- 缺点: 缺少 OptiX 9.x 的新特性（SER, ARM 支持等）

**长期（生产环境）**: 升级驱动到 R590+，使用 OptiX 9.1
- 优点: 最新特性，更好的性能
- 缺点: 需要重启，可能影响其他应用

## 下一步

1. 选择方案 1 或方案 2
2. 按照步骤操作
3. 重新运行 `cornell_box_test.exe`
4. 验证能否成功输出 `cornell_box.png`

---

**注意**: 如果选择方案 2（OptiX 8.0），需要同步更新 `.cursor/rules/build-env.mdc` 和 `.cursor/rules/libvlrw-build.mdc` 中的 OptiX 版本信息。
