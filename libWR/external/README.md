# libWR 第三方库

本目录包含 libWR 使用的第三方 header-only 库。

## 包含的库

### GLM (OpenGL Mathematics) v1.0.1

- **GitHub**: https://github.com/g-truc/glm
- **许可证**: MIT
- **用途**: 向量、矩阵、变换等数学运算
- **文件**: 只保留 `glm/` 头文件目录和 LICENSE.txt

**安装**:
```bash
# 下载并解压
wget https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
unzip 1.0.1.zip

# 只复制头文件
cp -r glm-1.0.1/glm ./glm/
cp glm-1.0.1/copying.txt ./glm/LICENSE.txt

# 删除其他文件（文档、测试、CMake 等）
```

### STB (Sean Barrett's libraries)

- **GitHub**: https://github.com/nothings/stb
- **许可证**: Public Domain / MIT
- **用途**: 图像保存（PNG）
- **文件**: 只保留 `stb_image_write.h`

**安装**:
```bash
# 下载单个文件
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -O stb/stb_image_write.h
```

## 目录结构

```
external/
├── glm/                    # GLM 数学库
│   ├── glm.hpp            # 主头文件
│   ├── gtc/               # 稳定扩展
│   ├── gtx/               # 实验性扩展
│   ├── detail/            # 内部实现
│   ├── ext/               # 扩展
│   ├── simd/              # SIMD 优化
│   └── LICENSE.txt        # MIT 许可证
│
└── stb/                    # STB 图像库
    └── stb_image_write.h  # PNG 保存
```

## 维护指南

### 更新 GLM

```bash
# 1. 下载新版本
wget https://github.com/g-truc/glm/archive/refs/tags/X.Y.Z.zip

# 2. 解压并替换
unzip X.Y.Z.zip
rm -rf glm/*
cp -r glm-X.Y.Z/glm/* glm/
cp glm-X.Y.Z/copying.txt glm/LICENSE.txt

# 3. 删除不需要的文件
rm -rf glm/CMakeLists.txt
```

### 更新 STB

```bash
# 下载最新版本
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -O stb/stb_image_write.h
```

## 为什么只保留头文件？

1. **减小仓库大小** - GLM 完整包含文档、测试等约 10 MB，只保留头文件约 2 MB
2. **加快克隆速度** - 更少的文件
3. **避免干扰** - GLM 的 CMakeLists.txt 可能与项目冲突
4. **Header-only** - 这些库本身就是 header-only，不需要编译

## 许可证

所有第三方库的许可证文件都保留在各自目录中：

- `glm/LICENSE.txt` - MIT License
- `stb/stb_image_write.h` - Public Domain / MIT (在文件头部)
