// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/scene/texture_manager.h>
#include "../utils/checks.h"
#include <iostream>
#include <windows.h>
#include <wincodec.h>

namespace optixw {

// 辅助函数：UTF-8 转宽字符
static inline std::wstring utf8ToWide(const char* text) {
    if (text == nullptr || text[0] == '\0') {
        return std::wstring();
    }
    const int len = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (len <= 0) {
        throw std::runtime_error("Failed to convert UTF-8 path to wide string.");
    }
    std::wstring wide(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, wide.data(), len);
    wide.pop_back();
    return wide;
}

// 辅助函数：sRGB 转线性
static inline float srgbToLinear(float x) {
    x = fmaxf(0.0f, fminf(1.0f, x));
    return (x <= 0.04045f) ? (x / 12.92f) : powf((x + 0.055f) / 1.055f, 2.4f);
}

// 加载图像（WIC）
static bool loadImageRGBA32F(const char* filePath, bool decodeSRGB, std::vector<float4>* outPixels, uint32_t* outWidth, uint32_t* outHeight) {
    if (!outPixels || !outWidth || !outHeight) {
        return false;
    }

    std::wstring path = utf8ToWide(filePath);
    if (path.empty()) {
        return false;
    }

    HRESULT hrInit = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool didInit = SUCCEEDED(hrInit);

    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;
    bool ok = false;
    UINT w = 0;
    UINT h = 0;
    std::vector<uint8_t> rgba;

    HRESULT hr = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = factory->CreateDecoderFromFilename(
        path.c_str(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnDemand,
        &decoder);
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = factory->CreateFormatConverter(&converter);
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = converter->Initialize(
        frame,
        GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone,
        nullptr,
        0.0,
        WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) {
        goto Cleanup;
    }

    hr = converter->GetSize(&w, &h);
    if (FAILED(hr) || w == 0 || h == 0) {
        goto Cleanup;
    }

    rgba.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4u);
    hr = converter->CopyPixels(
        nullptr,
        w * 4u,
        static_cast<UINT>(rgba.size()),
        rgba.data());
    if (FAILED(hr)) {
        goto Cleanup;
    }

    outPixels->resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (size_t i = 0; i < outPixels->size(); ++i) {
        float r = rgba[i * 4 + 0] / 255.0f;
        float g = rgba[i * 4 + 1] / 255.0f;
        float b = rgba[i * 4 + 2] / 255.0f;
        float a = rgba[i * 4 + 3] / 255.0f;
        if (decodeSRGB) {
            r = srgbToLinear(r);
            g = srgbToLinear(g);
            b = srgbToLinear(b);
        }
        (*outPixels)[i] = make_float4(r, g, b, a);
    }

    *outWidth = static_cast<uint32_t>(w);
    *outHeight = static_cast<uint32_t>(h);
    ok = true;

Cleanup:
    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (didInit) CoUninitialize();
    return ok;
}

TextureManager::TextureManager(TaskScheduler* scheduler)
    : m_scheduler(scheduler)
    , m_d_textures(0)
{
}

TextureManager::~TextureManager() {
    clear();
}

uint32_t TextureManager::loadTexture2D(const std::string& path, bool decodeSRGB) {
    TextureData tex;
    
    if (!loadImageRGBA32F(path.c_str(), decodeSRGB, &tex.pixels, &tex.width, &tex.height)) {
        throw std::runtime_error("Failed to load texture: " + path);
    }

    tex.d_pixels = 0;
    
    const uint32_t id = static_cast<uint32_t>(m_textures.size());
    m_textures.push_back(std::move(tex));
    
    std::cout << "[TextureManager] 加载纹理 " << id << ": " << path 
              << " (" << tex.width << "x" << tex.height << ")" << std::endl;
    
    return id;
}

std::vector<uint32_t> TextureManager::loadTextures(const std::vector<std::string>& paths, bool decodeSRGB) {
    std::vector<uint32_t> ids;
    ids.reserve(paths.size());

    if (paths.empty()) {
        return ids;
    }

    std::cout << "[TextureManager] 并行加载 " << paths.size() << " 个纹理..." << std::endl;

    // 使用 Taskflow 并行加载
    tf::Taskflow& taskflow = m_scheduler->createTaskflow();
    
    // 为每个纹理创建加载任务
    std::vector<tf::Task> tasks;
    std::vector<TextureData> tempTextures(paths.size());
    
    for (size_t i = 0; i < paths.size(); ++i) {
        tasks.push_back(taskflow.emplace([this, &paths, &tempTextures, i, decodeSRGB]() {
            if (!loadImageRGBA32F(paths[i].c_str(), decodeSRGB, 
                                  &tempTextures[i].pixels, 
                                  &tempTextures[i].width, 
                                  &tempTextures[i].height)) {
                std::cerr << "[TextureManager] 加载纹理失败: " << paths[i] << std::endl;
                tempTextures[i].width = 1;
                tempTextures[i].height = 1;
                tempTextures[i].pixels.resize(1, make_float4(1.0f, 0.0f, 1.0f, 1.0f));  // 错误纹理（洋红色）
            }
            tempTextures[i].d_pixels = 0;
        }));
    }

    // 执行并等待完成
    m_scheduler->run(taskflow);

    // 添加到纹理列表
    for (size_t i = 0; i < tempTextures.size(); ++i) {
        const uint32_t id = static_cast<uint32_t>(m_textures.size());
        m_textures.push_back(std::move(tempTextures[i]));
        ids.push_back(id);
    }

    std::cout << "[TextureManager] 并行加载完成" << std::endl;
    
    return ids;
}

void TextureManager::uploadToDevice() {
    if (m_textures.empty()) return;

    std::cout << "[TextureManager] 上传 " << m_textures.size() << " 个纹理到 GPU..." << std::endl;

    // 上传每个纹理的像素数据
    for (auto& tex : m_textures) {
        if (!tex.pixels.empty() && tex.d_pixels == 0) {
            const size_t size = tex.pixels.size() * sizeof(float4);
            CU_CHECK(cuMemAlloc(&tex.d_pixels, size));
            CU_CHECK(cuMemcpyHtoD(tex.d_pixels, tex.pixels.data(), size));
        }
    }

    // 创建纹理描述符数组
    std::vector<Texture2DData> textureDescs(m_textures.size());
    for (size_t i = 0; i < m_textures.size(); ++i) {
        textureDescs[i].pixels = reinterpret_cast<const float4*>(m_textures[i].d_pixels);
        textureDescs[i].width = m_textures[i].width;
        textureDescs[i].height = m_textures[i].height;
        textureDescs[i].isSRGB = 0;  // 使用isSRGB而不是pad字段
    }

    // 上传纹理描述符数组
    const size_t descSize = textureDescs.size() * sizeof(Texture2DData);
    if (m_d_textures) {
        CU_CHECK(cuMemFree(m_d_textures));
    }
    CU_CHECK(cuMemAlloc(&m_d_textures, descSize));
    CU_CHECK(cuMemcpyHtoD(m_d_textures, textureDescs.data(), descSize));

    std::cout << "[TextureManager] 纹理上传完成" << std::endl;
}

void TextureManager::clear() {
    freeDeviceBuffers();
    m_textures.clear();
}

void TextureManager::freeDeviceBuffers() {
    // 释放每个纹理的像素缓冲
    for (auto& tex : m_textures) {
        if (tex.d_pixels) {
            CU_CHECK(cuMemFree(tex.d_pixels));
            tex.d_pixels = 0;
        }
    }

    // 释放纹理描述符数组
    if (m_d_textures) {
        CU_CHECK(cuMemFree(m_d_textures));
        m_d_textures = 0;
    }
}

// ==================== 结构体版本的方法实现 ====================

uint32_t TextureManager::loadTexture2D(const TextureLoadSingleParams& params) {
    return loadTexture2D(params.path, params.decodeSRGB);
}

std::vector<uint32_t> TextureManager::loadTextures(const TextureLoadBatchParams& params) {
    return loadTextures(params.paths, params.decodeSRGB);
}

} // namespace optixw
