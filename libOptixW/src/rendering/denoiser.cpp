// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/rendering/denoiser.h>
#include "../utils/checks.h"
#include <iostream>

namespace optixw {

Denoiser::Denoiser(OptixDeviceContext context)
    : m_context(context)
    , m_denoiser(nullptr)
    , m_d_state(0)
    , m_d_scratch(0)
    , m_d_intensity(0)
    , m_stateSize(0)
    , m_scratchSize(0)
    , m_d_tileBuffer(0)
    , m_d_tileInputBuffer(0)
    , m_d_tileAlbedoBuffer(0)
    , m_d_tileNormalBuffer(0)
    , m_width(0)
    , m_height(0)
    , m_useAlbedo(false)
    , m_useNormal(false)
    , m_setup(false)
{
}

Denoiser::~Denoiser() {
    destroy();
}

void Denoiser::setup(uint32_t width, uint32_t height, bool useAlbedo, bool useNormal) {
    if (m_setup) {
        destroy();
    }

    m_width = width;
    m_height = height;
    m_useAlbedo = useAlbedo;
    m_useNormal = useNormal;

    std::cout << "[Denoiser] 设置降噪器 " << width << "x" << height << std::endl;

    // 创建降噪器选项
    OptixDenoiserOptions options = {};
    options.guideAlbedo = useAlbedo ? 1 : 0;
    options.guideNormal = useNormal ? 1 : 0;

    // 创建降噪器
    OPTIX_CHECK(optixDenoiserCreate(m_context, OPTIX_DENOISER_MODEL_KIND_LDR, &options, &m_denoiser));

    // 查询内存需求
    OptixDenoiserSizes sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, width, height, &sizes));

    m_stateSize = sizes.stateSizeInBytes;
    m_scratchSize = sizes.withoutOverlapScratchSizeInBytes;

    // 分配缓冲
    allocateBuffers();

    // 设置降噪器
    OPTIX_CHECK(optixDenoiserSetup(
        m_denoiser,
        0,  // stream
        width,
        height,
        m_d_state,
        m_stateSize,
        m_d_scratch,
        m_scratchSize
    ));

    m_setup = true;
    std::cout << "[Denoiser] Setup complete" << std::endl;
}

void Denoiser::denoise(
    CUdeviceptr inputColor,
    CUdeviceptr inputAlbedo,
    CUdeviceptr inputNormal,
    CUdeviceptr output,
    CUstream stream)
{
    if (!m_setup) {
        throw std::runtime_error("Denoiser not setup");
    }

    // 设置输入层
    OptixDenoiserLayer layer = {};
    layer.input.data = inputColor;
    layer.input.width = m_width;
    layer.input.height = m_height;
    layer.input.rowStrideInBytes = m_width * sizeof(float) * 4;
    layer.input.pixelStrideInBytes = sizeof(float) * 4;
    layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    layer.output.data = output;
    layer.output.width = m_width;
    layer.output.height = m_height;
    layer.output.rowStrideInBytes = m_width * sizeof(float) * 4;
    layer.output.pixelStrideInBytes = sizeof(float) * 4;
    layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // 设置引导层
    OptixDenoiserGuideLayer guideLayer = {};
    if (m_useAlbedo) {
        guideLayer.albedo.data = inputAlbedo;
        guideLayer.albedo.width = m_width;
        guideLayer.albedo.height = m_height;
        guideLayer.albedo.rowStrideInBytes = m_width * sizeof(float) * 3;
        guideLayer.albedo.pixelStrideInBytes = sizeof(float) * 3;
        guideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    }

    if (m_useNormal) {
        guideLayer.normal.data = inputNormal;
        guideLayer.normal.width = m_width;
        guideLayer.normal.height = m_height;
        guideLayer.normal.rowStrideInBytes = m_width * sizeof(float) * 3;
        guideLayer.normal.pixelStrideInBytes = sizeof(float) * 3;
        guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    }

    // Execute denoising
    OptixDenoiserParams params = {};
    params.hdrIntensity = m_d_intensity;
    params.blendFactor = 0.0f;

    OPTIX_CHECK(optixDenoiserInvoke(
        m_denoiser,
        stream,
        &params,
        m_d_state,
        m_stateSize,
        &guideLayer,
        &layer,
        1,
        0, 0,
        m_d_scratch,
        m_scratchSize
    ));
}

void Denoiser::denoiseTiled(
    CUdeviceptr inputColor,
    CUdeviceptr inputAlbedo,
    CUdeviceptr inputNormal,
    CUdeviceptr output,
    uint32_t tileWidth,
    uint32_t tileHeight,
    uint32_t overlap,
    CUstream stream)
{
    // 分块降噪实现（简化版本）
    // 实际应该循环处理每个分块
    denoise(inputColor, inputAlbedo, inputNormal, output, stream);
}

void Denoiser::destroy() {
    if (!m_setup) return;

    freeBuffers();

    if (m_denoiser) {
        OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
        m_denoiser = nullptr;
    }

    m_setup = false;
    std::cout << "[Denoiser] Destroyed" << std::endl;
}

void Denoiser::allocateBuffers() {
    // 分配状态和临时缓冲
    if (m_stateSize > 0) {
        CU_CHECK(cuMemAlloc(&m_d_state, m_stateSize));
    }
    if (m_scratchSize > 0) {
        CU_CHECK(cuMemAlloc(&m_d_scratch, m_scratchSize));
    }

    // 分配强度缓冲
    CU_CHECK(cuMemAlloc(&m_d_intensity, sizeof(float)));
    float intensity = 1.0f;
    CU_CHECK(cuMemcpyHtoD(m_d_intensity, &intensity, sizeof(float)));
}

void Denoiser::freeBuffers() {
    if (m_d_state) {
        CU_CHECK(cuMemFree(m_d_state));
        m_d_state = 0;
    }
    if (m_d_scratch) {
        CU_CHECK(cuMemFree(m_d_scratch));
        m_d_scratch = 0;
    }
    if (m_d_intensity) {
        CU_CHECK(cuMemFree(m_d_intensity));
        m_d_intensity = 0;
    }
    if (m_d_tileBuffer) {
        CU_CHECK(cuMemFree(m_d_tileBuffer));
        m_d_tileBuffer = 0;
    }
    if (m_d_tileInputBuffer) {
        CU_CHECK(cuMemFree(m_d_tileInputBuffer));
        m_d_tileInputBuffer = 0;
    }
    if (m_d_tileAlbedoBuffer) {
        CU_CHECK(cuMemFree(m_d_tileAlbedoBuffer));
        m_d_tileAlbedoBuffer = 0;
    }
    if (m_d_tileNormalBuffer) {
        CU_CHECK(cuMemFree(m_d_tileNormalBuffer));
        m_d_tileNormalBuffer = 0;
    }
}

} // namespace optixw
