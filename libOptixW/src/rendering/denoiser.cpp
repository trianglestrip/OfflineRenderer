// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/rendering/denoiser.h>
#include "../utils/checks.h"
#include <iostream>

namespace optixw {

Denoiser::Denoiser(OptixDeviceContext context)
    : m_context(context)
    , m_denoiser(nullptr)
    , m_buffers({0, 0, 0, 0, 0})
    , m_tileBuffers({0, 0, 0, 0})
    , m_params({0, 0, false, false, false})
{
}

Denoiser::~Denoiser() {
    destroy();
}

void Denoiser::setup(uint32_t width, uint32_t height, bool useAlbedo, bool useNormal) {
    if (m_params.setup) {
        destroy();
    }

    m_params.width = width;
    m_params.height = height;
    m_params.useAlbedo = useAlbedo;
    m_params.useNormal = useNormal;

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

    m_buffers.stateSize = sizes.stateSizeInBytes;
    m_buffers.scratchSize = sizes.withoutOverlapScratchSizeInBytes;

    // 分配缓冲
    allocateBuffers();

    // 设置降噪器
    OPTIX_CHECK(optixDenoiserSetup(
        m_denoiser,
        0,  // stream
        width,
        height,
        m_buffers.d_state,
        m_buffers.stateSize,
        m_buffers.d_scratch,
        m_buffers.scratchSize
    ));

    m_params.setup = true;
    std::cout << "[Denoiser] Setup complete" << std::endl;
}

void Denoiser::denoise(
    CUdeviceptr inputColor,
    CUdeviceptr inputAlbedo,
    CUdeviceptr inputNormal,
    CUdeviceptr output,
    CUstream stream)
{
    if (!m_params.setup) {
        throw std::runtime_error("Denoiser not setup");
    }

    // 设置输入层
    OptixDenoiserLayer layer = {};
    layer.input.data = inputColor;
    layer.input.width = m_params.width;
    layer.input.height = m_params.height;
    layer.input.rowStrideInBytes = m_params.width * sizeof(float) * 4;
    layer.input.pixelStrideInBytes = sizeof(float) * 4;
    layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    layer.output.data = output;
    layer.output.width = m_params.width;
    layer.output.height = m_params.height;
    layer.output.rowStrideInBytes = m_params.width * sizeof(float) * 4;
    layer.output.pixelStrideInBytes = sizeof(float) * 4;
    layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // 设置引导层
    OptixDenoiserGuideLayer guideLayer = {};
    if (m_params.useAlbedo) {
        guideLayer.albedo.data = inputAlbedo;
        guideLayer.albedo.width = m_params.width;
        guideLayer.albedo.height = m_params.height;
        guideLayer.albedo.rowStrideInBytes = m_params.width * sizeof(float) * 3;
        guideLayer.albedo.pixelStrideInBytes = sizeof(float) * 3;
        guideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    }

    if (m_params.useNormal) {
        guideLayer.normal.data = inputNormal;
        guideLayer.normal.width = m_params.width;
        guideLayer.normal.height = m_params.height;
        guideLayer.normal.rowStrideInBytes = m_params.width * sizeof(float) * 3;
        guideLayer.normal.pixelStrideInBytes = sizeof(float) * 3;
        guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    }

    // Execute denoising
    OptixDenoiserParams params = {};
    params.hdrIntensity = m_buffers.d_intensity;
    params.blendFactor = 0.0f;

    OPTIX_CHECK(optixDenoiserInvoke(
        m_denoiser,
        stream,
        &params,
        m_buffers.d_state,
        m_buffers.stateSize,
        &guideLayer,
        &layer,
        1,
        0, 0,
        m_buffers.d_scratch,
        m_buffers.scratchSize
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
    if (!m_params.setup) return;

    freeBuffers();

    if (m_denoiser) {
        OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
        m_denoiser = nullptr;
    }

    m_params.setup = false;
    std::cout << "[Denoiser] Destroyed" << std::endl;
}

void Denoiser::allocateBuffers() {
    // 分配状态和临时缓冲
    if (m_buffers.stateSize > 0) {
        CU_CHECK(cuMemAlloc(&m_buffers.d_state, m_buffers.stateSize));
    }
    if (m_buffers.scratchSize > 0) {
        CU_CHECK(cuMemAlloc(&m_buffers.d_scratch, m_buffers.scratchSize));
    }

    // 分配强度缓冲
    CU_CHECK(cuMemAlloc(&m_buffers.d_intensity, sizeof(float)));
    float intensity = 1.0f;
    CU_CHECK(cuMemcpyHtoD(m_buffers.d_intensity, &intensity, sizeof(float)));
}

void Denoiser::freeBuffers() {
    if (m_buffers.d_state) {
        CU_CHECK(cuMemFree(m_buffers.d_state));
        m_buffers.d_state = 0;
    }
    if (m_buffers.d_scratch) {
        CU_CHECK(cuMemFree(m_buffers.d_scratch));
        m_buffers.d_scratch = 0;
    }
    if (m_buffers.d_intensity) {
        CU_CHECK(cuMemFree(m_buffers.d_intensity));
        m_buffers.d_intensity = 0;
    }
    if (m_tileBuffers.tileBuffer) {
        CU_CHECK(cuMemFree(m_tileBuffers.tileBuffer));
        m_tileBuffers.tileBuffer = 0;
    }
    if (m_tileBuffers.tileInputBuffer) {
        CU_CHECK(cuMemFree(m_tileBuffers.tileInputBuffer));
        m_tileBuffers.tileInputBuffer = 0;
    }
    if (m_tileBuffers.tileAlbedoBuffer) {
        CU_CHECK(cuMemFree(m_tileBuffers.tileAlbedoBuffer));
        m_tileBuffers.tileAlbedoBuffer = 0;
    }
    if (m_tileBuffers.tileNormalBuffer) {
        CU_CHECK(cuMemFree(m_tileBuffers.tileNormalBuffer));
        m_tileBuffers.tileNormalBuffer = 0;
    }
}

// ==================== 结构体版本的方法实现 ====================

void Denoiser::setup(const DenoiserSetupParams& params) {
    setup(params.width, params.height, params.useAlbedo, params.useNormal);
}

void Denoiser::denoise(const DenoiserParams& params) {
    denoise(params.inputColor, params.inputAlbedo, params.inputNormal, params.output, params.stream);
}

void Denoiser::denoiseTiled(const DenoiserTiledParams& params) {
    denoiseTiled(
        params.inputColor, 
        params.inputAlbedo, 
        params.inputNormal, 
        params.output, 
        params.tileWidth, 
        params.tileHeight, 
        params.overlap, 
        params.stream
    );
}

} // namespace optixw
