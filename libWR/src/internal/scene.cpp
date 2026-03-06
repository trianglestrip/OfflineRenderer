#include "wr/scene.h"
#include "internal/gpu_types.h"
#include "internal/cuda_utils.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace wr {

using namespace internal;

extern OptixDeviceContext getOptixContext();

struct TextureData {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
    cudaArray_t d_array = nullptr;
    cudaTextureObject_t texObj = 0;
};

struct SceneImpl {
    std::vector<float> vertices;
    std::vector<float> normals;     // Vertex normals (3 floats per vertex)
    std::vector<float> uvs;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> triangleMaterialIds;
    std::vector<MaterialData> materials;
    std::vector<TextureData> textures;
    Vec3 environmentRadiance{0.0f, 0.0f, 0.0f};
    
    // HDR Environment Map
    cudaArray_t d_envArray = nullptr;
    cudaTextureObject_t envMapTexture = 0;
    int envMapWidth = 0;
    int envMapHeight = 0;
    
    std::vector<uint32_t> emissiveTriangles;
    std::vector<float> emissiveTriangleCDF;
    
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_normals = 0;      // Device normals buffer
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_triangleMaterialIds = 0;
    CUdeviceptr d_materials = 0;
    CUdeviceptr d_uvs = 0;
    CUdeviceptr d_textureObjects = 0;
    CUdeviceptr d_gasOutput = 0;
    OptixTraversableHandle gasHandle = 0;
    
    CUdeviceptr d_emissiveTriangles = 0;
    CUdeviceptr d_emissiveTriangleCDF = 0;
    
    uint32_t numVertices = 0;
    uint32_t numTriangles = 0;
    bool hasUVs = false;
    bool hasNormals = false;        // Track if normals are provided
    bool finalized = false;
    
    ~SceneImpl() {
        for (TextureData& t : textures) {
            if (t.texObj) cudaDestroyTextureObject(t.texObj);
            if (t.d_array) cudaFreeArray(t.d_array);
        }
        if (d_vertices) cuMemFree(d_vertices);
        if (d_normals) cuMemFree(d_normals);
        if (d_indices) cuMemFree(d_indices);
        if (d_triangleMaterialIds) cuMemFree(d_triangleMaterialIds);
        if (d_materials) cuMemFree(d_materials);
        if (d_uvs) cuMemFree(d_uvs);
        if (d_textureObjects) cuMemFree(d_textureObjects);
        if (d_gasOutput) cuMemFree(d_gasOutput);
        if (d_emissiveTriangles) cuMemFree(d_emissiveTriangles);
        if (d_emissiveTriangleCDF) cuMemFree(d_emissiveTriangleCDF);
        if (envMapTexture) cudaDestroyTextureObject(envMapTexture);
        if (d_envArray) cudaFreeArray(d_envArray);
    }
};

Scene::Scene() : m_impl(new SceneImpl()) {}
Scene::~Scene() { delete m_impl; }

uint32_t Scene::addLambertianMaterial(const Vec3& albedo) {
    return addLambertianMaterial(albedo, 0);
}

uint32_t Scene::addLambertianMaterial(const Vec3& albedo, uint32_t albedoTextureId) {
    MaterialData mat = {};
    mat.albedo = make_float4(albedo.x, albedo.y, albedo.z, 0.0f);
    mat.emission = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    mat.ior = 1.0f;
    mat.type = MaterialType::Lambertian;
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.albedoTextureId = albedoTextureId;
    mat.normalTextureId = 0;
    
    m_impl->materials.push_back(mat);
    return static_cast<uint32_t>(m_impl->materials.size() - 1);
}

uint32_t Scene::addEmissiveMaterial(const Vec3& emission) {
    MaterialData mat = {};
    mat.albedo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    mat.emission = make_float4(emission.x, emission.y, emission.z, 0.0f);
    mat.ior = 1.0f;
    mat.type = MaterialType::Emissive;
    mat.roughness = 0.0f;
    mat.metallic = 0.0f;
    mat.albedoTextureId = 0;
    mat.normalTextureId = 0;
    
    m_impl->materials.push_back(mat);
    return static_cast<uint32_t>(m_impl->materials.size() - 1);
}

uint32_t Scene::addGlassMaterial(const Vec3& albedo, float ior) {
    MaterialData mat = {};
    mat.albedo = make_float4(albedo.x, albedo.y, albedo.z, 0.0f);
    mat.emission = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    mat.ior = ior;
    mat.type = MaterialType::Glass;
    mat.roughness = 0.0f;
    mat.metallic = 0.0f;
    mat.albedoTextureId = 0;
    mat.normalTextureId = 0;
    
    m_impl->materials.push_back(mat);
    return static_cast<uint32_t>(m_impl->materials.size() - 1);
}

uint32_t Scene::addGGXReflectionMaterial(const Vec3& albedo, float roughness, float metallic) {
    MaterialData mat = {};
    mat.albedo = make_float4(albedo.x, albedo.y, albedo.z, 0.0f);
    mat.emission = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    mat.ior = 1.5f;  // Default IOR for dielectric
    mat.type = MaterialType::GGXReflection;
    mat.roughness = roughness;
    mat.metallic = metallic;
    mat.albedoTextureId = 0;
    mat.normalTextureId = 0;
    
    m_impl->materials.push_back(mat);
    return static_cast<uint32_t>(m_impl->materials.size() - 1);
}

uint32_t Scene::addGGXTransmissionMaterial(const Vec3& albedo, float roughness, float ior) {
    MaterialData mat = {};
    mat.albedo = make_float4(albedo.x, albedo.y, albedo.z, 0.0f);
    mat.emission = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    mat.ior = ior;
    mat.type = MaterialType::GGXTransmission;
    mat.roughness = roughness;
    mat.metallic = 0.0f;
    mat.albedoTextureId = 0;
    mat.normalTextureId = 0;
    
    m_impl->materials.push_back(mat);
    return static_cast<uint32_t>(m_impl->materials.size() - 1);
}

void Scene::setMaterialNormalMap(uint32_t materialId, uint32_t normalTextureId) {
    if (materialId >= m_impl->materials.size()) {
        throw std::runtime_error("Invalid material ID");
    }
    m_impl->materials[materialId].normalTextureId = normalTextureId;
}

void Scene::addTriangleMesh(std::span<const float> verts,
                            std::span<const uint32_t> inds,
                            uint32_t materialId) {
    addTriangleMesh(verts, inds, std::span<const float>(), materialId);
}

void Scene::addTriangleMesh(std::span<const float> verts,
                            std::span<const uint32_t> inds,
                            std::span<const float> uvData,
                            uint32_t materialId) {
    addTriangleMesh(verts, inds, std::span<const float>(), uvData, materialId);
}

void Scene::addTriangleMesh(std::span<const float> verts,
                            std::span<const uint32_t> inds,
                            std::span<const float> normalData,
                            std::span<const float> uvData,
                            uint32_t materialId) {
    uint32_t vertexOffset = static_cast<uint32_t>(m_impl->vertices.size() / 3);
    
    // Add vertices
    for (float v : verts) {
        m_impl->vertices.push_back(v);
    }
    
    size_t numNewVerts = verts.size() / 3;
    
    // Add normals (if provided)
    if (!normalData.empty() && normalData.size() >= numNewVerts * 3) {
        m_impl->hasNormals = true;
        for (float n : normalData) {
            m_impl->normals.push_back(n);
        }
    } else {
        // Pad with zeros if no normals (will use geometric normal)
        for (size_t i = 0; i < numNewVerts * 3; ++i) {
            m_impl->normals.push_back(0.0f);
        }
    }
    
    // Add UVs
    if (!uvData.empty() && uvData.size() >= numNewVerts * 2) {
        m_impl->hasUVs = true;
        for (size_t i = 0; i < numNewVerts; ++i) {
            m_impl->uvs.push_back(uvData[i * 2 + 0]);
            m_impl->uvs.push_back(uvData[i * 2 + 1]);
        }
    } else {
        for (size_t i = 0; i < numNewVerts; ++i) {
            m_impl->uvs.push_back(0.0f);
            m_impl->uvs.push_back(0.0f);
        }
    }
    
    // Add indices
    for (uint32_t idx : inds) {
        m_impl->indices.push_back(vertexOffset + idx);
    }
    
    // Add material IDs
    uint32_t numTris = static_cast<uint32_t>(inds.size() / 3);
    for (uint32_t i = 0; i < numTris; ++i) {
        m_impl->triangleMaterialIds.push_back(materialId);
    }
    
    m_impl->numTriangles += numTris;
    m_impl->numVertices = static_cast<uint32_t>(m_impl->vertices.size() / 3);
}

uint32_t Scene::loadTexture(const std::string& path) {
    int w, h, comp;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!data) {
        throw std::runtime_error("Failed to load texture: " + path);
    }
    
    TextureData tex;
    tex.width = w;
    tex.height = h;
    tex.pixels.resize(w * h * 4);
    std::memcpy(tex.pixels.data(), data, tex.pixels.size());
    stbi_image_free(data);
    
    m_impl->textures.push_back(std::move(tex));
    uint32_t id = static_cast<uint32_t>(m_impl->textures.size());
    std::cout << "[Scene] Loaded texture: " << path << " (" << w << "x" << h << "), id=" << id << std::endl;
    return id;
}

void Scene::destroyTexture(uint32_t textureId) {
    if (textureId == 0 || textureId > m_impl->textures.size()) return;
    size_t idx = textureId - 1;
    TextureData& tex = m_impl->textures[idx];
    if (tex.texObj) cudaDestroyTextureObject(tex.texObj);
    tex.texObj = 0;
    if (tex.d_array) cudaFreeArray(tex.d_array);
    tex.d_array = nullptr;
    tex.pixels.clear();
}

uint32_t Texture2D::create(Scene* scene, const std::string& path) {
    return scene->loadTexture(path);
}

void Texture2D::destroy(Scene* scene, uint32_t textureId) {
    scene->destroyTexture(textureId);
}

void Scene::setEnvironmentRadiance(const Vec3& radiance) {
    m_impl->environmentRadiance = radiance;
}

void Scene::setEnvironmentMap(const std::string& hdrPath) {
    // Load HDR image using stb_image
    int width, height, channels;
    float* data = stbi_loadf(hdrPath.c_str(), &width, &height, &channels, 4);  // Force RGBA
    
    if (!data) {
        throw std::runtime_error("Failed to load HDR environment map: " + hdrPath);
    }
    
    std::cout << "[Scene] Loaded HDR environment map: " << hdrPath 
              << " (" << width << "x" << height << ", " << channels << " channels)\n";
    
    // Create CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t d_array;
    cudaMallocArray(&d_array, &channelDesc, width, height);
    
    // Copy data to CUDA array
    cudaMemcpy2DToArray(
        d_array,
        0, 0,
        data,
        width * sizeof(float4),
        width * sizeof(float4),
        height,
        cudaMemcpyHostToDevice
    );
    
    stbi_image_free(data);
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;  // Wrap horizontally
    texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp vertically
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    
    // Clean up old environment map if exists
    if (m_impl->envMapTexture) {
        cudaDestroyTextureObject(m_impl->envMapTexture);
    }
    if (m_impl->d_envArray) {
        cudaFreeArray(m_impl->d_envArray);
    }
    
    m_impl->d_envArray = d_array;
    m_impl->envMapTexture = texObj;
    m_impl->envMapWidth = width;
    m_impl->envMapHeight = height;
    
    std::cout << "[Scene] Environment map texture created (GPU)\n";
}

void Scene::finalize(const SceneBuildConfig& config) {
    if (m_impl->finalized) return;
    
    std::cout << "[Scene] Finalizing scene: " << m_impl->numTriangles 
              << " triangles, " << m_impl->numVertices << " vertices" << std::endl;
    
    if (m_impl->vertices.empty() || m_impl->indices.empty()) {
        throw std::runtime_error("Scene has no geometry");
    }
    
    size_t vertSize = m_impl->vertices.size() * sizeof(float);
    CU_CHECK(cuMemAlloc(&m_impl->d_vertices, vertSize));
    CU_CHECK(cuMemcpyHtoD(m_impl->d_vertices, m_impl->vertices.data(), vertSize));
    
    size_t indSize = m_impl->indices.size() * sizeof(uint32_t);
    CU_CHECK(cuMemAlloc(&m_impl->d_indices, indSize));
    CU_CHECK(cuMemcpyHtoD(m_impl->d_indices, m_impl->indices.data(), indSize));
    
    size_t matIdSize = m_impl->triangleMaterialIds.size() * sizeof(uint32_t);
    CU_CHECK(cuMemAlloc(&m_impl->d_triangleMaterialIds, matIdSize));
    CU_CHECK(cuMemcpyHtoD(m_impl->d_triangleMaterialIds, m_impl->triangleMaterialIds.data(), matIdSize));
    
    size_t matSize = m_impl->materials.size() * sizeof(MaterialData);
    CU_CHECK(cuMemAlloc(&m_impl->d_materials, matSize));
    CU_CHECK(cuMemcpyHtoD(m_impl->d_materials, m_impl->materials.data(), matSize));
    
    // Upload normals if present
    if (m_impl->hasNormals && !m_impl->normals.empty()) {
        size_t normalSize = m_impl->normals.size() * sizeof(float);
        CU_CHECK(cuMemAlloc(&m_impl->d_normals, normalSize));
        CU_CHECK(cuMemcpyHtoD(m_impl->d_normals, m_impl->normals.data(), normalSize));
        std::cout << "[Scene] Uploaded normals: " << m_impl->normals.size() / 3 << " vertices" << std::endl;
    }
    
    // Upload UVs if present
    if (m_impl->hasUVs && !m_impl->uvs.empty()) {
        size_t uvSize = m_impl->uvs.size() * sizeof(float);
        CU_CHECK(cuMemAlloc(&m_impl->d_uvs, uvSize));
        CU_CHECK(cuMemcpyHtoD(m_impl->d_uvs, m_impl->uvs.data(), uvSize));
        std::cout << "[Scene] Uploaded UVs: " << m_impl->uvs.size() / 2 << " vertices" << std::endl;
    }
    
    // Create CUDA texture objects for each texture
    if (!m_impl->textures.empty()) {
        for (TextureData& tex : m_impl->textures) {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            CUDA_CHECK(cudaMallocArray(&tex.d_array, &channelDesc, tex.width, tex.height));
            CUDA_CHECK(cudaMemcpy2DToArray(tex.d_array, 0, 0, tex.pixels.data(), tex.width * 4, tex.width * 4, tex.height, cudaMemcpyHostToDevice));
            
            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = tex.d_array;
            
            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = 1;
            
            CUDA_CHECK(cudaCreateTextureObject(&tex.texObj, &resDesc, &texDesc, nullptr));
        }
        
        size_t texObjSize = m_impl->textures.size() * sizeof(cudaTextureObject_t);
        std::vector<cudaTextureObject_t> hostTexObjs(m_impl->textures.size());
        for (size_t i = 0; i < m_impl->textures.size(); ++i) {
            hostTexObjs[i] = m_impl->textures[i].texObj;
        }
        CU_CHECK(cuMemAlloc(&m_impl->d_textureObjects, texObjSize));
        CU_CHECK(cuMemcpyHtoD(m_impl->d_textureObjects, hostTexObjs.data(), texObjSize));
        std::cout << "[Scene] Created " << m_impl->textures.size() << " texture objects" << std::endl;
    }
    
    // Build emissive triangle list for NEE
    m_impl->emissiveTriangles.clear();
    m_impl->emissiveTriangleCDF.clear();
    
    float totalPower = 0.0f;
    for (uint32_t triIdx = 0; triIdx < m_impl->numTriangles; ++triIdx) {
        uint32_t matId = m_impl->triangleMaterialIds[triIdx];
        const MaterialData& mat = m_impl->materials[matId];
        
        if (mat.type == MaterialType::Emissive) {
            m_impl->emissiveTriangles.push_back(triIdx);
            
            // Calculate triangle area
            uint32_t i0 = m_impl->indices[triIdx * 3 + 0];
            uint32_t i1 = m_impl->indices[triIdx * 3 + 1];
            uint32_t i2 = m_impl->indices[triIdx * 3 + 2];
            
            float3 v0 = make_float3(
                m_impl->vertices[i0 * 3 + 0],
                m_impl->vertices[i0 * 3 + 1],
                m_impl->vertices[i0 * 3 + 2]
            );
            float3 v1 = make_float3(
                m_impl->vertices[i1 * 3 + 0],
                m_impl->vertices[i1 * 3 + 1],
                m_impl->vertices[i1 * 3 + 2]
            );
            float3 v2 = make_float3(
                m_impl->vertices[i2 * 3 + 0],
                m_impl->vertices[i2 * 3 + 1],
                m_impl->vertices[i2 * 3 + 2]
            );
            
            float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
            float3 cross_e1_e2 = make_float3(
                e1.y * e2.z - e1.z * e2.y,
                e1.z * e2.x - e1.x * e2.z,
                e1.x * e2.y - e1.y * e2.x
            );
            float area = 0.5f * sqrtf(cross_e1_e2.x * cross_e1_e2.x + 
                                      cross_e1_e2.y * cross_e1_e2.y + 
                                      cross_e1_e2.z * cross_e1_e2.z);
            
            // Power = emission * area
            float power = (mat.emission.x + mat.emission.y + mat.emission.z) * area;
            totalPower += power;
            m_impl->emissiveTriangleCDF.push_back(totalPower);
        }
    }
    
    // Normalize CDF
    if (!m_impl->emissiveTriangleCDF.empty()) {
        for (float& cdf : m_impl->emissiveTriangleCDF) {
            cdf /= totalPower;
        }
        
        // Upload to GPU
        size_t emissiveTriSize = m_impl->emissiveTriangles.size() * sizeof(uint32_t);
        size_t emissiveCDFSize = m_impl->emissiveTriangleCDF.size() * sizeof(float);
        
        CU_CHECK(cuMemAlloc(&m_impl->d_emissiveTriangles, emissiveTriSize));
        CU_CHECK(cuMemcpyHtoD(m_impl->d_emissiveTriangles, m_impl->emissiveTriangles.data(), emissiveTriSize));
        
        CU_CHECK(cuMemAlloc(&m_impl->d_emissiveTriangleCDF, emissiveCDFSize));
        CU_CHECK(cuMemcpyHtoD(m_impl->d_emissiveTriangleCDF, m_impl->emissiveTriangleCDF.data(), emissiveCDFSize));
        
        std::cout << "[Scene] Built light list: " << m_impl->emissiveTriangles.size() << " emissive triangles" << std::endl;
    }
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = 0;
    if (config.allowUpdate) {
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    }
    if (config.allowCompaction) {
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    }
    if (config.preferFastTrace) {
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    CUdeviceptr d_vertexBuffers[1] = { m_impl->d_vertices };
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexBuffers = d_vertexBuffers;
    buildInput.triangleArray.numVertices = m_impl->numVertices;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    
    buildInput.triangleArray.indexBuffer = m_impl->d_indices;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    buildInput.triangleArray.numIndexTriplets = m_impl->numTriangles;
    
    uint32_t buildInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.triangleArray.flags = buildInputFlags;
    buildInput.triangleArray.numSbtRecords = 1;
    
    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        getOptixContext(),
        &accelOptions,
        &buildInput,
        1,
        &bufferSizes
    ));
    
    CUdeviceptr d_tempBuffer;
    CU_CHECK(cuMemAlloc(&d_tempBuffer, bufferSizes.tempSizeInBytes));
    CU_CHECK(cuMemAlloc(&m_impl->d_gasOutput, bufferSizes.outputSizeInBytes));
    
    OPTIX_CHECK(optixAccelBuild(
        getOptixContext(),
        0,
        &accelOptions,
        &buildInput,
        1,
        d_tempBuffer,
        bufferSizes.tempSizeInBytes,
        m_impl->d_gasOutput,
        bufferSizes.outputSizeInBytes,
        &m_impl->gasHandle,
        nullptr,
        0
    ));
    
    CU_CHECK(cuMemFree(d_tempBuffer));
    
    std::cout << "[Scene] GAS built, handle: " << m_impl->gasHandle << std::endl;
    
    m_impl->finalized = true;
}

DevicePtr Scene::getVerticesBuffer() const { return static_cast<DevicePtr>(m_impl->d_vertices); }
DevicePtr Scene::getNormalsBuffer() const { return static_cast<DevicePtr>(m_impl->d_normals); }
DevicePtr Scene::getIndicesBuffer() const { return static_cast<DevicePtr>(m_impl->d_indices); }
DevicePtr Scene::getTriangleMaterialIdsBuffer() const { return static_cast<DevicePtr>(m_impl->d_triangleMaterialIds); }
DevicePtr Scene::getMaterialsBuffer() const { return static_cast<DevicePtr>(m_impl->d_materials); }
TraversableHandle Scene::getGASHandle() const { return static_cast<TraversableHandle>(m_impl->gasHandle); }
uint32_t Scene::getNumMaterials() const { return static_cast<uint32_t>(m_impl->materials.size()); }
Vec3 Scene::getEnvironmentRadiance() const { return m_impl->environmentRadiance; }

uint64_t Scene::getEnvironmentMapTexture() const { return static_cast<uint64_t>(m_impl->envMapTexture); }
uint32_t Scene::getEnvironmentMapWidth() const { return m_impl->envMapWidth; }
uint32_t Scene::getEnvironmentMapHeight() const { return m_impl->envMapHeight; }

DevicePtr Scene::getEmissiveTrianglesBuffer() const { return static_cast<DevicePtr>(m_impl->d_emissiveTriangles); }
DevicePtr Scene::getEmissiveTriangleCDFBuffer() const { return static_cast<DevicePtr>(m_impl->d_emissiveTriangleCDF); }
uint32_t Scene::getNumEmissiveTriangles() const { return static_cast<uint32_t>(m_impl->emissiveTriangles.size()); }

DevicePtr Scene::getTexturesBuffer() const { return static_cast<DevicePtr>(m_impl->d_textureObjects); }
uint32_t Scene::getNumTextures() const { return static_cast<uint32_t>(m_impl->textures.size()); }
DevicePtr Scene::getUVsBuffer() const { return static_cast<DevicePtr>(m_impl->d_uvs); }

} // namespace wr
