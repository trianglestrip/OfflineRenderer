#pragma once

#include "wr/types.h"  // For PhotonMapConfig
#include "gpu_types.h"
#include <cuda_runtime.h>
#include <optix.h>

namespace wr {
namespace internal {

// PipelineImpl is defined in pipeline.cpp, not needed here

// Photon mapper class
class PhotonMapper {
public:
    PhotonMapper();
    ~PhotonMapper();
    
    // Initialize photon mapper
    void initialize(const wr::PhotonMapConfig& config);
    
    // Build photon map for a scene
    void buildPhotonMap(
        OptixTraversableHandle traversable,
        const GeometryBuffers& geometry,
        const MaterialData* materials,
        uint32_t numMaterials,
        const void* textures,
        uint32_t numTextures,
        const uint32_t* emissiveTriangles,
        const float* emissiveTriangleCDF,
        uint32_t numEmissiveTriangles
    );
    
    // Get photon map parameters for rendering
    PhotonMapParams getPhotonMapParams() const;
    CausticPhotonMapParams getCausticMapParams() const;
    
    // Check if photon map is valid
    bool isValid() const { return m_valid; }
    
    // Get number of stored photons
    uint32_t getNumPhotons() const;
    uint32_t getNumCausticPhotons() const;
    
private:
    void allocatePhotonBuffers();
    void freePhotonBuffers();
    void createPhotonPipeline();
    void destroyPhotonPipeline();

    wr::PhotonMapConfig m_config;
    
    // GPU buffers
    CUdeviceptr d_photons = 0;
    CUdeviceptr d_causticPhotons = 0;
    CUdeviceptr d_photonCounter = 0;
    CUdeviceptr d_causticCounter = 0;
    
    // Host counters
    uint32_t m_numPhotons = 0;
    uint32_t m_numCausticPhotons = 0;
    
    // OptiX pipeline for photon tracing (simplified - not using separate pipeline for now)
    CUmodule m_photonModule = nullptr;
    
    bool m_valid = false;
    bool m_initialized = false;
};

} // namespace internal
} // namespace wr
