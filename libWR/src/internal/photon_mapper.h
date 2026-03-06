#pragma once

#include "gpu_types.h"
#include <cuda_runtime.h>
#include <optix.h>

namespace wr {
namespace internal {

// Forward declaration
struct PipelineImpl;

// Photon mapping configuration
struct PhotonMapConfig {
    bool enabled = false;
    uint32_t numPhotons = 100000;        // Total photons to emit
    uint32_t maxBounces = 8;             // Max photon bounces
    float searchRadius = 0.05f;          // Search radius for k-NN
    uint32_t maxPhotonsPerQuery = 100;   // Max photons to gather per query
    uint32_t causticPhotons = 50000;     // Max caustic photons
    float causticSearchRadius = 0.02f;   // Smaller radius for sharper caustics
};

// Photon mapper class
class PhotonMapper {
public:
    PhotonMapper();
    ~PhotonMapper();
    
    // Initialize photon mapper
    void initialize(const PhotonMapConfig& config);
    
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
    
    PhotonMapConfig m_config;
    
    // GPU buffers
    CUdeviceptr d_photons = 0;
    CUdeviceptr d_causticPhotons = 0;
    CUdeviceptr d_photonCounter = 0;
    CUdeviceptr d_causticCounter = 0;
    
    // Host counters
    uint32_t m_numPhotons = 0;
    uint32_t m_numCausticPhotons = 0;
    
    // OptiX pipeline for photon tracing
    PipelineImpl* m_photonPipeline = nullptr;
    CUmodule m_photonModule = nullptr;
    
    bool m_valid = false;
    bool m_initialized = false;
};

} // namespace internal
} // namespace wr
