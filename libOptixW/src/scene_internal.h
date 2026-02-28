#pragma once

#include "optixw/optixw.h"
#include <optix.h>
#include <cuda.h>

namespace optixw {

// Access to scene internals for renderer
class SceneAccessor {
public:
    static OptixTraversableHandle getGasHandle(Scene* scene);
    static CUdeviceptr getVerticesPtr(Scene* scene);
    static CUdeviceptr getTexcoordsPtr(Scene* scene);
    static CUdeviceptr getIndicesPtr(Scene* scene);
    static CUdeviceptr getMaterialsPtr(Scene* scene);
    static CUdeviceptr getTexturesPtr(Scene* scene);
    static CUdeviceptr getTriangleMaterialIdsPtr(Scene* scene);
    static uint32_t getMaterialCount(Scene* scene);
    static uint32_t getTextureCount(Scene* scene);
    static uint32_t getTriangleCount(Scene* scene);
    static Vec3 getEnvironmentRadiance(Scene* scene);
    static CUdeviceptr getEnvironmentMapPtr(Scene* scene);
    static uint32_t getEnvironmentMapWidth(Scene* scene);
    static uint32_t getEnvironmentMapHeight(Scene* scene);
    static float getEnvironmentMapScale(Scene* scene);
};

} // namespace optixw
