#pragma once
#include "internal/gpu_types.h"
#include "vector_math.cuh"

namespace wr {
namespace internal {

// Vertex in a bidirectional path
struct PathVertex {
    float3 position;
    float3 normal;
    float3 geometricNormal;
    float3 throughput;
    float3 direction;      // Direction to next vertex
    float pdfFwd;          // PDF of sampling this vertex from previous
    float pdfRev;          // PDF of sampling previous vertex from this
    uint32_t materialId;
    uint32_t depth;
    bool isDelta;          // Is this a delta interaction?
};

// Light path (from light source)
struct LightPath {
    PathVertex vertices[16];  // Max 16 vertices per light path
    uint32_t length;
    float3 emission;          // Initial emission from light
};

// Eye path (from camera)
struct EyePath {
    PathVertex vertices[16];  // Max 16 vertices per eye path
    uint32_t length;
    uint32_t pixelIndex;
};

// Connection result between eye and light paths
struct PathConnection {
    float3 contribution;
    float misWeight;
    bool valid;
};

} // namespace internal
} // namespace wr
