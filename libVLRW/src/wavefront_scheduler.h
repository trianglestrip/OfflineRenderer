#pragma once

#include <cstdint>
#include <cuda.h>
#include <functional>
#include <vector>
#include <string_view>

namespace vlrw {

// Wavefront Scheduler: manages buffers, queues, OptiX launches, compaction (C++20 style)
class WavefrontScheduler {
public:
    WavefrontScheduler() = default;
    ~WavefrontScheduler();

    // Initialize buffers: ray, hit, pathState, accum, shadow, etc.
    void initialize(uint32_t width, uint32_t height, uint32_t maxDepth);

    // Single iteration: RayGen -> Intersection -> MaterialEval -> ShadowTrace -> Compact
    // Returns number of active rays after this bounce
    void iterate(
        uint32_t bounce,
        uint32_t* outNumActive,
        CUstream stream = nullptr);

    uint32_t getActiveCount() const { return m_numActive; }

    // Taskflow integration: register stage callbacks
    using StageCallback = std::function<void(uint32_t bounce, CUstream stream)>;
    void registerStageCallback(std::string_view stageName, StageCallback callback);

private:
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_maxDepth = 0;
    uint32_t m_numActive = 0;

    // GPU buffers
    CUdeviceptr m_rayPool = 0;
    CUdeviceptr m_activeQueue = 0;
    CUdeviceptr m_nextQueue = 0;
    CUdeviceptr m_shadowQueue = 0;
    CUdeviceptr m_accumBuffer = 0;
    CUdeviceptr m_queueCounters = 0;

    // Taskflow: stage callbacks
    std::vector<std::pair<std::string, StageCallback>> m_stageCallbacks;
};

} // namespace vlrw
