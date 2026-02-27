#include "wavefront_scheduler.h"
#include <cuda.h>

namespace vlrw {

WavefrontScheduler::~WavefrontScheduler() {
    if (m_rayPool) cuMemFree(m_rayPool);
    if (m_activeQueue) cuMemFree(m_activeQueue);
    if (m_nextQueue) cuMemFree(m_nextQueue);
    if (m_shadowQueue) cuMemFree(m_shadowQueue);
    if (m_accumBuffer) cuMemFree(m_accumBuffer);
    if (m_queueCounters) cuMemFree(m_queueCounters);
}

void WavefrontScheduler::initialize(uint32_t width, uint32_t height, uint32_t maxDepth) {
    m_width = width;
    m_height = height;
    m_maxDepth = maxDepth;
    m_numActive = width * height;
    
    // Allocate GPU buffers (placeholder)
}

void WavefrontScheduler::iterate(uint32_t bounce, uint32_t* outNumActive, CUstream stream) {
    // Placeholder: iterate through Wavefront stages
    
    // Taskflow integration: call registered callbacks
    for (const auto& [name, callback] : m_stageCallbacks) {
        callback(bounce, stream);
    }
    
    *outNumActive = m_numActive;
}

void WavefrontScheduler::registerStageCallback(std::string_view stageName, StageCallback callback) {
    m_stageCallbacks.emplace_back(std::string(stageName), callback);
}

} // namespace vlrw
