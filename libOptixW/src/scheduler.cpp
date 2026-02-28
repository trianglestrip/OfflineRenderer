#include "optixw/optixw.h"
#include "optixw/types.h"
#include <vector>
#include <algorithm>

namespace optixw {

// Wavefront scheduler: manages ray queues and stages
class WavefrontScheduler {
public:
    WavefrontScheduler(uint32_t maxRays) : m_maxRays(maxRays) {
        m_activeIndices.reserve(maxRays);
    }
    
    // Initialize with primary rays
    void initPrimaryRays(uint32_t numPixels) {
        m_activeIndices.clear();
        for (uint32_t i = 0; i < numPixels; ++i) {
            m_activeIndices.push_back(i);
        }
    }
    
    // Get active ray count
    uint32_t getActiveCount() const {
        return static_cast<uint32_t>(m_activeIndices.size());
    }
    
    // Compact active rays (remove terminated)
    void compact(const RayState* rayStates) {
        auto newEnd = std::remove_if(
            m_activeIndices.begin(),
            m_activeIndices.end(),
            [rayStates](uint32_t idx) {
                return rayStates[idx].stage == RayState::Terminated;
            }
        );
        m_activeIndices.erase(newEnd, m_activeIndices.end());
    }
    
    const std::vector<uint32_t>& getActiveIndices() const {
        return m_activeIndices;
    }
    
private:
    uint32_t m_maxRays;
    std::vector<uint32_t> m_activeIndices;
};

} // namespace optixw
