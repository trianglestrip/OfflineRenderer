#pragma once

#include <cstdint>

namespace wr {

// Forward declarations
class Scene;
class Renderer;

// Context configuration
struct ContextConfig {
    int deviceId = 0;
    int logLevel = 4;  // OptiX log level (0-4)
    bool enableValidation = false;
};

// Context manages OptiX device context and CUDA context
class Context {
public:
    Context(const ContextConfig& config = {});
    ~Context();

    Scene* createScene();
    Renderer* createRenderer();
    
    // Get current device ID
    int getDeviceId() const;
};

} // namespace wr
