#pragma once

#include "vlrw/vlrw.h"

namespace vlrw {

class SceneImpl;
class ContextImpl;

class RendererImpl {
public:
    RendererImpl(ContextImpl* context);
    ~RendererImpl();

    void render(
        SceneImpl* scene,
        const Camera& camera,
        RGB* outputBuffer,
        uint32_t width,
        uint32_t height,
        uint32_t spp);

private:
    ContextImpl* m_context;
};

} // namespace vlrw
