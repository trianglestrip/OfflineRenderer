#include <VLRM/VLRM.h>
#include "shared/common_internal.h"

namespace vlrm {
    struct RendererImpl {
        Context* context;
        Scene* scene;
        
        Point3D cameraPosition;
        Vector3D cameraForward;
        Vector3D cameraUp;
        float fovY;
        
        RendererImpl(Context* ctx, Scene* scn) 
            : context(ctx), scene(scn), fovY(45.0f * (VLRM_M_PI / 180.0f)) {}
    };

    Renderer::Renderer(Context* context, Scene* scene) {
    }

    Renderer::~Renderer() {
    }

    void Renderer::setCamera(
        const Point3D& position,
        const Vector3D& forward,
        const Vector3D& up,
        float fovY
    ) {
        VLRMAssert_NotImplemented();
    }

    void Renderer::render(
        uint32_t width,
        uint32_t height,
        uint32_t spp,
        uint32_t maxDepth,
        RGB* outputBuffer
    ) {
        VLRMAssert_NotImplemented();
    }
}
