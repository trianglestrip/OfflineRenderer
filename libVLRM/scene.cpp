#include <VLRM/VLRM.h>
#include "shared/common_internal.h"

namespace vlrm {
    struct SceneImpl {
        Context* context;
        std::vector<float> vertices;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> materialIndices;
        std::vector<MaterialData> materials;
        
        SceneImpl(Context* ctx) : context(ctx) {}
    };

    Scene::Scene(Context* context) {
    }

    Scene::~Scene() {
    }

    void Scene::addTriangleMesh(
        std::span<const float> vertices,
        std::span<const uint32_t> indices,
        uint32_t materialId
    ) {
        VLRMAssert_NotImplemented();
    }

    void Scene::addMaterial(
        const RGB& albedo,
        const RGB& emission
    ) {
        VLRMAssert_NotImplemented();
    }

    void Scene::buildAccelerationStructure() {
        VLRMAssert_NotImplemented();
    }
}
