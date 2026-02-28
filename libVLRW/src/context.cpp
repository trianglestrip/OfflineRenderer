#include "vlrw/vlrw.h"
#include "optix_context.h"
#include "scene.h"
#include "renderer.h"

namespace vlrw {

Context* Context::create(void* cuContext) {
    auto impl = new ContextImpl(static_cast<CUcontext>(cuContext));
    return new Context(impl);
}

Context::~Context() {
    delete m_impl;
}

Scene* Context::createScene() {
    auto impl = new SceneImpl(m_impl);
    return new Scene(impl);
}

Renderer* Context::createRenderer() {
    auto impl = new RendererImpl(m_impl);
    return new Renderer(impl);
}

Scene::~Scene() {
    delete m_impl;
}

void Scene::addTriangleMesh(
    std::span<const float> vertices,
    std::span<const uint32_t> indices,
    uint32_t materialId)
{
    m_impl->addTriangleMesh(vertices, indices, materialId);
}

void Scene::addTriangleMeshMove(
    std::vector<float>&& vertices,
    std::vector<uint32_t>&& indices,
    uint32_t materialId)
{
    m_impl->addTriangleMeshMove(std::move(vertices), std::move(indices), materialId);
}

void Scene::setMaterial(uint32_t id, const MaterialDesc& desc) {
    m_impl->setMaterial(id, desc);
}

void Scene::addPointLight(const PointLightDesc& light) {
    m_impl->addPointLight(light);
}

void Scene::addAreaLight(const AreaLightDesc& light) {
    m_impl->addAreaLight(light);
}

void Scene::finalize() {
    m_impl->finalize();
}

} // namespace vlrw
