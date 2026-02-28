#include <optix.h>
#include <cuda_runtime.h>
#include <optixw/wavefront_kernel_params.h>
#include "vector_math.cuh"

using namespace optixw;

static constexpr float kPi = 3.14159265f;
static constexpr float kRayEps = 0.001f;

enum MaterialTag : uint32_t {
    kMatte = 0,
    kLambertianScattering = 1,
    kSpecularReflection = 2,
    kSpecularScattering = 3,
    kMicrofacetReflection = 4,
    kMicrofacetScattering = 5,
    kUE4 = 6,
    kOldStyle = 7,
    kDiffuseEmitter = 8,
    kDirectionalEmitter = 9,
    kPointEmitter = 10,
    kMulti = 11,
    kEnvironmentEmitter = 12
};

struct BSDFSample {
    float3 wi;
    float3 weight; // f * cos / pdf
    float pdf;
    uint32_t isDelta;
    uint32_t valid;
};

__device__ inline float randf(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 16) / 65536.0f;
}

__device__ inline float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

__device__ inline float saturate(float x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

__device__ inline float max3(const float3& v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

__device__ inline float3 black3() { return make_float3(0.0f, 0.0f, 0.0f); }
__device__ inline float3 white3() { return make_float3(1.0f, 1.0f, 1.0f); }
__device__ inline bool hasValue(const float3& v) { return v.x > 0.0f || v.y > 0.0f || v.z > 0.0f; }
__device__ inline float3 lerp3(const float3& a, const float3& b, float t) { return a * (1.0f - t) + b * t; }
__device__ inline float3 cdiv3(const float3& a, const float3& b) {
    return make_float3(
        a.x / fmaxf(b.x, 1e-6f),
        a.y / fmaxf(b.y, 1e-6f),
        a.z / fmaxf(b.z, 1e-6f));
}

__device__ inline float3 sampleCosineHemisphere(float u1, float u2, float* pdf) {
    float phi = 2.0f * kPi * u1;
    float cosTheta = sqrtf(u2);
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - u2));
    *pdf = cosTheta / kPi;
    return make_float3(cosf(phi) * sinTheta, cosTheta, sinf(phi) * sinTheta);
}

__device__ inline float3 toWorld(const float3& local, const float3& normal) {
    float3 tangent;
    if (fabsf(normal.y) < 0.9f) {
        tangent = normalize(cross(make_float3(0, 1, 0), normal));
    } else {
        tangent = normalize(cross(make_float3(1, 0, 0), normal));
    }
    float3 bitangent = cross(normal, tangent);
    return tangent * local.x + normal * local.y + bitangent * local.z;
}

__device__ inline float3 reflectDir(const float3& wi, const float3& n) {
    return normalize(wi - 2.0f * dot(wi, n) * n);
}

__device__ inline bool refractDir(const float3& wi, const float3& n, float eta, float3* wt) {
    float cosi = saturate(-dot(wi, n));
    float sin2t = eta * eta * fmaxf(0.0f, 1.0f - cosi * cosi);
    if (sin2t >= 1.0f) {
        return false;
    }
    float cost = sqrtf(fmaxf(0.0f, 1.0f - sin2t));
    *wt = normalize(eta * wi + (eta * cosi - cost) * n);
    return true;
}

__device__ inline float dielectricFresnel(float cosThetaI, float etaI, float etaT) {
    cosThetaI = saturate(cosThetaI);
    float sinThetaI2 = fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
    float eta = etaI / etaT;
    float sinThetaT2 = eta * eta * sinThetaI2;
    if (sinThetaT2 >= 1.0f) {
        return 1.0f;
    }
    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT2));
    float rs = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);
    float rp = (etaI * cosThetaT - etaT * cosThetaI) / (etaI * cosThetaT + etaT * cosThetaI);
    return 0.5f * (rs * rs + rp * rp);
}

__device__ inline float3 fresnelConductor(float cosThetaI, const float3& eta, const float3& k) {
    cosThetaI = saturate(cosThetaI);
    float cos2 = cosThetaI * cosThetaI;
    float3 one = white3();
    float3 eta2 = eta * eta;
    float3 k2 = k * k;
    float3 t0 = eta2 + k2;
    float3 t1 = t0 * cos2;
    float3 t2 = 2.0f * eta * cosThetaI;
    float3 rs = cdiv3(t0 - t2 + one, t0 + t2 + one);
    float3 rp = cdiv3(t1 - t2 + one, t1 + t2 + one);
    return 0.5f * (rs + rp);
}

__device__ inline float3 schlickFresnel(float cosTheta, const float3& F0) {
    return F0 + (white3() - F0) * pow5(1.0f - saturate(cosTheta));
}

__device__ inline float ggxD(float alpha, float NoH) {
    float a2 = alpha * alpha;
    float d = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(kPi * d * d, 1e-6f);
}

__device__ inline float ggxG1(float alpha, float NoV) {
    float a2 = alpha * alpha;
    float denom = NoV + sqrtf(a2 + (1.0f - a2) * NoV * NoV);
    return (2.0f * NoV) / fmaxf(denom, 1e-6f);
}

__device__ inline float ggxG(float alpha, float NoV, float NoL) {
    return ggxG1(alpha, NoV) * ggxG1(alpha, NoL);
}

__device__ inline float materialAlpha(const MaterialData& mat) {
    float roughness = fmaxf(0.03f, mat.roughness);
    return roughness * roughness;
}

__device__ inline float3 sampleGGXHalf(const float3& n, float alpha, uint32_t& seed, float* outPdfH) {
    float u1 = randf(seed);
    float u2 = randf(seed);
    float phi = 2.0f * kPi * u1;
    float a2 = alpha * alpha;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (a2 - 1.0f) * u2));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float3 hLocal = make_float3(cosf(phi) * sinTheta, cosTheta, sinf(phi) * sinTheta);
    float3 h = toWorld(hLocal, n);
    float NoH = fmaxf(0.0f, dot(n, h));
    float D = ggxD(alpha, NoH);
    *outPdfH = D * NoH;
    return normalize(h);
}

__device__ inline bool isValidMatId(uint32_t id, uint32_t numMaterials) {
    return id < numMaterials;
}

__device__ inline uint32_t gatherMultiChildren(
    const MaterialData& mat,
    const MaterialData* materials,
    uint32_t numMaterials,
    uint32_t outIds[4])
{
    uint32_t count = mat.numSubMaterials;
    if (count > 4u) {
        count = 4u;
    }
    uint32_t validCount = 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t id = mat.subMaterialIndices[i];
        if (isValidMatId(id, numMaterials) && materials[id].type != kEnvironmentEmitter) {
            outIds[validCount++] = id;
        }
    }
    return validCount;
}

__device__ inline bool isEmitterMaterial(
    const MaterialData& mat,
    const MaterialData* materials,
    uint32_t numMaterials)
{
    if (mat.type == kMulti) {
        uint32_t ids[4] = {};
        uint32_t count = gatherMultiChildren(mat, materials, numMaterials, ids);
        for (uint32_t i = 0; i < count; ++i) {
            const MaterialData& child = materials[ids[i]];
            if (child.type >= kDiffuseEmitter || hasValue(child.emission)) {
                return true;
            }
        }
        return false;
    }
    return mat.type >= kDiffuseEmitter || hasValue(mat.emission);
}

__device__ inline float3 emitterRadiance(const MaterialData& mat, const float3& lightToTargetDir) {
    float3 base = mat.emission * fmaxf(mat.emitterScale, 0.0f);
    if (mat.type == kDirectionalEmitter) {
        float3 d = normalize(mat.emitterDirection);
        float cone = powf(fmaxf(dot(d, lightToTargetDir), 0.0f), 64.0f);
        return base * cone;
    }
    return base;
}

__device__ inline float3 emitterRadianceResolved(
    const MaterialData& mat,
    const MaterialData* materials,
    uint32_t numMaterials,
    const float3& lightToTargetDir)
{
    if (mat.type == kMulti) {
        uint32_t ids[4] = {};
        uint32_t count = gatherMultiChildren(mat, materials, numMaterials, ids);
        if (count == 0) {
            return black3();
        }
        float3 sum = black3();
        for (uint32_t i = 0; i < count; ++i) {
            const MaterialData& child = materials[ids[i]];
            if (child.type != kMulti) {
                sum = sum + emitterRadiance(child, lightToTargetDir);
            }
        }
        return sum / fmaxf(static_cast<float>(count), 1.0f);
    }
    return emitterRadiance(mat, lightToTargetDir);
}

__device__ inline float3 loadVertex(const float* vertices, uint32_t vertexIndex) {
    return make_float3(
        vertices[vertexIndex * 3 + 0],
        vertices[vertexIndex * 3 + 1],
        vertices[vertexIndex * 3 + 2]);
}

__device__ bool sampleEmissiveTriangle(
    const ShadeKernelParams& params,
    uint32_t& seed,
    float3* outPosition,
    float3* outNormal,
    uint32_t* outMaterialId,
    float* outPdfArea)
{
    if (params.indices == nullptr || params.vertices == nullptr || params.triangleMaterialIds == nullptr) {
        return false;
    }
    if (params.numTriangles == 0 || params.numMaterials == 0) {
        return false;
    }

    uint32_t chosenTri = 0;
    uint32_t emissiveCount = 0;
    for (uint32_t tri = 0; tri < params.numTriangles; ++tri) {
        uint32_t matId = params.triangleMaterialIds[tri];
        if (matId >= params.numMaterials) {
            continue;
        }
        if (!isEmitterMaterial(params.materials[matId], params.materials, params.numMaterials)) {
            continue;
        }
        ++emissiveCount;
        if (randf(seed) < (1.0f / emissiveCount)) {
            chosenTri = tri;
        }
    }

    if (emissiveCount == 0) {
        return false;
    }

    const uint32_t i0 = params.indices[chosenTri * 3 + 0];
    const uint32_t i1 = params.indices[chosenTri * 3 + 1];
    const uint32_t i2 = params.indices[chosenTri * 3 + 2];
    float3 v0 = loadVertex(params.vertices, i0);
    float3 v1 = loadVertex(params.vertices, i1);
    float3 v2 = loadVertex(params.vertices, i2);

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 triN = cross(e1, e2);
    float area = 0.5f * length(triN);
    if (area <= 1e-8f) {
        return false;
    }

    float u = randf(seed);
    float v = randf(seed);
    float su = sqrtf(u);
    float b0 = 1.0f - su;
    float b1 = su * (1.0f - v);
    float b2 = su * v;

    *outPosition = v0 * b0 + v1 * b1 + v2 * b2;
    *outNormal = normalize(triN);
    *outMaterialId = params.triangleMaterialIds[chosenTri];
    *outPdfArea = (1.0f / emissiveCount) * (1.0f / area);
    return true;
}

__device__ inline void queueShadowRay(
    RayState& ray,
    const HitInfo& hit,
    const float3& shadowDir,
    float shadowDist,
    const float3& pendingDirect,
    const float3& nextOrigin,
    const float3& nextDirection,
    const float3& nextThroughput,
    uint32_t terminateAfterShadow)
{
    ray.pendingDirect = pendingDirect;
    ray.nextOrigin = nextOrigin;
    ray.nextDirection = nextDirection;
    ray.nextThroughput = nextThroughput;
    ray.terminateAfterShadow = terminateAfterShadow;
    const float3 offsetN = dot(shadowDir, hit.normal) > 0.0f ? hit.normal : -hit.normal;
    ray.origin = hit.position + offsetN * kRayEps;
    ray.direction = shadowDir;
    ray.tMin = kRayEps;
    ray.tMax = fmaxf(kRayEps * 2.0f, shadowDist - 2.0f * kRayEps);
    ray.stage = RayState::Shadow;
}

__device__ inline void maybeRussianRoulette(RayState& ray, bool* terminate) {
    *terminate = false;
    if (ray.depth < 3) {
        return;
    }
    float surviveProb = fminf(0.95f, fmaxf(0.2f, max3(ray.throughput)));
    if (randf(ray.seed) > surviveProb) {
        *terminate = true;
        return;
    }
    ray.throughput = ray.throughput / surviveProb;
}

__device__ inline float iorToF0(float etaI, float etaT) {
    float a = (etaT - etaI) / fmaxf(etaT + etaI, 1e-6f);
    return a * a;
}

__device__ inline void ue4Params(const MaterialData& mat, float3* kd, float3* F0, float* alpha) {
    float metallic = saturate(mat.metallic);
    float roughness = fmaxf(0.03f, mat.roughness);
    *alpha = roughness * roughness;
    *F0 = lerp3(make_float3(0.04f, 0.04f, 0.04f), mat.baseColor, metallic);
    *kd = mat.baseColor * (1.0f - metallic) * saturate(mat.occlusion);
}

__device__ inline void oldStyleParams(const MaterialData& mat, float3* kd, float3* F0, float* alpha) {
    *kd = mat.baseColor;
    *F0 = mat.specularColor;
    float roughness = sqrtf(fmaxf(0.0005f, 1.0f - saturate(mat.glossiness)));
    *alpha = roughness * roughness;
}

__device__ inline float3 evalGGXSpec(
    float3 F0, float alpha, const float3& n, const float3& wo, const float3& wi)
{
    float NoV = fmaxf(0.0f, dot(n, wo));
    float NoL = fmaxf(0.0f, dot(n, wi));
    if (NoV <= 0.0f || NoL <= 0.0f) {
        return black3();
    }
    float3 h = normalize(wi + wo);
    float NoH = fmaxf(0.0f, dot(n, h));
    float VoH = fmaxf(0.0f, dot(wo, h));
    float D = ggxD(alpha, NoH);
    float G = ggxG(alpha, NoV, NoL);
    float3 F = schlickFresnel(VoH, F0);
    return F * (D * G / fmaxf(4.0f * NoV * NoL, 1e-6f));
}

__device__ inline bool isDeltaMaterial(
    const MaterialData& mat,
    const MaterialData* materials,
    uint32_t numMaterials)
{
    if (mat.type == kMulti) {
        uint32_t ids[4] = {};
        uint32_t count = gatherMultiChildren(mat, materials, numMaterials, ids);
        if (count == 0) {
            return false;
        }
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t t = materials[ids[i]].type;
            if (!(t == kSpecularReflection || t == kSpecularScattering)) {
                return false;
            }
        }
        return true;
    }
    return mat.type == kSpecularReflection || mat.type == kSpecularScattering;
}

__device__ inline float3 evalMaterialBSDFSingle(
    const MaterialData& mat,
    const float3& n, const float3& wo, const float3& wi)
{
    float NoV = dot(n, wo);
    float NoL = dot(n, wi);
    float absNoV = fabsf(NoV);
    float absNoL = fabsf(NoL);
    if (absNoV <= 0.0f || absNoL <= 0.0f) {
        return black3();
    }

    if (mat.type == kMatte) {
        if (NoV <= 0.0f || NoL <= 0.0f) {
            return black3();
        }
        return mat.baseColor * (1.0f / kPi);
    }

    if (mat.type == kLambertianScattering) {
        const float F = saturate(mat.specularF0 + (1.0f - mat.specularF0) * pow5(1.0f - absNoV));
        const bool sameHemisphere = NoV * NoL > 0.0f;
        return mat.baseColor * ((sameHemisphere ? F : (1.0f - F)) * (1.0f / kPi));
    }

    if (mat.type == kUE4) {
        if (NoV <= 0.0f || NoL <= 0.0f) {
            return black3();
        }
        float3 kd, F0;
        float alpha;
        ue4Params(mat, &kd, &F0, &alpha);
        return kd * (1.0f / kPi) + evalGGXSpec(F0, alpha, n, wo, wi);
    }

    if (mat.type == kOldStyle) {
        if (NoV <= 0.0f || NoL <= 0.0f) {
            return black3();
        }
        float3 kd, F0;
        float alpha;
        oldStyleParams(mat, &kd, &F0, &alpha);
        return kd * (1.0f / kPi) + evalGGXSpec(F0, alpha, n, wo, wi);
    }

    if (mat.type == kMicrofacetReflection) {
        if (NoV <= 0.0f || NoL <= 0.0f) {
            return black3();
        }
        const float3 F0 = fresnelConductor(1.0f, mat.eta, mat.k) * mat.baseColor;
        return evalGGXSpec(F0, materialAlpha(mat), n, wo, wi);
    }

    if (mat.type == kMicrofacetScattering) {
        float etaI = fmaxf(1.0f, mat.iorExt);
        float etaT = fmaxf(1.01f, mat.iorInt);
        float F0 = iorToF0(etaI, etaT);
        if (NoV * NoL > 0.0f) {
            if (NoV <= 0.0f || NoL <= 0.0f) {
                return black3();
            }
            return evalGGXSpec(make_float3(F0, F0, F0), materialAlpha(mat), n, wo, wi) * mat.baseColor;
        }

        float F = saturate(F0 + (1.0f - F0) * pow5(1.0f - absNoV));
        return mat.baseColor * ((1.0f - F) * (1.0f / kPi));
    }

    return black3();
}

__device__ inline float3 evalMaterialBSDF(
    const MaterialData& mat, const MaterialData* materials, uint32_t numMaterials,
    const float3& n, const float3& wo, const float3& wi)
{
    if (mat.type != kMulti) {
        return evalMaterialBSDFSingle(mat, n, wo, wi);
    }

    uint32_t ids[4] = {};
    uint32_t count = gatherMultiChildren(mat, materials, numMaterials, ids);
    if (count == 0) {
        return black3();
    }

    float3 sum = black3();
    uint32_t used = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const MaterialData& child = materials[ids[i]];
        if (child.type == kMulti || child.type == kEnvironmentEmitter) {
            continue;
        }
        sum = sum + evalMaterialBSDFSingle(child, n, wo, wi);
        ++used;
    }

    if (used == 0) {
        return black3();
    }
    return sum / static_cast<float>(used);
}

__device__ inline BSDFSample sampleDiffuse(const float3& coeff, const float3& n, uint32_t& seed) {
    BSDFSample s = {};
    float pdf = 0.0f;
    float3 local = sampleCosineHemisphere(randf(seed), randf(seed), &pdf);
    s.wi = normalize(toWorld(local, n));
    s.pdf = pdf;
    s.weight = coeff;
    s.valid = (pdf > 1e-8f);
    s.isDelta = 0;
    return s;
}

__device__ inline BSDFSample sampleGGXSpecular(
    const float3& n, const float3& wo, float3 F0, float alpha, uint32_t& seed)
{
    BSDFSample s = {};
    float pdfH = 0.0f;
    float3 h = sampleGGXHalf(n, alpha, seed, &pdfH);
    float3 wi = reflectDir(-wo, h);
    float NoL = fmaxf(0.0f, dot(n, wi));
    float NoV = fmaxf(0.0f, dot(n, wo));
    float VoH = fmaxf(0.0f, dot(wo, h));
    float NoH = fmaxf(0.0f, dot(n, h));
    if (NoL <= 0.0f || NoV <= 0.0f || VoH <= 0.0f || NoH <= 0.0f) {
        s.valid = 0;
        return s;
    }

    float D = ggxD(alpha, NoH);
    float G = ggxG(alpha, NoV, NoL);
    float3 F = schlickFresnel(VoH, F0);
    float3 f = F * (D * G / fmaxf(4.0f * NoV * NoL, 1e-6f));
    float pdf = pdfH / fmaxf(4.0f * VoH, 1e-6f);
    s.wi = wi;
    s.pdf = pdf;
    s.weight = f * (NoL / fmaxf(pdf, 1e-8f));
    s.valid = (pdf > 1e-8f);
    s.isDelta = 0;
    return s;
}

__device__ inline BSDFSample sampleMaterialSingle(
    const MaterialData& mat,
    const HitInfo& hit,
    RayState& ray,
    const float3& wo,
    uint32_t& seed)
{
    BSDFSample s = {};
    const float3 n = hit.normal;

    if (mat.type == kMatte) {
        return sampleDiffuse(mat.baseColor, n, seed);
    }

    if (mat.type == kLambertianScattering) {
        float pdf = 0.0f;
        float3 local = sampleCosineHemisphere(randf(seed), randf(seed), &pdf);
        float3 wiReflect = normalize(toWorld(local, n));
        float3 wiTransmit = -wiReflect;

        float cosNV = fabsf(dot(n, wo));
        float F = saturate(mat.specularF0 + (1.0f - mat.specularF0) * pow5(1.0f - cosNV));
        float reflectProb = fminf(0.95f, fmaxf(0.05f, F));

        if (randf(seed) < reflectProb) {
            s.wi = wiReflect;
            s.weight = mat.baseColor * (F / reflectProb);
            s.pdf = pdf * reflectProb;
            s.valid = (s.pdf > 1e-8f);
            s.isDelta = 0;
            return s;
        }

        s.wi = wiTransmit;
        s.weight = mat.baseColor * ((1.0f - F) / fmaxf(1.0f - reflectProb, 1e-6f));
        s.pdf = pdf * (1.0f - reflectProb);
        s.valid = (s.pdf > 1e-8f);
        s.isDelta = 0;
        return s;
    }

    if (mat.type == kUE4) {
        float3 kd, F0;
        float alpha;
        ue4Params(mat, &kd, &F0, &alpha);
        float specProb = fmaxf(0.1f, fminf(0.9f, max3(F0)));
        if (randf(seed) < specProb) {
            s = sampleGGXSpecular(n, wo, F0, alpha, seed);
            s.weight = s.weight / specProb;
        } else {
            s = sampleDiffuse(kd, n, seed);
            s.weight = s.weight / (1.0f - specProb);
        }
        return s;
    }

    if (mat.type == kOldStyle) {
        float3 kd, F0;
        float alpha;
        oldStyleParams(mat, &kd, &F0, &alpha);
        float specProb = fmaxf(0.1f, fminf(0.9f, max3(F0)));
        if (randf(seed) < specProb) {
            s = sampleGGXSpecular(n, wo, F0, alpha, seed);
            s.weight = s.weight / specProb;
        } else {
            s = sampleDiffuse(kd, n, seed);
            s.weight = s.weight / (1.0f - specProb);
        }
        return s;
    }

    if (mat.type == kSpecularReflection) {
        float cosNV = fabsf(dot(n, wo));
        float3 F = fresnelConductor(cosNV, mat.eta, mat.k) * mat.baseColor;
        s.wi = reflectDir(-wo, n);
        s.weight = F;
        s.pdf = 1.0f;
        s.isDelta = 1;
        s.valid = 1;
        return s;
    }

    if (mat.type == kSpecularScattering) {
        const bool inside = ray.insideMedium != 0;
        float etaI = inside ? mat.iorInt : mat.iorExt;
        float etaT = inside ? mat.iorExt : mat.iorInt;
        float Fr = dielectricFresnel(fabsf(dot(n, wo)), etaI, etaT);
        float3 reflected = reflectDir(-wo, n);
        float3 refracted = reflected;
        bool canRefract = refractDir(-wo, n, etaI / etaT, &refracted);

        float reflectProb = canRefract ? saturate(Fr) : 1.0f;
        bool chooseRefl = !canRefract || randf(seed) < reflectProb;
        if (chooseRefl) {
            s.wi = reflected;
            s.weight = mat.baseColor * (Fr / fmaxf(reflectProb, 1e-6f));
        } else {
            float transProb = fmaxf(1.0f - reflectProb, 1e-6f);
            float etaScale = (etaI / etaT) * (etaI / etaT);
            s.wi = refracted;
            s.weight = mat.baseColor * ((1.0f - Fr) * etaScale / transProb);
            ray.insideMedium = inside ? 0u : 1u;
        }
        s.pdf = 1.0f;
        s.isDelta = 1;
        s.valid = 1;
        return s;
    }

    if (mat.type == kMicrofacetReflection) {
        float3 F0 = fresnelConductor(1.0f, mat.eta, mat.k) * mat.baseColor;
        return sampleGGXSpecular(n, wo, F0, materialAlpha(mat), seed);
    }

    if (mat.type == kMicrofacetScattering) {
        const bool inside = ray.insideMedium != 0;
        float etaI = inside ? mat.iorInt : mat.iorExt;
        float etaT = inside ? mat.iorExt : mat.iorInt;
        float Fr = dielectricFresnel(fabsf(dot(n, wo)), etaI, etaT);
        float reflectProb = fminf(0.95f, fmaxf(0.05f, Fr));

        if (randf(seed) < reflectProb) {
            float F0 = iorToF0(etaI, etaT);
            s = sampleGGXSpecular(n, wo, make_float3(F0, F0, F0), materialAlpha(mat), seed);
            s.weight = s.weight * mat.baseColor / reflectProb;
            return s;
        }

        float3 refracted;
        bool canRefract = refractDir(-wo, n, etaI / etaT, &refracted);
        if (!canRefract) {
            s.wi = reflectDir(-wo, n);
            s.weight = mat.baseColor;
            s.pdf = 1.0f;
            s.isDelta = 1;
            s.valid = 1;
            return s;
        }

        if (mat.roughness <= 0.02f) {
            float etaScale = (etaI / etaT) * (etaI / etaT);
            s.wi = refracted;
            s.weight = mat.baseColor * ((1.0f - Fr) * etaScale / fmaxf(1.0f - reflectProb, 1e-6f));
            s.pdf = 1.0f;
            s.isDelta = 1;
            s.valid = 1;
            ray.insideMedium = inside ? 0u : 1u;
            return s;
        }

        float pdf = 0.0f;
        float3 local = sampleCosineHemisphere(randf(seed), randf(seed), &pdf);
        float3 lobe = normalize(toWorld(local, refracted));
        if (dot(lobe, n) * dot(wo, n) > 0.0f) {
            lobe = -lobe;
        }

        s.wi = lobe;
        s.pdf = pdf * (1.0f - reflectProb);
        s.weight = mat.baseColor * ((1.0f - Fr) / fmaxf(1.0f - reflectProb, 1e-6f));
        s.isDelta = 0;
        s.valid = (s.pdf > 1e-8f);
        ray.insideMedium = inside ? 0u : 1u;
        return s;
    }

    return sampleDiffuse(mat.baseColor, n, seed);
}

__device__ inline BSDFSample sampleMaterial(
    const MaterialData& mat,
    const MaterialData* materials,
    uint32_t numMaterials,
    const HitInfo& hit,
    RayState& ray,
    const float3& wo,
    uint32_t& seed)
{
    if (mat.type != kMulti) {
        return sampleMaterialSingle(mat, hit, ray, wo, seed);
    }

    uint32_t ids[4] = {};
    uint32_t count = gatherMultiChildren(mat, materials, numMaterials, ids);
    if (count == 0) {
        return sampleDiffuse(mat.baseColor, hit.normal, seed);
    }

    uint32_t validIds[4] = {};
    uint32_t validCount = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const MaterialData& child = materials[ids[i]];
        if (child.type == kMulti || child.type == kEnvironmentEmitter) {
            continue;
        }
        validIds[validCount++] = ids[i];
    }
    if (validCount == 0) {
        return sampleDiffuse(mat.baseColor, hit.normal, seed);
    }

    uint32_t sel = min(static_cast<uint32_t>(randf(seed) * validCount), validCount - 1);
    BSDFSample inner = sampleMaterialSingle(materials[validIds[sel]], hit, ray, wo, seed);
    if (inner.valid) {
        inner.weight = inner.weight * static_cast<float>(validCount);
    }
    return inner;
}

extern "C" __global__ void shade(ShadeKernelParams params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.numActive) {
        return;
    }

    uint32_t rayIndex = params.activeIndices[idx];
    RayState& ray = params.rayPool[rayIndex];

    if (ray.stage == RayState::Terminated) {
        params.accumBuffer[ray.pixelIndex] = params.accumBuffer[ray.pixelIndex] + ray.radiance;
        return;
    }
    if (ray.stage != RayState::Shade) {
        return;
    }

    if (params.materials == nullptr || params.numMaterials == 0) {
        ray.stage = RayState::Terminated;
        params.accumBuffer[ray.pixelIndex] = params.accumBuffer[ray.pixelIndex] + ray.radiance;
        return;
    }

    const HitInfo& hit = params.hitBuffer[rayIndex];
    if (hit.materialId >= params.numMaterials) {
        ray.stage = RayState::Terminated;
        params.accumBuffer[ray.pixelIndex] = params.accumBuffer[ray.pixelIndex] + ray.radiance;
        return;
    }

    const MaterialData& mat = params.materials[hit.materialId];
    if (isEmitterMaterial(mat, params.materials, params.numMaterials)) {
        float3 Le = emitterRadianceResolved(mat, params.materials, params.numMaterials, -ray.direction);
        ray.radiance = ray.radiance + ray.throughput * Le;
        ray.stage = RayState::Terminated;
        params.accumBuffer[ray.pixelIndex] = params.accumBuffer[ray.pixelIndex] + ray.radiance;
        return;
    }

    const float3 wo = -ray.direction;
    float3 directContribution = black3();
    bool hasDirectSample = false;
    float3 shadowWi = black3();
    float shadowDist = 0.0f;

    BSDFSample bs = sampleMaterial(mat, params.materials, params.numMaterials, hit, ray, wo, ray.seed);
    if (!bs.valid) {
        ray.stage = RayState::Terminated;
        params.accumBuffer[ray.pixelIndex] = params.accumBuffer[ray.pixelIndex] + ray.radiance;
        return;
    }

    // NEE with true visibility for non-delta material families.
    if (!bs.isDelta && !isDeltaMaterial(mat, params.materials, params.numMaterials)) {
        float3 lightPos, lightNormal;
        uint32_t lightMatId = 0;
        float lightPdfArea = 0.0f;
        if (sampleEmissiveTriangle(params, ray.seed, &lightPos, &lightNormal, &lightMatId, &lightPdfArea)) {
            float3 toLight = lightPos - hit.position;
            float dist2 = dot(toLight, toLight);
            if (dist2 > 1e-8f && lightPdfArea > 1e-8f && lightMatId < params.numMaterials) {
                shadowWi = normalize(toLight);
                float cosSurfaceAbs = fabsf(dot(hit.normal, shadowWi));
                float cosLight = fmaxf(0.0f, dot(lightNormal, -shadowWi));
                if (cosSurfaceAbs > 0.0f && cosLight > 0.0f) {
                    const MaterialData& lightMat = params.materials[lightMatId];
                    float3 Le = emitterRadianceResolved(
                        lightMat, params.materials, params.numMaterials, -shadowWi);
                    if (hasValue(Le)) {
                        float3 f = evalMaterialBSDF(
                            mat, params.materials, params.numMaterials, hit.normal, wo, shadowWi);
                        if (hasValue(f)) {
                            float geom = (cosSurfaceAbs * cosLight) / dist2;
                            directContribution = ray.throughput * f * Le * (geom / lightPdfArea);
                            hasDirectSample = hasValue(directContribution);
                            shadowDist = sqrtf(dist2);
                        }
                    }
                }
            }
        }
    }

    float3 nextThroughput = ray.throughput * bs.weight;
    float3 offsetN = dot(bs.wi, hit.normal) > 0.0f ? hit.normal : -hit.normal;
    float3 nextOrigin = hit.position + offsetN * kRayEps;
    float3 nextDirection = bs.wi;

    ray.throughput = nextThroughput;
    ray.depth++;

    bool terminateIndirect = false;
    maybeRussianRoulette(ray, &terminateIndirect);
    nextThroughput = ray.throughput;

    if (terminateIndirect) {
        if (hasDirectSample) {
            queueShadowRay(
                ray, hit, shadowWi, shadowDist, directContribution,
                ray.origin, ray.direction, ray.throughput, 1);
            return;
        }
        ray.stage = RayState::Terminated;
        params.accumBuffer[ray.pixelIndex] = params.accumBuffer[ray.pixelIndex] + ray.radiance;
        return;
    }

    if (hasDirectSample) {
        queueShadowRay(
            ray, hit, shadowWi, shadowDist, directContribution,
            nextOrigin, nextDirection, nextThroughput, 0);
        return;
    }

    ray.origin = nextOrigin;
    ray.direction = nextDirection;
    ray.throughput = nextThroughput;
    ray.tMin = kRayEps;
    ray.tMax = 1e20f;
    ray.stage = RayState::Trace;
}
