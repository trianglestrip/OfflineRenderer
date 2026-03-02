#include "optixw/types.h"
#include "optixw/wavefront_kernel_params.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define M_1_PI (1.0f / M_PI)

// Material types
#define kMatte 0
#define kLambertianScattering 1
#define kUE4 2
#define kOldStyle 3
#define kSpecularReflection 4
#define kSpecularScattering 5
#define kMicrofacetReflection 6
#define kMicrofacetScattering 7
#define kDiffuseEmitter 8
#define kSpecularEmitter 9
#define kEnvironmentEmitter 10
#define kMulti 11

// BSDF sample structure
struct BSDFSample {
    float3 wi;
    float3 weight;
    float pdf;
    uint32_t valid;
    uint32_t isDelta;
};

__device__ inline float randf(uint32_t& seed) {
    seed = 1103515245 * seed + 12345;
    return static_cast<float>(seed & 0x7FFFFFFF) / 2147483648.0f;
}

__device__ inline float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

__device__ inline float saturate(float x) {
    return fmaxf(0.0f, fminf(1.0f, x));
}

__device__ inline float max3(const float3& v) {
    return fmaxf(fmaxf(v.x, v.y), v.z);
}



__device__ inline float3 black3() {
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ inline float3 white3() {
    return make_float3(1.0f, 1.0f, 1.0f);
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ inline float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ inline float3 operator+(const float3& a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__device__ inline float3 sqrtf(const float3& v) {
    return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ inline float3 normalize(const float3& v) {
    float l = length(v);
    if (l < 1e-8f) {
        return black3();
    }
    return v / l;
}

__device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float3 reflectDir(const float3& wi, const float3& n) {
    return wi - n * 2.0f * dot(wi, n);
}

__device__ inline bool refractDir(const float3& wi, const float3& n, float eta, float3* wt) {
    float cosThetaI = dot(-wi, n);
    float sin2ThetaI = fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;
    if (sin2ThetaT >= 1.0f) {
        return false;
    }
    float cosThetaT = sqrtf(1.0f - sin2ThetaT);
    *wt = normalize(wi * eta + n * (eta * cosThetaI - cosThetaT));
    return true;
}

__device__ inline float3 schlickFresnel(float cosTheta, const float3& F0) {
    float exponent = powf(1.0f - cosTheta, 5.0f);
    return F0 + (white3() - F0) * exponent;
}

__device__ inline float dielectricFresnel(float cosTheta, float etaI, float etaT) {
    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0f) {
        return 1.0f;
    }
    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    float rPar = (etaT * cosTheta - etaI * cosThetaT) / (etaT * cosTheta + etaI * cosThetaT);
    float rPerp = (etaI * cosTheta - etaT * cosThetaT) / (etaI * cosTheta + etaT * cosThetaT);
    return (rPar * rPar + rPerp * rPerp) * 0.5f;
}

__device__ inline float3 fresnelConductor(float cosTheta, const float3& eta, const float3& k) {
    float cosTheta2 = cosTheta * cosTheta;
    float sinTheta2 = 1.0f - cosTheta2;
    float3 eta2 = eta * eta;
    float3 k2 = k * k;
    float3 t1 = eta2 - k2 - make_float3(sinTheta2, sinTheta2, sinTheta2);
    float3 a2plusb2 = sqrtf(t1 * t1 + make_float3(4.0f, 4.0f, 4.0f) * eta2 * k2);
    float3 t2 = a2plusb2 + make_float3(cosTheta2, cosTheta2, cosTheta2);
    float3 a = sqrtf((a2plusb2 + t1) * 0.5f);
    float3 t3 = make_float3(2.0f, 2.0f, 2.0f) * a * cosTheta;
    float3 Rs = (t2 - t3) / (t2 + t3);
    float3 t4 = a2plusb2 * cosTheta2 + make_float3(sinTheta2 * sinTheta2, sinTheta2 * sinTheta2, sinTheta2 * sinTheta2);
    float3 t5 = t3 * sinTheta2;
    float3 Rp = Rs * (t4 - t5) / (t4 + t5);
    return (Rp + Rs) * 0.5f;
}

__device__ inline float iorToF0(float etaI, float etaT) {
    float f = (etaI - etaT) / (etaI + etaT);
    return f * f;
}

__device__ inline float matToLinear(float x) {
    if (x <= 0.04045f) {
        return x / 12.92f;
    } else {
        return powf((x + 0.055f) / 1.055f, 2.4f);
    }
}

__device__ inline float3 srgbToLinear(const float3& c) {
    return make_float3(
        matToLinear(c.x),
        matToLinear(c.y),
        matToLinear(c.z)
    );
}

__device__ inline float3 linearToSrgb(const float3& c) {
    auto clamp = [](float x) { return fmaxf(0.0f, fminf(1.0f, x)); };
    auto toSrgb = [&clamp](float x) {
        if (x <= 0.0031308f) {
            return x * 12.92f;
        } else {
            return 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
        }
    };
    return make_float3(
        clamp(toSrgb(c.x)),
        clamp(toSrgb(c.y)),
        clamp(toSrgb(c.z))
    );
}

__device__ inline float3 acesFilmic(const float3& x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    float3 numerator = x * (a * x + b);
    float3 denominator = x * (c * x + d) + e;
    return make_float3(
        numerator.x / denominator.x,
        numerator.y / denominator.y,
        numerator.z / denominator.z
    );
}

__device__ inline float3 sampleCosineHemisphere(float u1, float u2, float* pdf) {
    float r = sqrtf(u1);
    float theta = 2.0f * M_PI * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    *pdf = z * M_1_PI;
    return make_float3(x, y, z);
}

__device__ inline float3 toWorld(const float3& local, const float3& n) {
    float3 up = fabsf(n.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);
    return local.x * tangent + local.y * bitangent + local.z * n;
}

__device__ inline float ggxD(float alpha, float NoH) {
    float alpha2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float denominator = NoH2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (M_PI * denominator * denominator);
}

__device__ inline float ggxG1(float alpha, float NoV) {
    float alpha2 = alpha * alpha;
    float NoV2 = NoV * NoV;
    float denominator = NoV + sqrtf(NoV2 + alpha2 * (1.0f - NoV2));
    return 2.0f * NoV / denominator;
}

__device__ inline float ggxG(float alpha, float NoV, float NoL) {
    return ggxG1(alpha, NoV) * ggxG1(alpha, NoL);
}

__device__ inline float3 sampleGGXHalf(const float3& n, float alpha, uint32_t& seed, float* pdf) {
    float u1 = randf(seed);
    float u2 = randf(seed);
    float phi = 2.0f * M_PI * u2;
    float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha * alpha - 1.0f) * u1));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float3 h = make_float3(
        sinTheta * cosf(phi),
        sinTheta * sinf(phi),
        cosTheta
    );
    *pdf = ggxD(alpha, cosTheta) * cosTheta;
    return toWorld(h, n);
}

__device__ inline optixw::MaterialData resolveMaterial(
    const optixw::MaterialData& mat,
    const optixw::HitInfo& hit,
    const optixw::Texture2DData* textures,
    uint32_t numTextures
) {
    optixw::MaterialData result = mat;
    if (mat.baseColorTextureId < numTextures && textures != nullptr) {
        const optixw::Texture2DData& tex = textures[mat.baseColorTextureId];
        if (tex.pixels != nullptr && tex.width > 0 && tex.height > 0) {
            float u = hit.texCoord.x;
            float v = hit.texCoord.y;
            u = fmodf(u, 1.0f);
            v = fmodf(v, 1.0f);
            if (u < 0.0f) u += 1.0f;
            if (v < 0.0f) v += 1.0f;
            uint32_t x = static_cast<uint32_t>(u * tex.width);
            uint32_t y = static_cast<uint32_t>((1.0f - v) * tex.height);
            uint32_t idx = y * tex.width + x;
            if (idx < tex.width * tex.height) {
                float4 texel = tex.pixels[idx];
                result.baseColor = srgbToLinear(make_float3(texel.x, texel.y, texel.z)) * mat.baseColor;
            }
        }
    }
    return result;
}

__device__ inline uint32_t gatherMultiChildren(
    const optixw::MaterialData& mat,
    const optixw::MaterialData* materials,
    uint32_t numMaterials,
    uint32_t* ids
) {
    uint32_t count = 0;
    if (mat.numSubMaterials > 0 && materials != nullptr) {
        for (uint32_t i = 0; i < mat.numSubMaterials && count < 4; ++i) {
            uint32_t childId = mat.subMaterialIndices[i];
            if (childId < numMaterials) {
                ids[count++] = childId;
            }
        }
    }
    return count;
}

__device__ inline void ue4Params(const optixw::MaterialData& mat, float3* kd, float3* F0, float* alpha) {
    *kd = mat.baseColor * (1.0f - mat.metallic);
    *F0 = make_float3(0.04f, 0.04f, 0.04f) * (1.0f - mat.metallic) + mat.baseColor * mat.metallic;
    *alpha = mat.roughness * mat.roughness;
}

__device__ inline void oldStyleParams(const optixw::MaterialData& mat, float3* kd, float3* F0, float* alpha) {
    *kd = mat.baseColor;
    *F0 = mat.specularColor;
    *alpha = 1.0f - mat.glossiness;
    *alpha *= *alpha;
}

__device__ inline float materialAlpha(const optixw::MaterialData& mat) {
    return mat.roughness * mat.roughness;
}

__device__ inline float3 evalMaterialBSDFSingle(
    const optixw::MaterialData& inMat,
    const optixw::HitInfo& hit,
    const optixw::Texture2DData* textures,
    uint32_t numTextures,
    const float3& n,
    const float3& wo,
    const float3& wi
) {
    const optixw::MaterialData mat = resolveMaterial(inMat, hit, textures, numTextures);
    float NoL = fmaxf(0.0f, dot(n, wi));
    float NoV = fmaxf(0.0f, dot(n, wo));
    if (NoL <= 0.0f || NoV <= 0.0f) {
        return black3();
    }

    if (mat.type == kMatte) {
        return mat.baseColor * M_1_PI * NoL;
    }

    if (mat.type == kLambertianScattering) {
        float cosNV = fabsf(dot(n, wo));
        float F = saturate(mat.specularF0 + (1.0f - mat.specularF0) * pow5(1.0f - cosNV));
        float3 diffuse = mat.baseColor * M_1_PI * NoL;
        float3 specular = schlickFresnel(dot(n, normalize(wo + wi)), make_float3(F, F, F)) * NoL;
        return diffuse + specular;
    }

    if (mat.type == kUE4) {
        float3 kd, F0;
        float alpha;
        ue4Params(mat, &kd, &F0, &alpha);
        float3 H = normalize(wo + wi);
        float NoH = fmaxf(0.0f, dot(n, H));
        float VoH = fmaxf(0.0f, dot(wo, H));
        float D = ggxD(alpha, NoH);
        float G = ggxG(alpha, NoV, NoL);
        float3 F = schlickFresnel(VoH, F0);
        float3 spec = F * (D * G / fmaxf(4.0f * NoV * NoL, 1e-6f));
        float3 diff = kd * M_1_PI;
        return (diff + spec) * NoL;
    }

    if (mat.type == kOldStyle) {
        float3 kd, F0;
        float alpha;
        oldStyleParams(mat, &kd, &F0, &alpha);
        float3 H = normalize(wo + wi);
        float NoH = fmaxf(0.0f, dot(n, H));
        float VoH = fmaxf(0.0f, dot(wo, H));
        float D = ggxD(alpha, NoH);
        float G = ggxG(alpha, NoV, NoL);
        float3 F = schlickFresnel(VoH, F0);
        float3 spec = F * (D * G / fmaxf(4.0f * NoV * NoL, 1e-6f));
        float3 diff = kd * M_1_PI;
        return (diff + spec) * NoL;
    }

    if (mat.type == kSpecularReflection) {
        float3 H = normalize(wo + wi);
        if (fabsf(dot(n, H)) < 0.999f) {
            return black3();
        }
        float cosNV = fabsf(dot(n, wo));
        float3 F = fresnelConductor(cosNV, mat.eta, mat.k) * mat.baseColor;
        return F * NoL;
    }

    if (mat.type == kSpecularScattering) {
        const bool inside = false;
        float3 normal = inside ? -n : n;
        float etaI = inside ? mat.iorInt : mat.iorExt;
        float etaT = inside ? mat.iorExt : mat.iorInt;
        float3 reflected = reflectDir(-wo, normal);
        float3 refracted;
        bool canRefract = refractDir(-wo, normal, etaI / etaT, &refracted);
        if (dot(wi, reflected) > 0.999f) {
            float cosNV = fabsf(dot(n, wo));
            float Fr = dielectricFresnel(cosNV, etaI, etaT);
            return mat.baseColor * Fr * NoL;
        } else if (canRefract && dot(wi, refracted) > 0.999f) {
            float cosNV = fabsf(dot(n, wo));
            float Fr = dielectricFresnel(cosNV, etaI, etaT);
            float etaScale = (etaI / etaT) * (etaI / etaT);
            return mat.baseColor * (1.0f - Fr) * etaScale * NoL;
        } else {
            return black3();
        }
    }

    if (mat.type == kMicrofacetReflection) {
        float3 F0 = fresnelConductor(1.0f, mat.eta, mat.k) * mat.baseColor;
        float3 H = normalize(wo + wi);
        float NoH = fmaxf(0.0f, dot(n, H));
        float VoH = fmaxf(0.0f, dot(wo, H));
        float alpha = materialAlpha(mat);
        float D = ggxD(alpha, NoH);
        float G = ggxG(alpha, NoV, NoL);
        float3 F = schlickFresnel(VoH, F0);
        return F * (D * G / fmaxf(4.0f * NoV * NoL, 1e-6f)) * NoL;
    }

    if (mat.type == kMicrofacetScattering) {
        const bool inside = false;
        float3 normal = inside ? -n : n;
        float etaI = inside ? mat.iorInt : mat.iorExt;
        float etaT = inside ? mat.iorExt : mat.iorInt;
        float3 H = normalize(wo + wi);
        float NoH = fmaxf(0.0f, dot(n, H));
        float VoH = fmaxf(0.0f, dot(wo, H));
        float cosNV = fabsf(dot(n, wo));
        float Fr = dielectricFresnel(cosNV, etaI, etaT);
        float alpha = materialAlpha(mat);
        float D = ggxD(alpha, NoH);
        float G = ggxG(alpha, NoV, NoL);
        float3 F = schlickFresnel(VoH, make_float3(iorToF0(etaI, etaT), iorToF0(etaI, etaT), iorToF0(etaI, etaT)));
        float3 spec = F * (D * G / fmaxf(4.0f * NoV * NoL, 1e-6f));
        float3 diff = mat.baseColor * M_1_PI;
        return (diff + spec) * NoL;
    }

    return mat.baseColor * M_1_PI * NoL;
}

__device__ inline float3 evalMaterialBSDF(
    const optixw::MaterialData& mat, const optixw::MaterialData* materials, uint32_t numMaterials,
    const optixw::HitInfo& hit, const optixw::Texture2DData* textures, uint32_t numTextures,
    const float3& n, const float3& wo, const float3& wi
) {
    if (mat.type != kMulti) {
        return evalMaterialBSDFSingle(mat, hit, textures, numTextures, n, wo, wi);
    }

    uint32_t ids[4] = {};
    uint32_t count = gatherMultiChildren(mat, materials, numMaterials, ids);
    if (count == 0) {
        return black3();
    }

    float3 sum = black3();
    uint32_t used = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const optixw::MaterialData& child = materials[ids[i]];
        if (child.type == kMulti || child.type == kEnvironmentEmitter) {
            continue;
        }
        float3 bsdf = evalMaterialBSDFSingle(child, hit, textures, numTextures, n, wo, wi);
        sum = sum + bsdf;
        ++used;
    }

    if (used == 0) {
        return black3();
    }
    return sum / static_cast<float>(used);
}

__device__ inline float evalMaterialPDFSingle(
    const optixw::MaterialData& inMat,
    const optixw::HitInfo& hit,
    const optixw::Texture2DData* textures,
    uint32_t numTextures,
    const float3& n,
    const float3& wo,
    const float3& wi
) {
    const optixw::MaterialData mat = resolveMaterial(inMat, hit, textures, numTextures);
    float NoL = fmaxf(0.0f, dot(n, wi));
    float NoV = fmaxf(0.0f, dot(n, wo));
    if (NoL <= 0.0f || NoV <= 0.0f) {
        return 0.0f;
    }

    if (mat.type == kMatte) {
        return NoL * M_1_PI;
    }

    if (mat.type == kLambertianScattering) {
        return NoL * M_1_PI;
    }

    if (mat.type == kUE4) {
        float3 kd, F0;
        float alpha;
        ue4Params(mat, &kd, &F0, &alpha);
        float specProb = fmaxf(0.1f, fminf(0.9f, max3(F0)));
        float3 H = normalize(wo + wi);
        float NoH = fmaxf(0.0f, dot(n, H));
        float VoH = fmaxf(0.0f, dot(wo, H));
        float pdfH = ggxD(alpha, NoH) * NoH;
        float pdfSpec = pdfH / (4.0f * VoH);
        float pdfDiff = NoL * M_1_PI;
        return specProb * pdfSpec + (1.0f - specProb) * pdfDiff;
    }

    if (mat.type == kOldStyle) {
        float3 kd, F0;
        float alpha;
        oldStyleParams(mat, &kd, &F0, &alpha);
        float specProb = fmaxf(0.1f, fminf(0.9f, max3(F0)));
        float3 H = normalize(wo + wi);
        float NoH = fmaxf(0.0f, dot(n, H));
        float VoH = fmaxf(0.0f, dot(wo, H));
        float pdfH = ggxD(alpha, NoH) * NoH;
        float pdfSpec = pdfH / (4.0f * VoH);
        float pdfDiff = NoL * M_1_PI;
        return specProb * pdfSpec + (1.0f - specProb) * pdfDiff;
    }

    if (mat.type == kSpecularReflection || mat.type == kSpecularScattering) {
        return 0.0f;
    }

    if (mat.type == kMicrofacetReflection) {
        float3 H = normalize(wo + wi);
        float NoH = fmaxf(0.0f, dot(n, H));
        float VoH = fmaxf(0.0f, dot(wo, H));
        float alpha = materialAlpha(mat);
        float pdfH = ggxD(alpha, NoH) * NoH;
        return pdfH / (4.0f * VoH);
    }

    if (mat.type == kMicrofacetScattering) {
        float3 H = normalize(wo + wi);
        float NoH = fmaxf(0.0f, dot(n, H));
        float VoH = fmaxf(0.0f, dot(wo, H));
        float alpha = materialAlpha(mat);
        float pdfH = ggxD(alpha, NoH) * NoH;
        return pdfH / (4.0f * VoH);
    }

    return NoL * M_1_PI;
}

__device__ inline float evalMaterialPDF(
    const optixw::MaterialData& mat, const optixw::MaterialData* materials, uint32_t numMaterials,
    const optixw::HitInfo& hit, const optixw::Texture2DData* textures, uint32_t numTextures,
    const float3& n, const float3& wo, const float3& wi
) {
    if (mat.type != kMulti) {
        return evalMaterialPDFSingle(mat, hit, textures, numTextures, n, wo, wi);
    }

    uint32_t ids[4] = {};
    uint32_t count = gatherMultiChildren(mat, materials, numMaterials, ids);
    if (count == 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    uint32_t used = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const optixw::MaterialData& child = materials[ids[i]];
        if (child.type == kMulti || child.type == kEnvironmentEmitter) {
            continue;
        }
        float pdf = evalMaterialPDFSingle(child, hit, textures, numTextures, n, wo, wi);
        sum = sum + pdf;
        ++used;
    }

    if (used == 0) {
        return 0.0f;
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
    const float3& n, const float3& wo, float3 F0, float alpha, uint32_t& seed
) {
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
    float denom = fmaxf(4.0f * NoV * NoL, 1e-6f);
    float3 f = make_float3(F.x * (D * G / denom), F.y * (D * G / denom), F.z * (D * G / denom));
    float pdf = pdfH / fmaxf(4.0f * VoH, 1e-6f);
    s.wi = wi;
    s.pdf = pdf;
    float weightDenom = fmaxf(pdf, 1e-8f);
    s.weight = make_float3(f.x * (NoL / weightDenom), f.y * (NoL / weightDenom), f.z * (NoL / weightDenom));
    s.valid = (pdf > 1e-8f);
    s.isDelta = 0;
    return s;
}

__device__ inline BSDFSample sampleMaterialSingle(
    const optixw::MaterialData& inMat,
    const optixw::Texture2DData* textures,
    uint32_t numTextures,
    const optixw::HitInfo& hit,
    optixw::RayState& ray,
    const float3& wo,
    uint32_t& seed
) {
    BSDFSample s = {};
    const optixw::MaterialData mat = resolveMaterial(inMat, hit, textures, numTextures);
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
        float denom = fmaxf(1.0f - reflectProb, 1e-6f);
        s.weight = mat.baseColor * ((1.0f - F) / denom);
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
        float3 normal = inside ? -n : n;
        float etaI = inside ? mat.iorInt : mat.iorExt;
        float etaT = inside ? mat.iorExt : mat.iorInt;
        float Fr = dielectricFresnel(fabsf(dot(normal, wo)), etaI, etaT);
        float3 reflected = reflectDir(-wo, normal);
        float3 refracted = reflected;
        bool canRefract = refractDir(-wo, normal, etaI / etaT, &refracted);

        float reflectProb = canRefract ? saturate(Fr) : 1.0f;
        bool chooseRefl = !canRefract || randf(seed) < reflectProb;
        if (chooseRefl) {
            s.wi = reflected;
            float denom = fmaxf(reflectProb, 1e-6f);
            s.weight = mat.baseColor * (Fr / denom);
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
        float3 normal = inside ? -n : n;
        float etaI = inside ? mat.iorInt : mat.iorExt;
        float etaT = inside ? mat.iorExt : mat.iorInt;
        float Fr = dielectricFresnel(fabsf(dot(normal, wo)), etaI, etaT);
        float reflectProb = fminf(0.95f, fmaxf(0.05f, Fr));

        if (randf(seed) < reflectProb) {
            float F0 = iorToF0(etaI, etaT);
            s = sampleGGXSpecular(normal, wo, make_float3(F0, F0, F0), materialAlpha(mat), seed);
            s.weight = s.weight * mat.baseColor / reflectProb;
            return s;
        }

        float3 refracted;
        bool canRefract = refractDir(-wo, normal, etaI / etaT, &refracted);
        if (!canRefract) {
            s.wi = reflectDir(-wo, normal);
            s.weight = mat.baseColor;
            s.pdf = 1.0f;
            s.isDelta = 1;
            s.valid = 1;
            return s;
        }

        if (mat.roughness <= 0.02f) {
            float etaScale = (etaI / etaT) * (etaI / etaT);
            s.wi = refracted;
            float denom = fmaxf(1.0f - reflectProb, 1e-6f);
            s.weight = mat.baseColor * ((1.0f - Fr) * etaScale / denom);
            s.pdf = 1.0f;
            s.isDelta = 1;
            s.valid = 1;
            ray.insideMedium = inside ? 0u : 1u;
            return s;
        }

        float pdf = 0.0f;
        float3 local = sampleCosineHemisphere(randf(seed), randf(seed), &pdf);
        float3 lobe = normalize(toWorld(local, refracted));
        if (dot(lobe, normal) * dot(wo, normal) > 0.0f) {
            lobe = -lobe;
        }

        s.wi = lobe;
        s.pdf = pdf * (1.0f - reflectProb);
        float denom = fmaxf(1.0f - reflectProb, 1e-6f);
        s.weight = mat.baseColor * ((1.0f - Fr) / denom);
        s.isDelta = 0;
        s.valid = (s.pdf > 1e-8f);
        ray.insideMedium = inside ? 0u : 1u;
        return s;
    }

    return sampleDiffuse(mat.baseColor, n, seed);
}

__device__ inline BSDFSample sampleMaterial(
    const optixw::MaterialData& mat,
    const optixw::MaterialData* materials,
    uint32_t numMaterials,
    const optixw::Texture2DData* textures,
    uint32_t numTextures,
    const optixw::HitInfo& hit,
    optixw::RayState& ray,
    const float3& wo,
    uint32_t& seed
) {
    if (mat.type != kMulti) {
        return sampleMaterialSingle(mat, textures, numTextures, hit, ray, wo, seed);
    }

    uint32_t ids[4] = {};
    uint32_t count = gatherMultiChildren(mat, materials, numMaterials, ids);
    if (count == 0) {
        const optixw::MaterialData resolved = resolveMaterial(mat, hit, textures, numTextures);
        return sampleDiffuse(resolved.baseColor, hit.normal, seed);
    }

    uint32_t validIds[4] = {};
    uint32_t validCount = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const optixw::MaterialData& child = materials[ids[i]];
        if (child.type == kMulti || child.type == kEnvironmentEmitter) {
            continue;
        }
        validIds[validCount++] = ids[i];
    }
    if (validCount == 0) {
        const optixw::MaterialData resolved = resolveMaterial(mat, hit, textures, numTextures);
        return sampleDiffuse(resolved.baseColor, hit.normal, seed);
    }

    uint32_t sel = min(static_cast<uint32_t>(randf(seed) * validCount), validCount - 1);
    BSDFSample inner = sampleMaterialSingle(materials[validIds[sel]], textures, numTextures, hit, ray, wo, seed);
    if (inner.valid) {
        inner.weight = inner.weight * static_cast<float>(validCount);
    }
    return inner;
}

__device__ inline float3 sampleEmissiveTriangle(
    const float* vertices,
    const uint32_t* indices,
    const optixw::MaterialData* materials,
    uint32_t materialId,
    const float3& hitPoint,
    uint32_t& seed,
    float* pdf
) {
    if (materials == nullptr || materialId >= 10000) {
        *pdf = 0.0f;
        return black3();
    }

    const optixw::MaterialData& mat = materials[materialId];
    if (mat.type != kDiffuseEmitter && mat.type != kSpecularEmitter) {
        *pdf = 0.0f;
        return black3();
    }

    float u1 = randf(seed);
    float u2 = randf(seed);
    float su1 = sqrtf(u1);
    float3 barycentric = make_float3(1.0f - su1, u2 * su1, 1.0f - su1 - u2 * su1);
    
    if (indices == nullptr || vertices == nullptr) {
        *pdf = 0.0f;
        return black3();
    }
    
    float3 v0 = make_float3(vertices[0], vertices[1], vertices[2]);
    float3 v1 = make_float3(vertices[3], vertices[4], vertices[5]);
    float3 v2 = make_float3(vertices[6], vertices[7], vertices[8]);
    float3 p = barycentric.x * v0 + barycentric.y * v1 + barycentric.z * v2;
    float3 wi = normalize(p - hitPoint);

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 n = cross(e1, e2);
    float area = 0.5f * length(n);
    n = normalize(n);

    float NoL = fmaxf(0.0f, dot(n, -wi));
    if (NoL <= 0.0f) {
        *pdf = 0.0f;
        return black3();
    }

    float distanceSquared = dot(p - hitPoint, p - hitPoint);
    *pdf = distanceSquared / (NoL * area);

    if (mat.type == kDiffuseEmitter) {
        return mat.baseColor * mat.emitterScale;
    } else {
        float3 F = schlickFresnel(dot(n, -wi), mat.baseColor);
        return F * mat.emitterScale;
    }
}

__device__ inline float3 sampleEnvironment(
    const optixw::EnvironmentMappingData& env,
    const float3& hitPoint,
    uint32_t& seed,
    float* pdf
) {
    if (env.environmentMap == nullptr || env.environmentMapWidth == 0 || env.environmentMapHeight == 0) {
        *pdf = 0.0f;
        return black3();
    }

    float u = randf(seed);
    float v = randf(seed);
    uint32_t x = static_cast<uint32_t>(u * env.environmentMapWidth);
    uint32_t y = static_cast<uint32_t>(v * env.environmentMapHeight);
    uint32_t idx = y * env.environmentMapWidth + x;
    if (idx >= env.environmentMapWidth * env.environmentMapHeight) {
        *pdf = 0.0f;
        return black3();
    }

    float4 texel = env.environmentMap[idx];
    float3 color = srgbToLinear(make_float3(texel.x, texel.y, texel.z)) * env.environmentMapScale;

    float theta = v * M_PI;
    float phi = u * 2.0f * M_PI;
    float3 wi = make_float3(
        sinf(theta) * cosf(phi),
        sinf(theta) * sinf(phi),
        cosf(theta)
    );

    float sinTheta = sinf(theta);
    if (sinTheta < 1e-8f) {
        *pdf = 0.0f;
        return black3();
    }

    *pdf = 1.0f / (2.0f * M_PI * M_PI * sinTheta);
    return color;
}

__device__ inline uint32_t tea(uint32_t val0, uint32_t val1) {
    uint32_t v0 = val0;
    uint32_t v1 = val1;
    uint32_t s0 = 0;

    for (int n = 0; n < 4; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

extern "C" __global__ void shade(const optixw::ShadeKernelParams* params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params->renderBuffers.numActive) {
        return;
    }

    uint32_t rayIndex = params->renderBuffers.activeIndices[idx];
    optixw::RayState& ray = params->renderBuffers.rayPool[rayIndex];

    if (ray.stage == optixw::RayState::Terminated) {
        if (ray.radiance.x > 0.0f || ray.radiance.y > 0.0f || ray.radiance.z > 0.0f) {
            if (ray.pixelIndex < 262144) {
                params->renderBuffers.accumBuffer[ray.pixelIndex] =
                    params->renderBuffers.accumBuffer[ray.pixelIndex] + ray.radiance;
            }
        }
        return;
    }
    if (ray.stage != optixw::RayState::Shade) {
        return;
    }

    ray.radiance = make_float3(0.5f, 0.5f, 0.5f);
    ray.stage = optixw::RayState::Terminated;
}
