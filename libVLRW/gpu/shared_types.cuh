#pragma once

#include <cuda_runtime.h>
#include <optix.h>

// libVLRW Shared Types (Host + Device)
// 参考 libVLR/shared/shared.h

namespace vlrw {
namespace shared {

// ============================================================================
// Math Types (参考 libVLR basic_types)
// ============================================================================

struct Point3D {
    float x, y, z;
    
    __host__ __device__ Point3D() : x(0), y(0), z(0) {}
    __host__ __device__ Point3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    __host__ __device__ explicit Point3D(float3 v) : x(v.x), y(v.y), z(v.z) {}
    
    __host__ __device__ float3 toFloat3() const { return make_float3(x, y, z); }
};

struct Vector3D {
    float x, y, z;
    
    __host__ __device__ Vector3D() : x(0), y(0), z(0) {}
    __host__ __device__ Vector3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    __host__ __device__ explicit Vector3D(float3 v) : x(v.x), y(v.y), z(v.z) {}
    
    __host__ __device__ float3 toFloat3() const { return make_float3(x, y, z); }
    __host__ __device__ float length() const { return sqrtf(x*x + y*y + z*z); }
    __host__ __device__ Vector3D normalize() const {
        float len = length();
        return (len > 0) ? Vector3D(x/len, y/len, z/len) : Vector3D(0, 0, 0);
    }
};

struct Normal3D {
    float x, y, z;
    
    __host__ __device__ Normal3D() : x(0), y(0), z(1) {}
    __host__ __device__ Normal3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    __host__ __device__ explicit Normal3D(float3 v) : x(v.x), y(v.y), z(v.z) {}
    
    __host__ __device__ float3 toFloat3() const { return make_float3(x, y, z); }
    __host__ __device__ Normal3D normalize() const {
        float len = sqrtf(x*x + y*y + z*z);
        return (len > 0) ? Normal3D(x/len, y/len, z/len) : Normal3D(0, 0, 1);
    }
};

// Vector operations
__host__ __device__ inline float dot(const Vector3D& a, const Vector3D& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float dot(const Normal3D& a, const Vector3D& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vector3D cross(const Vector3D& a, const Vector3D& b) {
    return Vector3D(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline Vector3D operator+(const Vector3D& a, const Vector3D& b) {
    return Vector3D(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Vector3D operator-(const Vector3D& a, const Vector3D& b) {
    return Vector3D(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline Vector3D operator*(const Vector3D& a, float s) {
    return Vector3D(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline Vector3D operator*(float s, const Vector3D& a) {
    return Vector3D(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline Vector3D operator/(const Vector3D& a, float s) {
    return Vector3D(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline Vector3D operator-(const Vector3D& a) {
    return Vector3D(-a.x, -a.y, -a.z);
}

// ============================================================================
// RGB Spectrum (参考 libVLR rgb_spectrum_types.h)
// ============================================================================

struct RGBSpectrum {
    float r, g, b;
    
    __host__ __device__ RGBSpectrum() : r(0), g(0), b(0) {}
    __host__ __device__ RGBSpectrum(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}
    __host__ __device__ explicit RGBSpectrum(float v) : r(v), g(v), b(v) {}
    
    __host__ __device__ static RGBSpectrum Zero() { return RGBSpectrum(0, 0, 0); }
    __host__ __device__ static RGBSpectrum One() { return RGBSpectrum(1, 1, 1); }
    
    __host__ __device__ float luminance() const {
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
    
    __host__ __device__ bool hasNonZero() const {
        return r > 0 || g > 0 || b > 0;
    }
    
    __host__ __device__ RGBSpectrum operator+(const RGBSpectrum& other) const {
        return RGBSpectrum(r + other.r, g + other.g, b + other.b);
    }
    
    __host__ __device__ RGBSpectrum operator*(const RGBSpectrum& other) const {
        return RGBSpectrum(r * other.r, g * other.g, b * other.b);
    }
    
    __host__ __device__ RGBSpectrum operator*(float s) const {
        return RGBSpectrum(r * s, g * s, b * s);
    }
    
    __host__ __device__ RGBSpectrum operator/(float s) const {
        return RGBSpectrum(r / s, g / s, b / s);
    }
    
    __host__ __device__ void operator+=(const RGBSpectrum& other) {
        r += other.r; g += other.g; b += other.b;
    }
    
    __host__ __device__ void operator*=(float s) {
        r *= s; g *= s; b *= s;
    }
};

__host__ __device__ inline RGBSpectrum operator*(float s, const RGBSpectrum& spec) {
    return RGBSpectrum(spec.r * s, spec.g * s, spec.b * s);
}

// ============================================================================
// Coordinate Frame (参考 libVLR basic_types)
// ============================================================================

struct ReferenceFrame {
    Vector3D x, y, z;  // tangent, bitangent, normal
    
    __host__ __device__ ReferenceFrame() {}
    __host__ __device__ ReferenceFrame(const Vector3D& _x, const Vector3D& _y, const Vector3D& _z)
        : x(_x), y(_y), z(_z) {}
    
    // 从法线构建坐标系
    __host__ __device__ static ReferenceFrame fromNormal(const Normal3D& n) {
        Vector3D normal(n.x, n.y, n.z);
        
        // 选择一个不平行于法线的向量
        Vector3D up = (fabsf(normal.z) < 0.999f) ? Vector3D(0, 0, 1) : Vector3D(1, 0, 0);
        
        Vector3D tangent = cross(up, normal).normalize();
        Vector3D bitangent = cross(normal, tangent);
        
        return ReferenceFrame(tangent, bitangent, normal);
    }
    
    // 世界坐标 -> 局部坐标
    __host__ __device__ Vector3D toLocal(const Vector3D& v) const {
        return Vector3D(dot(v, x), dot(v, y), dot(v, z));
    }
    
    // 局部坐标 -> 世界坐标
    __host__ __device__ Vector3D fromLocal(const Vector3D& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
};

// ============================================================================
// Surface Point (参考 libVLR basic_types)
// ============================================================================

struct SurfacePoint {
    Point3D position;
    Normal3D geometricNormal;
    Normal3D shadingNormal;
    ReferenceFrame shadingFrame;
    float u, v;  // texture coordinates
    
    __host__ __device__ SurfacePoint() : u(0), v(0) {}
    
    __host__ __device__ void setShadingNormal(const Normal3D& n) {
        shadingNormal = n.normalize();
        shadingFrame = ReferenceFrame::fromNormal(shadingNormal);
    }
    
    __host__ __device__ float calcCosTerm(const Vector3D& dir) const {
        return fabsf(dot(shadingNormal, dir));
    }
};

// ============================================================================
// Material Types (参考 libVLR materials.h)
// ============================================================================

enum class MaterialType : uint32_t {
    Matte = 0,                  // 漫反射
    SpecularReflection,         // 镜面反射
    SpecularScattering,         // 镜面散射（玻璃）
    MicrofacetReflection,       // 微表面反射（粗糙金属）
    MicrofacetScattering,       // 微表面散射（粗糙玻璃）
    UE4,                        // UE4 PBR
    DiffuseEmitter,             // 漫射发光体
    NumTypes
};

// 材质描述符（参考 libVLR SurfaceMaterialDescriptor）
struct MaterialDescriptor {
    MaterialType type;
    uint32_t padding[3];  // 对齐到 16 字节
    
    // 材质参数（最多 28 个 uint32_t，共 112 字节）
    union {
        struct {  // Matte
            RGBSpectrum albedo;
        } matte;
        
        struct {  // SpecularReflection
            RGBSpectrum eta;  // 折射率
            RGBSpectrum k;    // 消光系数
        } specularReflection;
        
        struct {  // SpecularScattering
            RGBSpectrum etaExt;  // 外部折射率
            RGBSpectrum etaInt;  // 内部折射率
        } specularScattering;
        
        struct {  // MicrofacetReflection
            RGBSpectrum eta;
            RGBSpectrum k;
            float roughness;
            float anisotropy;
            float rotation;
        } microfacetReflection;
        
        struct {  // MicrofacetScattering
            RGBSpectrum etaExt;
            RGBSpectrum etaInt;
            float roughness;
            float anisotropy;
            float rotation;
        } microfacetScattering;
        
        struct {  // UE4
            RGBSpectrum baseColor;
            float roughness;
            float metallic;
            float specular;  // 0.5 = 4% reflectance
        } ue4;
        
        struct {  // DiffuseEmitter
            RGBSpectrum emittance;
            float scale;
        } diffuseEmitter;
        
        uint32_t rawData[28];
    };
    
    __host__ __device__ MaterialDescriptor() : type(MaterialType::Matte) {
        for (int i = 0; i < 28; ++i) rawData[i] = 0;
    }
};

// ============================================================================
// Direction Type (参考 libVLR basic_types)
// ============================================================================

struct DirectionType {
    enum Value : uint32_t {
        WholeSphere     = 0x00,
        Reflection      = 0x01,
        Transmission    = 0x02,
        Emission        = 0x04,
        
        LowFreq         = 0x08,
        HighFreq        = 0x10,
        Delta0D         = 0x20,
        Delta1D         = 0x40,
        
        AllFreq         = LowFreq | HighFreq | Delta0D | Delta1D,
        NonDelta        = LowFreq | HighFreq,
        Delta           = Delta0D | Delta1D,
        
        All             = 0xFF
    };
    
    uint32_t value;
    
    __host__ __device__ DirectionType() : value(WholeSphere) {}
    __host__ __device__ DirectionType(Value v) : value(v) {}
    
    __host__ __device__ bool matches(DirectionType other) const {
        return (value & other.value) != 0;
    }
    
    __host__ __device__ bool isDelta() const {
        return (value & Delta) != 0;
    }
    
    __host__ __device__ bool hasNonDelta() const {
        return (value & NonDelta) != 0;
    }
};

// ============================================================================
// BSDF Query (参考 libVLR basic_types)
// ============================================================================

struct BSDFQuery {
    Vector3D dirOut;           // 出射方向（局部坐标）
    Normal3D geometricNormal;  // 几何法线（局部坐标）
    DirectionType dirTypeFilter;
    
    __host__ __device__ BSDFQuery(
        const Vector3D& _dirOut,
        const Normal3D& _geomNormal,
        DirectionType filter = DirectionType::All)
        : dirOut(_dirOut), geometricNormal(_geomNormal), dirTypeFilter(filter) {}
};

struct BSDFSample {
    float uComponent;  // 选择 BSDF 分量
    float uDir[2];     // 采样方向
    
    __host__ __device__ BSDFSample(float _uComp, float _u0, float _u1)
        : uComponent(_uComp) {
        uDir[0] = _u0;
        uDir[1] = _u1;
    }
};

struct BSDFQueryResult {
    Vector3D dirIn;       // 入射方向（局部坐标）
    float dirPDF;         // 方向 PDF
    DirectionType sampledType;
    
    __host__ __device__ BSDFQueryResult() : dirPDF(0), sampledType(DirectionType::WholeSphere) {}
};

// ============================================================================
// EDF Query (参考 libVLR basic_types)
// ============================================================================

struct EDFQuery {
    DirectionType dirTypeFilter;
    
    __host__ __device__ EDFQuery(DirectionType filter = DirectionType::All)
        : dirTypeFilter(filter) {}
};

// ============================================================================
// Light Types
// ============================================================================

struct PointLight {
    Point3D position;
    RGBSpectrum intensity;
};

struct AreaLight {
    Point3D position;
    Normal3D normal;
    Vector3D tangent;
    float width;
    float height;
    RGBSpectrum emission;
    uint32_t doubleSided;
};

// ============================================================================
// Ray Types (参考 libVLR shared.h)
// ============================================================================

struct RayType {
    enum Value {
        Primary = 0,
        Shadow,
        NumTypes
    };
};

} // namespace shared
} // namespace vlrw
