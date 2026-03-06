#pragma once

#include <cstdint>
#include "common_internal.h"
#include "../include/VLRM/basic_types.h"

namespace vlrm {
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr Vector3D asVector3D(const float3 &v) {
        return Vector3D(v.x, v.y, v.z);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 asOptiXType(const Vector3D &v) {
        return make_float3(v.x, v.y, v.z);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr Normal3D asNormal3D(const float3 &v) {
        return Normal3D(v.x, v.y, v.z);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 asOptiXType(const Normal3D &n) {
        return make_float3(n.x, n.y, n.z);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr Point3D asPoint3D(const float3 &v) {
        return Point3D(v.x, v.y, v.z);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 asOptiXType(const Point3D &p) {
        return make_float3(p.x, p.y, p.z);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE int32_t floatToOrderedInt(float fVal) {
#if defined(VLRM_Host)
        int32_t iVal = *reinterpret_cast<int32_t*>(&fVal);
#else
        int32_t iVal = __float_as_int(fVal);
#endif
        return (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE float orderedIntToFloat(int32_t iVal) {
        int32_t orgVal = (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
#if defined(VLRM_Host)
        return *reinterpret_cast<float*>(&orgVal);
#else
        return __int_as_float(orgVal);
#endif
    }

    struct Point3DAsOrderedInt {
        int32_t x, y, z;

        CUDA_DEVICE_FUNCTION CUDA_INLINE Point3DAsOrderedInt() : x(0), y(0), z(0) {
        }
        CUDA_DEVICE_FUNCTION CUDA_INLINE Point3DAsOrderedInt(const Point3D &v) :
            x(floatToOrderedInt(v.x)), y(floatToOrderedInt(v.y)), z(floatToOrderedInt(v.z)) {
        }

        CUDA_DEVICE_FUNCTION CUDA_INLINE explicit operator Point3D() const {
            return Point3D(orderedIntToFloat(x), orderedIntToFloat(y), orderedIntToFloat(z));
        }
    };

#if defined(VLRM_Device) || defined(OPTIXU_Platform_CodeCompletion)
#if __CUDA_ARCH__ < 600
#   define atomicOr_block atomicOr
#   define atomicAnd_block atomicAnd
#   define atomicAdd_block atomicAdd
#   define atomicMin_block atomicMin
#   define atomicMax_block atomicMax
#endif

    CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMinPoint3D(Point3DAsOrderedInt* dst, const Point3DAsOrderedInt &v) {
        atomicMin(&dst->x, v.x);
        atomicMin(&dst->y, v.y);
        atomicMin(&dst->z, v.z);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMaxPoint3D(Point3DAsOrderedInt* dst, const Point3DAsOrderedInt &v) {
        atomicMax(&dst->x, v.x);
        atomicMax(&dst->y, v.y);
        atomicMax(&dst->z, v.z);
    }
#endif
}
