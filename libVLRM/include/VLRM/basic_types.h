#pragma once

#include "common.h"

namespace vlrm {
    template <typename RealType>
    struct Vector3DTemplate {
        RealType x, y, z;

#if defined(VLRM_Device)
        CUDA_DEVICE_FUNCTION constexpr Vector3DTemplate() { }
#else
        constexpr Vector3DTemplate() : x(0), y(0), z(0) { }
#endif
        CUDA_DEVICE_FUNCTION constexpr Vector3DTemplate(RealType v) : x(v), y(v), z(v) { }
        CUDA_DEVICE_FUNCTION constexpr Vector3DTemplate(RealType xx, RealType yy, RealType zz) : x(xx), y(yy), z(zz) { }

        CUDA_DEVICE_FUNCTION Vector3DTemplate operator+() const { return *this; }
        CUDA_DEVICE_FUNCTION Vector3DTemplate operator-() const { return Vector3DTemplate(-x, -y, -z); }

        CUDA_DEVICE_FUNCTION Vector3DTemplate operator+(const Vector3DTemplate &v) const { return Vector3DTemplate(x + v.x, y + v.y, z + v.z); }
        CUDA_DEVICE_FUNCTION Vector3DTemplate operator-(const Vector3DTemplate &v) const { return Vector3DTemplate(x - v.x, y - v.y, z - v.z); }
        CUDA_DEVICE_FUNCTION Vector3DTemplate operator*(const Vector3DTemplate &v) const { return Vector3DTemplate(x * v.x, y * v.y, z * v.z); }
        CUDA_DEVICE_FUNCTION Vector3DTemplate operator/(const Vector3DTemplate &v) const { return Vector3DTemplate(x / v.x, y / v.y, z / v.z); }
        CUDA_DEVICE_FUNCTION Vector3DTemplate operator*(RealType s) const { return Vector3DTemplate(x * s, y * s, z * s); }
        CUDA_DEVICE_FUNCTION Vector3DTemplate operator/(RealType s) const { RealType r = 1 / s; return Vector3DTemplate(x * r, y * r, z * r); }
        CUDA_DEVICE_FUNCTION friend Vector3DTemplate operator*(RealType s, const Vector3DTemplate &v) { return Vector3DTemplate(s * v.x, s * v.y, s * v.z); }

        CUDA_DEVICE_FUNCTION Vector3DTemplate &operator+=(const Vector3DTemplate &v) { x += v.x; y += v.y; z += v.z; return *this; }
        CUDA_DEVICE_FUNCTION Vector3DTemplate &operator-=(const Vector3DTemplate &v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
        CUDA_DEVICE_FUNCTION Vector3DTemplate &operator*=(RealType s) { x *= s; y *= s; z *= s; return *this; }
        CUDA_DEVICE_FUNCTION Vector3DTemplate &operator/=(RealType s) { RealType r = 1 / s; x *= r; y *= r; z *= r; return *this; }

        CUDA_DEVICE_FUNCTION bool operator==(const Vector3DTemplate &v) const { return x == v.x && y == v.y && z == v.z; }
        CUDA_DEVICE_FUNCTION bool operator!=(const Vector3DTemplate &v) const { return x != v.x || y != v.y || z != v.z; }

        CUDA_DEVICE_FUNCTION RealType &operator[](unsigned int index) {
            VLRMAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }
        CUDA_DEVICE_FUNCTION RealType operator[](unsigned int index) const {
            VLRMAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }

        CUDA_DEVICE_FUNCTION RealType length() const {
            return std::sqrt(x * x + y * y + z * z);
        }
        CUDA_DEVICE_FUNCTION RealType sqLength() const { return x * x + y * y + z * z; }
        CUDA_DEVICE_FUNCTION Vector3DTemplate& normalize() {
            RealType length = std::sqrt(x * x + y * y + z * z);
            return *this /= length;
        }

        CUDA_DEVICE_FUNCTION void makeCoordinateSystem(Vector3DTemplate<RealType>* vx, Vector3DTemplate<RealType>* vy) const {
            RealType sign = z >= 0 ? 1 : -1;
            const RealType a = -1 / (sign + z);
            const RealType b = x * y * a;
            *vx = Vector3DTemplate<RealType>(1 + sign * x * x * a, sign * b, -sign * x);
            *vy = Vector3DTemplate<RealType>(b, sign + y * y * a, -y);
        }

        CUDA_DEVICE_FUNCTION RealType maxValue() const { using std::fmax; return fmax(x, fmax(y, z)); }
        CUDA_DEVICE_FUNCTION RealType minValue() const { using std::fmin; return fmin(x, fmin(y, z)); }
        CUDA_DEVICE_FUNCTION bool hasNaN() const { using vlrm::isnan; return isnan(x) || isnan(y) || isnan(z); }
        CUDA_DEVICE_FUNCTION bool hasInf() const { using vlrm::isinf; return isinf(x) || isinf(y) || isinf(z); }
        CUDA_DEVICE_FUNCTION bool allFinite() const {
            return !hasNaN() && !hasInf();
        }

        CUDA_DEVICE_FUNCTION static constexpr Vector3DTemplate Zero() { return Vector3DTemplate(0); }
        CUDA_DEVICE_FUNCTION static constexpr Vector3DTemplate Ex() { return Vector3DTemplate(1, 0, 0); }
        CUDA_DEVICE_FUNCTION static constexpr Vector3DTemplate Ey() { return Vector3DTemplate(0, 1, 0); }
        CUDA_DEVICE_FUNCTION static constexpr Vector3DTemplate Ez() { return Vector3DTemplate(0, 0, 1); }
    };

    template <typename RealType>
    CUDA_DEVICE_FUNCTION Vector3DTemplate<RealType> normalize(const Vector3DTemplate<RealType> &v) {
        RealType l = v.length();
        return v / l;
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION RealType dot(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION Vector3DTemplate<RealType> cross(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        return Vector3DTemplate<RealType>(vec1.y * vec2.z - vec1.z * vec2.y,
                                          vec1.z * vec2.x - vec1.x * vec2.z,
                                          vec1.x * vec2.y - vec1.y * vec2.x);
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION RealType absDot(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        return std::fabs(vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z);
    }

    template <typename RealType>
    struct Normal3DTemplate {
        RealType x, y, z;

#if defined(VLRM_Device)
        CUDA_DEVICE_FUNCTION constexpr Normal3DTemplate() { }
#else
        constexpr Normal3DTemplate() : x(0), y(0), z(0) { }
#endif
        CUDA_DEVICE_FUNCTION constexpr Normal3DTemplate(RealType v) : x(v), y(v), z(v) { }
        CUDA_DEVICE_FUNCTION constexpr Normal3DTemplate(RealType xx, RealType yy, RealType zz) : x(xx), y(yy), z(zz) { }
        CUDA_DEVICE_FUNCTION constexpr Normal3DTemplate(const Vector3DTemplate<RealType> &v) : x(v.x), y(v.y), z(v.z) { }

        CUDA_DEVICE_FUNCTION Normal3DTemplate operator+() const { return *this; }
        CUDA_DEVICE_FUNCTION Normal3DTemplate operator-() const { return Normal3DTemplate(-x, -y, -z); }

        CUDA_DEVICE_FUNCTION Normal3DTemplate operator+(const Normal3DTemplate &n) const { return Normal3DTemplate(x + n.x, y + n.y, z + n.z); }
        CUDA_DEVICE_FUNCTION Normal3DTemplate operator-(const Normal3DTemplate &n) const { return Normal3DTemplate(x - n.x, y - n.y, z - n.z); }
        CUDA_DEVICE_FUNCTION Normal3DTemplate operator*(RealType s) const { return Normal3DTemplate(x * s, y * s, z * s); }
        CUDA_DEVICE_FUNCTION Normal3DTemplate operator/(RealType s) const { RealType r = 1 / s; return Normal3DTemplate(x * r, y * r, z * r); }
        CUDA_DEVICE_FUNCTION friend Normal3DTemplate operator*(RealType s, const Normal3DTemplate &n) { return Normal3DTemplate(s * n.x, s * n.y, s * n.z); }

        CUDA_DEVICE_FUNCTION Normal3DTemplate &operator+=(const Normal3DTemplate &n) { x += n.x; y += n.y; z += n.z; return *this; }
        CUDA_DEVICE_FUNCTION Normal3DTemplate &operator-=(const Normal3DTemplate &n) { x -= n.x; y -= n.y; z -= n.z; return *this; }
        CUDA_DEVICE_FUNCTION Normal3DTemplate &operator*=(RealType s) { x *= s; y *= s; z *= s; return *this; }
        CUDA_DEVICE_FUNCTION Normal3DTemplate &operator/=(RealType s) { RealType r = 1 / s; x *= r; y *= r; z *= r; return *this; }

        CUDA_DEVICE_FUNCTION RealType length() const {
            return std::sqrt(x * x + y * y + z * z);
        }
        CUDA_DEVICE_FUNCTION RealType sqLength() const { return x * x + y * y + z * z; }
        CUDA_DEVICE_FUNCTION Normal3DTemplate& normalize() {
            RealType length = std::sqrt(x * x + y * y + z * z);
            return *this /= length;
        }

        CUDA_DEVICE_FUNCTION explicit operator Vector3DTemplate<RealType>() const { return Vector3DTemplate<RealType>(x, y, z); }
    };

    template <typename RealType>
    struct Point3DTemplate {
        RealType x, y, z;

#if defined(VLRM_Device)
        CUDA_DEVICE_FUNCTION constexpr Point3DTemplate() { }
#else
        constexpr Point3DTemplate() : x(0), y(0), z(0) { }
#endif
        CUDA_DEVICE_FUNCTION constexpr Point3DTemplate(RealType v) : x(v), y(v), z(v) { }
        CUDA_DEVICE_FUNCTION constexpr Point3DTemplate(RealType xx, RealType yy, RealType zz) : x(xx), y(yy), z(zz) { }

        CUDA_DEVICE_FUNCTION Point3DTemplate operator+() const { return *this; }
        CUDA_DEVICE_FUNCTION Point3DTemplate operator-() const { return Point3DTemplate(-x, -y, -z); }

        CUDA_DEVICE_FUNCTION Point3DTemplate operator+(const Vector3DTemplate<RealType> &v) const { return Point3DTemplate(x + v.x, y + v.y, z + v.z); }
        CUDA_DEVICE_FUNCTION Point3DTemplate operator-(const Vector3DTemplate<RealType> &v) const { return Point3DTemplate(x - v.x, y - v.y, z - v.z); }
        CUDA_DEVICE_FUNCTION Vector3DTemplate<RealType> operator-(const Point3DTemplate &p) const { return Vector3DTemplate<RealType>(x - p.x, y - p.y, z - p.z); }
        CUDA_DEVICE_FUNCTION Point3DTemplate operator*(RealType s) const { return Point3DTemplate(x * s, y * s, z * s); }
        CUDA_DEVICE_FUNCTION Point3DTemplate operator/(RealType s) const { RealType r = 1 / s; return Point3DTemplate(x * r, y * r, z * r); }
        CUDA_DEVICE_FUNCTION friend Point3DTemplate operator*(RealType s, const Point3DTemplate &p) { return Point3DTemplate(s * p.x, s * p.y, s * p.z); }

        CUDA_DEVICE_FUNCTION Point3DTemplate &operator+=(const Vector3DTemplate<RealType> &v) { x += v.x; y += v.y; z += v.z; return *this; }
        CUDA_DEVICE_FUNCTION Point3DTemplate &operator-=(const Vector3DTemplate<RealType> &v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
        CUDA_DEVICE_FUNCTION Point3DTemplate &operator*=(RealType s) { x *= s; y *= s; z *= s; return *this; }
        CUDA_DEVICE_FUNCTION Point3DTemplate &operator/=(RealType s) { RealType r = 1 / s; x *= r; y *= r; z *= r; return *this; }

        CUDA_DEVICE_FUNCTION bool operator==(const Point3DTemplate &p) const { return x == p.x && y == p.y && z == p.z; }
        CUDA_DEVICE_FUNCTION bool operator!=(const Point3DTemplate &p) const { return x != p.x || y != p.y || z != p.z; }

        CUDA_DEVICE_FUNCTION RealType &operator[](unsigned int index) {
            VLRMAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }
        CUDA_DEVICE_FUNCTION RealType operator[](unsigned int index) const {
            VLRMAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }

        CUDA_DEVICE_FUNCTION RealType maxValue() const { using std::fmax; return fmax(x, fmax(y, z)); }
        CUDA_DEVICE_FUNCTION RealType minValue() const { using std::fmin; return fmin(x, fmin(y, z)); }
        CUDA_DEVICE_FUNCTION bool hasNaN() const { using vlrm::isnan; return isnan(x) || isnan(y) || isnan(z); }
        CUDA_DEVICE_FUNCTION bool hasInf() const { using vlrm::isinf; return isinf(x) || isinf(y) || isinf(z); }
        CUDA_DEVICE_FUNCTION bool allFinite() const {
            return !hasNaN() && !hasInf();
        }
    };

    template <typename RealType>
    struct RGBTemplate {
        RealType r, g, b;

#if defined(VLRM_Device)
        CUDA_DEVICE_FUNCTION constexpr RGBTemplate() { }
#else
        constexpr RGBTemplate() : r(0), g(0), b(0) { }
#endif
        CUDA_DEVICE_FUNCTION constexpr RGBTemplate(RealType v) : r(v), g(v), b(v) { }
        CUDA_DEVICE_FUNCTION constexpr RGBTemplate(RealType rr, RealType gg, RealType bb) : r(rr), g(gg), b(bb) { }

        CUDA_DEVICE_FUNCTION RGBTemplate operator+() const { return *this; }
        CUDA_DEVICE_FUNCTION RGBTemplate operator-() const { return RGBTemplate(-r, -g, -b); }

        CUDA_DEVICE_FUNCTION RGBTemplate operator+(const RGBTemplate &c) const { return RGBTemplate(r + c.r, g + c.g, b + c.b); }
        CUDA_DEVICE_FUNCTION RGBTemplate operator-(const RGBTemplate &c) const { return RGBTemplate(r - c.r, g - c.g, b - c.b); }
        CUDA_DEVICE_FUNCTION RGBTemplate operator*(const RGBTemplate &c) const { return RGBTemplate(r * c.r, g * c.g, b * c.b); }
        CUDA_DEVICE_FUNCTION RGBTemplate operator/(const RGBTemplate &c) const { return RGBTemplate(r / c.r, g / c.g, b / c.b); }
        CUDA_DEVICE_FUNCTION RGBTemplate operator*(RealType s) const { return RGBTemplate(r * s, g * s, b * s); }
        CUDA_DEVICE_FUNCTION RGBTemplate operator/(RealType s) const { RealType rs = 1 / s; return RGBTemplate(r * rs, g * rs, b * rs); }
        CUDA_DEVICE_FUNCTION friend RGBTemplate operator*(RealType s, const RGBTemplate &c) { return RGBTemplate(s * c.r, s * c.g, s * c.b); }

        CUDA_DEVICE_FUNCTION RGBTemplate &operator+=(const RGBTemplate &c) { r += c.r; g += c.g; b += c.b; return *this; }
        CUDA_DEVICE_FUNCTION RGBTemplate &operator-=(const RGBTemplate &c) { r -= c.r; g -= c.g; b -= c.b; return *this; }
        CUDA_DEVICE_FUNCTION RGBTemplate &operator*=(const RGBTemplate &c) { r *= c.r; g *= c.g; b *= c.b; return *this; }
        CUDA_DEVICE_FUNCTION RGBTemplate &operator/=(const RGBTemplate &c) { r /= c.r; g /= c.g; b /= c.b; return *this; }
        CUDA_DEVICE_FUNCTION RGBTemplate &operator*=(RealType s) { r *= s; g *= s; b *= s; return *this; }
        CUDA_DEVICE_FUNCTION RGBTemplate &operator/=(RealType s) { RealType rs = 1 / s; r *= rs; g *= rs; b *= rs; return *this; }

        CUDA_DEVICE_FUNCTION RealType luminance() const {
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }

        CUDA_DEVICE_FUNCTION RealType maxValue() const { using std::fmax; return fmax(r, fmax(g, b)); }
        CUDA_DEVICE_FUNCTION RealType minValue() const { using std::fmin; return fmin(r, fmin(g, b)); }
        CUDA_DEVICE_FUNCTION bool hasNaN() const { using vlrm::isnan; return isnan(r) || isnan(g) || isnan(b); }
        CUDA_DEVICE_FUNCTION bool hasInf() const { using vlrm::isinf; return isinf(r) || isinf(g) || isinf(b); }
        CUDA_DEVICE_FUNCTION bool allFinite() const {
            return !hasNaN() && !hasInf();
        }

        CUDA_DEVICE_FUNCTION static constexpr RGBTemplate Zero() { return RGBTemplate(0); }
        CUDA_DEVICE_FUNCTION static constexpr RGBTemplate One() { return RGBTemplate(1); }
    };

    using Vector3D = Vector3DTemplate<float>;
    using Normal3D = Normal3DTemplate<float>;
    using Point3D = Point3DTemplate<float>;
    using RGB = RGBTemplate<float>;
}
