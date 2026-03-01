// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

namespace optixw {

// 3D vector type for positions/directions
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Camera parameters
struct Camera {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    float fovY;
    float aspect;
};

// Point light
struct PointLight {
    Vec3 position;
    Vec3 intensity;
};

// Area light
struct AreaLight {
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
    float width;
    float height;
    Vec3 emission;
    bool doubleSided;
};

} // namespace optixw
