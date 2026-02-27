#pragma once

#include <cstdint>

namespace vlrw {

// RGB 工作流（无光谱）
struct RGB {
    float r, g, b;

    RGB() : r(0), g(0), b(0) {}
    RGB(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

    static RGB black() { return {0, 0, 0}; }
    static RGB white() { return {1, 1, 1}; }

    RGB& operator+=(const RGB& o) { r += o.r; g += o.g; b += o.b; return *this; }
    RGB operator*(float s) const { return {r * s, g * s, b * s}; }
    RGB operator*(const RGB& o) const { return {r * o.r, g * o.g, b * o.b}; }
    
    float& operator[](int i) { return (&r)[i]; }
    const float& operator[](int i) const { return (&r)[i]; }
};

struct RenderParams {
    uint32_t width  = 1024;
    uint32_t height = 1024;
    uint32_t maxDepth = 8;
    uint32_t spp = 1;
};

} // namespace vlrw
