// 定义基本类型
typedef unsigned char uchar3;

// RGB颜色结构
struct RGB {
    float r, g, b;
    
    __host__ __device__ RGB() : r(0.0f), g(0.0f), b(0.0f) {}
    __host__ __device__ RGB(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
};

// Point3D结构
struct Point3D {
    float x, y, z;
    
    __host__ __device__ Point3D() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Vector3D结构
struct Vector3D {
    float x, y, z;
    
    __host__ __device__ Vector3D() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vector3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// 定义渲染参数结构
struct Params {
    RGB* image_buffer;
    int width;
    int height;
    int spp;
    int maxDepth;
    float cameraFovY;
    Point3D cameraPos;
    Vector3D cameraU;
    Vector3D cameraV;
    Vector3D cameraW;
};

// Ray Generation Data
struct RayGenData {
    // 相机参数
    Point3D origin;
    Vector3D u, v, w;  // 相机坐标系
};

// Miss Data
struct MissData {
    RGB backgroundColor;
};

// Hit Group Data
struct HitGroupData {
    RGB color;
    float emission;
};

// OptiX设备函数和变量
extern "C" {
    __constant__ Params params;
    __constant__ RayGenData rt_data;
}

// 光线生成程序
extern "C" __global__ void __raygen__rg_main()
{
    const uint32_t image_width = params.width;
    const uint32_t image_height = params.height;
    
    const uint32_t x = optixGetLaunchIndex().x;
    const uint32_t y = optixGetLaunchIndex().y;
    
    if(x >= image_width || y >= image_height) return;
    
    // 计算像素索引
    const uint32_t pixel_idx = y * image_width + x;
    
    // 简单的相机模型
    const float aspect_ratio = static_cast<float>(image_width) / static_cast<float>(image_height);
    const float half_h = tanf(params.cameraFovY * 0.5f);
    const float half_w = aspect_ratio * half_h;
    
    // 计算像素颜色 - 生成渐变效果
    RGB color;
    color.r = static_cast<float>(x) / static_cast<float>(image_width);
    color.g = static_cast<float>(y) / static_cast<float>(image_height);
    color.b = 0.5f;
    
    // 存储结果到输出缓冲区
    params.image_buffer[pixel_idx] = color;
}

// Miss程序 - 当光线未击中任何物体时调用
extern "C" __global__ void __miss__ms_main()
{
    const uint32_t x = optixGetLaunchIndex().x;
    const uint32_t y = optixGetLaunchIndex().y;
    const uint32_t image_width = params.width;
    
    const uint32_t pixel_idx = y * image_width + x;
    
    // 简单的背景色 - 蓝天
    params.image_buffer[pixel_idx] = RGB(0.1f, 0.2f, 0.8f);
}

// 最近击中程序 - 当光线击中最接近的物体时调用
extern "C" __global__ void __closesthit__ch_main()
{
    const uint32_t x = optixGetLaunchIndex().x;
    const uint32_t y = optixGetLaunchIndex().y;
    const uint32_t image_width = params.width;
    
    const uint32_t pixel_idx = y * image_width + x;
    
    // 获取交点数据
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    
    // 计算简单的着色
    params.image_buffer[pixel_idx] = data->color;
}

// 任意击中程序 - 当光线击中任何物体时调用（用于透明度等效果）
extern "C" __global__ void __anyhit__ah_main()
{
    // 对于简单光线追踪，我们不需要做任何特殊处理
    // 只有在需要透明度或alpha混合时才需要此程序
}