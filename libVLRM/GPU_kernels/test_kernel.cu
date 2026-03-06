// 简单的测试内核，用于生成PTX文件
extern "C" {
    struct Params {
        unsigned char* image_buffer;
        int width;
        int height;
        int spp;
        int maxDepth;
    };
    
    struct RGB {
        float r, g, b;
    };
}

// 光线生成程序
extern "C" __global__ void __raygen__rg_main()
{
    const int x = optixGetLaunchIndex().x;
    const int y = optixGetLaunchIndex().y;
    
    Params* params = (Params*)optixGetSbtDataPointer();
    
    if(x >= params->width || y >= params->height) return;
    
    // 简单的颜色渐变效果
    int idx = y * params->width + x;
    params->image_buffer[idx * 3 + 0] = (unsigned char)(255 * x / (float)params->width); // R
    params->image_buffer[idx * 3 + 1] = (unsigned char)(255 * y / (float)params->height); // G
    params->image_buffer[idx * 3 + 2] = 128; // B
}