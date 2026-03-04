#include <optixw/optixw.h>
#include <optixw/launch_params.h>
#include <optixw/types.h>
#include <iostream>

using namespace optixw;

int main() {
    std::cout << "sizeof(LaunchParams) = " << sizeof(LaunchParams) << std::endl;
    std::cout << "sizeof(GeometryBuffers) = " << sizeof(GeometryBuffers) << std::endl;
    std::cout << "sizeof(RenderBufferDataRW) = " << sizeof(RenderBufferDataRW) << std::endl;
    std::cout << "sizeof(CameraData) = " << sizeof(CameraData) << std::endl;
    std::cout << "sizeof(EnvironmentMappingData) = " << sizeof(EnvironmentMappingData) << std::endl;
    std::cout << "sizeof(RayState) = " << sizeof(RayState) << std::endl;
    std::cout << "sizeof(HitInfo) = " << sizeof(HitInfo) << std::endl;
    std::cout << "sizeof(MaterialData) = " << sizeof(MaterialData) << std::endl;
    return 0;
}
