#include <optixw/types.h>
#include <optixw/launch_params.h>
#include <iostream>

int main() {
    std::cout << "sizeof(optixw::LaunchParams) = " << sizeof(optixw::LaunchParams) << std::endl;
    std::cout << "sizeof(optixw::RenderBufferDataRW) = " << sizeof(optixw::RenderBufferDataRW) << std::endl;
    std::cout << "sizeof(optixw::GeometryBuffers) = " << sizeof(optixw::GeometryBuffers) << std::endl;
    std::cout << "sizeof(optixw::CameraData) = " << sizeof(optixw::CameraData) << std::endl;
    std::cout << "sizeof(optixw::EnvironmentMappingData) = " << sizeof(optixw::EnvironmentMappingData) << std::endl;
    return 0;
}
