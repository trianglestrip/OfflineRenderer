# Test executables for libVLRW

find_package(CUDAToolkit 13.1 REQUIRED)

# Cornell Box test
add_executable(cornell_box_test cornell_box_test.cpp)

target_include_directories(cornell_box_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ext/stb
)

target_link_libraries(cornell_box_test PRIVATE
    libVLRW
    CUDA::cuda_driver
    ${OPTIX_LIBRARY}
)

target_compile_features(cornell_box_test PRIVATE cxx_std_20)

target_compile_options(cornell_box_test PRIVATE
    /MP
    /Zc:__cplusplus
    /utf-8
)

set_target_properties(cornell_box_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)
