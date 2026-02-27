# libVLRW — VLR Wavefront（Wavefront Renderer Core，独立于 libVLR）
cmake_minimum_required(VERSION 3.26)
set(TARGET_NAME "libVLRW")

set(VLRW_HEADERS
    "include/vlrw/types.h"
    "include/vlrw/vlrw.h"
    "src/optix_context.h"
    "src/scene.h"
    "src/renderer.h"
    "src/wavefront_scheduler.h"
)
set(VLRW_SOURCES
    "src/context.cpp"
    "src/optix_context.cpp"
    "src/scene.cpp"
    "src/renderer.cpp"
    "src/wavefront_scheduler.cpp"
)

set(VLRW_CUDA_KERNELS
    "gpu/ray_gen.cu"
    "gpu/wavefront_kernel.cu"
    "gpu/compact.cu"
    "gpu/gather_active.cu"
    "gpu/visualize_rays.cu"
)
set(VLRW_OPTIX_KERNELS
    "gpu/trace_rays.cu"
    "gpu/shadow_trace.cu"
)

set(VLRW_GPU_DEPS
    "include/vlrw/types.h"
    "include/vlrw/vlrw.h"
    "gpu/wavefront_types.cuh"
    "gpu/materials.cuh"
    "gpu/optix_common.cuh"
)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(CUDAToolkit 13.1 REQUIRED)
include(nvcuda_compile_ptx)
include(nvcuda_compile_optix)

nvcuda_compile_ptx(
    SOURCES ${VLRW_CUDA_KERNELS}
    DEPENDENCIES ${VLRW_GPU_DEPS}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES VLRW_CUDA_PTXES
    NVCC_OPTIONS
        "-Xcompiler=/wd4819,/Zc:__cplusplus"
        "$<$<CONFIG:Debug>:-D_DEBUG=1>"
        "$<$<CONFIG:Debug>:-G>"
        "--gpu-architecture=compute_75"
        "-std=c++20"
        "-cudart=shared"
        "--use_fast_math"
        "-I${OPTIX_INCLUDE_DIR}"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/gpu"
)

nvcuda_compile_optix(
    SOURCES ${VLRW_OPTIX_KERNELS}
    DEPENDENCIES ${VLRW_GPU_DEPS}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/optixir"
    GENERATED_FILES VLRW_OPTIX_IR
    NVCC_OPTIONS
        "-Xcompiler=/wd4819,/Zc:__cplusplus"
        "$<$<CONFIG:Debug>:-D_DEBUG=1>"
        "$<$<CONFIG:Debug>:-lineinfo>"
        "--gpu-architecture=compute_75"
        "-std=c++20"
        "-cudart=shared"
        "--use_fast_math"
        "--relocatable-device-code=true"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/gpu"
)

add_library(${TARGET_NAME} STATIC
    ${VLRW_SOURCES}
    ${VLRW_CUDA_KERNELS}
)
add_dependencies(${TARGET_NAME} ${NVCUDA_PTX_LAST_TARGET})

target_include_directories(${TARGET_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${OPTIX_INCLUDE_DIR}"
)
target_link_libraries(${TARGET_NAME} PUBLIC CUDA::cuda_driver)
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/MP>"
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>"
    )
endif()

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "libVLRW")

# Add test subdirectory
add_subdirectory(test)
