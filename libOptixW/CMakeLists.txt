cmake_minimum_required(VERSION 3.26)

set(TARGET_NAME libOptixW)

# CUDA/OptiX paths (from parent CMakeLists.txt or rules)
find_package(CUDAToolkit 13.1 REQUIRED)

# OptiX SDK path
if(NOT DEFINED OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0" CACHE PATH "OptiX SDK install directory")
endif()

if(NOT EXISTS "${OptiX_INSTALL_DIR}/include/optix.h")
    message(FATAL_ERROR "OptiX SDK not found at ${OptiX_INSTALL_DIR}. Please set OptiX_INSTALL_DIR.")
endif()

message(STATUS "Using OptiX SDK: ${OptiX_INSTALL_DIR}")

# Sources
set(OPTIXW_SOURCES
    "src/context.cpp"
    "src/scene.cpp"
    "src/renderer.cpp"
    "src/renderer_pipeline.cpp"
    "src/renderer_loop.cpp"
    "src/scheduler.cpp"
)

set(OPTIXW_HEADERS
    "include/optixw/optixw.h"
    "include/optixw/types.h"
    "include/optixw/launch_params.h"
    "include/optixw/wavefront_kernel_params.h"
)

# CUDA kernels (will compile to PTX)
set(OPTIXW_KERNELS
    "kernels/ray_gen.cu"
    "kernels/trace.cu"
    "kernels/shade.cu"
    "kernels/compact.cu"
)

# CUDA kernels that are launched via CUDA Driver API.
# Build these to cubin to avoid PTX JIT/toolchain compatibility issues.
set(OPTIXW_DRIVER_KERNELS
    "kernels/shade.cu"
    "kernels/compact.cu"
    "kernels/scale.cu"
    "kernels/denoise_merge.cu"
)

# Create static library
add_library(${TARGET_NAME} STATIC
    ${OPTIXW_SOURCES}
    ${OPTIXW_HEADERS}
)

# Include directories
target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OptiX_INSTALL_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${TARGET_NAME}
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        windowscodecs
        ole32
)

# C++20 standard
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_20)

# Compile CUDA kernels to PTX
foreach(KERNEL ${OPTIXW_KERNELS})
    get_filename_component(KERNEL_NAME ${KERNEL} NAME_WE)
    set(PTX_FILE "${CMAKE_BINARY_DIR}/ptx/${KERNEL_NAME}.ptx")
    
    add_custom_command(
        OUTPUT ${PTX_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/ptx"
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -ptx
            -o ${PTX_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -I${OptiX_INSTALL_DIR}/include
            --gpu-architecture=compute_75
            --use_fast_math
            -std=c++20
            -Xcompiler=/wd4819
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
        COMMENT "Compiling ${KERNEL} to PTX"
    )
    
    list(APPEND PTX_FILES ${PTX_FILE})
endforeach()

# Add PTX files as dependency
add_custom_target(${TARGET_NAME}_ptx ALL DEPENDS ${PTX_FILES})
add_dependencies(${TARGET_NAME} ${TARGET_NAME}_ptx)

# Compile CUDA driver kernels to cubin
foreach(KERNEL ${OPTIXW_DRIVER_KERNELS})
    get_filename_component(KERNEL_NAME ${KERNEL} NAME_WE)
    set(CUBIN_FILE "${CMAKE_BINARY_DIR}/cubin/${KERNEL_NAME}.cubin")

    add_custom_command(
        OUTPUT ${CUBIN_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/cubin"
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -cubin
            -o ${CUBIN_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -I${OptiX_INSTALL_DIR}/include
            --gpu-architecture=sm_75
            --use_fast_math
            -std=c++20
            -Xcompiler=/wd4819
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
        COMMENT "Compiling ${KERNEL} to cubin"
    )

    list(APPEND CUBIN_FILES ${CUBIN_FILE})
endforeach()

add_custom_target(${TARGET_NAME}_cubin ALL DEPENDS ${CUBIN_FILES})
add_dependencies(${TARGET_NAME} ${TARGET_NAME}_cubin)

# Copy PTX files to runtime directory
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/ptx"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/ptx"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/ptx"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/cubin"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/cubin"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/cubin"
    COMMENT "Copying PTX/CUBIN files to runtime directory"
)
