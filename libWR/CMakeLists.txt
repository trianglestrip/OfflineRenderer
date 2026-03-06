cmake_minimum_required(VERSION 3.26)

set(TARGET_NAME libWR)

# CUDA/OptiX paths (inherited from parent)
find_package(CUDAToolkit 13.1 REQUIRED)

if(NOT DEFINED OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0" CACHE PATH "OptiX SDK install directory")
endif()

if(NOT EXISTS "${OptiX_INSTALL_DIR}/include/optix.h")
    message(FATAL_ERROR "OptiX SDK not found at ${OptiX_INSTALL_DIR}. Please set OptiX_INSTALL_DIR.")
endif()

message(STATUS "Using OptiX SDK: ${OptiX_INSTALL_DIR}")

# CUDA kernels
set(WR_OPTIX_KERNELS
    "kernels/trace.cu"
    # "kernels/photon_trace.cu"  # TODO: Fix compilation errors
)

set(WR_CUDA_KERNELS
    "kernels/shade.cu"
    "kernels/compact.cu"
)

# Set kernel output directory to bin/Release (same as test executable)
set(WR_KERNEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/Release")

# Compile OptiX kernels to PTX
foreach(KERNEL_FILE ${WR_OPTIX_KERNELS})
    get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME_WE)
    set(PTX_FILE "${WR_KERNEL_OUTPUT_DIR}/${KERNEL_NAME}.ptx")
    
    add_custom_command(
        OUTPUT ${PTX_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WR_KERNEL_OUTPUT_DIR}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -ptx
            -o ${PTX_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -I${CMAKE_CURRENT_SOURCE_DIR}/src
            -I${OptiX_INSTALL_DIR}/include
            --gpu-architecture=compute_75
            --use_fast_math
            -O3
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}
        COMMENT "Compiling ${KERNEL_FILE} to PTX"
    )
    
    list(APPEND WR_KERNEL_FILES ${PTX_FILE})
endforeach()

# Compile CUDA kernels to CUBIN
foreach(KERNEL_FILE ${WR_CUDA_KERNELS})
    get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME_WE)
    set(CUBIN_FILE "${WR_KERNEL_OUTPUT_DIR}/${KERNEL_NAME}.cubin")
    
    add_custom_command(
        OUTPUT ${CUBIN_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WR_KERNEL_OUTPUT_DIR}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -cubin
            -o ${CUBIN_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -I${CMAKE_CURRENT_SOURCE_DIR}/src
            -I${OptiX_INSTALL_DIR}/include
            --gpu-architecture=sm_75
            --use_fast_math
            -O3
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}
        COMMENT "Compiling ${KERNEL_FILE} to CUBIN"
    )
    
    list(APPEND WR_KERNEL_FILES ${CUBIN_FILE})
endforeach()

add_custom_target(wr_kernels ALL DEPENDS ${WR_KERNEL_FILES})

# Core sources (pure C++, no GPU dependencies)
# Note: Utils (geometry, image, file) moved to test/helpers/
set(WR_CORE_SOURCES
    # Empty - all utils moved to test helpers
)

# Internal sources (GPU/OptiX dependent)
set(WR_INTERNAL_SOURCES
    "src/internal/context.cpp"
    "src/internal/scene.cpp"
    "src/internal/renderer.cpp"
    "src/internal/pipeline.cpp"
    "src/internal/denoiser.cpp"
    "src/internal/photon_mapper.cpp"
)

# Public headers (core API only, no utils)
set(WR_PUBLIC_HEADERS
    "include/wr/wr.h"
    "include/wr/types.h"
    "include/wr/context.h"
    "include/wr/scene.h"
    "include/wr/renderer.h"
)

# Internal headers
set(WR_INTERNAL_HEADERS
    "src/internal/gpu_types.h"
    "src/internal/cuda_utils.h"
    "src/internal/denoiser.h"
)

add_library(${TARGET_NAME} STATIC
    ${WR_CORE_SOURCES}
    ${WR_INTERNAL_SOURCES}
    ${WR_PUBLIC_HEADERS}
    ${WR_INTERNAL_HEADERS}
)

# Set output directory for static library to bin/Release
set_target_properties(${TARGET_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)

add_dependencies(${TARGET_NAME} wr_kernels)

# include directories
target_include_directories(${TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
        ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
        ${OptiX_INSTALL_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

# link libraries
target_link_libraries(${TARGET_NAME}
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        windowscodecs
        ole32
)

# C++20 requirement
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_20)

# Add test subdirectory
add_subdirectory(test)
