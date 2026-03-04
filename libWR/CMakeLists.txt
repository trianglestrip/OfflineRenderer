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
)

set(WR_CUDA_KERNELS
    "kernels/shade.cu"
    "kernels/compact.cu"
)

# Compile OptiX kernels to PTX
foreach(KERNEL_FILE ${WR_OPTIX_KERNELS})
    get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME_WE)
    set(PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.ptx")
    
    add_custom_command(
        OUTPUT ${PTX_FILE}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -ptx
            -o ${PTX_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
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
    set(CUBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.cubin")
    
    add_custom_command(
        OUTPUT ${CUBIN_FILE}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -cubin
            -o ${CUBIN_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
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

# Minimal sources for libWR
set(WR_SOURCES
    "src/context.cpp"
    "src/scene.cpp"
    "src/renderer.cpp"
    "src/pipeline.cpp"
    "utils/image_utils.cpp"
    "utils/math_utils.cpp"
    "utils/file_utils.cpp"
)

# Public headers
set(WR_HEADERS
    "include/wr/wr.h"
    "include/wr/types.h"
)

add_library(${TARGET_NAME} STATIC
    ${WR_SOURCES}
    ${WR_HEADERS}
)

add_dependencies(${TARGET_NAME} wr_kernels)

# include directories
target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
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

# Copy PTX/CUBIN files to runtime directory
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying kernel files to runtime directory"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${WR_KERNEL_FILES}
        $<TARGET_FILE_DIR:${TARGET_NAME}>
    COMMENT "Copying kernel files"
)

# Add test subdirectory
add_subdirectory(test)
