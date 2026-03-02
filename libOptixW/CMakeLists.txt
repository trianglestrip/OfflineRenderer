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

# Sources - CPU 层代码（模块化组织）
set(OPTIXW_SOURCES
    # 核心模块（保留旧代码以便逐步迁移）
    "src/context.cpp"
    "src/scene.cpp"
    "src/renderer.cpp"
    "src/renderer_pipeline.cpp"
    "src/renderer_loop.cpp"
    
    # Core 模块 - 核心基础设施
    "src/core/task_scheduler.cpp"
    "src/core/buffer_manager.cpp"
    
    # Rendering 模块 - 渲染管线
    "src/rendering/pipeline.cpp"
    "src/rendering/kernel_manager.cpp"
    "src/rendering/denoiser.cpp"
    
    # Scene 模块 - 场景数据管理
    "src/scene/geometry_manager.cpp"
    "src/scene/material_manager.cpp"
    "src/scene/texture_manager.cpp"
    "src/scene/light_manager.cpp"
    
    # Utils 模块 - 工具函数
    "src/utils/path_utils.cpp"
)

set(OPTIXW_HEADERS
    # 公开头文件
    "include/optixw/optixw.h"
    "include/optixw/types.h"
    "include/optixw/launch_params.h"
    "include/optixw/wavefront_kernel_params.h"
    
    # Core 模块 - 核心基础设施
    "include/optixw/core/math_types.h"
    "include/optixw/core/task_scheduler.h"
    "include/optixw/core/buffer_manager.h"
    
    # Rendering 模块 - 渲染管线
    "include/optixw/rendering/pipeline.h"
    "include/optixw/rendering/kernel_manager.h"
    "include/optixw/rendering/denoiser.h"
    
    # Scene 模块 - 场景数据管理
    "include/optixw/scene/geometry_manager.h"
    "include/optixw/scene/material_manager.h"
    "include/optixw/scene/texture_manager.h"
    "include/optixw/scene/light_manager.h"
    
    # Utils 模块 - 工具函数
    "src/utils/checks.h"
    "src/utils/path_utils.h"
)

# CUDA kernels (will compile to PTX)
set(OPTIXW_KERNELS
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
set(INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OptiX_INSTALL_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include/taskflow"
)

message(STATUS "Include directories: ${INCLUDE_DIRECTORIES}")

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${INCLUDE_DIRECTORIES}
)

# Find and link Taskflow
find_package(Taskflow QUIET)
if(NOT Taskflow_FOUND)
    # 如果找不到预安装的Taskflow，则手动指定路径
    message(STATUS "Taskflow not found via find_package, using ext/taskflow")
    # 添加Taskflow的头文件路径
    list(APPEND INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/taskflow")
    # 不在target_link_libraries中添加Taskflow，因为它是头文件库
else()
    message(STATUS "Found Taskflow: ${Taskflow_VERSION}")
    # 添加Taskflow的头文件路径
    list(APPEND INCLUDE_DIRECTORIES ${Taskflow_INCLUDE_DIRS})
endif()

# Link libraries
target_link_libraries(${TARGET_NAME}
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        windowscodecs
        ole32
        # Taskflow is header-only, so we don't need to link it
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
