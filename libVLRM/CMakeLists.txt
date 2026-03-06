set(TARGET_NAME "libvlrm")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

set(
    CUDA_KERNELS
    "GPU_kernels/ray_gen.cu"
    "GPU_kernels/wavefront_kernel.cu"
    "GPU_kernels/compact.cu"
)

set(
    OPTIX_KERNELS
    "GPU_kernels/trace_rays.cu"
    "GPU_kernels/shadow_trace.cu"
)

set(
    GPU_KERNEL_DEPENDENCIES
    "utils/cuda_util.h"
    "utils/optix_util.h"
    "utils/optixu_on_cudau.h"
    "include/VLRM/common.h"
    "include/VLRM/basic_types.h"
    "shared/common_internal.h"
    "shared/basic_types_internal.h"
    "shared/wavefront_types.h"
)

nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819 /Zc:__cplusplus\""
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-G>"
    "--gpu-architecture=compute_75"
    "-std=c++20"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
    "-I\"include/VLRM\""
)

nvcuda_compile_ptx(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819 /Zc:__cplusplus\""
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-lineinfo>"
    "--gpu-architecture=compute_75"
    "-std=c++20"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
    "-I\"include/VLRM\""
)

source_group(
    "Host" REGULAR_EXPRESSION 
    ".*\.(h|c|hpp|cpp)")
source_group(
    "include" REGULAR_EXPRESSION 
    "include/VLRM/.*\.(h|c|hpp|cpp)")
source_group(
    "Shared" REGULAR_EXPRESSION 
    "shared/.*\.(h|c|hpp|cpp)")
source_group(
    "GPU Kernels" REGULAR_EXPRESSION 
    "GPU_kernels/.*\.(cuh|cu)")
source_group(
    "Utilities" REGULAR_EXPRESSION 
    "utils/.*\.(h|c|hpp|cpp)")

add_library(
    "${TARGET_NAME}" SHARED
    ${SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
)
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
    "VLRM_API_EXPORTS"
)
target_compile_options(
    "${TARGET_NAME}" PRIVATE
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/Zc:__cplusplus>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/utf-8>"
)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "include/VLRM"
    "${OPTIX_INCLUDE_DIR}"
)
target_link_libraries(
    "${TARGET_NAME}" PRIVATE
    CUDA::cuda_driver
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)

add_subdirectory(test)
