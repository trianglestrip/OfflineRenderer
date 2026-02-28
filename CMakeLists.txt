# Current CMakeLists has been tested only on Windows/MSVC.
# Manual copying executables and other files are required for other platforms.

cmake_minimum_required(VERSION 3.26)
project(
    VLR_cmake
    VERSION 1.0
    DESCRIPTION "GPU Monte Carlo Ray Tracing Renderer"
    LANGUAGES C CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_LIBRARY_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(
    CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH})
# CUDA 13.1 已安装（12.5+ 推荐用于 OptiX 8.0）
find_package(CUDAToolkit 13.1 REQUIRED)
find_package(OptiX80)
include("copy_files")
include("nvcuda_compile_ptx")

if (OptiX80_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX80_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK not found. Set OptiX_INSTALL_DIR to SDK root (8.x or 9.x).")
endif()



# ----------------------------------------------------------------
# JP: CMakeLists.txtのあるサブディレクトリ内のターゲット全てにフォルダーを設定する。
# https://stackoverflow.com/questions/45092198/cmake-how-do-i-change-properties-on-subdirectory-project-targets

# _dir以下の(CMakeが有効な)サブディレクトリにあるターゲットを再帰的に取得する。
function(get_all_targets _result _dir)
    get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        get_all_targets(${_result} "${_subdir}")
    endforeach()
    get_property(_sub_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    set(${_result} ${${_result}} ${_sub_targets} PARENT_SCOPE)
endfunction()

# _dir以下の(CMakeが有効な)サブディレクトリにあるターゲットにフォルダーを設定する。
function(set_project_group _folder_name _dir)
    get_all_targets(_targets "${_dir}")
    foreach(_target IN LISTS _targets)
        # message(${_target})
        set_target_properties(
            ${_target}
            PROPERTIES FOLDER "${_folder_name}"
        )
    endforeach()
endfunction()

# END: JP: CMakeLists.txtのあるサブディレクトリ内のターゲット全てにフォルダーを設定する。
# ----------------------------------------------------------------



# Taskflow (header-only library for task-parallel programming)
# Currently not used by libOptixW
# option(TF_BUILD_EXAMPLES "Build Taskflow examples" OFF)
# option(TF_BUILD_TESTS "Build Taskflow tests" OFF)
# option(TF_BUILD_BENCHMARKS "Build Taskflow benchmarks" OFF)
# add_subdirectory(ext/taskflow)
# set_project_group("External" "ext/taskflow")



# libOptixW: Wavefront Path Tracer wrapper for libVLR
add_subdirectory(libOptixW)
set_project_group("libOptixW" "libOptixW")
