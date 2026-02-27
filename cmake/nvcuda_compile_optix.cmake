# nvcuda_compile_optix.cmake - Compile OptiX .cu sources to OptiX IR (.optixir) with nvcc
# Sets GENERATED_FILES and NVCUDA_OPTIX_LAST_TARGET in parent scope.

function(nvcuda_compile_optix)
    cmake_parse_arguments(ARG "" "TARGET_PATH;GENERATED_FILES" "SOURCES;DEPENDENCIES;NVCC_OPTIONS" ${ARGN})
    if(NOT ARG_SOURCES OR NOT ARG_TARGET_PATH OR NOT ARG_GENERATED_FILES)
        message(FATAL_ERROR "nvcuda_compile_optix requires SOURCES, TARGET_PATH, GENERATED_FILES")
    endif()

    set(nvcc "${CMAKE_CUDA_COMPILER}")
    if(NOT nvcc)
        set(nvcc "$ENV{CUDA_PATH}/bin/nvcc.exe")
        if(NOT EXISTS "${nvcc}")
            set(nvcc "nvcc")
        endif()
    endif()

    set(optixir_outputs "")
    set(_all_depends ${ARG_DEPENDENCIES})

    foreach(cu_src IN LISTS ARG_SOURCES)
        get_filename_component(basename "${cu_src}" NAME_WE)
        set(optixir_out "${ARG_TARGET_PATH}/${basename}.optixir")
        list(APPEND optixir_outputs "${optixir_out}")

        set(inc_dirs "")
        if(OPTIX_INCLUDE_DIR)
            list(APPEND inc_dirs "-I${OPTIX_INCLUDE_DIR}")
        endif()
        if(CUDAToolkit_INCLUDE_DIRS)
            foreach(d IN LISTS CUDAToolkit_INCLUDE_DIRS)
                list(APPEND inc_dirs "-I${d}")
            endforeach()
        endif()

        add_custom_command(
            OUTPUT "${optixir_out}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_TARGET_PATH}"
            COMMAND ${nvcc} --optix-ir ${inc_dirs} ${ARG_NVCC_OPTIONS}
                -o "${optixir_out}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${cu_src}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${cu_src}" ${ARG_DEPENDENCIES}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Compiling ${cu_src} to OptiX IR"
        )
    endforeach()

    list(GET ARG_SOURCES 0 _first_cu)
    get_filename_component(_first_base "${_first_cu}" NAME_WE)
    string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" _tid "${ARG_TARGET_PATH}")
    set(_target_name "nvcuda_optix_${_first_base}")
    add_custom_target(${_target_name} ALL DEPENDS ${optixir_outputs})

    set(${ARG_GENERATED_FILES} ${optixir_outputs} PARENT_SCOPE)
    set(NVCUDA_OPTIX_LAST_TARGET ${_target_name})
    set(NVCUDA_OPTIX_LAST_TARGET ${_target_name} PARENT_SCOPE)
endfunction()
