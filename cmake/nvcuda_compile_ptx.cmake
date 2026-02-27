# nvcuda_compile_ptx.cmake - Compile .cu sources to .ptx with nvcc, output to TARGET_PATH.
# Sets GENERATED_FILES and NVCUDA_PTX_LAST_TARGET in parent scope.

function(nvcuda_compile_ptx)
    cmake_parse_arguments(ARG "" "TARGET_PATH;GENERATED_FILES" "SOURCES;DEPENDENCIES;NVCC_OPTIONS" ${ARGN})
    if(NOT ARG_SOURCES OR NOT ARG_TARGET_PATH OR NOT ARG_GENERATED_FILES)
        message(FATAL_ERROR "nvcuda_compile_ptx requires SOURCES, TARGET_PATH, GENERATED_FILES")
    endif()

    set(nvcc "${CMAKE_CUDA_COMPILER}")
    if(NOT nvcc)
        set(nvcc "$ENV{CUDA_PATH}/bin/nvcc.exe")
        if(NOT EXISTS "${nvcc}")
            set(nvcc "nvcc")
        endif()
    endif()

    set(ptx_outputs "")
    set(_all_depends ${ARG_DEPENDENCIES})

    foreach(cu_src IN LISTS ARG_SOURCES)
        get_filename_component(basename "${cu_src}" NAME_WE)
        set(ptx_out "${ARG_TARGET_PATH}/${basename}.ptx")
        list(APPEND ptx_outputs "${ptx_out}")

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
            OUTPUT "${ptx_out}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_TARGET_PATH}"
            COMMAND ${nvcc} -ptx ${inc_dirs} ${ARG_NVCC_OPTIONS}
                -o "${ptx_out}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${cu_src}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${cu_src}" ${ARG_DEPENDENCIES}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Compiling ${cu_src} to PTX"
        )
    endforeach()

    list(GET ARG_SOURCES 0 _first_cu)
    get_filename_component(_first_base "${_first_cu}" NAME_WE)
    string(REPLACE "/" "_" _tid "${ARG_TARGET_PATH}")
    string(REPLACE " " "_" _tid "${_tid}")
    set(_target_name "nvcuda_ptx_${_first_base}")
    add_custom_target(${_target_name} ALL DEPENDS ${ptx_outputs})

    set(${ARG_GENERATED_FILES} ${ptx_outputs} PARENT_SCOPE)
    set(NVCUDA_PTX_LAST_TARGET ${_target_name})
    set(NVCUDA_PTX_LAST_TARGET ${_target_name} PARENT_SCOPE)
endfunction()
