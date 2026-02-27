# copy_files.cmake - Copy SOURCES to TARGET_PATH at build time and set GENERATED_FILES in parent scope.
# Optionally set COPY_FILES_LAST_TARGET so caller can add_dependencies(target ${COPY_FILES_LAST_TARGET}).

function(copy_files)
    cmake_parse_arguments(ARG "" "TARGET_PATH;GENERATED_FILES" "SOURCES" ${ARGN})
    if(NOT ARG_SOURCES OR NOT ARG_TARGET_PATH OR NOT ARG_GENERATED_FILES)
        message(FATAL_ERROR "copy_files requires SOURCES, TARGET_PATH, GENERATED_FILES")
    endif()

    set(outputs "")
    foreach(src IN LISTS ARG_SOURCES)
        get_filename_component(fname "${src}" NAME)
        set(dst "${ARG_TARGET_PATH}/${fname}")
        list(APPEND outputs "${dst}")
        add_custom_command(
            OUTPUT "${dst}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_TARGET_PATH}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${src}" "${dst}"
            DEPENDS "${src}"
            COMMENT "Copy ${fname} to ${ARG_TARGET_PATH}"
        )
    endforeach()

    # 清理路径中的特殊字符（包括生成器表达式）
    string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" _tid "${ARG_TARGET_PATH}")
    add_custom_target(copy_files_${_tid} ALL DEPENDS ${outputs})

    set(${ARG_GENERATED_FILES} ${outputs} PARENT_SCOPE)
    set(COPY_FILES_LAST_TARGET copy_files_${_tid})
    set(COPY_FILES_LAST_TARGET copy_files_${_tid} PARENT_SCOPE)
endfunction()
