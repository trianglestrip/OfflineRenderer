cmake_minimum_required(VERSION 3.26)

function(add_wr_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    target_include_directories(${TEST_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/libWR/include
        ${CMAKE_SOURCE_DIR}/libWR
        ${OptiX_INSTALL_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/libWR/stb
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${TEST_NAME}
        PRIVATE
            libWR
    )

    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}"
    )

    if(MSVC)
        target_link_options(${TEST_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
    endif()

    add_custom_command(TARGET ${TEST_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying kernel files to ${TEST_NAME} runtime directory"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/libWR/trace.ptx
            ${CMAKE_BINARY_DIR}/libWR/shade.cubin
            ${CMAKE_BINARY_DIR}/libWR/compact.cubin
            $<TARGET_FILE_DIR:${TEST_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/render_config.ini"
            "$<TARGET_FILE_DIR:${TEST_NAME}>/render_config.ini"
        COMMENT "Copying kernel files to ${TEST_NAME} runtime directory"
    )
endfunction()

add_wr_test(wr_cornell_box_var_test cornell_box_var_test.cpp)
