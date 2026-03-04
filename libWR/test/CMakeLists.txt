cmake_minimum_required(VERSION 3.26)

function(add_wr_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    target_include_directories(${TEST_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/utils
        ${OptiX_INSTALL_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/stb
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${TEST_NAME}
        PRIVATE
            libWR
    )

    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    if(MSVC)
        target_link_options(${TEST_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
    endif()

    # 将 render_config.ini 复制到测试运行时目录
    # 内核文件 (ptx/cubin) 直接构建到 bin/Release
    add_custom_command(TARGET ${TEST_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/render_config.ini"
            "$<TARGET_FILE_DIR:${TEST_NAME}>/render_config.ini"
        COMMENT "Copying render_config.ini to ${TEST_NAME} runtime directory"
    )
endfunction()

add_wr_test(wr_cornell_box_var_test cornell_box_var_test.cpp)
