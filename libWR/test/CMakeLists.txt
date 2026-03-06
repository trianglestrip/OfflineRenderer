cmake_minimum_required(VERSION 3.26)

# Test helper sources
set(TEST_HELPERS
    "helpers/geometry.cpp"
    "helpers/image.cpp"
    "helpers/file.cpp"
)

function(add_wr_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE} ${TEST_HELPERS})

    target_include_directories(${TEST_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/glm
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/stb
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

    # Copy render_config.ini and res/ to test runtime directory
    add_custom_command(TARGET ${TEST_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/render_config.ini"
            "$<TARGET_FILE_DIR:${TEST_NAME}>/render_config.ini"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:${TEST_NAME}>/res"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/res/checkerboard_line.png"
            "$<TARGET_FILE_DIR:${TEST_NAME}>/res/checkerboard_line.png"
        COMMENT "Copying config and resources to ${TEST_NAME} runtime directory"
    )
endfunction()

add_wr_test(wr_cornell_box_var_test cornell_box_var_test.cpp)
