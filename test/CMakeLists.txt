cmake_minimum_required(VERSION 3.26)

function(add_optixw_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    target_include_directories(${TEST_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/libOptixW/include
            ${OptiX_INSTALL_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/ext/stb
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${TEST_NAME}
        PRIVATE
            libOptixW
    )

    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}"
    )

    add_custom_command(TARGET ${TEST_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/ptx"
            "$<TARGET_FILE_DIR:${TEST_NAME}>/ptx"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/cubin"
            "$<TARGET_FILE_DIR:${TEST_NAME}>/cubin"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/render_config.ini"
            "$<TARGET_FILE_DIR:${TEST_NAME}>/render_config.ini"
        COMMENT "Copying PTX/CUBIN files to ${TEST_NAME} runtime directory"
    )
endfunction()

add_optixw_test(optixw_test cornell_box_test.cpp)
add_optixw_test(cornell_box_var_test cornell_box_var_test.cpp)

add_test(NAME optixw_cornell_box COMMAND optixw_test)
add_test(NAME optixw_cornell_box_var COMMAND cornell_box_var_test)
