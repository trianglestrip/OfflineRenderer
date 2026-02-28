cmake_minimum_required(VERSION 3.26)

# Helper function to create test executable
function(add_optixw_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    
    # Include directories
    target_include_directories(${TEST_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
            ${OptiX_INSTALL_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(${TEST_NAME}
        PRIVATE
            libOptixW
    )
    
    # Copy test executable to bin directory
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}"
    )
    
    # Copy PTX files to test runtime directory
    add_custom_command(TARGET ${TEST_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/ptx"
            "$<TARGET_FILE_DIR:${TEST_NAME}>/ptx"
        COMMENT "Copying PTX files to ${TEST_NAME} runtime directory"
    )
endfunction()

# Test executables
add_optixw_test(optixw_test cornell_box_test.cpp)
add_optixw_test(cornell_box_var_test cornell_box_var_test.cpp)

# Add tests
add_test(NAME optixw_cornell_box COMMAND optixw_test)
add_test(NAME optixw_cornell_box_var COMMAND cornell_box_var_test)
