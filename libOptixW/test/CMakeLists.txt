cmake_minimum_required(VERSION 3.26)

# Test executable
add_executable(optixw_test cornell_box_test.cpp)

# Include directories
target_include_directories(optixw_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${OptiX_INSTALL_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(optixw_test
    PRIVATE
        libOptixW
)

# Copy test executable to bin directory
set_target_properties(optixw_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}"
)

# Copy PTX files to test runtime directory
add_custom_command(TARGET optixw_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/ptx"
        "$<TARGET_FILE_DIR:optixw_test>/ptx"
    COMMENT "Copying PTX files to test runtime directory"
)

# Add test
add_test(NAME optixw_cornell_box COMMAND optixw_test)
