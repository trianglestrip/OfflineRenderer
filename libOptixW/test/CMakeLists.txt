cmake_minimum_required(VERSION 3.26)

# Test executable
add_executable(optixw_test cornell_box_test.cpp)

target_link_libraries(optixw_test
    PRIVATE
        libOptixW
)

# Copy test executable to bin directory
set_target_properties(optixw_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}"
)

# Add test
add_test(NAME optixw_cornell_box COMMAND optixw_test)
