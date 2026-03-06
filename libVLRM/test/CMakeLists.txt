set(TARGET_NAME "cornell_box_vlrm_test")

set(SOURCES
    "cornell_box_test.cpp"
)

add_executable(
    "${TARGET_NAME}"
    ${SOURCES}
)

target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
)

target_compile_options(
    "${TARGET_NAME}" PRIVATE
    "$<$<C_COMPILER_ID:MSVC>:/MP>"
    "$<$<CXX_COMPILER_ID:MSVC>:/MP>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>"
    "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)

target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/libVLRM/include"
)

target_link_libraries(
    "${TARGET_NAME}" PRIVATE
    libvlrm
)

target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)

set_target_properties(
    "${TARGET_NAME}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)
