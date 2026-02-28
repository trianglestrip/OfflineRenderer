# FindOptiX80.cmake - Locate OptiX SDK 8.x/9.x (headers only, no DLL to link)
# Set OptiX_INSTALL_DIR or environment OptiX_INSTALL_DIR to SDK root (e.g. .../OptiX SDK 9.1.0)

set(OptiX_INSTALL_DIR "" CACHE PATH "Path to OptiX SDK root (8.x or 9.x, e.g. .../OptiX SDK 9.1.0)")
if(DEFINED ENV{OptiX_INSTALL_DIR} AND NOT OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}" CACHE PATH "Path to OptiX SDK (from env)" FORCE)
endif()

if(OptiX_INSTALL_DIR)
    find_path(OPTIX80_INCLUDE_DIR
        NAMES optix.h
        PATHS "${OptiX_INSTALL_DIR}/include"
        NO_DEFAULT_PATH
    )
else()
    find_path(OPTIX80_INCLUDE_DIR
        NAMES optix.h
    )
endif()

if(OPTIX80_INCLUDE_DIR)
    set(OptiX80_FOUND TRUE)
else()
    set(OptiX80_FOUND FALSE)
endif()

if(OptiX80_FOUND AND NOT OptiX80_FIND_QUIETLY)
    message(STATUS "Found OptiX: ${OPTIX80_INCLUDE_DIR}")
endif()

if(NOT OptiX80_FOUND AND OptiX80_FIND_REQUIRED)
    message(FATAL_ERROR "OptiX SDK not found. Set OptiX_INSTALL_DIR or env OptiX_INSTALL_DIR to the SDK root (e.g. .../OptiX SDK 9.1.0)")
endif()
