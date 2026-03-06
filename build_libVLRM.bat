@echo off
REM Build script for libVLRM library

echo ========================================
echo Building libVLRM (Wavefront Renderer)
echo ========================================

REM Force rebuild libVLRM library
cmake --build build --config Release --target libvlrm --clean-first

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] libVLRM build failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo libVLRM build completed successfully!
echo ========================================
