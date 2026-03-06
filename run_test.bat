@echo off
REM Build and run libWR Cornell Box test
REM Usage: run_test.bat [Debug|Release]

setlocal

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release

echo ========================================
echo Building and Running libWR Test (%CONFIG%)
echo ========================================

REM Configure CMake if build directory doesn't exist
if not exist build (
    echo Configuring CMake...
    cmake -B build ^
        -G "Visual Studio 17 2022" ^
        -A x64 ^
        -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" ^
        -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"
    
    if errorlevel 1 (
        echo CMake configuration failed!
        exit /b 1
    )
)

REM Build test (this will also rebuild libWR if needed)
echo Building test program...
cmake --build build --config %CONFIG% --target wr_cornell_box_var_test

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo Test executable built successfully.

echo.
echo ========================================
echo Running test...
echo ========================================
echo.

REM Run test (must run from the same directory as PTX/CUBIN files)
cd build\bin\%CONFIG%
.\wr_cornell_box_var_test.exe
cd ..\..\..


if errorlevel 1 (
    echo Test execution failed!
    exit /b 1
)

echo.
echo ========================================
echo Test completed successfully!
echo Check gallery\wr_cornell.png for output
echo ========================================

endlocal
