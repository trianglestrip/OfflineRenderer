@echo off
REM Build libWR library
REM Usage: build_libWR.bat [Debug|Release]

setlocal

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release

echo ========================================
echo Building libWR (%CONFIG%)
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

REM Build libWR (clean CUDA artifacts to force recompilation)
echo Building libWR library (cleaning CUDA artifacts first)...

REM Delete CUDA compiled artifacts to force kernel recompilation
if exist "build\libWR\Release\*.obj" del /Q "build\libWR\Release\*.obj" 2>nul
if exist "build\bin\Release\*.cubin" del /Q "build\bin\Release\*.cubin" 2>nul
if exist "build\bin\Release\*.ptx" del /Q "build\bin\Release\*.ptx" 2>nul

REM Build libWR (without --clean-first to preserve test executables)
cmake --build build --config %CONFIG% --target libWR

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo Output: build\bin\%CONFIG%\libWR.lib
echo ========================================

endlocal
