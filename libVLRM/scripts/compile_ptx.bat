@echo off
setlocal enabledelayedexpansion

REM Set NVCC path (assuming CUDA is installed in default location)
set NVCC_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe"

REM If the above path doesn't exist, try common paths
if not exist %NVCC_PATH% (
    where nvcc >nul 2>nul
    if !errorlevel! equ 0 (
        for /f "delims=" %%i in ('where nvcc 2^>nul') do set NVCC_PATH="%%i"
    )
)

if not exist %NVCC_PATH% (
    echo Error: Could not find nvcc compiler. Please ensure CUDA is installed.
    exit /b 1
)

echo Using NVCC compiler: %NVCC_PATH%

REM Compile CUDA kernel to PTX
%NVCC_PATH% -ptx -arch=sm_50 ^
    -I"d:\gitProject\OfflineRenderer\libVLRM\include\VLRM" ^
    -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include" ^
    -I"C:\Program Files\NVIDIA Corporation\OptiX SDK 7.4.0\include" ^
    "d:\gitProject\OfflineRenderer\libVLRM\GPU_kernels\ray_tracing_kernel.cu" ^
    -o "d:\gitProject\OfflineRenderer\libVLRM\GPU_kernels\ray_tracing.ptx"

if %ERRORLEVEL% EQU 0 (
    echo Successfully compiled ray_tracing_kernel.cu to PTX format.
) else (
    echo Failed to compile ray_tracing_kernel.cu to PTX format.
    exit /b 1
)