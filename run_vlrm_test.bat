@echo off
REM Build and run Cornell Box test for libVLRM

echo ========================================
echo Building Cornell Box Test (libVLRM)
echo ========================================

cmake --build build --config Release --target cornell_box_vlrm_test

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Test build failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Running Cornell Box Test...
echo ========================================

cd build\bin\Release
cornell_box_vlrm_test.exe

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Test execution failed!
    cd ..\..\..
    exit /b %ERRORLEVEL%
)

cd ..\..\..

echo.
echo ========================================
echo Test completed! Check cornell_box_vlrm.ppm
echo ========================================
