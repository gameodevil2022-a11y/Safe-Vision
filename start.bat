@echo off
echo ========================================
echo    SafeVision Server - Starting...
echo ========================================
echo.

:: Speed up CUDA startup by lazy-loading GPU kernels
set CUDA_MODULE_LOADING=LAZY

:: Activate venv and run
cd /d "%~dp0"
call .venv\Scripts\activate
python try.py

pause
