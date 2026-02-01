@echo off
REM Windows Build Script for Magic Body
REM Run this on a Windows PC with Python installed

echo =========================================
echo   Building Magic Body for Windows
echo =========================================

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install opencv-python mediapipe numpy pyinstaller

REM Check model files
if not exist "pose_landmarker.task" (
    echo ERROR: pose_landmarker.task not found!
    pause
    exit /b 1
)
if not exist "selfie_segmenter.tflite" (
    echo ERROR: selfie_segmenter.tflite not found!
    pause
    exit /b 1
)

REM Build
echo Building executable...
pyinstaller --onefile --windowed --name MagicBody --add-data "pose_landmarker.task;." --add-data "selfie_segmenter.tflite;." app.py

echo.
echo =========================================
echo   Build Complete!
echo =========================================
echo   Output: dist\MagicBody.exe
echo.
pause
