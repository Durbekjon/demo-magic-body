#!/bin/bash
# Build script for Magic Body desktop app

echo "========================================="
echo "  Building Magic Body Desktop App"
echo "========================================="

# Check for PyInstaller
if ! command -v pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Check for required model files
if [ ! -f "pose_landmarker.task" ]; then
    echo "ERROR: pose_landmarker.task not found!"
    echo "Please ensure the model file is in the current directory."
    exit 1
fi

if [ ! -f "selfie_segmenter.tflite" ]; then
    echo "ERROR: selfie_segmenter.tflite not found!"
    echo "Please ensure the model file is in the current directory."
    exit 1
fi

echo ""
echo "Building application..."
echo ""

# Build using spec file
pyinstaller magic_body.spec --clean --noconfirm

echo ""
echo "========================================="
echo "  Build Complete!"
echo "========================================="
echo ""
echo "Output locations:"
echo "  macOS: dist/MagicBody.app"
echo "  Windows: dist/MagicBody.exe"
echo ""
echo "To run on macOS: open dist/MagicBody.app"
echo "To run on Windows: dist\\MagicBody.exe"
