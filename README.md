# 🎭 Magic Body - Fun-House Body Effects

Real-time body deformation effects using AI pose tracking and segmentation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

## ✨ Effects

| Key | Effect | Description |
|-----|--------|-------------|
| `1` | Big Head | Cartoon-style head enlargement |
| `2` | Long Limbs | Stretched arms and legs |
| `3` | Rubber Body | Wobbly jelly-like motion |
| `4` | Alien | Big head + long limbs + compressed torso |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/demo-magic-body.git
cd demo-magic-body

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

## 🔨 Build Executable

### Windows (.exe)

```cmd
pip install pyinstaller
pyinstaller --onefile --windowed --name MagicBody --add-data "pose_landmarker.task;." --add-data "selfie_segmenter.tflite;." app.py
```

Output: `dist\MagicBody.exe`

### macOS (.app)

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name MagicBody --add-data "pose_landmarker.task:." --add-data "selfie_segmenter.tflite:." app.py
```

Output: `dist/MagicBody.app`

## 🎮 Controls

| Key | Action |
|-----|--------|
| `1-4` | Switch effect mode |
| `ESC` | Exit |

## 📁 Project Structure

```
demo-magic-body/
├── app.py                    # Main application
├── pose_landmarker.task      # MediaPipe pose model
├── selfie_segmenter.tflite   # MediaPipe segmentation model
├── requirements.txt          # Python dependencies
├── build.sh                  # macOS build script
└── build_windows.bat         # Windows build script
```

## 📋 Requirements

- `opencv-python>=4.8.0`
- `mediapipe>=0.10.0`
- `numpy>=1.24.0`

## 📄 License

MIT License
