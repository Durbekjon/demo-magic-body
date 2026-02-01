# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Magic Body app.
Build with: pyinstaller magic_body.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect MediaPipe data files
mediapipe_datas = collect_data_files('mediapipe')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        # MediaPipe model files
        ('pose_landmarker.task', '.'),
        ('selfie_segmenter.tflite', '.'),
        # MediaPipe package data
        *mediapipe_datas,
    ],
    hiddenimports=[
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
        'cv2',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'tensorflow',
        'tensorboard',
        'scipy',
        'matplotlib',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MagicBody',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want console output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if desired: 'icon.ico'
)

# For macOS, create an app bundle
app = BUNDLE(
    exe,
    name='MagicBody.app',
    icon=None,  # Add icon path here if desired: 'icon.icns'
    bundle_identifier='com.demo.magicbody',
    info_plist={
        'NSCameraUsageDescription': 'Magic Body needs camera access for body effects.',
        'CFBundleShortVersionString': '1.0.0',
    },
)
