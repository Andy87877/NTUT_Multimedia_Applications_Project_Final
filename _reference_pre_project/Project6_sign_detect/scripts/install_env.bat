@echo off
chcp 437 > nul

echo ============================================
echo   YOLO Sign Detection - Setup Environment
echo   NTUT Multimedia Applications Project 6
echo ============================================
echo.

echo [Step 1/2] Installing ultralytics (YOLOv11)...
pip install ultralytics
if errorlevel 1 (
    echo [ERROR] Failed to install ultralytics.
    echo         Please make sure pip is available.
    pause
    exit /b 1
)
echo [OK] ultralytics installed.

echo.
echo [Step 2/2] Installing other dependencies...
pip install pyyaml matplotlib opencv-python
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.

echo.
echo ============================================
echo   Setup complete! You can now train:
echo.
echo   Option 1 - Command line:
echo     python train_yolo.py --mode train
echo.
echo   Option 2 - Jupyter Notebook:
echo     jupyter notebook train_yolo.ipynb
echo.
echo   Option 3 - Google Colab:
echo     Upload train_yolo.ipynb to Colab
echo ============================================
echo.
pause
