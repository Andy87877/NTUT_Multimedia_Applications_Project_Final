@echo off
title YOLOv4-tiny V2 Training - NTUT Project 6

set "ROOT=%~dp0"
pushd "%ROOT%"

echo ============================================================
echo   YOLOv4-tiny V2 Local GPU Training (High-Speed PyTorch)
echo   NTUT Multimedia Applications - Project 6 (V2 Advanced)
echo ============================================================
echo.

echo [1/3] Preparing dataset and config files...
python "%ROOT%scripts\train_yolov4tiny_darknet.py" --mode prepare
if errorlevel 1 (
    echo ERROR: Preparation failed. Check Python environment.
    pause
    exit /b 1
)

echo.
echo [2/3] Starting High-Speed PyTorch GPU Training (V2 Advanced)...
echo.
echo   Dataset : _SignDetection.yolov4pytorch (151 images)
echo   Partition: Train (80 percent) / Val (10 percent) / Test (10 percent)
echo   Augmentations: HSV Shift, Random Translation, Geometric Flip, Blur
echo   Optimizations: CIoU Loss, Cosine Annealing, Adam with Weight Decay
echo   Device  : NVIDIA GPU (GeForce GTX 1650) via CUDA
echo.

python "%ROOT%scripts\train_pytorch_yolov4tiny_v2.py"
if errorlevel 1 (
    echo.
    echo ERROR: V2 Training failed. Check CUDA and PyTorch installation.
    pause
    exit /b 1
)

echo.
echo [3/3] Training and Verification Completed!
echo.
echo ============================================================
echo   YOLOv4-tiny V2 Model Ready!
echo ============================================================
echo.
echo   JetBot deploy files successfully written to: jetbot_deploy\
echo     - yolov4-tiny-416.cfg      (Original Darknet Config)
echo     - yolov4-tiny-416.weights  (Trained Darknet Best weights)
echo     - obj.names                (Class Names)
echo.
echo   Excel and Markdown reports successfully written to:
echo     - See Excel history: runs\predict_vis_yolov4tiny_v2\training_history.csv
echo     - See Markdown report: runs\predict_vis_yolov4tiny_v2\training_metrics_report.md
echo ============================================================
pause

popd
