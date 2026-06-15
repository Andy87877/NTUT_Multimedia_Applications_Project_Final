@echo off
chcp 437 > nul
title YOLOv4-tiny Training - NTUT Project 6

set "ROOT=%~dp0.."
pushd "%ROOT%"

echo.
echo ============================================================
echo   YOLOv4-tiny Local GPU Training (High-Speed PyTorch)
echo   NTUT Multimedia Applications - Project 6
echo ============================================================
echo.

:: ── STEP 1: Prepare dataset, cfg, weights ─────────────────────
echo [1/3] Preparing dataset + config files + pretrained weights...
echo.
python "%~dp0train_yolov4tiny_darknet.py" --mode prepare
if errorlevel 1 (
    echo.
    echo ERROR: Preparation failed. Check Python environment.
    pause & exit /b 1
)

:: ── STEP 2: PyTorch GPU Training ──────────────────────────────
echo.
echo [2/3] Starting High-Speed PyTorch GPU Training ...
echo.
echo   Dataset : _SignDetection.yolov4pytorch (151 images)
echo   Classes : 0=stop  1=rail  2=pedestrian  3=blocked
echo   Device  : NVIDIA GPU (GeForce GTX 1650) via CUDA
echo.
echo   Training is extremely fast using local GPU (~15-30 seconds!).
echo   Wait a moment...
echo.

python "%~dp0train_pytorch_yolov4tiny.py"
if errorlevel 1 (
    echo.
    echo ERROR: Training failed. Check CUDA and PyTorch installation.
    pause & exit /b 1
)

:: ── STEP 3: Verification ──────────────────────────────────────
echo.
echo [3/3] Training and Verification Completed!
echo.
echo ============================================================
echo   YOLOv4-tiny Model Ready!
echo ============================================================
echo.
echo   JetBot deploy files successfully written to: jetbot_deploy\
echo     - yolov4-tiny-416.cfg      (Original Darknet Config)
echo     - yolov4-tiny-416.weights  (Trained Darknet Binary Weights)
echo     - obj.names                (Class Names)
echo     - DEPLOY.txt               (Step-by-step deploy instructions)
echo.
echo   A verification prediction was performed. Review the results:
echo     - See 'inference_pytorch.jpg' in Project6/ folder
echo       to confirm box coordinates and classification accuracy.
echo.
echo   Deployment Steps on JetBot (Section 7.2):
echo     1. Copy jetbot_deploy/* to trt_yolv4-tiny-master/yolo/
echo     2. Run on JetBot:
echo        python3 yolo_to_onnx.py -c 4 -m yolov4-tiny-416
echo        python3 onnx_to_tensorrt.py -c 4 -m yolov4-tiny-416
echo     3. Run in Project06.ipynb:
echo        trt_yolo = TRT_YOLO("yolov4-tiny-416", (416, 416), 4)
echo ============================================================
pause

popd
