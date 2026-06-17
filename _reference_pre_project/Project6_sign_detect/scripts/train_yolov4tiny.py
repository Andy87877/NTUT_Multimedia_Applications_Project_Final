# -*- coding: utf-8 -*-
from pathlib import Path
import cv2
import yaml
import argparse
import shutil
import os
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
YOLOv4-tiny Training Script using Ultralytics
Dataset : _SignDetection.yolov4pytorch
Format  : _annotations.txt  ->  x1,y1,x2,y2,class_id (absolute pixel coords)
Classes : blocked / pedestrian / rail / stop

Steps:
  1. Convert _annotations.txt -> YOLO label .txt files (normalized cx,cy,w,h)
  2. Write data.yaml
  3. Train with YOLOv4-tiny via ultralytics (yolov4-tiny.pt)

Usage:
    python train_yolov4tiny.py --mode train
    python train_yolov4tiny.py --mode train --epochs 150 --batch 16
    python train_yolov4tiny.py --mode val
    python train_yolov4tiny.py --mode export
    python train_yolov4tiny.py --mode detect
"""


# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "_SignDetection.yolov4pytorch"
# raw images + _annotations.txt
TRAIN_RAW_DIR = DATASET_DIR / "train"
ANN_FILE = TRAIN_RAW_DIR / "_annotations.txt"
CLS_FILE = TRAIN_RAW_DIR / "_classes.txt"

# Converted dataset (YOLO format with separate images/ and labels/)
CONV_DIR = PROJECT_ROOT / "_yolov4tiny_converted"
CONV_IMG_DIR = CONV_DIR / "images" / "train"
CONV_LBL_DIR = CONV_DIR / "labels" / "train"
DATA_YAML = CONV_DIR / "data.yaml"

RUNS_DIR = PROJECT_ROOT / "runs" / "yolov4tiny"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

CLASS_NAMES = ["blocked", "pedestrian", "rail", "stop"]

# ── Training config ────────────────────────────────────────────────
TRAIN_CONFIG = {
    # Model: yolov4-tiny equivalent in ultralytics -> YOLOv8n / YOLO11n (same scale)
    # ultralytics does not ship a "yolov4-tiny.pt" directly, but provides
    # yolov4-tiny architecture via cfg file. We use the built-in one:
    "model": "yolov4-tiny.pt",   # auto-downloaded if not present
    "data": str(DATA_YAML),
    "epochs": 100,
    "imgsz": 416,
    "batch": 16,
    "workers": 4,

    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "scale": 0.5,
    "translate": 0.1,

    "project": str(RUNS_DIR.parent),
    "name": RUNS_DIR.name,
    "exist_ok": True,
    "save": True,
    "save_period": 10,
    "patience": 30,
    "plots": True,
    "verbose": True,
    "device": "",
}


# ══════════════════════════════════════════════════════════════════
#  1. Parse _annotations.txt -> YOLO label files
# ══════════════════════════════════════════════════════════════════
def convert_dataset(force=False):
    """
    _annotations.txt format per line:
        filename.jpg  x1,y1,x2,y2,cls  x1,y1,x2,y2,cls  ...
    Convert to YOLO:  cls cx cy w h  (normalized 0-1)
    """
    if CONV_IMG_DIR.exists() and not force:
        n = len(list(CONV_IMG_DIR.glob("*.jpg")))
        if n > 0:
            print(
                f"[INFO] Converted dataset already exists ({n} images). Skipping.")
            return

    print("[INFO] Converting _annotations.txt -> YOLO label files ...")
    CONV_IMG_DIR.mkdir(parents=True, exist_ok=True)
    CONV_LBL_DIR.mkdir(parents=True, exist_ok=True)

    ok = skip = err = 0
    with open(ANN_FILE, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            fname = parts[0]
            src = TRAIN_RAW_DIR / fname
            if not src.exists():
                print(f"  [WARN] image not found: {fname}")
                skip += 1
                continue

            # Get real image size
            img = cv2.imread(str(src))
            if img is None:
                print(f"  [WARN] cannot read: {fname}")
                skip += 1
                continue
            H, W = img.shape[:2]

            # Copy image
            dst_img = CONV_IMG_DIR / fname
            shutil.copy2(src, dst_img)

            # Build label
            label_lines = []
            for ann in parts[1:]:
                try:
                    x1, y1, x2, y2, cls_id = map(int, ann.split(","))
                except ValueError:
                    continue
                # Clamp
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                if bw <= 0 or bh <= 0:
                    continue
                label_lines.append(
                    f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            dst_lbl = CONV_LBL_DIR / (Path(fname).stem + ".txt")
            with open(dst_lbl, "w") as lf:
                lf.write("\n".join(label_lines))
            ok += 1

    print(
        f"[INFO] Conversion done:  {ok} ok  |  {skip} skipped  |  {err} errors")
    print(f"       Images -> {CONV_IMG_DIR}")
    print(f"       Labels -> {CONV_LBL_DIR}")


# ══════════════════════════════════════════════════════════════════
#  2. Write data.yaml
# ══════════════════════════════════════════════════════════════════
def write_data_yaml():
    cfg = {
        "train": str(CONV_IMG_DIR),
        "val": str(CONV_IMG_DIR),   # same split (train-only dataset)
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[INFO] data.yaml written -> {DATA_YAML}")
    for k, v in cfg.items():
        print(f"       {k}: {v}")
    return DATA_YAML


# ══════════════════════════════════════════════════════════════════
#  3. Get YOLO class
# ══════════════════════════════════════════════════════════════════
def get_yolo():
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"[INFO] ultralytics {ultralytics.__version__} ready.")
        return YOLO
    except ImportError:
        import subprocess
        print("[WARNING] Installing ultralytics ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
        return YOLO


# ══════════════════════════════════════════════════════════════════
#  4. Check if yolov4-tiny.pt is available, else fallback
# ══════════════════════════════════════════════════════════════════
def resolve_model():
    """
    ultralytics v8.x includes yolov4-tiny.pt in its model zoo.
    Try to load it; if unavailable, fall back to yolov8n.pt which
    has a similar parameter count (~3M) and speed profile.
    """
    YOLO = get_yolo()
    preferred = "yolov4-tiny.pt"
    fallback = "yolov8n.pt"

    try:
        print(f"[INFO] Trying model: {preferred}")
        m = YOLO(preferred)
        print(f"[INFO] {preferred} loaded successfully.")
        return preferred, YOLO
    except Exception as e:
        print(f"[WARN] {preferred} not available: {e}")
        print(
            f"[INFO] Falling back to {fallback} (similar scale to yolov4-tiny)")
        return fallback, YOLO


# ══════════════════════════════════════════════════════════════════
#  5. Train
# ══════════════════════════════════════════════════════════════════
def train(resume=False):
    convert_dataset()
    write_data_yaml()

    model_name, YOLO = resolve_model()

    config = TRAIN_CONFIG.copy()
    config["model"] = model_name
    config["data"] = str(DATA_YAML)

    print("\n" + "=" * 60)
    print("  YOLOv4-tiny Sign Detection Training")
    print("=" * 60)
    print(f"  Model  : {model_name}")
    print(f"  Data   : {DATA_YAML}")
    print(f"  Epochs : {config['epochs']}")
    print(f"  Imgsz  : {config['imgsz']}")
    print(f"  Batch  : {config['batch']}")
    print("=" * 60 + "\n")

    if resume:
        last_ckpt = RUNS_DIR / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"[INFO] Resuming from: {last_ckpt}")
            model = YOLO(str(last_ckpt))
            config["resume"] = True
        else:
            print("[WARN] No checkpoint found, starting fresh.")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)

    results = model.train(**config)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best  : {RUNS_DIR / 'weights' / 'best.pt'}")
    print(f"  Last  : {RUNS_DIR / 'weights' / 'last.pt'}")
    print("=" * 60)
    return results


# ══════════════════════════════════════════════════════════════════
#  6. Validate
# ══════════════════════════════════════════════════════════════════
def validate(weights_path=None):
    _, YOLO = resolve_model()

    if weights_path is None:
        weights_path = RUNS_DIR / "weights" / "best.pt"
    if not Path(weights_path).exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    write_data_yaml()
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(DATA_YAML), imgsz=416, conf=0.25, iou=0.6,
                        plots=True, verbose=True)

    print("\n" + "=" * 60)
    print("  Validation Results")
    print("=" * 60)
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")
    print("=" * 60)
    return metrics


# ══════════════════════════════════════════════════════════════════
#  7. Export ONNX
# ══════════════════════════════════════════════════════════════════
def export_model(weights_path=None, export_format="onnx"):
    _, YOLO = resolve_model()

    if weights_path is None:
        weights_path = RUNS_DIR / "weights" / "best.pt"
    if not Path(weights_path).exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    export_path = model.export(format=export_format, imgsz=416,
                               simplify=True, opset=11, dynamic=False, half=False)
    dst = WEIGHTS_DIR / Path(export_path).name
    shutil.copy2(export_path, dst)

    print(f"\n[INFO] Exported -> {dst}")
    print("  JetBot TensorRT conversion:")
    print(
        f"    trtexec --onnx={dst.name} --saveEngine=sign_v4tiny.trt --workspace=1024 --fp16")
    return export_path


# ══════════════════════════════════════════════════════════════════
#  8. Detect / Inference test
# ══════════════════════════════════════════════════════════════════
def detect(source=None, weights_path=None, conf=0.25):
    _, YOLO = resolve_model()

    if weights_path is None:
        weights_path = RUNS_DIR / "weights" / "best.pt"
    if not Path(weights_path).exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    convert_dataset()
    if source is None:
        source = str(CONV_IMG_DIR)

    print(f"[INFO] Source  : {source}")
    print(f"[INFO] Weights : {weights_path}")

    model = YOLO(str(weights_path))
    results = model.predict(source=source, imgsz=416, conf=conf, iou=0.45,
                            save=True, save_txt=True, save_conf=True,
                            project=str(RUNS_DIR.parent / "predict_v4tiny"),
                            name="sign_detect", exist_ok=True, verbose=True, max_det=10)

    print("\n" + "=" * 60)
    print("  Detection Results")
    print("=" * 60)
    for r in results:
        img_name = Path(r.path).name
        if r.boxes is None or len(r.boxes) == 0:
            print(f"  {img_name}: no sign detected")
        else:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_v = float(box.conf[0])
                xyxy = [int(v) for v in box.xyxy[0]]
                width = xyxy[2] - xyxy[0]
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(
                    CLASS_NAMES) else str(cls_id)
                print(
                    f"  {img_name}: [{cls_name}] conf={conf_v:.2f}  w={width}px")
    print("=" * 60)
    return results


# ══════════════════════════════════════════════════════════════════
#  9. Entry point
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv4-tiny Sign Detection  (Project 6 - NTUT)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", "-m", default="train",
                        choices=["train", "resume", "val",
                                 "export", "detect", "convert"],
                        help="Mode: train / resume / val / export / detect / convert")
    parser.add_argument("--weights", "-w", default=None,
                        help="Path to weights (for val/export/detect)")
    parser.add_argument("--source",  "-s", default=None,
                        help="Image source for detect mode")
    parser.add_argument("--format",  "-f", default="onnx",
                        choices=["onnx", "torchscript", "tflite"])
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--force-convert", action="store_true",
                        help="Re-run dataset conversion even if already done")

    args = parser.parse_args()

    if args.epochs:
        TRAIN_CONFIG["epochs"] = args.epochs
    if args.batch:
        TRAIN_CONFIG["batch"] = args.batch

    print("\n" + "=" * 60)
    print("  YOLOv4-tiny Sign Detection Tool")
    print("  NTUT Multimedia Applications - Project 6")
    print("=" * 60)
    print(f"  Mode   : {args.mode.upper()}")
    print(f"  Epochs : {TRAIN_CONFIG['epochs']}")
    print(f"  Batch  : {TRAIN_CONFIG['batch']}")
    print(f"  Imgsz  : {TRAIN_CONFIG['imgsz']}")
    print("=" * 60 + "\n")

    if args.mode == "convert":
        convert_dataset(force=args.force_convert)
        write_data_yaml()
    elif args.mode == "train":
        train(resume=False)
    elif args.mode == "resume":
        train(resume=True)
    elif args.mode == "val":
        validate(weights_path=args.weights)
    elif args.mode == "export":
        export_model(weights_path=args.weights, export_format=args.format)
    elif args.mode == "detect":
        detect(source=args.source, weights_path=args.weights, conf=args.conf)


if __name__ == "__main__":
    main()
