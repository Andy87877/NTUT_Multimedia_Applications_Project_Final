# -*- coding: utf-8 -*-
# fix Windows console encoding
from pathlib import Path
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
YOLO Sign Detection Training Script
Dataset : _SignDetection.yolo26  (151 images)
Classes : blocked / pedestrian / rail / stop

Usage:
    python train_yolo.py --mode train
    python train_yolo.py --mode resume
    python train_yolo.py --mode val
    python train_yolo.py --mode export
    python train_yolo.py --mode detect
"""


# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "_SignDetection.yolo26"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "sign_detection"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# ── Training hyper-parameters ──────────────────────────────────────
TRAIN_CONFIG = {
    # Basic
    "model": "yolo11n.pt",  # nano pretrained weights, fast enough for JetBot
    "data": str(DATA_YAML),
    "epochs": 100,
    "imgsz": 416,           # matches TRT_YOLO("yolov4-tiny-416") in ipynb
    "batch": 16,
    "workers": 4,

    # Optimizer
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # Loss weights
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # Augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,

    # Output
    "project": str(RUNS_DIR.parent),
    "name": RUNS_DIR.name,
    "exist_ok": True,
    "save": True,
    "save_period": 10,            # save checkpoint every 10 epochs
    "patience": 30,            # early stopping patience
    "plots": True,
    "verbose": True,
    "device": "",            # auto: GPU if available, else CPU
}

CLASS_NAMES = ["blocked", "pedestrian", "rail", "stop"]


# ══════════════════════════════════════════════════════════════════
#  1. Fix data.yaml paths
# ══════════════════════════════════════════════════════════════════
def fix_data_yaml():
    """
    The original data.yaml uses relative paths and expects valid/test dirs
    that don't exist in this dataset (train-only).
    We rewrite it with absolute paths and fall back val -> train.
    """
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_dir = DATASET_DIR / "train" / "images"
    valid_dir = DATASET_DIR / "valid" / "images"
    test_dir = DATASET_DIR / "test" / "images"

    cfg["train"] = str(train_dir)
    cfg["val"] = str(valid_dir) if valid_dir.exists() else str(train_dir)
    cfg["test"] = str(test_dir) if test_dir.exists() else str(train_dir)

    fixed = DATASET_DIR / "data_fixed.yaml"
    with open(fixed, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] data_fixed.yaml written -> {fixed}")
    print(f"       train : {cfg['train']}")
    print(f"       val   : {cfg['val']}")
    print(f"       nc    : {cfg['nc']}  names: {cfg['names']}")
    return fixed


# ══════════════════════════════════════════════════════════════════
#  2. Dataset summary
# ══════════════════════════════════════════════════════════════════
def show_dataset_info():
    imgs = list((DATASET_DIR / "train" / "images").glob("*.jpg"))
    lbls = list((DATASET_DIR / "train" / "labels").glob("*.txt"))

    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"  Path          : {DATASET_DIR}")
    print(f"  Train images  : {len(imgs)}")
    print(f"  Train labels  : {len(lbls)}")
    print(f"  Classes (4)   : blocked / pedestrian / rail / stop")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════
#  3. Import ultralytics
# ══════════════════════════════════════════════════════════════════
def get_yolo():
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"[INFO] ultralytics {ultralytics.__version__} ready.")
        return YOLO
    except ImportError:
        print("[WARNING] ultralytics not found, installing...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
        return YOLO


# ══════════════════════════════════════════════════════════════════
#  4. Train
# ══════════════════════════════════════════════════════════════════
def train(resume=False):
    YOLO = get_yolo()

    print("\n" + "=" * 60)
    print("  Starting YOLO training (Sign Detection)")
    print("=" * 60)

    fixed_yaml = fix_data_yaml()
    show_dataset_info()

    config = TRAIN_CONFIG.copy()
    config["data"] = str(fixed_yaml)

    if resume:
        last_ckpt = RUNS_DIR / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"[INFO] Resuming from checkpoint: {last_ckpt}")
            model = YOLO(str(last_ckpt))
            config["resume"] = True
        else:
            print("[WARNING] No checkpoint found, starting from pretrained weights.")
            model = YOLO(config["model"])
    else:
        print(f"[INFO] Loading pretrained weights: {config['model']}")
        model = YOLO(config["model"])

    results = model.train(**config)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best model : {RUNS_DIR / 'weights' / 'best.pt'}")
    print(f"  Last model : {RUNS_DIR / 'weights' / 'last.pt'}")
    print("=" * 60)
    return results


# ══════════════════════════════════════════════════════════════════
#  5. Validate
# ══════════════════════════════════════════════════════════════════
def validate(weights_path=None):
    YOLO = get_yolo()

    if weights_path is None:
        weights_path = RUNS_DIR / "weights" / "best.pt"

    if not Path(weights_path).exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("        Run training first:  python train_yolo.py --mode train")
        return

    fixed_yaml = fix_data_yaml()
    print(f"\n[INFO] Validating: {weights_path}")

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(fixed_yaml),
        imgsz=TRAIN_CONFIG["imgsz"],
        conf=0.25,
        iou=0.6,
        plots=True,
        verbose=True,
    )

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
#  6. Export (ONNX -> TensorRT on JetBot)
# ══════════════════════════════════════════════════════════════════
def export_model(weights_path=None, export_format="onnx"):
    """
    Export PyTorch model to ONNX.
    Then on JetBot run:
        trtexec --onnx=best.onnx --saveEngine=sign_detect.trt --workspace=1024 --fp16
    """
    YOLO = get_yolo()

    if weights_path is None:
        weights_path = RUNS_DIR / "weights" / "best.pt"

    if not Path(weights_path).exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("        Run training first:  python train_yolo.py --mode train")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Exporting {weights_path} -> {export_format.upper()}")
    model = YOLO(str(weights_path))

    export_path = model.export(
        format=export_format,
        imgsz=TRAIN_CONFIG["imgsz"],
        simplify=True,   # simplify ONNX graph
        opset=11,     # JetPack compatible
        dynamic=False,
        half=False,  # FP32 here; convert to FP16 on JetBot
    )

    dst = WEIGHTS_DIR / Path(export_path).name
    shutil.copy2(export_path, dst)

    print("\n" + "=" * 60)
    print("  Export complete!")
    print(f"  ONNX model : {dst}")
    print()
    print("  JetBot deployment:")
    print("    1. scp the .onnx file to JetBot")
    print("    2. On JetBot, run:")
    print(f"       trtexec --onnx={dst.name} \\")
    print("               --saveEngine=sign_detect.trt \\")
    print("               --workspace=1024 --fp16")
    print("=" * 60)
    return export_path


# ══════════════════════════════════════════════════════════════════
#  7. Detect / Inference test
# ══════════════════════════════════════════════════════════════════
def detect(source=None, weights_path=None, conf=0.25):
    YOLO = get_yolo()

    if weights_path is None:
        weights_path = RUNS_DIR / "weights" / "best.pt"

    if not Path(weights_path).exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    if source is None:
        source = str(DATASET_DIR / "train" / "images")

    print(f"\n[INFO] Source  : {source}")
    print(f"[INFO] Weights : {weights_path}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=source,
        imgsz=TRAIN_CONFIG["imgsz"],
        conf=conf,
        iou=0.45,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(RUNS_DIR.parent / "predict"),
        name="sign_detect",
        exist_ok=True,
        verbose=True,
        max_det=10,
    )

    print("\n" + "=" * 60)
    print("  Detection Results")
    print("=" * 60)
    for r in results:
        img_name = Path(r.path).name
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  {img_name}: no sign detected")
        else:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_v = float(box.conf[0])
                xyxy = [int(v) for v in box.xyxy[0]]
                width = xyxy[2] - xyxy[0]
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(
                    CLASS_NAMES) else str(cls_id)
                print(
                    f"  {img_name}: [{cls_name}] conf={conf_v:.2f}  w={width}px  bbox={xyxy}")
    print("=" * 60)
    return results


# ══════════════════════════════════════════════════════════════════
#  8. Entry point
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="YOLO Sign Detection Training  (Project 6 - NTUT)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="train",
        choices=["train", "resume", "val", "export", "detect"],
        help=(
            "Mode:\n"
            "  train  - Train from scratch (default)\n"
            "  resume - Resume from last checkpoint\n"
            "  val    - Validate model performance\n"
            "  export - Export to ONNX for JetBot TensorRT\n"
            "  detect - Run inference test on training images\n"
        ),
    )
    parser.add_argument("--weights", "-w", type=str, default=None,
                        help="Path to model weights (for val/export/detect)")
    parser.add_argument("--source",  "-s", type=str, default=None,
                        help="Image source for detect mode (file or folder)")
    parser.add_argument("--format",  "-f", type=str, default="onnx",
                        choices=["onnx", "torchscript", "tflite"],
                        help="Export format (default: onnx)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold for detect (default: 0.25)")
    parser.add_argument("--epochs",  type=int,   default=None,
                        help="Override number of training epochs (default: 100)")
    parser.add_argument("--batch",   type=int,   default=None,
                        help="Override batch size (default: 16)")

    args = parser.parse_args()

    if args.epochs is not None:
        TRAIN_CONFIG["epochs"] = args.epochs
    if args.batch is not None:
        TRAIN_CONFIG["batch"] = args.batch

    print("\n" + "=" * 60)
    print("  YOLO Sign Detection Tool")
    print("  NTUT Multimedia Applications - Project 6")
    print("=" * 60)
    print(f"  Mode   : {args.mode.upper()}")
    print(f"  Epochs : {TRAIN_CONFIG['epochs']}")
    print(f"  Batch  : {TRAIN_CONFIG['batch']}")
    print(f"  Imgsz  : {TRAIN_CONFIG['imgsz']}")
    print("=" * 60 + "\n")

    if args.mode == "train":
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
