# -*- coding: utf-8 -*-
"""
train_local_yolo26.py
=====================
YOLO26 local training script (Ultralytics framework)

YOLO26 is the latest YOLO model released by Ultralytics in January 2026,
featuring NMS-free end-to-end inference, ProgLoss + STAL loss functions,
with improved accuracy and speed over YOLO11 / YOLOv8.

This script uses the Ultralytics Python API for custom car detection training,
replacing the traditional Darknet framework manual training pipeline.

Prerequisites:
  1. pip install ultralytics
  2. python prepare_dataset_yolo26.py  (generate data.yaml and standard format labels)

Usage:
  python train_local_yolo26.py

Custom parameters:
  python train_local_yolo26.py --epochs 100 --imgsz 640 --batch 16
  python train_local_yolo26.py --model yolo26s.pt  (use Small model)
"""

import os
import sys
import argparse

# Project root directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Parse command line arguments for YOLO26 training."""
    parser = argparse.ArgumentParser(
        description="YOLO26 car detection training (Ultralytics)"
    )
    parser.add_argument(
        "--model", type=str, default="yolo26n.pt",
        help="Pretrained model (yolo26n/yolo26s/yolo26m/yolo26l/yolo26x, default: yolo26n.pt)"
    )
    parser.add_argument(
        "--data", type=str,
        default=os.path.join(
            PROJECT_DIR, "dataset_merged_yolo26", "data.yaml"),
        help="Dataset config file (data.yaml)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16, reduce to 8/4 if GPU OOM)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto/cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Data loader workers (default: 2, recommend <=4 on Windows)"
    )
    parser.add_argument(
        "--project", type=str,
        default=os.path.join(PROJECT_DIR, "runs_yolo26"),
        help="Output directory for training results"
    )
    parser.add_argument(
        "--name", type=str, default="car_detect_merged",
        help="Experiment name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ============================================================
    # Environment check
    # ============================================================
    print("=" * 60)
    print("  YOLO26 Car Detection - Training Script")
    print("=" * 60)

    # Check ultralytics installation
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"\n  Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("\n[ERROR] ultralytics package not found!")
        print("  Please run: pip install ultralytics")
        sys.exit(1)

    # Check GPU availability - REQUIRED
    import torch
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA is NOT available! This script REQUIRES an NVIDIA GPU.")
        print("  Please install CUDA version of PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB) - REQUIRED")

    # Check data.yaml existence
    if not os.path.exists(args.data):
        print(f"\n[ERROR] Dataset config not found: {args.data}")
        print("  Please run first: python prepare_dataset_yolo26.py")
        sys.exit(1)

    print(f"\n  Model:      {args.model}")
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Output:     {args.project}/{args.name}")
    print(f"{'=' * 60}\n")

    # ============================================================
    # Load pretrained YOLO26 model
    # ============================================================
    # YOLO26 automatically downloads pretrained weights on first use.
    # Available sizes: n(nano), s(small), m(medium), l(large), x(xlarge)
    print("Loading pretrained YOLO26 model...")
    model = YOLO(args.model)
    print(f"  Model loaded: {args.model}\n")

    # ============================================================
    # Start training
    # ============================================================
    # Ultralytics handles all training internals:
    #   - Data augmentation (mosaic, mixup, HSV, flip, etc.)
    #   - Learning rate scheduling (cosine annealing)
    #   - EMA (exponential moving average)
    #   - Auto anchor computation
    #   - Best/last checkpoint saving
    print("Starting YOLO26 training...\n")

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,        # Allow overwriting same-name experiments
        "pretrained": True,      # Transfer learning from COCO pretrained weights
        "verbose": True,         # Detailed training logs
        "save": True,            # Save best.pt and last.pt
        "plots": True,           # Generate loss/metric plots
    }

    # Set compute device if explicitly specified
    if args.device:
        train_kwargs["device"] = args.device

    # Resume from last checkpoint if requested
    if args.resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)

    # ============================================================
    # Training complete - print summary
    # ============================================================
    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")

    print(f"\n{'=' * 60}")
    print("  YOLO26 Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Best weights: {best_weights}")
    print(f"  Results dir:  {args.project}/{args.name}/")
    print(f"\n  Next steps - run inference:")
    print(
        f"    python detect_image_yolo26.py --image test.jpg --weights {best_weights}")
    print(
        f"    python detect_video_yolo26.py --video input.mp4 --weights {best_weights}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
