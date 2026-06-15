# -*- coding: utf-8 -*-
"""
detect_image_yolo26.py
======================
Use trained YOLO26 model to detect cars in a single image.

YOLO26 (Ultralytics, Jan 2026) provides NMS-free end-to-end inference,
making the detection pipeline much simpler than the traditional Darknet approach.

Usage:
  python detect_image_yolo26.py --image path/to/image.jpg
  python detect_image_yolo26.py --image path/to/image.jpg --weights runs_yolo26/car_detect/weights/best.pt
  python detect_image_yolo26.py --image path/to/image.jpg --conf 0.3
"""

import os
import sys
import argparse

# Project root directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_image(weights, imgfile, conf_thresh=0.4, iou_thresh=0.6, output_dir=None):
    """
    Detect objects in a single image using YOLO26.

    Args:
        weights: Path to trained YOLO26 weights (.pt file)
        imgfile: Path to the input image
        conf_thresh: Confidence threshold (default: 0.4)
        iou_thresh: IoU threshold for NMS (default: 0.6)
        output_dir: Directory to save results (default: alongside input image)
    """
    from ultralytics import YOLO
    import cv2

    # ============================================================
    # Load YOLO26 model
    # ============================================================
    print(f"Loading YOLO26 model: {weights}")
    model = YOLO(weights)

    # Force GPU usage
    import torch
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is NOT available! This script requires an NVIDIA GPU.")
        print("  Please install the CUDA version of PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ============================================================
    # Run detection
    # ============================================================
    print(f"\nInput image: {imgfile}")

    # Read image to show dimensions
    img = cv2.imread(imgfile)
    if img is None:
        print(f"[ERROR] Cannot read image: {imgfile}")
        return
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # YOLO26 inference - handles all preprocessing internally:
    #   - Resize to model input size
    #   - Normalize pixel values
    #   - NMS-free end-to-end detection
    print("Running YOLO26 detection...")
    results = model.predict(
        source=imgfile,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        verbose=False,
    )

    # ============================================================
    # Process and display results
    # ============================================================
    result = results[0]  # Single image, single result
    boxes = result.boxes
    n_detections = len(boxes)

    print(f"\nDetected {n_detections} objects:")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_name = result.names[cls_id]
        print(f"  {cls_name}: {conf:.2f} at ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")

    # ============================================================
    # Save result image with bounding boxes drawn
    # ============================================================
    # Ultralytics provides a convenient plot() method that draws
    # bounding boxes, labels, and confidence scores on the image
    result_img = result.plot()

    if output_dir is None:
        base, ext = os.path.splitext(imgfile)
        output_path = base + "_yolo26" + ext
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(imgfile))

    cv2.imwrite(output_path, result_img)
    print(f"\nResult saved to: {output_path}")

    return result_img


def main():
    parser = argparse.ArgumentParser(description="YOLO26 image detection")
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to the image to detect"
    )
    parser.add_argument(
        "--weights", "-w", type=str,
        default=os.path.join(PROJECT_DIR, "runs_yolo26", "car_detect", "weights", "best.pt"),
        help="Path to YOLO26 weights file (.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold (default: 0.4)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.6,
        help="IoU threshold (default: 0.6)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for result images"
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights not found: {args.weights}")
        print("  Please train first: python train_local_yolo26.py")
        sys.exit(1)

    detect_image(args.weights, args.image, args.conf, args.iou, args.output)


if __name__ == "__main__":
    main()