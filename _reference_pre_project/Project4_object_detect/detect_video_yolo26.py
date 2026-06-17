# -*- coding: utf-8 -*-
"""
detect_video_yolo26.py
======================
Use trained YOLO26 model to detect cars in a video and output annotated video.

YOLO26 (Ultralytics, Jan 2026) provides NMS-free end-to-end inference,
making the detection pipeline much simpler than the traditional Darknet approach.

Usage:
  python detect_video_yolo26.py --video path/to/video.mp4
  python detect_video_yolo26.py --video path/to/video.mp4 --output result.mp4
  python detect_video_yolo26.py --video path/to/video.mp4 --weights runs_yolo26/car_detect/weights/best.pt
"""

import os
import sys
import argparse
import time

# Project root directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_video(weights, videofile, conf_thresh=0.4, iou_thresh=0.6, output_path=None):
    """
    Detect objects in a video using YOLO26, frame by frame.

    Args:
        weights: Path to trained YOLO26 weights (.pt file)
        videofile: Path to the input video
        conf_thresh: Confidence threshold (default: 0.4)
        iou_thresh: IoU threshold for NMS (default: 0.6)
        output_path: Path for the output video (default: input_detected.mp4)
    """
    from ultralytics import YOLO
    import cv2
    import torch

    # ============================================================
    # Load YOLO26 model
    # ============================================================
    print(f"Loading YOLO26 model: {weights}")
    model = YOLO(weights)

    # Force GPU usage
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is NOT available! This script requires an NVIDIA GPU.")
        print("  Please install the CUDA version of PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ============================================================
    # Open input video
    # ============================================================
    cap = cv2.VideoCapture(videofile)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {videofile}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\nVideo info:")
    print(f"  File:       {videofile}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps:.1f}")
    print(f"  Frames:     {total_frames}")
    print(f"  Duration:   {duration:.1f} sec")

    # ============================================================
    # Setup output video writer
    # ============================================================
    if output_path is None:
        base, ext = os.path.splitext(videofile)
        output_path = base + "_yolo26" + ext

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[ERROR] Cannot create output video: {output_path}")
        cap.release()
        return

    # ============================================================
    # Process video frame by frame
    # ============================================================
    print(f"\nProcessing video...")
    print(f"Output: {output_path}\n")

    frame_count = 0
    total_detections = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO26 inference on single frame
        # Handles all preprocessing internally (resize, normalize, etc.)
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            verbose=False,
        )

        result = results[0]
        n_detections = len(result.boxes)
        total_detections += n_detections

        # Draw bounding boxes using Ultralytics built-in plot()
        annotated_frame = result.plot()

        # Add frame info overlay in top-left corner
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {n_detections}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Write annotated frame to output video
        out.write(annotated_frame)

        # Progress display every 30 frames
        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
            print(f"  Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                  f"FPS: {fps_actual:.1f} | ETA: {eta:.0f}s")

    cap.release()
    out.release()

    # ============================================================
    # Print summary
    # ============================================================
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  YOLO26 Video Detection Complete!")
    print(f"{'=' * 60}")
    print(f"  Processed frames: {frame_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections/frame: {total_detections / max(frame_count, 1):.1f}")
    print(f"  Processing time: {elapsed:.1f} sec")
    print(f"  Average FPS: {frame_count / max(elapsed, 1):.1f}")
    print(f"  Output file: {output_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="YOLO26 video detection")
    parser.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path to the video to detect"
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
        help="Output video path"
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights not found: {args.weights}")
        print("  Please train first: python train_local_yolo26.py")
        sys.exit(1)

    detect_video(args.weights, args.video, args.conf, args.iou, args.output)


if __name__ == "__main__":
    main()