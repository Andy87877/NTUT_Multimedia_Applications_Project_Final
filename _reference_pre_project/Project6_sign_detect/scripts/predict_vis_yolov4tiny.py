# -*- coding: utf-8 -*-
from _reference_pre_project.Project6_sign_detect.scripts.train_pytorch_yolov4tiny import YoloV4Tiny, load_darknet_weights, ANCHORS
import sys
from pathlib import Path
import random
import numpy as np
import cv2
import torch

# Append scripts folder to path to import YoloV4Tiny
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))


# Paths
OBJ_DIR = PROJECT_ROOT / "obj"
WEIGHTS_PATH = PROJECT_ROOT / "jetbot_deploy" / "yolov4-tiny-416.weights"
OUT_DIR = PROJECT_ROOT / "runs" / "predict_vis_yolov4tiny"

CLASS_NAMES = ["stop", "rail", "pedestrian", "blocked"]

# Color per class: BGR
CLASS_COLORS = {
    0: (100, 230, 120),   # stop       -> green-ish
    1: (80,  255, 220),   # rail       -> yellow-ish
    2: (255, 200,  80),   # pedestrian -> cyan-ish
    3: (80,  100, 255),   # blocked    -> red-ish
}


def nms(boxes, iou_threshold=0.45):
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        remaining = []
        for box in boxes:
            # If classes differ, we keep them anyway
            if box["class"] != best["class"]:
                remaining.append(box)
                continue
            # Calculate IoU
            b1 = best["bbox"]
            b2 = box["bbox"]
            xi1 = max(b1[0], b2[0])
            yi1 = max(b1[1], b2[1])
            xi2 = min(b1[2], b2[2])
            yi2 = min(b1[3], b2[3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / (union_area + 1e-6)
            if iou < iou_threshold:
                remaining.append(box)
        boxes = remaining
    return keep


def draw_gt(img, label_path):
    """Draw ground-truth boxes (grey outline) onto img copy."""
    out = img.copy()
    h, w = out.shape[:2]
    if not label_path.exists():
        return out
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            c = CLASS_COLORS.get(cls_id, (200, 200, 200))
            # Draw standard box
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 1)
            label = CLASS_NAMES[cls_id] if cls_id < len(
                CLASS_NAMES) else str(cls_id)
            cv2.putText(out, f"GT:{label}", (x1, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


def draw_pred(img, detections):
    """Draw prediction boxes (solid colored) onto img copy."""
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class"]
        score = det["score"]
        c = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = CLASS_NAMES[cls_id] if cls_id < len(
            CLASS_NAMES) else str(cls_id)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+4, y1), c, -1)
        cv2.putText(out, text, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def make_card(img_path, label_path, detections, target_h=416):
    """Create a side-by-side comparison card."""
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        return None

    h, w = img_orig.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    img_r = cv2.resize(img_orig, (new_w, target_h))

    # Scale detections boxes to panel size
    panel_detections = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        panel_detections.append({
            "bbox": [int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)],
            "class": det["class"],
            "score": det["score"]
        })

    panel_clean = img_r.copy()
    panel_gt = draw_gt(img_r, label_path)
    panel_pred = draw_pred(img_r, panel_detections)

    # Add headers
    def add_header(panel, text, color):
        p = panel.copy()
        cv2.rectangle(p, (0, 0), (new_w, 26), color, -1)
        cv2.putText(p, text, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return p

    panel_clean = add_header(panel_clean, "Original", (60, 60, 60))
    panel_gt = add_header(panel_gt, "Ground Truth", (30, 100, 30))
    panel_pred = add_header(panel_pred, "YOLOv4-tiny", (30, 30, 140))

    # Combine with divider lines
    div = np.full((target_h, 3, 3), 40, dtype=np.uint8)
    card = np.hstack([panel_clean, div, panel_gt, div, panel_pred])
    return card


def main():
    print("\n" + "=" * 60)
    print("  YOLOv4-tiny PyTorch Local Prediction Visualizer")
    print("=" * 60)

    # 1. Check weights
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Weights file not found: {WEIGHTS_PATH}")
        return

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = YoloV4Tiny(num_classes=4).to(device)

    success = load_darknet_weights(model, str(WEIGHTS_PATH))
    if not success:
        print("[ERROR] Failed to load weights.")
        return
    model.eval()
    print("[INFO] Model loaded successfully.")

    # 3. Get images from obj/ directory
    all_imgs = sorted(list(OBJ_DIR.glob("*.jpg")) +
                      list(OBJ_DIR.glob("*.png")))
    if not all_imgs:
        print(f"[ERROR] No images found in {OBJ_DIR}")
        return
    print(f"[INFO] Found {len(all_imgs)} images in obj/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run inference and generate cards
    stats = {"total": 0, "detected": 0}
    all_cards = []

    # Let's process a representative set (e.g. 24 random images or all if small)
    random.seed(42)
    # Process up to 30 images to keep grid size reasonable but representative
    num_to_process = min(len(all_imgs), 30)
    selected_imgs = random.sample(all_imgs, num_to_process)
    selected_imgs = sorted(selected_imgs)
    print(f"[INFO] Processing {num_to_process} representative images...")

    for idx, img_path in enumerate(selected_imgs):
        label_path = img_path.with_suffix(".txt")
        stats["total"] += 1

        # Load & preprocess image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read {img_path}")
            continue
        H, W = img.shape[:2]
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_in, (416, 416))
        img_t = torch.from_numpy(img_in).permute(
            2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        # Run inference
        with torch.no_grad():
            out1, out2 = model(img_t)

        # Decode detections
        detections = []
        for scale_idx, out in enumerate([out1, out2]):
            _, _, H_out, W_out = out.shape
            out = out.view(1, 3, 5 + 4, H_out, W_out).permute(0, 1, 3, 4, 2)

            xy = torch.sigmoid(out[..., 0:2])
            wh = torch.exp(out[..., 2:4])
            conf = torch.sigmoid(out[..., 4])
            cls_probs = torch.sigmoid(out[..., 5:])

            scale_anchors = ANCHORS[scale_idx]

            for a in range(3):
                for y in range(H_out):
                    for x in range(W_out):
                        score = conf[0, a, y, x].item()
                        class_probs = cls_probs[0, a, y, x]
                        class_id = torch.argmax(class_probs).item()
                        class_score = class_probs[class_id].item()

                        final_score = score * class_score
                        # Using 0.25 confidence threshold
                        if final_score > 0.25:
                            aw, ah = scale_anchors[a]
                            gx = (xy[0, a, y, x, 0].item() + x) / W_out
                            gy = (xy[0, a, y, x, 1].item() + y) / H_out
                            gw = (wh[0, a, y, x, 0].item() * aw) / 416
                            gh = (wh[0, a, y, x, 1].item() * ah) / 416

                            # Calculate raw bounding boxes
                            x1 = int((gx - gw/2) * W)
                            y1 = int((gy - gh/2) * H)
                            x2 = int((gx + gw/2) * W)
                            y2 = int((gy + gh/2) * H)

                            # Clamp to image size
                            x1 = max(0, min(W-1, x1))
                            y1 = max(0, min(H-1, y1))
                            x2 = max(0, min(W-1, x2))
                            y2 = max(0, min(H-1, y2))

                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "score": final_score,
                                "class": class_id
                            })

        # Apply NMS
        detections = nms(detections, iou_threshold=0.45)
        if len(detections) > 0:
            stats["detected"] += 1

        # Create card
        card = make_card(img_path, label_path, detections)
        if card is not None:
            out_path = OUT_DIR / f"vis_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), card, [cv2.IMWRITE_JPEG_QUALITY, 92])
            all_cards.append(card)

            cls_str = ", ".join(
                f"{CLASS_NAMES[d['class']]}({d['score']:.2f})" for d in detections
            ) if detections else "-- none --"
            print(
                f"  [{idx+1:2d}/{num_to_process}] {img_path.name[:35]:<35} -> {cls_str}")

    # Build and save a single large grid image of all cards (e.g. 5x6 grid)
    if all_cards:
        COLS = 3
        ROWS = (len(all_cards) + COLS - 1) // COLS
        max_w = max(c.shape[1] for c in all_cards)
        max_h = max(c.shape[0] for c in all_cards)
        padded = []
        for c in all_cards:
            h, w = c.shape[:2]
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[:h, :w] = c
            padded.append(canvas)
        while len(padded) % COLS != 0:
            padded.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
        rows_imgs = []
        for i in range(ROWS):
            row = np.hstack(padded[i*COLS:(i+1)*COLS])
            rows_imgs.append(row)
        grid = np.vstack(rows_imgs)
        grid_path = OUT_DIR / "grid_all.jpg"
        cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"\n[INFO] Grid saved -> {grid_path}")

    # Summary
    print()
    print("=" * 60)
    print("  YOLOv4-tiny Inference Summary")
    print("=" * 60)
    print(f"  Processed images  : {stats['total']}")
    print(
        f"  Detected images   : {stats['detected']} ({stats['detected']/stats['total']*100:.1f}%)")
    print(f"  Output folder     : {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
