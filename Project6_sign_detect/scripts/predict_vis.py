# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import random
import numpy as np
import cv2
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Inference visualization on training images using best.pt
- Load clean training images (no labels drawn)
- Run YOLO11n best.pt inference
- Display: original | detected side-by-side
- Save results to runs/predict_vis/
"""


# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "_SignDetection.yolo26"
IMG_DIR = DATASET_DIR / "train" / "images"
LBL_DIR = DATASET_DIR / "train" / "labels"
BEST_MODEL = PROJECT_ROOT / "runs" / "sign_detection" / "weights" / "best.pt"
OUT_DIR = PROJECT_ROOT / "runs" / "predict_vis"

CLASS_NAMES = ["blocked", "pedestrian", "rail", "stop"]

# Color per class: BGR
CLASS_COLORS = {
    0: (80,  100, 255),   # blocked    -> red-ish
    1: (255, 200,  80),   # pedestrian -> cyan-ish
    2: (80,  255, 220),   # rail       -> yellow-ish
    3: (100, 230, 120),   # stop       -> green-ish
}

# ── Draw GT boxes from YOLO label file ─────────────────────────────


def draw_gt(img, label_path, color=(180, 180, 180)):
    """Draw ground-truth boxes (grey dashed outline) onto img copy."""
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
            # dashed box via segments
            for (sx1, sy1, sx2, sy2) in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]:
                pts = np.linspace(0, 1, 20)
                for i in range(0, len(pts)-1, 2):
                    px1 = int(sx1 + pts[i] * (sx2-sx1))
                    py1 = int(sy1 + pts[i] * (sy2-sy1))
                    px2 = int(sx1 + pts[i+1] * (sx2-sx1))
                    py2 = int(sy1 + pts[i+1] * (sy2-sy1))
                    cv2.line(out, (px1, py1), (px2, py2), c, 1)
            label = CLASS_NAMES[cls_id] if cls_id < len(
                CLASS_NAMES) else str(cls_id)
            cv2.putText(out, f"GT:{label}", (x1, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


# ── Draw PRED boxes ────────────────────────────────────────────────
def draw_pred(img, boxes_list):
    """Draw prediction boxes (solid colored) onto img copy."""
    out = img.copy()
    for cls_id, conf, x1, y1, x2, y2 in boxes_list:
        c = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = CLASS_NAMES[cls_id] if cls_id < len(
            CLASS_NAMES) else str(cls_id)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+4, y1), c, -1)
        cv2.putText(out, text, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return out


# ── Make side-by-side comparison card ─────────────────────────────
def make_card(img_path, label_path, pred_boxes, target_h=416):
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        return None

    # Resize to target height while keeping aspect ratio
    h, w = img_orig.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    img_r = cv2.resize(img_orig, (new_w, target_h))

    # Rescale pred boxes too
    boxes_scaled = []
    for cls_id, conf, x1, y1, x2, y2 in pred_boxes:
        boxes_scaled.append((cls_id, conf,
                             int(x1*scale), int(y1*scale),
                             int(x2*scale), int(y2*scale)))

    panel_clean = img_r.copy()                      # left  : clean (no box)
    panel_gt = draw_gt(img_r, label_path)        # middle: GT dashed
    panel_pred = draw_pred(img_r, boxes_scaled)    # right : prediction

    # Header bar
    def add_header(panel, text, color):
        p = panel.copy()
        cv2.rectangle(p, (0, 0), (new_w, 26), color, -1)
        cv2.putText(p, text, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return p

    panel_clean = add_header(panel_clean, "Original (clean)",  (60,  60,  60))
    panel_gt = add_header(panel_gt,    "Ground Truth (GT)", (30,  100, 30))
    panel_pred = add_header(panel_pred,  "Prediction (best.pt)", (30, 30, 140))

    # Divider line
    div = np.full((target_h, 3, 3), 40, dtype=np.uint8)
    card = np.hstack([panel_clean, div, panel_gt, div, panel_pred])
    return card


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Visualize best.pt on training images")
    parser.add_argument("--n",     type=int,   default=20,
                        help="Number of images to process (default 20, 0=all)")
    parser.add_argument("--conf",  type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed for image selection")
    parser.add_argument("--show",  action="store_true",
                        help="Pop-up window for each image (requires display)")
    parser.add_argument("--grid",  action="store_true",
                        help="Also save a single large grid image of all cards")
    args = parser.parse_args()

    # ── Check model ──────────────────────────────────────────────
    if not BEST_MODEL.exists():
        print(f"[ERROR] Model not found: {BEST_MODEL}")
        print("        Run training first: python train_yolo.py --mode train")
        return

    # ── Load YOLO ────────────────────────────────────────────────
    from ultralytics import YOLO
    print(f"[INFO] Loading model: {BEST_MODEL}")
    model = YOLO(str(BEST_MODEL))
    print(f"[INFO] Model loaded. Classes: {CLASS_NAMES}")

    # ── Pick images ──────────────────────────────────────────────
    all_imgs = sorted(IMG_DIR.glob("*.jpg"))
    if not all_imgs:
        print(f"[ERROR] No images found in {IMG_DIR}")
        return

    random.seed(args.seed)
    if args.n == 0 or args.n >= len(all_imgs):
        selected = all_imgs
    else:
        selected = random.sample(all_imgs, args.n)
    selected = sorted(selected)

    print(
        f"[INFO] Processing {len(selected)} / {len(all_imgs)} images  (conf={args.conf})")

    # ── Run batch inference ───────────────────────────────────────
    img_paths = [str(p) for p in selected]
    results = model.predict(
        source=img_paths,
        imgsz=416,
        conf=args.conf,
        iou=0.45,
        verbose=False,
        device="",
    )

    # ── Build cards & save ────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "detected": 0, "correct_cls": 0}
    all_cards = []

    for img_path, result in zip(selected, results):
        label_path = LBL_DIR / (img_path.stem + ".txt")
        stats["total"] += 1

        # Parse predictions
        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            stats["detected"] += 1
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                pred_boxes.append((cls_id, conf, x1, y1, x2, y2))

        # Check if at least one pred class matches GT
        if label_path.exists() and pred_boxes:
            gt_classes = set()
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.add(int(parts[0]))
            pred_classes = {b[0] for b in pred_boxes}
            if gt_classes & pred_classes:
                stats["correct_cls"] += 1

        # Build and save card
        card = make_card(img_path, label_path, pred_boxes)
        if card is None:
            continue

        out_path = OUT_DIR / f"vis_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), card, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if args.show:
            cv2.imshow("Detection Result (press any key)", card)
            cv2.waitKey(0)

        all_cards.append(card)

        # Console log
        cls_str = ", ".join(
            f"{CLASS_NAMES[b[0]]}({b[1]:.2f})" for b in pred_boxes
        ) if pred_boxes else "-- none --"
        print(
            f"  [{stats['total']:3d}/{len(selected)}] {img_path.name[:45]:<45} -> {cls_str}")

    if args.show:
        cv2.destroyAllWindows()

    # ── Optional grid image ───────────────────────────────────────
    if args.grid and all_cards:
        COLS = 3
        ROWS = (len(all_cards) + COLS - 1) // COLS
        # Pad to uniform width
        max_w = max(c.shape[1] for c in all_cards)
        max_h = max(c.shape[0] for c in all_cards)
        padded = []
        for c in all_cards:
            h, w = c.shape[:2]
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[:h, :w] = c
            padded.append(canvas)
        # Fill empty slots
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

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Inference Summary")
    print("=" * 60)
    print(f"  Images processed  : {stats['total']}")
    print(
        f"  Images w/ detects : {stats['detected']}  ({stats['detected']/max(stats['total'],1)*100:.1f}%)")
    print(
        f"  Class match w/ GT : {stats['correct_cls']}  ({stats['correct_cls']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Output folder     : {OUT_DIR}")
    print("=" * 60)
    print()
    print("  Each output file has 3 panels side by side:")
    print("    LEFT   : Original clean image (no boxes)")
    print("    MIDDLE : Ground Truth (GT dashed boxes)")
    print("    RIGHT  : Model Prediction (solid boxes)")


if __name__ == "__main__":
    main()
