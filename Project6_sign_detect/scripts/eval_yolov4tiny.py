# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

# Append scripts folder to path
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))

from train_pytorch_yolov4tiny import YoloV4Tiny, load_darknet_weights, ANCHORS

# Paths
OBJ_DIR = PROJECT_ROOT / "obj"
WEIGHTS_PATH = PROJECT_ROOT / "jetbot_deploy" / "yolov4-tiny-416.weights"

CLASS_NAMES = ["stop", "rail", "pedestrian", "blocked"]

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

def calculate_iou(box1, box2):
    """Calculate IoU of two boxes [x1, y1, x2, y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)

def main():
    print("=" * 60)
    print("  YOLOv4-tiny Model Quantitative Evaluation")
    print("=" * 60)

    # 1. Load weights
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Weights file not found: {WEIGHTS_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV4Tiny(num_classes=4).to(device)
    success = load_darknet_weights(model, str(WEIGHTS_PATH))
    if not success:
        print("[ERROR] Failed to load weights.")
        return
    model.eval()

    # 2. Get images
    all_imgs = sorted(list(OBJ_DIR.glob("*.jpg")) + list(OBJ_DIR.glob("*.png")))
    if not all_imgs:
        print("[ERROR] No images found.")
        return

    # Metrics variables: TP, FP, FN per class
    # class_id: {TP: int, FP: int, FN: int}
    metrics = {i: {"TP": 0, "FP": 0, "FN": 0, "GT": 0} for i in range(4)}

    print(f"[INFO] Evaluating model on {len(all_imgs)} images...")
    
    CONF_TH = 0.3  # Confidence threshold for evaluation
    IOU_TH = 0.5   # IoU threshold for matching

    for idx, img_path in enumerate(all_imgs):
        # Load ground truths
        gts = []
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    gts.append({"class": cls_id, "bbox": [cx, cy, bw, bh]})
                    metrics[cls_id]["GT"] += 1

        # Load & preprocess image
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_in, (416, 416))
        img_t = torch.from_numpy(img_in).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        # Run inference
        with torch.no_grad():
            out1, out2 = model(img_t)

        # Decode detections
        detections = []
        for scale_idx, out in enumerate([out1, out2]):
            _, _, H_out, W_out = out.shape
            out = out.view(1, 3, 9, H_out, W_out).permute(0, 1, 3, 4, 2)

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
                        if final_score > CONF_TH:
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

                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "score": final_score,
                                "class": class_id
                            })

        # Apply NMS
        detections = nms(detections, iou_threshold=0.45)

        # Convert GT normalized boxes to pixel coordinates
        gt_pixel_boxes = []
        for gt in gts:
            cx, cy, bw, bh = gt["bbox"]
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            gt_pixel_boxes.append({"class": gt["class"], "bbox": [x1, y1, x2, y2], "matched": False})

        # Match predictions to GTs
        for det in detections:
            det_box = det["bbox"]
            det_cls = det["class"]

            best_iou = -1
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_pixel_boxes):
                if gt["class"] == det_cls and not gt["matched"]:
                    iou = calculate_iou(det_box, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= IOU_TH:
                # Matched! It is a True Positive (TP)
                metrics[det_cls]["TP"] += 1
                gt_pixel_boxes[best_gt_idx]["matched"] = True
            else:
                # No match! It is a False Positive (FP)
                metrics[det_cls]["FP"] += 1

        # Any unmatched GT is a False Negative (FN)
        for gt in gt_pixel_boxes:
            if not gt["matched"]:
                metrics[gt["class"]]["FN"] += 1

    print("\n" + "=" * 60)
    print("  Quantitative Metrics Results")
    print("=" * 60)
    print(f"{'Class Name':<12} | {'GT':<4} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8}")
    print("-" * 75)

    sum_p, sum_r, sum_f1 = 0, 0, 0
    total_gts = 0

    for cls_id in range(4):
        m = metrics[cls_id]
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        sum_p += precision
        sum_r += recall
        sum_f1 += f1
        total_gts += m["GT"]

        name = CLASS_NAMES[cls_id]
        print(f"{name:<12} | {m['GT']:<4d} | {tp:<4d} | {fp:<4d} | {fn:<4d} | {precision:<9.4f} | {recall:<6.4f} | {f1:<8.4f}")

    print("-" * 75)
    mean_p = sum_p / 4
    mean_r = sum_r / 4
    mean_f1 = sum_f1 / 4
    print(f"{'mAP@0.5/Avg':<12} | {total_gts:<4d} | {'-':<4} | {'-':<4} | {'-':<4} | {mean_p:<9.4f} | {mean_r:<6.4f} | {mean_f1:<8.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
