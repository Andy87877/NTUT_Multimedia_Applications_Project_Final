# -*- coding: utf-8 -*-
import sys
import random
from pathlib import Path
import cv2
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.append(str(SCRIPTS_DIR))

from train_pytorch_yolov4tiny_v2 import YoloV4Tiny, load_darknet_weights, ANCHORS, nms, calculate_iou

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBJ_DIR = PROJECT_ROOT / "obj"
WEIGHTS_PATH = PROJECT_ROOT / "jetbot_deploy" / "yolov4-tiny-416.weights"

def evaluate_at_threshold(model, test_imgs, device, conf_th):
    tps = 0
    fps = 0
    fns = 0
    gts_total = 0

    for img_path in test_imgs:
        label_path = img_path.with_suffix(".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        
        gts = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = int((cx - bw / 2) * W)
                        y1 = int((cy - bh / 2) * H)
                        x2 = int((cx + bw / 2) * W)
                        y2 = int((cy + bh / 2) * H)
                        gts.append({"class": cls_id, "bbox": [x1, y1, x2, y2], "matched": False})
                        gts_total += 1

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_in, (416, 416))
        img_t = torch.from_numpy(img_in).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            out1, out2 = model(img_t)

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
                        
                        if final_score > conf_th:
                            aw, ah = scale_anchors[a]
                            gx = (xy[0, a, y, x, 0].item() + x) / W_out
                            gy = (xy[0, a, y, x, 1].item() + y) / H_out
                            gw = (wh[0, a, y, x, 0].item() * aw) / 416
                            gh = (wh[0, a, y, x, 1].item() * ah) / 416

                            x1 = max(0, min(W-1, int((gx - gw/2) * W)))
                            y1 = max(0, min(H-1, int((gy - gh/2) * H)))
                            x2 = max(0, min(W-1, int((gx + gw/2) * W)))
                            y2 = max(0, min(H-1, int((gy + gh/2) * H)))

                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "score": final_score,
                                "class": class_id
                            })

        detections = nms(detections, iou_threshold=0.45)

        for det in detections:
            det_box = det["bbox"]
            det_cls = det["class"]

            best_iou = -1
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts):
                if gt["class"] == det_cls and not gt["matched"]:
                    iou = calculate_iou(det_box, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= 0.5:
                tps += 1
                gts[best_gt_idx]["matched"] = True
            else:
                fps += 1

        for gt in gts:
            if not gt["matched"]:
                fns += 1

    precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return tps, fps, fns, precision, recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV4Tiny(num_classes=4).to(device)
    load_darknet_weights(model, str(WEIGHTS_PATH))
    model.eval()

    all_imgs = sorted(list(OBJ_DIR.glob("*.jpg")) + list(OBJ_DIR.glob("*.png")))
    random.seed(42)
    random.shuffle(all_imgs)

    n_total = len(all_imgs)
    n_val = int(n_total * 0.10)
    n_test = int(n_total * 0.10)
    n_train = n_total - n_val - n_test
    test_imgs = all_imgs[n_train + n_val:]

    print("\n" + "=" * 60)
    print("  Threshold Sweep Analysis on Test Set (15 images)")
    print("=" * 60)
    print(f"{'Threshold':<10} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8}")
    print("-" * 65)

    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tp, fp, fn, p, r, f1 = evaluate_at_threshold(model, test_imgs, device, th)
        print(f"{th:<10.2f} | {tp:<4d} | {fp:<4d} | {fn:<4d} | {p:<9.4f} | {r:<6.4f} | {f1:<8.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
