# -*- coding: utf-8 -*-
"""
evaluate_test_detail.py
========================
Generates a detailed image-by-image analysis report comparing the V2 model predictions 
to the Ground Truth annotations for all 15 Test Set images.
"""

import sys
import random
from pathlib import Path
import cv2
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.append(str(SCRIPTS_DIR))

from train_pytorch_yolov4tiny_v2 import YoloV4Tiny, load_darknet_weights, ANCHORS, nms, calculate_iou

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBJ_DIR = PROJECT_ROOT / "obj"
WEIGHTS_PATH = PROJECT_ROOT / "jetbot_deploy" / "yolov4-tiny-416.weights"
OUT_REPORT = PROJECT_ROOT / "runs" / "predict_vis_yolov4tiny_v2" / "test_detailed_analysis.md"
ART_REPORT = Path("C:/Users/andy8/.gemini/antigravity/brain/a7848b0e-88be-4e87-b0ff-7df17f12bfb9/test_set_detailed_analysis.md")

CLASS_NAMES = ["stop", "rail", "pedestrian", "blocked"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV4Tiny(num_classes=4).to(device)
    
    if not load_darknet_weights(model, str(WEIGHTS_PATH)):
        print("[ERROR] Failed to load weights.")
        return
    model.eval()

    all_imgs = sorted(list(OBJ_DIR.glob("*.jpg")) + list(OBJ_DIR.glob("*.png")))
    random.seed(42)
    random.shuffle(all_imgs)

    n_total = len(all_imgs)
    n_val = int(n_total * 0.10)
    n_test = int(n_total * 0.10)
    n_train = n_total - n_val - n_test
    test_imgs = all_imgs[n_train + n_val:]

    md_lines = []
    md_lines.append("# YOLOv4-tiny V2 測試集逐張影像詳細對照報告")
    md_lines.append("\n本報告針對測試集共 15 張圖片，提供手動標註 (GT) 與 V2 模型預測結果的**逐一比對分析**。所有影像皆在信心度過濾門檻 **`0.30`** 下執行推論。")
    md_lines.append("\n---\n")

    for idx, img_path in enumerate(test_imgs):
        label_path = img_path.with_suffix(".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        
        # Load GT
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

        # Run inference
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
                        
                        if final_score > 0.3:
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

        # Match logic
        matching_results = []
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
                matching_results.append({
                    "pred_class": CLASS_NAMES[det_cls],
                    "score": det["score"],
                    "pred_box": det_box,
                    "gt_box": gts[best_gt_idx]["bbox"],
                    "status": "✅ 成功辨識 (True Positive)",
                    "iou": best_iou
                })
                gts[best_gt_idx]["matched"] = True
            else:
                matching_results.append({
                    "pred_class": CLASS_NAMES[det_cls],
                    "score": det["score"],
                    "pred_box": det_box,
                    "gt_box": "N/A",
                    "status": "⚠️ 虛警/背景多餘框 (False Positive)",
                    "iou": best_iou if best_iou > 0 else 0.0
                })

        for gt in gts:
            if not gt["matched"]:
                matching_results.append({
                    "pred_class": "N/A",
                    "score": 0.0,
                    "pred_box": "N/A",
                    "gt_box": gt["bbox"],
                    "status": "❌ 漏檢 (False Negative) - 未偵測到該物體",
                    "iou": 0.0
                })

        # Format markdown for this image
        md_lines.append(f"### 📷 Image {idx+1}: `{img_path.name}`")
        md_lines.append(f"* **影像尺寸 (H x W)**: {H} x {W}")
        md_lines.append("\n| 狀態 (Status) | 預測類別 | 信心度 | 預測邊界框 `[x1, y1, x2, y2]` | 真實邊界框 `[x1, y1, x2, y2]` | 重疊率 (IoU) |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        
        if not matching_results:
            md_lines.append("| 💡 無任何目標 | - | - | - | - | - |")
        else:
            for res in matching_results:
                score_str = f"{res['score']:.4f}" if res['score'] > 0 else "-"
                iou_str = f"{res['iou']:.4f}" if res['iou'] > 0 else "-"
                pred_box_str = str(res['pred_box']) if isinstance(res['pred_box'], list) else res['pred_box']
                gt_box_str = str(res['gt_box']) if isinstance(res['gt_box'], list) else res['gt_box']
                md_lines.append(f"| {res['status']} | {res['pred_class']} | {score_str} | `{pred_box_str}` | `{gt_box_str}` | {iou_str} |")
        md_lines.append("\n")

    report_content = "\n".join(md_lines)
    
    # Save files
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    ART_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(ART_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[INFO] Detailed markdown analysis generated at:\n  - {OUT_REPORT}\n  - {ART_REPORT}")

if __name__ == "__main__":
    main()
