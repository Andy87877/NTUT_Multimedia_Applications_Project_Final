# -*- coding: utf-8 -*-
"""
prepare_dataset_yolo26.py
=========================
將 Roboflow 匯出的 YOLOv4-PyTorch 格式資料集，
轉換成 Ultralytics YOLO26 所需的標準 YOLO 格式。

Roboflow YOLOv4-PyTorch 格式 (_annotations.txt):
  image.jpg x1,y1,x2,y2,cls x1,y1,x2,y2,cls ...

YOLO26 標準格式 (每張圖對應一個 .txt):
  cls cx cy w h  (全部正規化為 0~1)

同時自動產生 data.yaml 供 YOLO26 訓練使用。

使用方式:
  python prepare_dataset_yolo26.py
"""

import os
import shutil
from PIL import Image

# ============================================================
# 路徑設定
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOFLOW_DIR = os.path.join(PROJECT_DIR, "project4_yolo_v3.v1i.yolov4pytorch")

# YOLO26 資料集輸出目錄
OUTPUT_DIR = os.path.join(PROJECT_DIR, "dataset_yolo26")

# 類別名稱 (與 cfg/obj.names 一致)
CLASS_NAMES = ["Car"]


def convert_split(split_name):
    """
    轉換單一資料分割 (train / valid / test)。
    
    讀取 _annotations.txt，將每張圖的 xyxy 像素座標
    轉成 YOLO 格式 (cls cx cy w h)，正規化至 [0, 1]。
    圖片複製到 images/ 子目錄，標籤寫入 labels/ 子目錄。
    """
    src_dir = os.path.join(ROBOFLOW_DIR, split_name)
    ann_file = os.path.join(src_dir, "_annotations.txt")

    if not os.path.exists(ann_file):
        print(f"  [跳過] 找不到 {ann_file}")
        return 0

    img_out_dir = os.path.join(OUTPUT_DIR, split_name, "images")
    lbl_out_dir = os.path.join(OUTPUT_DIR, split_name, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    count = 0
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(" ")
            img_name = parts[0]
            bboxes_raw = parts[1:]  # 每個元素: "x1,y1,x2,y2,cls"

            # 讀取圖片尺寸
            img_path = os.path.join(src_dir, img_name)
            if not os.path.exists(img_path):
                print(f"  [警告] 圖片不存在: {img_path}")
                continue

            img = Image.open(img_path)
            img_w, img_h = img.size

            # 複製圖片
            shutil.copy2(img_path, os.path.join(img_out_dir, img_name))

            # 轉換標籤
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(lbl_out_dir, label_name)

            with open(label_path, "w", encoding="utf-8") as lf:
                for bbox_str in bboxes_raw:
                    if not bbox_str.strip():
                        continue
                    vals = bbox_str.split(",")
                    if len(vals) < 5:
                        continue

                    x1 = float(vals[0])
                    y1 = float(vals[1])
                    x2 = float(vals[2])
                    y2 = float(vals[3])
                    cls_id = int(vals[4])

                    # 轉換: xyxy 像素 → cx cy w h 正規化
                    cx = ((x1 + x2) / 2.0) / img_w
                    cy = ((y1 + y2) / 2.0) / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h

                    # 確保數值在 [0, 1] 範圍內
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))

                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            count += 1

    return count


def create_data_yaml():
    """
    產生 YOLO26 訓練所需的 data.yaml 設定檔。
    """
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")

    # 使用絕對路徑確保 Ultralytics 能正確找到資料
    abs_output = os.path.abspath(OUTPUT_DIR).replace("\\", "/")

    content = f"""# YOLO26 車輛偵測資料集設定
# 由 prepare_dataset_yolo26.py 自動產生

path: {abs_output}
train: train/images
val: valid/images
test: test/images

names:
"""
    for i, name in enumerate(CLASS_NAMES):
        content += f"  {i}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  data.yaml 已產生: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("  YOLO26 資料集準備工具")
    print("=" * 60)
    print(f"\n來源目錄: {ROBOFLOW_DIR}")
    print(f"輸出目錄: {OUTPUT_DIR}\n")

    # 轉換各資料分割
    for split in ["train", "valid", "test"]:
        print(f"正在轉換 [{split}] ...")
        n = convert_split(split)
        print(f"  完成: {n} 張圖片\n")

    # 產生 data.yaml
    print("正在產生 data.yaml ...")
    yaml_path = create_data_yaml()

    print(f"\n{'=' * 60}")
    print("  全部完成！")
    print(f"  資料集路徑: {OUTPUT_DIR}")
    print(f"  設定檔路徑: {yaml_path}")
    print(f"\n  接下來請執行:")
    print(f"    python train_local_yolo26.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
