# -*- coding: utf-8 -*-
"""
prepare_dataset.py
==================
將 Roboflow 匯出的 YOLO Darknet 格式資料集，整理成 pytorch-YOLOv4 訓練所需的格式。

Roboflow 匯出的 YOLO Darknet 格式結構：
    dataset/
    ├── train/
    │   ├── images/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img001.txt   (YOLO 格式: class_id cx cy w h)
    │       └── ...
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

pytorch-YOLOv4 的 train.txt 格式：
    image_path x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...

本腳本會：
1. 讀取 YOLO 格式標記 (cx, cy, w, h normalized)
2. 轉換成 xyxy 像素座標
3. 產生 train.txt 和 test.txt
"""

import os
import sys
import glob
import random
from PIL import Image

# ===== 設定 =====
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
CFG_DIR = os.path.join(PROJECT_DIR, "cfg")

TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VALID_IMG_DIR = os.path.join(DATASET_DIR, "valid", "images")
VALID_LBL_DIR = os.path.join(DATASET_DIR, "valid", "labels")
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test", "images")
TEST_LBL_DIR = os.path.join(DATASET_DIR, "test", "labels")

TRAIN_TXT = os.path.join(CFG_DIR, "train.txt")
TEST_TXT = os.path.join(CFG_DIR, "test.txt")

# 支援的圖片格式
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """將 YOLO 正規化座標 (cx, cy, w, h) 轉換為像素 xyxy 座標"""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    # 確保不超出圖片邊界
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2


def process_split(img_dir, lbl_dir, output_txt):
    """處理某一個 split (train/valid/test)"""
    if not os.path.exists(img_dir):
        print(f"  [跳過] 圖片資料夾不存在: {img_dir}")
        return 0

    lines = []
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])

    skipped = 0
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_file)

        if not os.path.exists(lbl_path):
            skipped += 1
            continue

        # 讀取圖片尺寸
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"  [錯誤] 無法讀取圖片 {img_path}: {e}")
            skipped += 1
            continue

        # 讀取標記
        bboxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                bboxes.append(f"{x1},{y1},{x2},{y2},{class_id}")

        if bboxes:
            # 使用絕對路徑
            abs_img_path = os.path.abspath(img_path)
            line = abs_img_path + " " + " ".join(bboxes)
            lines.append(line)

    # 寫入檔案
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    if skipped > 0:
        print(f"  [警告] {skipped} 張圖片沒有對應的標記檔，已跳過")

    return len(lines)


def main():
    print("=" * 60)
    print("  YOLOv4 資料集準備工具")
    print("=" * 60)
    print(f"\n專案目錄: {PROJECT_DIR}")
    print(f"資料集目錄: {DATASET_DIR}")

    # 檢查資料集是否存在
    if not os.path.exists(DATASET_DIR):
        print(f"\n[錯誤] 找不到資料集目錄: {DATASET_DIR}")
        print("請先將 Roboflow 匯出的資料集解壓到 dataset/ 資料夾中")
        print("結構應該像這樣:")
        print("  dataset/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  ├── valid/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  └── test/")
        print("      ├── images/")
        print("      └── labels/")
        sys.exit(1)

    # 處理各個 split
    print("\n--- 處理 Train 資料集 ---")
    n_train = process_split(TRAIN_IMG_DIR, TRAIN_LBL_DIR, TRAIN_TXT)
    print(f"  ✅ 產生 {n_train} 筆訓練資料 -> {TRAIN_TXT}")

    # valid + test 合併寫入 test.txt
    print("\n--- 處理 Valid/Test 資料集 ---")
    n_valid = process_split(VALID_IMG_DIR, VALID_LBL_DIR, TEST_TXT)
    print(f"  ✅ 產生 {n_valid} 筆驗證資料 -> {TEST_TXT}")

    # 總結
    print("\n" + "=" * 60)
    print(f"  總計: {n_train} 訓練 + {n_valid} 驗證 = {n_train + n_valid} 筆資料")
    print("=" * 60)

    if n_train == 0:
        print("\n⚠️  訓練資料為 0！")
        print("請確認你已經將標記好的圖片放入 dataset/train/images/ 和 dataset/train/labels/")
    else:
        print("\n✅ 資料集準備完成！接下來可以執行 train.py 開始訓練")


if __name__ == "__main__":
    main()
