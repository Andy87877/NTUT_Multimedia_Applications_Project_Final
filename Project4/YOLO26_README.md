# YOLO26 車輛偵測使用說明書

> **YOLO26** 是 Ultralytics 於 2026 年 1 月發布的最新物件偵測模型，具備 NMS-free 端對端推理、ProgLoss + STAL 損失函數等新特性，在精度與速度上均優於前代 YOLO11 / YOLOv8。

---

## 目錄

1. [環境安裝](#1-環境安裝)
2. [專案檔案結構](#2-專案檔案結構)
3. [資料集準備](#3-資料集準備)
4. [模型訓練](#4-模型訓練)
5. [圖片偵測](#5-圖片偵測)
6. [影片偵測](#6-影片偵測)
7. [YOLO26 vs YOLOv4-tiny 差異比較](#7-yolo26-vs-yolov4-tiny-差異比較)
8. [常見問題 FAQ](#8-常見問題-faq)

---

## 1. 環境安裝

### 1.1 前置需求

| 項目 | 版本需求 |
|------|---------|
| Python | >= 3.8 |
| PyTorch | >= 1.8 |
| CUDA (選配) | 11.x 或 12.x |

### 1.2 安裝 Ultralytics

```bash
pip install ultralytics
```

若要升級到最新版：

```bash
pip install -U ultralytics
```

### 1.3 驗證安裝

```bash
python -c "from ultralytics import YOLO; print('YOLO26 ready!')"
```

若有 NVIDIA GPU，可同時驗證 CUDA：

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 2. 專案檔案結構

```
Project4/
├── prepare_dataset_yolo26.py   # Step 1: 資料集格式轉換工具
├── train_local_yolo26.py       # Step 2: YOLO26 訓練腳本
├── detect_image_yolo26.py      # Step 3a: 圖片偵測腳本
├── detect_video_yolo26.py      # Step 3b: 影片偵測腳本
│
├── project4_YoloV4.v1i.yolov4pytorch/   # Roboflow 匯出的原始資料集
│   ├── train/
│   │   ├── _annotations.txt
│   │   └── *.jpg
│   ├── valid/
│   └── test/
│
├── dataset_yolo26/             # (自動產生) YOLO26 格式資料集
│   ├── data.yaml               # 資料集設定檔
│   ├── train/
│   │   ├── images/
│   │   └── labels/             # 標準 YOLO 格式 (cls cx cy w h)
│   ├── valid/
│   └── test/
│
└── runs_yolo26/                # (自動產生) 訓練結果
    └── car_detect/
        ├── weights/
        │   ├── best.pt         # 最佳權重
        │   └── last.pt         # 最後一輪權重
        ├── results.png         # 訓練曲線圖
        └── ...
```

---

## 3. 資料集準備

### 3.1 為什麼需要轉換？

| 格式 | 說明 | 範例 |
|------|------|------|
| Roboflow YOLOv4-PyTorch | `x1,y1,x2,y2,cls` (像素絕對值) | `247,167,263,186,0` |
| **YOLO26 標準格式** | `cls cx cy w h` (正規化 0~1) | `0 0.498 0.617 0.031 0.066` |

YOLO26 要求每張圖片對應一個 `.txt` 標籤檔，座標必須正規化到 `[0, 1]` 範圍。

### 3.2 執行轉換

```bash
python prepare_dataset_yolo26.py
```

執行後會：
1. 讀取 `project4_YoloV4.v1i.yolov4pytorch/` 中的 `_annotations.txt`
2. 將 xyxy 像素座標轉換為 YOLO 格式 (cx, cy, w, h 正規化)
3. 圖片複製到 `dataset_yolo26/{split}/images/`
4. 標籤寫入 `dataset_yolo26/{split}/labels/`
5. 自動產生 `dataset_yolo26/data.yaml`

### 3.3 驗證轉換結果

轉換完成後，可以打開任一 `.txt` 檔確認格式：

```bash
# 查看標籤檔 (數值應在 0~1 之間)
type dataset_yolo26\train\labels\screenshots_4_17_jpg.rf.7471f3352814a1dcc6e6ef92037c4bf5.txt
```

預期輸出類似：
```
0 0.498047 0.616783 0.031250 0.066434
0 0.539063 0.594406 0.031250 0.062937
```

---

## 4. 模型訓練

### 4.1 基本用法

```bash
python train_local_yolo26.py
```

預設參數：
- **模型**: `yolo26n.pt` (Nano，最輕量)
- **輪數**: 100 epochs
- **影像尺寸**: 640×640
- **批次大小**: 16

### 4.2 自訂參數

```bash
# 使用 Small 模型 + 更多輪數
python train_local_yolo26.py --model yolo26s.pt --epochs 200

# GPU 記憶體不足時降低 batch size
python train_local_yolo26.py --batch 8

# 指定使用 CPU
python train_local_yolo26.py --device cpu

# 從中斷處繼續訓練
python train_local_yolo26.py --resume
```

### 4.3 可用的 YOLO26 模型大小

| 模型 | 參數量 | 速度 | 精度 | 適用場景 |
|------|--------|------|------|---------|
| `yolo26n.pt` | 最少 | ⚡ 最快 | ★★★ | 即時偵測 / 邊緣裝置 |
| `yolo26s.pt` | 少 | ⚡ 快 | ★★★★ | 一般應用 |
| `yolo26m.pt` | 中 | 中等 | ★★★★★ | 平衡選擇 |
| `yolo26l.pt` | 多 | 較慢 | ★★★★★★ | 高精度需求 |
| `yolo26x.pt` | 最多 | 最慢 | ★★★★★★★ | 最高精度 |

> **建議**: GTX 1650 (4GB) 使用 `yolo26n.pt` 或 `yolo26s.pt`

### 4.4 訓練完成後

訓練結果儲存在 `runs_yolo26/car_detect/`：
- `weights/best.pt` → **最佳權重** (用於推理)
- `weights/last.pt` → 最後一輪權重
- `results.png` → Loss / mAP 曲線圖
- `confusion_matrix.png` → 混淆矩陣

---

## 5. 圖片偵測

### 5.1 基本用法

```bash
python detect_image_yolo26.py --image path/to/image.jpg
```

### 5.2 完整參數

```bash
python detect_image_yolo26.py \
    --image test_photo.jpg \
    --weights runs_yolo26/car_detect/weights/best.pt \
    --conf 0.4 \
    --iou 0.6 \
    --output results/
```

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--image` | 輸入圖片路徑 | (必填) |
| `--weights` | 權重檔路徑 | `runs_yolo26/car_detect/weights/best.pt` |
| `--conf` | 信心度門檻 | `0.4` |
| `--iou` | IoU 門檻 | `0.6` |
| `--output` | 輸出目錄 | 與輸入圖片同目錄 |

### 5.3 輸出結果

偵測結果圖片自動儲存為 `原檔名_yolo26.jpg`，圖上會標註：
- 偵測框 (bounding box)
- 類別名稱
- 信心度分數

---

## 6. 影片偵測

### 6.1 基本用法

```bash
python detect_video_yolo26.py --video "TAIPEI TAIWAN DRIVE AROUND.mp4"
```

### 6.2 完整參數

```bash
python detect_video_yolo26.py \
    --video input.mp4 \
    --weights runs_yolo26/car_detect/weights/best.pt \
    --conf 0.4 \
    --output output_detected.mp4
```

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--video` | 輸入影片路徑 | (必填) |
| `--weights` | 權重檔路徑 | `runs_yolo26/car_detect/weights/best.pt` |
| `--conf` | 信心度門檻 | `0.4` |
| `--iou` | IoU 門檻 | `0.6` |
| `--output` | 輸出影片路徑 | `原檔名_yolo26.mp4` |

### 6.3 執行過程

腳本會顯示即時進度：
```
  Progress: 25.0% | Frame: 1500/6000 | FPS: 18.3 | ETA: 246s
  Progress: 50.0% | Frame: 3000/6000 | FPS: 18.5 | ETA: 162s
  ...
```

---

## 7. YOLO26 vs YOLOv4-tiny 差異比較

| 項目 | YOLOv4-tiny (Darknet) | YOLO26 (Ultralytics) |
|------|----------------------|---------------------|
| **框架** | 自建 PyTorch Darknet | Ultralytics 官方套件 |
| **設定檔** | 手動 `.cfg` 檔 (286行) | 自動管理，無需設定 |
| **資料格式** | xyxy 像素值 | cx cy w h 正規化 |
| **前處理** | 手動 resize + normalize | 框架自動處理 |
| **後處理** | 手動 NMS | NMS-free 端對端推理 |
| **損失函數** | 手動實作 YoloLoss | 內建 ProgLoss + STAL |
| **訓練程式碼** | ~520 行 | ~160 行 |
| **偵測程式碼** | ~150 行 | ~130 行 |
| **預訓練權重** | `yolov4-tiny.conv.29` | COCO 預訓練 (自動下載) |
| **輸入尺寸** | 416×416 | 640×640 (可調整) |
| **精度 (COCO mAP)** | ~21.7% | ~38%+ (nano) |

---

## 8. 常見問題 FAQ

### Q1: `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

### Q2: GPU 記憶體不足 (CUDA out of memory)
降低 batch size：
```bash
python train_local_yolo26.py --batch 8
# 還是不夠就再降
python train_local_yolo26.py --batch 4
```

### Q3: 找不到 `data.yaml`
先執行資料集準備：
```bash
python prepare_dataset_yolo26.py
```

### Q4: 訓練很慢怎麼辦？
- 使用較小的模型 (`yolo26n.pt`)
- 降低影像尺寸 (`--imgsz 416`)
- 減少 epochs (`--epochs 50`)

### Q5: 信心度門檻怎麼調？
- 門檻太高 → 漏偵測 (少框)
- 門檻太低 → 誤偵測 (多框)
- **建議**: 從 `0.4` 開始，視結果微調

### Q6: 如何用自己訓練好的權重？
```bash
python detect_image_yolo26.py --image test.jpg --weights runs_yolo26/car_detect/weights/best.pt
```

---

## 快速開始 (TL;DR)

```bash
# 0. 安裝
pip install ultralytics

# 1. 準備資料集
python prepare_dataset_yolo26.py

# 2. 訓練模型
python train_local_yolo26.py

# 3. 偵測圖片
python detect_image_yolo26.py --image test.jpg

# 4. 偵測影片
python detect_video_yolo26.py --video input.mp4
```

就這四步，搞定！🚗
