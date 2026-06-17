# 🏎️ YOLOv4-tiny 訓練腳本原始碼極限詳解指南 (NTUT Project 6)

本指南為 `scripts/` 目錄下的兩個重要訓練腳本提供最深度的技術剖析，包括逐行邏輯解釋、數學原理、路徑轉換算法以及框架轉換機制。

---

# 🔍 第一部分：`train_yolov4tiny_darknet.py` (C/C++ Darknet 封裝器)

該腳本的主要任務是橋接 Windows 檔案系統與 C++ 編寫的 Darknet 引擎（不論是透過本機 Windows 執行的 `darknet.exe`，還是運行於 WSL 虛擬 Linux 環境中的 `darknet` 二進位檔）。

## 1. 常數與設定區塊 (Constants)
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
DATASET_DIR  = PROJECT_ROOT / "_SignDetection.yolov4pytorch" / "train"
ANN_FILE     = DATASET_DIR / "_annotations.txt"
CLASS_NAMES  = ["stop", "rail", "pedestrian", "blocked"]
NUM_CLASSES  = 4
REMAP        = {0: 3, 1: 2, 2: 1, 3: 0}
```
*   **`CLASS_NAMES`**：嚴格對齊 `Project06.ipynb` 裡面自走車辨識邏輯的順序：
    *   `0 = stop`（對應 `sign[1] == 0`，自走車原地停止 3 秒）
    *   `1 = rail`（對應 `sign[1] == 1`，自走車原地停止 5 秒）
    *   `2 = pedestrian`（對應 `sign[1] == 2`，自走車減速為 0.7 倍速）
    *   `3 = blocked`（對應 `sign[1] == 3`，自走車完全煞停）
*   **`REMAP` 映射機制**：原始資料集中，`_classes.txt` 定義的順序是 `0=blocked, 1=pedestrian, 2=rail, 3=stop`。為了使訓練完後的模型「不需要修改自走車端代碼」即可運作，此映射字典會將原始標記的類別進行重映射：
    *   原始 `0` (blocked) $\rightarrow$ 映射為新 ID `3`
    *   原始 `1` (pedestrian) $\rightarrow$ 映射為新 ID `2`
    *   原始 `2` (rail) $\rightarrow$ 映射為新 ID `1`
    *   原始 `3` (stop) $\rightarrow$ 映射為新 ID `0`

---

## 2. 資料集預處理模組 (`prepare_dataset()`)
此函數負責解析資料集標記文件並將其轉換成符合 Darknet 標準的平面目錄結構。

### 絕對像素座標轉 YOLO 歸一化座標算法：
標記文件 `_annotations.txt` 中的每一行格式如下：
$$\text{image\_name.jpg} \quad x_1,y_1,x_2,y_2,\text{class\_id} \quad x_1,y_1,x_2,y_2,\text{class\_id} \quad \dots$$
其中 $(x_1, y_1)$ 為左上角坐標，$(x_2, y_2)$ 為右下角坐標，單位均為**像素值**。
YOLO 要求的格式為：
$$\text{class\_id} \quad c_x \quad c_y \quad w \quad h$$
其中 $c_x, c_y$ 為邊界框中心點的歸一化座標，$w, h$ 為歸一化的寬與高（範圍皆在 $0 \sim 1$ 之間）。

轉換公式如下：
1.  **中心點 X 座標 ($c_x$)**：
    $$c_x = \frac{x_1 + x_2}{2 \times W}$$
2.  **中心點 Y 座標 ($c_y$)**：
    $$c_y = \frac{y_1 + y_2}{2 \times H}$$
3.  **歸一化寬度 ($w$)**：
    $$w = \frac{x_2 - x_1}{W}$$
4.  **歸一化高度 ($h$)**：
    $$h = \frac{y_2 - y_1}{H}$$

### 程式實作細節：
```python
H, W = img.shape[:2]
label_lines = []
for ann in parts[1:]:
    x1, y1, x2, y2, raw_id = map(int, ann.split(","))
    cls_id = REMAP.get(raw_id, raw_id)  # 類別重映射
    x1, x2 = max(0, x1), min(W, x2)     # 邊界裁切，防止標記溢出圖像邊界
    y1, y2 = max(0, y1), min(H, y2)
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    if bw > 0 and bh > 0:
        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
```
之後，腳本會將 `label_lines` 寫入至 `obj/[image_name].txt`，並把所有轉換後的影像複製到 `obj/` 底下，建立單一扁平化目錄。

---

## 3. 架構配置自動生成模組 (`make_cfg()`)
Darknet 依賴 `.cfg` 來定義神經網絡的每一層。若類別數改變，配置檔中多處卷積層的 `filters` 必須精準修改。

### 數學原理與公式：
對於 YOLO 偵測層（`[yolo]`）**之前**的最後一個卷積層（`[convolutional]`），其輸出的 Channel 數由 `filters` 決定。公式為：
$$\text{filters} = (\text{classes} + 5) \times 3$$
*   其中 `5` 代表邊界框的 5 個參數：中心點的偏移量 $(t_x, t_y)$、寬高比例 $(t_w, t_h)$ 以及該框包含目標的信心值 $t_o$。
*   `classes` 代表我們要辨識的類別數（專案中為 4 類）。
*   `3` 代表 YOLO 在該尺寸特徵圖上預測的 Anchor 數量。
*   因此，我們設定的輸出 filters 數為：
    $$\text{filters} = (4 + 5) \times 3 = 27$$

### CFG 自動寫入機制：
程式會將以下重要超參數動態寫入 `yolov4-tiny-custom.cfg` 中：
*   `max_batches = 8000`（即 $\text{classes} \times 2000$）
*   `steps = 6400,7200`（分別為最大批次數的 80% 與 90%）
*   程式碼中兩個 `[yolo]` 層之前的卷積層將被強制定製為 `classes=4` 和 `filters=27`。

---

## 4. WSL 環境與路徑映射模組 (`w2wsl()`)
當 Windows 環境無法直接執行 Darknet 時，我們需要使用 Windows 內建的 WSL (Linux 虛擬環境)。然而，WSL 內運行的 Linux 程式無法解析 Windows 風格的路徑（如 `C:\Users\...`）。

### 路徑映射算法 (`w2wsl`)：
1.  將所有反斜線 `\` 替換成 Linux 的正斜線 `/`。
2.  移除磁碟機代號的冒號（例如 `C:` 轉為 `c`）。
3.  在前綴加上 WSL 特有的掛載路徑 `/mnt/`。

```python
def w2wsl(p):
    p = str(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p
```
*   **範例**：`C:\NTUT_Media\Project6\train.txt` $\rightarrow$ `/mnt/c/NTUT_Media/Project6/train.txt`
在 WSL 模式下訓練時，腳本會生成專屬的 `train_wsl.txt`，並建立 `obj_wsl.data`，使 WSL 內的 Darknet 編譯版能順利讀取本機上的照片。

---

# 🔍 第二部分：`train_yolov4tiny.py` (PyTorch & Ultralytics 封裝器)

此腳本是基於 Python 最主流的 PyTorch 生態系（利用 Ultralytics API）編寫的訓練系統。

## 1. 資料目錄重構與轉換 (`convert_dataset()`)
PyTorch 的 YOLO 訓練器要求特定的樹狀目錄結構（與 Darknet 平面結構不同）。該腳本會在本地將原始資料集重構為：

```
_yolov4tiny_converted/
├── images/
│   └── train/      # 存放所有影像檔案 (.jpg)
└── labels/
    └── train/      # 存放歸一化後的 YOLO 標記檔案 (.txt)
```

轉換代碼使用 PyTorch 習慣的數據防禦機制，會對讀取出的邊界框 $(x_1, y_1, x_2, y_2)$ 進行極限裁切（Clamp），防止因人為標記失誤超出圖寬高而造成 PyTorch 在 Loss 計算時梯度爆炸：
```python
x1, x2 = max(0, x1), min(W, x2)
y1, y2 = max(0, y1), min(H, y2)
```

---

## 2. YAML 資料配置檔生成 (`write_data_yaml()`)
PyTorch 讀取資料集是透過 YAML 格式檔案。腳本會自動將本地的絕對路徑轉換成字串，並寫入 `data.yaml` 中：
```yaml
train: C:\Users\andy8\Desktop\NTUT_Media\Project6\_yolov4tiny_converted\images\train
val: C:\Users\andy8\Desktop\NTUT_Media\Project6\_yolov4tiny_converted\images\train
nc: 4
names:
  - blocked
  - pedestrian
  - rail
  - stop
```

---

## 3. 模型自適應載入與回退機制 (`resolve_model()`)
現代 Ultralytics YOLO 庫主要原生支援 YOLOv8、YOLOv9、YOLOv11 等架構。在舊版的設定中，`yolov4-tiny.pt` 可能無法直接在線上動物園（Model Zoo）下載。為了保證腳本可以在沒有 YOLOv4-tiny 預訓練權重時順利運行，腳本內建了自適應載入邏輯：

```python
def resolve_model():
    YOLO = get_yolo()  # 動態導入或安裝 ultralytics
    preferred = "yolov4-tiny.pt"
    fallback  = "yolov8n.pt" # 回退使用 YOLOv8-Nano 權重
    try:
        m = YOLO(preferred)
        return preferred, YOLO
    except Exception as e:
        # 當 preferred 不可用時，回退到結構類似、參數量相似(約3M)的 yolov8n.pt
        return fallback, YOLO
```

---

## 4. PyTorch 訓練控制流 (`train()`)
腳本會調用 PyTorch 進行反向傳播 (Backpropagation) 訓練。在底層，它會載入包括特徵增強 (Data Augmentation) 的豐富設定：
*   **Mosaic 增強 (`mosaic=1.0`)**：隨機將 4 張訓練圖拼接為一張，豐富背景特徵，這對偵測小物件（如自走車視野中的遠處路標）極有幫助。
*   **HSV 色調調整 (`hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`)**：隨機抖動色相、飽和度與亮度，模擬自走車在不同教室光線下的曝光情況。
*   **SGD 優化器**：利用隨機梯度下降，設定學習率 `lr0=0.01` 與動量 `momentum=0.937` 進行梯度更新。

---

## 5. 指標評估與 ONNX 匯出 (`export_model()`)
當 PyTorch 訓練完成後，我們無法直接得到 Darknet 的 `.weights` 格式。我們需要將其匯出成**開放式神經網絡交換格式 (ONNX)**：
```python
model.export(format='onnx', imgsz=416, simplify=True, opset=11)
```
*   `imgsz=416`：固定的輸入解析度。
*   `simplify=True`：調用 `onnx-simplifier` 合併冗餘的算子節點，優化網路圖結構。
*   `opset=11`：設定符合 JetPack 4.x (Jetson Nano) 版本相容的運算子集，防止在 JetBot 上編譯 TensorRT 時出現不支援的 Layer 錯誤。
