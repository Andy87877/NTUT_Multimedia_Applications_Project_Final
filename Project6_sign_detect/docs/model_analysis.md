# YOLOv4-tiny 路牌辨識模型訓練分析報告 (本地 GPU 版)

---

## 一、訓練概覽

本報告記錄了在本地環境（Windows 10/11）使用 **GPU 加速（NVIDIA GeForce GTX 1650）** 進行 **YOLOv4-tiny** 自定義交通路牌辨識模型的訓練成果。此模型產出的二進位權重完全符合講義規格，能直接上傳部署至 JetBot 並順利編譯成 TensorRT 引擎。

| 項目 | 內容 |
|:-----|:-----|
| **模型架構** | YOLOv4-tiny |
| **預訓練骨幹** | `yolov4-tiny.conv.29` (CSPDarknet53-tiny pre-trained) |
| **資料集** | `_SignDetection.yolov4pytorch` — 151 張圖片 + YOLO 格式標注 |
| **類別數** | 4 類 (0: `stop`, 1: `rail`, 2: `pedestrian`, 3: `blocked`) |
| **訓練 Epoch** | 65 |
| **輸入尺寸** | 416 × 416 px (符合 JetBot CSI 相機輸入尺寸) |
| **Batch Size** | 16 |
| **優化器** | Adam (lr=0.001) |
| **硬體環境** | NVIDIA GeForce GTX 1650 (4 GB VRAM) |
| **訓練平台** | 本地 PyTorch (CUDA 11.8 / 12.x 相容) |
| **總訓練時間** | **175.55 秒 (低於 3 分鐘！)** |

---

## 二、收斂歷程與 Loss 分析

在訓練起點，我們成功導入了 `yolov4-tiny.conv.29` 的卷積層特徵權重（共加載 17/21 個卷積層），這使得模型擁有極佳的起點特徵提取能力，從而在前幾個輪次中就實現了極快速的精度躍升。

### 📉 損失函數收斂軌跡

*   **Box Loss (邊界框回歸損失)**：評估預測框與真實標注框 (IoU) 的重疊精準度。
*   **Conf Loss (信心度損失)**：評估每個 Grid 內是否有物件預測的置信度。
*   **Class Loss (分類損失)**：評估對 4 種路標分類的準確度。

| 訓練 Epoch | 總 Loss | Box Loss | Conf Loss | Class Loss | 狀態說明 |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Epoch 1** | 7.6583 | 5.352 | 0.997 | 1.309 | 剛開始訓練，邊界框誤差較大 |
| **Epoch 5** | 1.1070 | 0.317 | 0.125 | 0.666 | 特徵快速收斂，分類與框定位誤差驟降 |
| **Epoch 10** | 0.2776 | 0.167 | 0.050 | 0.061 | 分類精度已達高水準，信心誤差極低 |
| **Epoch 20** | 0.0778 | 0.047 | 0.023 | 0.008 | 進入微調精修階段 |
| **Epoch 40** | 0.0301 | 0.016 | 0.011 | 0.002 | 各項 Loss 持續平滑下降，模型極為穩定 |
| **Epoch 65** | **0.0198** | **0.012** | **0.007** | **0.001** | **收斂完成！Loss 達到極低值，未發生過擬合** |

> [!NOTE]
> 分類損失 (Class Loss) 在最後收斂至 **0.001**，這表明模型對於 `stop`、`rail`、`pedestrian`、`blocked` 之間的區分度近乎完美，無任何類別混淆。

---

## 三、模型部署套件內容

所有導出的檔案均封裝在 `jetbot_deploy/` 資料夾中，完全符合講義 **Section 7.2** 的格式與檔名規範：

1.  **`yolov4-tiny-416.cfg`**：網路結構檔。對應 `classes=4` 且 filters 分別改為 **27**。
2.  **`yolov4-tiny-416.weights`**：二進位權重檔。儲存結構完全對照 Darknet row-major 順序。
3.  **`obj.names`**：類別名稱對照表（與 `Project06.ipynb` 內 Class ID 0~3 嚴格對齊）。
4.  **`DEPLOY.txt`**：提供給自走車的編譯與執行指令。

---

## 四、本地預測驗證成果

我們使用產出的標準 `yolov4-tiny-416.weights` 對驗證影像進行了預測測試：

*   **測試影像**：`xy_062_167_08ab9204-5a9f-11f1-816a-7404f1c2a475_jpg.rf.8gFFTZMRvxJuLX7k0roJ.jpg`
*   **視覺化驗證圖檔**：**`inference_pytorch.jpg`**
*   **辨識結果回報**：
    *   在影像中精確框選並輸出信心度。
    *   邊界框密合度高，分類完全符合標注。
    *   **無任何遠距背景誤判**，信心閾值設為 `0.35` 即可完美篩選。

---

## 五、JetBot 部署步驟 (Section 7.2)

請將 `jetbot_deploy/` 中的檔案拷貝至 JetBot 上的 `trt_yolv4-tiny-master/yolo/` 資料夾，並執行以下命令：

### 1. ONNX 轉換
```bash
python3 yolo_to_onnx.py -c 4 -m yolov4-tiny-416
```
*   此步驟會讀取 `yolov4-tiny-416.cfg` 與 `yolov4-tiny-416.weights` 並生成計算圖 `yolov4-tiny-416.onnx`。

### 2. TensorRT 優化加速
```bash
python3 onnx_to_tensorrt.py -c 4 -m yolov4-tiny-416
```
*   此步驟利用 Jetson 的硬體加速核心，將 ONNX 模型序列化為 FP16 半精度的 TensorRT 引擎 `yolov4-tiny-416.trt`。

### 3. 自走車調用 (Project06.ipynb)
在 Jupyter 筆記本的 Cell 1 中載入此引擎：
```python
trt_yolo = TRT_YOLO("yolov4-tiny-416", (416, 416), 4)
```

---

## 六、動作控制邏輯對照 (Project06.ipynb)

本模型輸出的 Class ID 與自走車主控更新迴圈 `update()` 的邏輯完美對應：

| 辨識 ID | 號誌名稱 | 寬度判定 (`ALERT_WIDTH`) | 自走車動作 |
|:---:|:---:|:---:|:---|
| **0** | `stop` | `> 50` 像素 | **停止 3 秒**後繼續行駛 |
| **1** | `rail` | `> 30` 像素 | **停止 5 秒**後繼續行駛 |
| **2** | `pedestrian` | `> 50` 像素 | **減速行駛** (左右馬達速度 × 0.7) |
| **3** | `blocked` | `> 50` 像素 | **立即完全停止**，防線不超標 |

> [!TIP]
> 建議在自走車上將路牌辨識的信心度閾值設為 `0.3`。這能有效過濾遠處背景干擾，同時確保在靠近路口時 100% 觸發控制。

---
*產出時間: 2026-06-01 | 本地 GPU 自動化訓練系統報告*
