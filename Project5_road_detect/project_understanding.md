# 文件閱讀摘要：Project5 AI 道路辨識

## 一、AI.md 內容摘要

### 專案目標
建立一個 **迴歸模型 (Regression Model)**，讓 JetBot 的攝影機看到的每一張畫面，都能轉換成微型車馬達所需的 **轉向座標 (X, Y)**。

### 資料收集 (Data Collection)
- 透過 JetBot 前方相機，擷取 **224×224** 像素的影像
- 用 Gamepad 或滑桿在畫面上標記 **目標點 (X, Y)**，代表車子「應該要開過去的理想位置」
- 座標寫在檔名中，格式為：`xy_XXX_YYY_uuid.jpg`
  - X, Y 值經由 `int(x * 50 + 50)` 縮放至 0–100+ 範圍（原始值為 -1 ~ 1）
- 建議收集 **50 ~ 200 張**帶有座標的照片

### 模型訓練 (Model Training)
| 項目 | 設定 |
|------|------|
| 模型架構 | **ResNet-18** (pretrained) |
| 最後一層修改 | `model.fc = torch.nn.Linear(512, 2)` → 輸出 X, Y 兩個值 |
| 損失函數 | **MSE Loss** (均方誤差) |
| 最佳化器 | **Adam** |
| Batch Size | **8** |
| Epochs | **70** |
| Train/Test 比 | **90% / 10%** |
| 資料增強 | 隨機水平翻轉、色彩擾動、正規化 |
| 輸出模型 | `best_steering_model_xy.pth` |

### PD 控制邏輯
- 用 `np.arctan2(x, y)` 計算目標角度
- PD 控制器：`pid = angle * p_gain + (angle - angle_last) * d_gain`
- 左右馬達出力 = `speed_gain ± steering`

---

## 二、PDF 內容摘要 (Project5_AI道路辨識.pdf)

> [!NOTE]
> PDF 為陳彥霖教授 (National Taipei University of Technology, Spring 2026) 的課程投影片，部分中文因編碼問題顯示為亂碼，但結構可辨識。

### 課程流程 (共 29 頁)

| 步驟 | 對應 Notebook | 說明 |
|------|--------------|------|
| 1. 資料收集 | `data_collection_gamepad.ipynb` | 透過遊戲手把標記座標，存成 `xy_X_Y_uuid.jpg` |
| 2. 模型訓練 | `train_model.ipynb` | 用 PyTorch + ResNet18 訓練 |
| 3. TensorRT 優化 | `live_demo_build_trt.ipynb` | 將 PyTorch 模型轉為 TensorRT 加速 |
| 4. 即時推論 | `live_demo_trt.ipynb` | 即時影像輸入 → 道路跟隨 |

### PDF 重點摘要
1. **Page 5 (流程圖)**：Input 224×224 → PyTorch Training (ResNet18) → `best_steering_model_xy.pth` → TensorRT → Live Demo
2. **Page 8**：Camera 設定 224×224，slider 範圍 -1~1 映射到像素座標
3. **Page 12**：建議收集 **50~200** 張影像
4. **Page 25–28**：PyTorch 訓練細節
   - 使用 `torch.utils.data.Dataset` 自定義 `XYDataset` 類別
   - `__getitem__` 負責讀取影像、從檔名解析 x/y 座標、影像前處理
   - `random_hflip` 資料增強
5. **Page 29**：訓練/測試分割 = **90% / 10%**

---

## 三、dataset_xy 資料集統計

| 項目 | 數值 |
|------|------|
| 影像總數 | **450 張** |
| 影像尺寸 | **224 × 224** (RGB) |
| 檔名格式 | `xy_{X}_{Y}_{uuid}.jpg` |
| X 值範圍 | 0 ~ 224 |
| Y 值範圍 | 0 ~ 187+ |
| 檔案大小 | 約 17KB ~ 40KB 每張 |

> [!IMPORTANT]
> 資料集有 **450 張**，超過建議的 50~200 張，這是一個不錯的資料量，有助於提升模型精度。

---

## 四、接下來的工作

根據以上理解，我將建立 `train_model.ipynb`，包含：
1. **XYDataset** 自定義資料集類別（從檔名解析座標）
2. 資料增強（水平翻轉 + 色彩擾動 + 正規化）
3. 90/10 訓練/測試分割
4. ResNet-18 模型修改（`fc → Linear(512, 2)`）
5. Adam + MSE Loss 訓練迴圈
6. 儲存最佳模型 `best_steering_model_xy.pth`
