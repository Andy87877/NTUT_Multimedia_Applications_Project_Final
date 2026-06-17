# JetBot AI 道路辨識專案

這是一個非常典型的「軟體演算法控制硬體設備」的實作流程。在這個專案中，我們要建立一個迴歸模型（Regression Model），讓攝影機看到的每一張畫面，都能轉換成微型車馬達所需的轉向座標。

我們把這個過程由淺入深分為「資料收集」與「模型訓練」兩個階段來看。

---

## 第一部分：資料收集 (Data Collection)

### 要收集什麼樣的資料？

在 `data_collection_gamepad.ipynb` 腳本中，你的目標是建立一個「看到特定畫面，就該往哪個座標走」的對應關係。

#### 視覺輸入 (Input)

透過 JetBot 前方的相機，即時擷取 $224 \times 224$ 像素的影像。

#### 座標標記 (Labeling)

這一步需要你透過外接遊戲手把（Gamepad）或介面上的滑桿（x_slider, y_slider），在即時畫面上標記出一個綠色的目標點，也就是車子當下「應該要開過去的理想位置 $(X, Y)$」。

#### 儲存方式

當你按下紀錄鍵時，系統會把當下的影像存成 `.jpg` 檔。**最特別的是，它會直接把 $(X, Y)$ 的座標數值寫在檔名裡面**（例如：`xy_050_050_uuid.jpg`）。

#### 資料量建議

投影片建議大約收集 **50 到 200 張**這樣帶有座標檔名的照片。這組資料集就是軟體認知與實體空間的橋樑，你給的座標越精準，後續馬達跑起來就越順。

---

## 第二部分：模型訓練 (Model Training)

### 訓練流程概述

收集好資料後，接著開啟 `train_model.ipynb`，利用 PyTorch 深度學習框架來訓練模型，讓它學會你剛才示範的駕駛邏輯。

### 1. 資料預處理 (Data Preprocessing)

**切割訓練集與測試集**

- 程式首先會將你收集到的資料，隨機切割為 **90% 的訓練集 (Train Set)** 與 **10% 的測試集 (Test Set)**

**資料擴增 (Data Augmentation)**

- 為了讓模型適應不同的環境光線和視角，會對影像進行以下操作：
  - 隨機的水平翻轉 (random horizontal flip)
  - 色彩擾動 (color jitter)
  - 將影像轉為張量 (Tensor) 並進行正規化

### 2. 改造預訓練模型 (Model Architecture)

**模型選擇**

- 專案使用的是現成的 **ResNet-18** 影像辨識模型

**模型修改**

- 由於我們不是要做分類（比如分辨貓或狗），而是要預測具體的方向數值
- 程式會將模型最後的全連接層 (Fully Connected Layer, `fc`) 重新初始化
- 接收 512 個特徵，**輸出 2 個特徵** (`model.fc = torch.nn.Linear(512, 2)`)
- 這兩個輸出特徵正是對應我們剛才收集的 $X$ 和 $Y$ 座標

### 3. 設定參數與在 GPU 上訓練

**批次與回合**

- 每次取 8 張圖片作為一個 Batch 進行訓練 (`batch_size=8`)
- 整個資料集預設會反覆學習 70 個回合 (`NUM_EPOCHS = 70`)

**最佳化與損失函數**

- 採用 Adam 最佳化演算法 (`optim.Adam`)
- 訓練時，會計算模型預測的座標與你檔名上真實座標之間的「均方誤差」(`F.mse_loss`)

**硬體加速**

- 整個訓練過程會透過 CUDA 搬移到 GPU 的記憶體上執行，大幅提升運算效率

### 4. 儲存最佳結果

- 在每一個回合結束時，模型會用測試集來評估目前的誤差值 (`test_loss`)
- 只要當前的誤差打破了歷史最低紀錄，程式就會自動把當下的模型權重儲存下來
- 命名為 `best_steering_model_xy.pth`
- 這個檔案就是你最終要拿來控制實體車輛的大腦

---

## 核心實作程式碼

### 1. 資料收集邏輯

在收集資料時，檔名的格式至關重要，因為模型訓練時會從檔名解析出標準答案 ($X, Y$ 座標)。

```python
import uuid
import os

def save_snapshot(directory, x, y, image_value):
    """
    儲存帶有座標資訊的影像檔案
    
    Args:
        directory: 儲存目錄
        x: X 座標 (通常範圍 -1 到 1)
        y: Y 座標 (通常範圍 -1 到 1)
        image_value: 影像資料
    """
    # 將座標縮放到 0-100 的整數方便檔名儲存
    x_int = int(x * 50 + 50)
    y_int = int(y * 50 + 50)
    
    # 格式：xy_XXX_YYY_uuid.jpg
    filename = 'xy_%03d_%03d_%s.jpg' % (x_int, y_int, uuid.uuid1())
    image_path = os.path.join(directory, filename)
    
    with open(image_path, 'wb') as f:
        f.write(image_value)
```

### 2. 模型訓練代碼

這部分的核心是修改 ResNet-18 的最後一層，使其從「分類」轉變為「迴歸」輸出。

```python
import torch
import torchvision.models as models

# 1. 載入預訓練的 ResNet-18 模型
model = models.resnet18(pretrained=True)

# 2. 修改全連接層 (Fully Connected Layer)
# ResNet-18 最後一層輸入是 512，我們需要輸出 X 和 Y 兩個數值
model.fc = torch.nn.Linear(512, 2)

# 3. 將模型搬移至 GPU
device = torch.device('cuda')
model = model.to(device)

# 4. 定義損失函數 (迴歸任務使用 MSE) 與最佳化器
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()  # 手冊中使用 F.mse_loss
```

### 3. PD 控制邏輯

這是最感興趣的「韌體控制」部分。我們會根據模型推論出的目標點，計算馬達該給多少出力。

```python
import numpy as np

# 初始化全域變數紀錄上一次的角度，用於計算 D (微分) 項
angle_last = 0.0

def compute_steering(x, y, speed_gain, p_gain, d_gain, bias):
    """
    計算轉向角度和馬達出力
    
    Args:
        x, y: 模型預測的目標座標
        speed_gain: 速度增益 (0~1)
        p_gain: 比例項增益
        d_gain: 微分項增益
        bias: 馬達公差修正值
        
    Returns:
        left_motor: 左馬達出力 (0~1)
        right_motor: 右馬達出力 (0~1)
    """
    global angle_last
    
    # 1. 計算目標角度
    angle = np.arctan2(x, y)
    
    # 2. PD 控制器計算
    # P 項：當前誤差 * 增益
    # D 項：(當前角度 - 上次角度) * 增益
    pid = angle * p_gain + (angle - angle_last) * d_gain
    angle_last = angle
    
    # 3. 加入馬達公差修正 (Bias)
    steering = pid + bias
    
    # 4. 轉換為左右馬達出力
    left_motor = max(min(speed_gain + steering, 1.0), 0.0)
    right_motor = max(min(speed_gain - steering, 1.0), 0.0)
    
    return left_motor, right_motor
```

---

## PID 控制原理

### 概念說明

為了讓你更直觀地理解為什麼需要 `steering_dgain` (微分項) 來穩定車身，請觀察當 P 增益太大時，車子會如何震盪，而加入 D 增益後又是如何平滑路徑的。這對於你在 EVI Lab 調整無人機姿態控制的邏輯是完全通用的。

### 三個主要元件

**比例響應 (P - Proportional)**

- 基於目標與實際位置之間的誤差進行調整
- 誤差變大時，轉向角度也變大，使車輛快速對齊目標路徑

**積分響應 (I - Integral)**

- 累積過去的誤差來逐步修正長期的小偏差
- 讓車輛更精確地接近目標位置

**微分響應 (D - Derivative)**

- 根據誤差變化的速度進行調整
- 減少車輛的過度轉向或晃動

---

## 實作小叮嚨

### 馬達限速

第一次執行 `execute` 函數時，建議先把 `speed_gain` 調小（例如 0.15），避免 JetBot 因為模型預測不準而直接衝出桌面。

### 相機釋放

在 Jupyter Notebook 切換不同的 `.ipynb` 時，記得執行 `camera.stop()`，否則相機資源會被佔用導致另一個檔案無法開啟。

### 座標對齊

在收集資料時，確保 X 軸的 -1 代表左、1 代表右，這跟 OpenCV 的座標系統可能需要轉換。

---

## 常見問題

### 資料收集階段

- **問題**：如何在跑道轉彎處收集座標資料才能讓模型學會平滑過彎？
- **建議**：在轉彎時多收集不同角度和位置的影像，確保座標標記準確。

### 模型訓練階段

- **問題**：在撰寫 `XYDataset` 的 `__getitem__` 影像前處理部分遇到困難？
- **建議**：仔細檢查 `normalize` 的參數，確保使用與訓練集相同的正規化參數。

---

## 相關資源

此份代碼的邏輯對於未來在 EVI Lab 開發無人機與自走車的韌體會非常有幫助。
