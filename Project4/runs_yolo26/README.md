# YOLO26n 車輛偵測模型訓練分析報告

> **模型架構**：YOLO26n (`yolo26n.pt`)  
> **任務**：單類別車輛偵測（Car）  
> **訓練週期**：100 Epochs  
> **分析日期**：2026-05-05

---

## 目錄

1. [實驗概覽](#1-實驗概覽)
2. [訓練設定對比](#2-訓練設定對比)
3. [資料集資訊](#3-資料集資訊)
4. [訓練時間與速度分析](#4-訓練時間與速度分析)
5. [最終 Epoch 指標對比](#5-最終-epoch-指標對比)
6. [最佳模型 (Best Checkpoint) 表現](#6-最佳模型-best-checkpoint-表現)
7. [損失函數曲線分析](#7-損失函數曲線分析)
8. [mAP 成長歷程](#8-map-成長歷程)
9. [Precision / Recall 分析](#9-precision--recall-分析)
10. [學習率排程](#10-學習率排程)
11. [視覺化結果](#11-視覺化結果)
12. [結論與建議](#12-結論與建議)

---

## 1. 實驗概覽

本次實驗針對同一模型架構（YOLO26n）、相同超參數設定，分別在**兩個不同資料集**上進行訓練，以評估資料集規模與來源對模型性能的影響：

| 實驗名稱 | 資料集 | 結果資料夾 |
|---|---|---|
| **car_detect** | `dataset_yolo26`（單一來源） | `runs_yolo26/car_detect/` |
| **car_detect_merged** | `dataset_merged_yolo26`（合併自兩位貢獻者） | `runs_yolo26/car_detect_merged/` |

兩個模型皆採用預訓練權重（`pretrained: true`）進行遷移學習，偵測類別均為單一類別 `Car`。

---

## 2. 訓練設定對比

| 超參數 | car_detect | car_detect_merged |
|---|---|---|
| **模型** | yolo26n.pt | yolo26n.pt |
| **資料集** | dataset_yolo26 | dataset_merged_yolo26 |
| **Epochs** | 100 | 100 |
| **Batch Size** | 16 | 16 |
| **影像尺寸** | 640 × 640 | 640 × 640 |
| **Optimizer** | Auto | Auto |
| **初始學習率 (lr0)** | 0.01 | 0.01 |
| **最終學習率 (lrf)** | 0.01 | 0.01 |
| **Momentum** | 0.937 | 0.937 |
| **Weight Decay** | 0.0005 | 0.0005 |
| **Warmup Epochs** | 3.0 | 3.0 |
| **IoU 閾值** | 0.7 | 0.7 |
| **Close Mosaic (epoch)** | 10 | 10 |
| **AMP（混合精度）** | ✅ 啟用 | ✅ 啟用 |
| **預訓練** | ✅ 啟用 | ✅ 啟用 |
| **資料增強 - Mosaic** | 1.0 | 1.0 |
| **資料增強 - HSV-H** | 0.015 | 0.015 |
| **資料增強 - HSV-S** | 0.7 | 0.7 |
| **資料增強 - Scale** | 0.5 | 0.5 |
| **資料增強 - FlipLR** | 0.5 | 0.5 |
| **資料增強 - Erasing** | 0.4 | 0.4 |

> **結論**：兩次實驗的超參數**完全相同**，差異僅在於訓練資料集，保證了對比的公平性。

---

## 3. 資料集資訊

### car_detect — `dataset_yolo26`

```yaml
# 來源：prepare_dataset_yolo26.py 自動產生（單一貢獻者）
path: dataset_yolo26
train: train/images
val:   valid/images
test:  test/images
names:
  0: Car
```

- **資料分割**：`train/` / `valid/` / `test/`
- **類別數**：1（Car）
- **來源**：單一標注者的 CCTV 截圖資料

### car_detect_merged — `dataset_merged_yolo26`

```yaml
# 來源：合併自兩位貢獻者
#   113820020_林政德.yolo26 (前綴: 20_)
#   113820033_謝奕宏.yolo26 (前綴: 33_)
path: dataset_merged_yolo26
nc: 1
names:
  0: Car
```

- **資料分割**：`train/` / `valid/` / `test/`
- **類別數**：1（Car）
- **來源**：兩位貢獻者合併，資料集規模約為前者的 **2.1 倍**（依訓練 batch 數量推算）

> **推算依據**：`train_batch1262.jpg`（car_detect 最後 batch）vs `train_batch2702.jpg`（car_detect_merged 最後 batch），顯示合併資料集每個 epoch 的批次數約為 **2703 批次**，而原始為 **1263 批次**，訓練圖像量約達原始的 2.14 倍。

---

## 4. 訓練時間與速度分析

| 指標 | car_detect | car_detect_merged |
|---|---|---|
| **總訓練時間** | 2693.1 秒（≈ **44.9 分鐘**） | 4042.2 秒（≈ **67.4 分鐘**） |
| **每 Epoch 平均時間** | ~26.9 秒 | ~40.4 秒 |
| **首 Epoch 時間** | 26.8 秒 | 72.0 秒 |
| **第 2 Epoch 時間** | 59.7 秒（累積） | 132.4 秒（累積） |
| **每 Epoch 額外耗時** | 約 +0s（穩定） | 約 +0s（穩定） |
| **時間比例（merged/原始）** | — | **1.50×** |

**觀察**：

- merged 資料集每 epoch 耗時比原始多約 **50%**，但批次數量多出約 **114%**，代表 GPU 利用率在更大資料集上更高效率。
- 首 Epoch 時間差異顯著（72.0s vs 26.8s），推測是資料載入與預處理的差異。
- 兩者整體訓練時間均在 **1 小時內**完成，計算效率佳。

---

## 5. 最終 Epoch 指標對比

以下為 **Epoch 100**（最後一個 epoch）的指標：

| 指標 | car_detect | car_detect_merged | 差異 |
|---|---|---|---|
| **Precision (B)** | 0.787 | **0.922** | merged +13.5% ✅ |
| **Recall (B)** | **0.782** | 0.690 | car_detect +9.2% ✅ |
| **mAP@50 (B)** | **0.840** | 0.763 | car_detect +7.7% ✅ |
| **mAP@50-95 (B)** | **0.510** | 0.464 | car_detect +4.6% ✅ |
| **Train Box Loss** | **1.213** | 1.254 | car_detect 低 3.4% ✅ |
| **Train Cls Loss** | **0.569** | 0.570 | 幾乎相同 |
| **Train DFL Loss** | **0.00223** | 0.00294 | car_detect 低 24.1% ✅ |
| **Val Box Loss** | **1.347** | 1.586 | car_detect 低 15.1% ✅ |
| **Val Cls Loss** | **0.916** | 0.965 | car_detect 低 5.1% ✅ |
| **Val DFL Loss** | **0.00393** | 0.00452 | car_detect 低 15.0% ✅ |

**重點解讀**：

- **car_detect** 在 mAP（平均精準度）、Recall、所有 Loss 指標上均優於 merged。
- **car_detect_merged** 的 Precision 顯著較高（0.922 vs 0.787），代表其預測框在命中率上更精準，但代價是 Recall 較低（漏偵測較多）。
- 此現象反映 merged 模型在 validation set 上呈現「保守型」偵測風格（高精準、低召回）。

---

## 6. 最佳模型 (Best Checkpoint) 表現

YOLO26 訓練過程中自動儲存 `weights/best.pt`（以 mAP50 為準）。

### car_detect — 最佳 Epoch：**88**

| 指標 | 數值 |
|---|---|
| **Epoch** | 88 |
| **Precision** | 0.851 |
| **Recall** | 0.811 |
| **mAP@50** | **0.864** |
| **mAP@50-95** | 0.521 |
| **Val Box Loss** | 1.370 |
| **Val Cls Loss** | 0.867 |
| **Train Box Loss** | 1.282 |
| **Train Cls Loss** | 0.609 |

### car_detect_merged — 最佳 Epoch：**59**

| 指標 | 數值 |
|---|---|
| **Epoch** | 59 |
| **Precision** | 0.916 |
| **Recall** | 0.729 |
| **mAP@50** | **0.809** |
| **mAP@50-95** | 0.476 |
| **Val Box Loss** | 1.607 |
| **Val Cls Loss** | 0.888 |
| **Train Box Loss** | 1.560 |
| **Train Cls Loss** | 0.801 |

### 最佳 Best Checkpoint 對比

| 指標 | car_detect (ep88) | car_detect_merged (ep59) | 差異 |
|---|---|---|---|
| **Best Epoch** | 88 | 59 | merged 更早收斂 |
| **Precision** | 0.851 | **0.916** | merged +6.5% |
| **Recall** | **0.811** | 0.729 | car_detect +8.2% |
| **mAP@50** | **0.864** | 0.809 | car_detect +5.5% |
| **mAP@50-95** | **0.521** | 0.476 | car_detect +4.5% |

**觀察**：

- car_detect_merged 的最佳 epoch 出現在 epoch 59，**比 car_detect（epoch 88）早 29 個 epoch**，反映出更大資料集的快速初期收斂效果。
- 但最終 Best mAP@50 差距約 5.5%（0.864 vs 0.809），顯示原始資料集在目標任務上的資料品質或場景一致性更佳。

---

## 7. 損失函數曲線分析

### 7.1 訓練損失（Train Loss）

#### Box Loss（定位損失）

| Epoch | car_detect | car_detect_merged |
|---|---|---|
| 1 | 1.809 | 2.011 |
| 10 | 1.713 | 1.860 |
| 20 | 1.641 | 1.782 |
| 30 | 1.542 | 1.703 |
| 40 | 1.523 | 1.631 |
| 50 | 1.451 | 1.540 |
| 60 | 1.368 | 1.494 |
| 70 | 1.406 | 1.458 |
| 80 | 1.330 | 1.447 |
| 90 | 1.243 | 1.370 |
| 100 | 1.213 | 1.254 |
| **下降幅度** | **-33.1%** | **-37.6%** |

- 兩者 box loss 均穩定下降，merged 從更高起點（2.011）降至與 car_detect 接近的水準（1.254 vs 1.213）。
- merged 訓練初期（epoch 1-5）的 box loss 下降速率較快，反映更豐富的資料幫助模型更快學習定位特徵。

#### Classification Loss（分類損失）

| Epoch | car_detect | car_detect_merged |
|---|---|---|
| 1 | 4.903 | 4.338 |
| 10 | 1.971 | 1.547 |
| 20 | 1.198 | 1.150 |
| 30 | 0.961 | 1.001 |
| 40 | 0.837 | 0.926 |
| 50 | 0.778 | 0.841 |
| 60 | 0.670 | 0.792 |
| 70 | 0.648 | 0.751 |
| 80 | 0.658 | 0.729 |
| 90 | 0.604 | 0.682 |
| 100 | 0.569 | 0.570 |
| **下降幅度** | **-88.4%** | **-86.8%** |

- 兩者分類損失均大幅下降，最終收斂至**幾乎相同水準**（約 0.57）。
- car_detect 初期（epoch 1）的分類損失更高（4.903 vs 4.338），說明原始資料集的類別分布或標注風格對模型初始更具挑戰性。

### 7.2 驗證損失（Validation Loss）

#### Val Box Loss

| Epoch | car_detect | car_detect_merged |
|---|---|---|
| 1 | 1.565 | 1.616 |
| 10 | 1.617 | 1.629 |
| 20 | 1.646 | 1.687 |
| 30 | 1.540 | 1.632 |
| 40 | 1.470 | 1.622 |
| 50 | 1.415 | 1.607 |
| 60 | 1.393 | 1.622 |
| 70 | 1.458 | 1.642 |
| 80 | 1.343 | 1.592 |
| 90 | 1.361 | 1.558 |
| 100 | 1.347 | 1.586 |
| **下降幅度** | **-14.0%** | **-1.9%** |

> ⚠️ **重要發現**：car_detect_merged 的 validation box loss 從 epoch 1 的 1.616 到 epoch 100 的 1.586，整個訓練過程中幾乎**沒有顯著改善**（僅下降 1.9%），而且多個 epoch 出現 val box loss 反而高於訓練初始值的情況。這與 car_detect 的 -14% 改善形成明顯對比，暗示 **merged 模型在驗證集上的泛化能力（定位精度）不足**。

#### Val Classification Loss

| Epoch | car_detect | car_detect_merged |
|---|---|---|
| 1 | 5.349 | 5.109 |
| 10 | 2.229 | 1.493 |
| 20 | 1.310 | 1.219 |
| 30 | 1.135 | 1.031 |
| 50 | 0.918 | 0.999 |
| 70 | 0.954 | 1.013 |
| 90 | 0.847 | 0.970 |
| 100 | 0.916 | 0.965 |
| **下降幅度** | **-82.9%** | **-81.1%** |

- 分類損失兩者均有顯著改善，差距不大。

---

## 8. mAP 成長歷程

### mAP@50 各關鍵 Epoch

| Epoch | car_detect | car_detect_merged |
|---|---|---|
| 1 | 0.144 | 0.112 |
| 5 | 0.253 | 0.607 |
| 6 | 0.562 | 0.645 |
| 7 | 0.640 | 0.601 |
| 10 | 0.725 | 0.621 |
| 15 | 0.695 | 0.683 |
| 20 | 0.764 | 0.725 |
| 25 | 0.715 | 0.708 |
| 30 | 0.794 | 0.758 |
| 35 | 0.783 | 0.707 |
| 40 | 0.855 | 0.755 |
| 45 | 0.793 | 0.764 |
| 50 | 0.831 | 0.755 |
| **59** | 0.847 | **0.809** ← merged 最佳 |
| 60 | 0.839 | 0.790 |
| 65 | 0.847 | 0.802 |
| 70 | 0.816 | 0.764 |
| 75 | 0.796 | 0.767 |
| 80 | 0.812 | 0.777 |
| 85 | 0.845 | 0.785 |
| **88** | **0.864** ← car_detect 最佳 | 0.776 |
| 90 | 0.848 | 0.777 |
| 95 | 0.826 | 0.779 |
| 100 | 0.840 | 0.763 |

**觀察**：

1. **car_detect** 的 mAP@50 在 epoch 6 出現一次**跳躍式提升**（0.253 → 0.562），接著持續震盪上升，最終在 epoch 88 達到 **0.864** 的峰值。
2. **car_detect_merged** 的 mAP@50 在 epoch 5 出現第一次大幅提升（0.112 → 0.607），收斂比原始資料集更快，但後期（epoch 60 以後）出現**平原期（plateau）**，在 0.76–0.81 區間震盪，無法突破。
3. car_detect 在訓練後期（epoch 80-100）仍能維持 0.83+ 的高水準，顯示模型在訓練集上學到了更穩定的特徵。

### mAP@50-95 各關鍵 Epoch

| Epoch | car_detect | car_detect_merged |
|---|---|---|
| 1 | 0.122 | 0.078 |
| 10 | 0.404 | 0.335 |
| 20 | 0.399 | 0.372 |
| 30 | 0.444 | 0.446 |
| 40 | 0.498 | 0.433 |
| 50 | 0.503 | 0.447 |
| 60 | 0.496 | 0.452 |
| 70 | 0.473 | 0.441 |
| 80 | 0.504 | 0.466 |
| **87** | **0.527** ← car_detect 最佳 | 0.475 |
| 90 | 0.518 | 0.473 |
| **76** | 0.483 | **0.485** ← merged 最佳 |
| 100 | 0.510 | 0.464 |

- mAP@50-95 衡量在更嚴格 IoU 閾值（0.5~0.95）下的綜合性能，car_detect 最終以 **0.527**（epoch 87）領先 merged 的 **0.485**（epoch 76）約 **8.7%**。

---

## 9. Precision / Recall 分析

### 最終 Epoch (100) Precision 與 Recall

```
car_detect:         P=0.787  R=0.782  F1≈0.784
car_detect_merged:  P=0.922  R=0.690  F1≈0.789
```

計算 F1-score（調和平均）：

- **car_detect**：$F1 = \frac{2 \times 0.787 \times 0.782}{0.787 + 0.782} \approx 0.784$
- **car_detect_merged**：$F1 = \frac{2 \times 0.922 \times 0.690}{0.922 + 0.690} \approx 0.789$

> F1 分數相近（0.784 vs 0.789），說明兩模型整體平衡性接近，差異主要在 **Precision-Recall 的取捨方向**。

### Precision 與 Recall 趨勢分析

#### car_detect — 選取 Epochs

| Epoch | Precision | Recall |
|---|---|---|
| 1 | 0.006 | 0.378 |
| 5 | 0.550 | 0.095 |
| 6 | 0.865 | 0.284 |
| 10 | 0.745 | 0.656 |
| 20 | 0.712 | 0.744 |
| 30 | 0.836 | 0.738 |
| 50 | 0.870 | 0.689 |
| 66 | **0.931** | 0.752 |
| 88 | 0.851 | 0.811 |
| 100 | 0.787 | 0.782 |

#### car_detect_merged — 選取 Epochs

| Epoch | Precision | Recall |
|---|---|---|
| 1 | 0.005 | 0.357 |
| 5 | 0.624 | 0.543 |
| 10 | 0.669 | 0.597 |
| 20 | 0.788 | 0.651 |
| 30 | 0.781 | 0.659 |
| 50 | 0.746 | 0.729 |
| 59 | 0.916 | 0.729 |
| 80 | 0.814 | 0.729 |
| 98 | **0.918** | 0.690 |
| 100 | 0.922 | 0.690 |

**解讀**：

- car_detect 的 Precision 在中後期（epoch 60-90）維持在 0.80-0.93 之間，Recall 也在 0.73-0.82，兩者維持較佳的**均衡狀態**。
- car_detect_merged 在後期 Precision 雖高（0.90+），但 Recall 固定在約 0.69 無法提升，形成**高精準低召回**的模式，代表模型設定了更保守的偵測門檻（難以偵測到模糊/遮擋的車輛）。
- 這可能是因為合併資料集中兩個來源的標注風格不一致（部分場景的車輛樣態差異大），導致模型無法對所有情境都高信心地預測。

---

## 10. 學習率排程

兩模型採用相同的學習率排程策略：

| 階段 | 描述 |
|---|---|
| **Warmup（Epoch 1-3）** | 從極低值線性增加至 lr0=0.01 |
| **Decay（Epoch 3-100）** | 餘弦退火（Cosine Annealing）從 0.01 → lrf×lr0 = 0.01×0.01 = 1e-4 |

**實際 lr/pg0 追蹤（car_detect）**：

| Epoch | lr/pg0 |
|---|---|
| 1 | 0.000260 |
| 2 | 0.000535 |
| 3 | 0.000804（Warmup 結束） |
| 4 | 0.001067 |
| 10 | 0.001822 |
| 50 | 0.001030 |
| 90 | 0.000238 |
| 100 | 0.0000398 |

> 兩者的學習率曲線完全一致（相同設定），epoch 3 結束 warmup 後進入餘弦退火，最終降至約 **4×10⁻⁵**。

---

## 11. 視覺化結果

以下為訓練過程產生的視覺化圖表，位於各模型資料夾中：

### car_detect

| 圖表 | 檔案 | 說明 |
|---|---|---|
| 訓練曲線總覽 | [results.png](car_detect/results.png) | 所有 loss 與 metric 在 100 epoch 的走勢 |
| Box F1 曲線 | [BoxF1_curve.png](car_detect/BoxF1_curve.png) | 不同信心閾值下的 F1 分數 |
| Box PR 曲線 | [BoxPR_curve.png](car_detect/BoxPR_curve.png) | Precision-Recall 曲線 |
| Precision 曲線 | [BoxP_curve.png](car_detect/BoxP_curve.png) | 不同閾值下的 Precision |
| Recall 曲線 | [BoxR_curve.png](car_detect/BoxR_curve.png) | 不同閾值下的 Recall |
| 混淆矩陣 | [confusion_matrix.png](car_detect/confusion_matrix.png) | 最終模型的混淆矩陣 |
| 混淆矩陣（正規化） | [confusion_matrix_normalized.png](car_detect/confusion_matrix_normalized.png) | 正規化版本 |
| 資料集標籤分布 | [labels.jpg](car_detect/labels.jpg) | 標注框的空間分布與大小分布 |
| 訓練樣本（初始） | [train_batch0.jpg](car_detect/train_batch0.jpg) | 第一個 epoch 的訓練批次 |
| 訓練樣本（最終） | [train_batch1262.jpg](car_detect/train_batch1262.jpg) | 最後一個 epoch 的訓練批次 |
| 驗證預測 | [val_batch0_pred.jpg](car_detect/val_batch0_pred.jpg) | 模型在驗證集上的預測結果 |
| 驗證標注 | [val_batch0_labels.jpg](car_detect/val_batch0_labels.jpg) | 驗證集的真實標注 |

### car_detect_merged

| 圖表 | 檔案 | 說明 |
|---|---|---|
| 訓練曲線總覽 | [results.png](car_detect_merged/results.png) | 所有 loss 與 metric 在 100 epoch 的走勢 |
| Box F1 曲線 | [BoxF1_curve.png](car_detect_merged/BoxF1_curve.png) | 不同信心閾值下的 F1 分數 |
| Box PR 曲線 | [BoxPR_curve.png](car_detect_merged/BoxPR_curve.png) | Precision-Recall 曲線 |
| Precision 曲線 | [BoxP_curve.png](car_detect_merged/BoxP_curve.png) | 不同閾值下的 Precision |
| Recall 曲線 | [BoxR_curve.png](car_detect_merged/BoxR_curve.png) | 不同閾值下的 Recall |
| 混淆矩陣 | [confusion_matrix.png](car_detect_merged/confusion_matrix.png) | 最終模型的混淆矩陣 |
| 混淆矩陣（正規化） | [confusion_matrix_normalized.png](car_detect_merged/confusion_matrix_normalized.png) | 正規化版本 |
| 資料集標籤分布 | [labels.jpg](car_detect_merged/labels.jpg) | 標注框的空間分布與大小分布 |
| 訓練樣本（初始） | [train_batch0.jpg](car_detect_merged/train_batch0.jpg) | 第一個 epoch 的訓練批次 |
| 訓練樣本（最終） | [train_batch2702.jpg](car_detect_merged/train_batch2702.jpg) | 最後一個 epoch 的訓練批次 |
| 驗證預測 | [val_batch0_pred.jpg](car_detect_merged/val_batch0_pred.jpg) | 模型在驗證集上的預測結果 |
| 驗證標注 | [val_batch0_labels.jpg](car_detect_merged/val_batch0_labels.jpg) | 驗證集的真實標注 |

---

## 12. 結論與建議

### 12.1 綜合指標對比總結

| 評估面向 | 優勝者 | 說明 |
|---|---|---|
| **最終 mAP@50** | ✅ car_detect (0.840) | 高出 merged 7.7% |
| **最佳 mAP@50** | ✅ car_detect (0.864) | 高出 merged 5.5% |
| **最終 mAP@50-95** | ✅ car_detect (0.510) | 高出 merged 9.9% |
| **Precision（最終）** | ✅ car_detect_merged (0.922) | 高出 car_detect 13.5% |
| **Recall（最終）** | ✅ car_detect (0.782) | 高出 merged 9.2% |
| **F1-Score（最終）** | 🤝 接近（0.784 vs 0.789） | 差距 <1% |
| **Val Box Loss** | ✅ car_detect (1.347) | 低出 merged 15.1% |
| **訓練效率** | ✅ car_detect (44.9 min) | 較 merged 快 33% |
| **收斂速度（達最佳）** | ✅ car_detect_merged (ep59) | 較 car_detect 早 29 epochs |

### 12.2 問題診斷

#### car_detect_merged 表現不如預期的原因分析

1. **驗證集不匹配**：merged 的 val box loss 幾乎未改善（-1.9% vs car_detect 的 -14%），暗示驗證資料與訓練資料分布可能存在差異（兩個來源的標注風格/攝影機角度不同）。

2. **標注一致性問題**：兩位貢獻者的標注可能有不同標準（框的緊密程度、遮擋處理方式），導致模型學到的特徵較為混雜。

3. **Precision-Recall 失衡**：merged 模型在高精準但低召回的狀態下收斂，代表模型學到「謹慎預測」的策略，而非「全面偵測」。

4. **資料量雖然更多，但不保證更好**：數量增加但品質/一致性下降時，更多資料反而可能引入雜訊，干擾模型學習穩定的特徵。

### 12.3 建議

| 建議 | 說明 |
|---|---|
| **部署建議** | 若優先考慮整體偵測準確度（mAP），選用 **`car_detect/weights/best.pt`（epoch 88）** |
| **高精準應用** | 若應用場景需要「確定有車才回報」（低誤報），可考慮 **`car_detect_merged/weights/best.pt`（epoch 59）** |
| **資料品質優先** | 擴充資料集前，應統一標注規範（框的邊界定義、最小車輛尺寸等） |
| **延長訓練** | car_detect_merged 在 epoch 59 達最佳後持續衰退，建議加入 Early Stopping 或降低 patience |
| **分離驗證集** | 合併資料集時，建議為每個來源維持獨立的 validation set，以更準確評估泛化能力 |
| **資料增強調整** | 針對 merged 資料集，可嘗試增加 `copy_paste`、`mixup` 等增強手段提升泛化性 |
| **下一步實驗** | 嘗試更大型模型（如 yolo26s / yolo26m）或更長訓練（200 epochs）以觀察是否進一步提升 |

---

## 附錄：各 Epoch 完整指標

完整的逐 epoch 訓練資料請參閱：

- [car_detect/results.csv](car_detect/results.csv)
- [car_detect_merged/results.csv](car_detect_merged/results.csv)

---

*本報告由訓練資料自動分析生成，資料來源：`results.csv`、`args.yaml`、`data.yaml`*
