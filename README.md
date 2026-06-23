# 🏎️ JetBot 智能避障與多功能循跡期末專案 (Final Project)

本專案為 NTUT 多媒體應用期末專案。目標是結合本學期所學，在 **JetBot (Jetson Nano)** 自走車平台上部署一個雙模型並行推論系統，實現**自主道路跟隨**與**即時交通路牌辨識與自動控制**。

---

## 🎯 專案任務與評分標準 (70% 實車 DEMO + 30% 報告)

本專案的評分標準與規定詳見期末專案手冊：[Final_Project.pdf](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/Final_Project.pdf)

### 1. 道路自動駕駛功能 (30%)
* 讓 Jetbot 自動追蹤車道線並順暢行駛完賽道。
* 輪胎壓線一次扣 1 分。
* 輪胎完全超出車道一次扣 3 分。若完全脫離車道，需從脫離點重新開始。

### 2. 交通路牌辨識與防禦動作 (40%)
根據相機辨識出的交通路牌執行對應動作：
* 🛑 **停車再開 (Stop & Go)**：偵測到此號誌，原地停止 2 秒後繼續前進。(7.5%)
* 🚂 **鐵路平交道 (Railway)**：偵測到此號誌，原地停止 5 秒後繼續前進。(7.5%)
* 🚶 **當心行人 (Pedestrian)**：偵測到此號誌，自動減速為 0.7x 基礎速度行駛，通過後自動恢復。(7.5%)
* 🚧 **道路封閉 (Blocked)**：偵測到此號誌，於號誌牌前安全停車，並永久結束程式運算。(7.5%)
* 🛡️ **防假路標干擾機制**：自走車應忽略賽道旁遠處、背景或假的路標貼紙。若對假路標產生誤動作，每次扣 2 分。(10%)

### 3. 書面報告與實體 DEMO (30%)
* 實體驗收演示（期末考）。
* 繳交小組報告與個人報告。

---

## 📂 專案目錄與檔案結構

為了方便您後續的修改與維護，專案目錄已整理如下：

```
Final/
├── Final_team_1.ipynb            # 🚀 主部署 Jupyter Notebook (包含分單元測試與最終合併版)
├── 使用說明.md                   # 📝 車端部署、模型轉換與滑桿調參詳細說明書
├── 小組報告.md                   # 📊 期末小組書面報告草稿 (符合大綱要求)
├── Final_Project.pdf             # 📕 期末專案規範官方 PDF
├── README.md                     # 🏠 專案導覽與快速入口 (本檔案)
└── _reference_pre_project/       # 📂 先前專案與訓練程式碼參考庫
    ├── Project4_object_detect/   # 🔍 Project 4: 基礎 YOLO 物件偵測
    ├── Project5_road_detect/     # 🛣️ Project 5: ResNet-18 道路循跡訓練與蒐集
    └── Project6_sign_detect/     # 🛑 Project 6: YOLOv4-tiny 號誌偵測訓練與權重
```

### 📍 核心檔案說明：
* **[Final_team_1.ipynb](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/Final_team_1.ipynb)**: **控制核心**。內含三個單元：
  * **單元一**：純道路跟隨（僅載入 ResNet 進行 PID 與轉向調校，不載入 YOLO）。
  * **單元二**：純交通號誌動作測試（僅載入 YOLO 直線前進測試停/走動作）。
  * **單元三**：最終雙模型合併版（雙模型並行推論，完整賽道自駕）。
* **[使用說明.md](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/%E4%BD%BF%E7%94%A8%E8%AA%AA%E6%98%8E.md)**: 提供實車部署時，如何傳輸檔案、編譯道路模型 `.engine` 與 YOLO `.trt` 引擎，以及滑桿參數的調參指南。
* **[小組報告.md](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/%E5%B0%8F%E7%B5%84%E5%A0%B1%E5%91%8A.md)**: 已為您撰寫好的小組書面報告草稿，包含分工、實驗結果與遭遇相機硬體問題的防錯應對。
* **[_reference_pre_project/](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/_reference_pre_project)**: 存放這學期 Project 4 ~ 6 的歷史程式碼與訓練權重，便於您回顧或重新訓練模型。

---

## 🚀 快速啟動與修改指引

當您需要修改或重新測試專案時：

1. **若要調整道路循跡模型**：
   * 進入 `_reference_pre_project/Project5_road_detect/` 下的 [train_model.ipynb](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/_reference_pre_project/Project5_road_detect/train_model.ipynb) 重新訓練或調整網路結構，生成新的 `best_steering_model_xy.pth`。
2. **若要重新訓練路標辨識 YOLO 模型**：
   * 進入 `_reference_pre_project/Project6_sign_detect/` 下的 [scripts/](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/_reference_pre_project/Project6_sign_detect/scripts/) 目錄修改設定與執行訓練，重新匯出 `.cfg` 與 `.weights` 權重檔。
3. **若要修改控制邏輯（例如調整 Cooldown 時間或速度比率）**：
   * 修改主目錄下的 [Final_team_1.ipynb](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/Final_team_1.ipynb) 的推論與馬達速度控制迴圈（對應單元一、二、三的推論函數）。
4. **實車部署**：
   * 請嚴格遵循 **[使用說明.md](file:///c:/Users/andy8/Desktop/NTUT_Media/Final/%E4%BD%BF%E7%94%A8%E8%AA%AA%E6%98%8E.md)** 的步驟，將檔案傳輸至 JetBot 上進行 TensorRT 編譯與現場測試。
