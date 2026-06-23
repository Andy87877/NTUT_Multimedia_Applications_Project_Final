# JetBot 專案部署目錄與 YOLO 引擎編譯指南

本說明文件整理了 JetBot 自走車端（Jetson Nano）的檔案目錄結構，以及如何編譯與放置 YOLOv4-tiny 號誌偵測模型，供您部署時快速查閱。

---

## 📂 JetBot 端完整目錄結構

請確保將檔案上傳至 JetBot 上的 `~/trt_yolov4-tiny-master/` 根目錄，並維持以下結構：

```text
~/trt_yolov4-tiny-master/
├── Final_team_1.ipynb                      # 合併控制主程式（包含控制滑桿與即時遙測介面）
├── utils/
│   └── yolo.py                             # YOLO 推論封裝模組
├── road_following_model/                   # 道路循跡模型資料夾
│   ├── best_steering_model_xy.pth          # PC 端訓練好的原始權重
│   ├── best_steering_model_xy.onnx         # 程式自動匯出的 ONNX 檔案
│   ├── best_steering_model_xy.engine       # trtexec 編譯出的引擎檔
│   └── best_steering_model_xy_trt.pth      # 封裝為 TRTModule 的最終載入檔
└── yolo/                                   # 🚥 YOLO 號誌偵測資料夾
    ├── yolov4-tiny-416.weights             # PC 端訓練好的 Darknet 原始權重
    ├── yolov4-tiny-416.cfg                 # 號誌偵測網路設定檔
    ├── obj.names                           # 號誌類別名稱檔（0=stop, 1=rail, 2=pedestrian, 3=blocked）
    └── yolov4-tiny-416.trt                 # 👈 編譯出來的 TensorRT 引擎檔
```

---

## 🛠️ YOLOv4-tiny 引擎編譯步驟

若您在車端尚未產生 `yolov4-tiny-416.trt`，請開啟 JetBot 的終端機（Terminal）並依序執行以下指令：

```bash
# 1. 切換至 yolo 資料夾
cd ~/trt_yolov4-tiny-master/yolo/

# 2. 將 Darknet 權重轉換為 ONNX 格式
# -c 代表類別數量（本專案為 4 類），-m 代表模型名稱
python3 yolo_to_onnx.py -c 4 -m yolov4-tiny-416

# 3. 將 ONNX 轉換為 TensorRT .trt 檔案
python3 onnx_to_tensorrt.py -c 4 -m yolov4-tiny-416
```

> [!NOTE]
> - 編譯過程大約需要 **2 ~ 5 分鐘**，編譯期間系統負載較高，請耐心等候。
> - 編譯完成後，同目錄下會生成 `yolov4-tiny-416.trt`，此時 `Final_team_1.ipynb` 便可載入 YOLO 模型。
