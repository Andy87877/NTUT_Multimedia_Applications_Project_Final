# -*- coding: utf-8 -*-
"""
detect_image.py
===============
使用訓練好的 YOLOv4-tiny 模型偵測單張圖片中的車輛

等效於 Colab 上的:
  !./darknet detect yolov4-custom.cfg backup/yolov4_custom_best.weights test.jpg

使用方式:
  python detect_image_yoloV4.py --image path/to/image.jpg
  python detect_image_yoloV4.py --image path/to/image.jpg --weights backup/yolov4-tiny-custom_best.pth
"""

from tool.torch_utils import do_detect
from tool.utils import load_class_names
from tool.darknet2pytorch import Darknet
import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image

# 將 darknet 加入路徑
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DARKNET_DIR = os.path.join(PROJECT_DIR, "darknet")
sys.path.insert(0, DARKNET_DIR)


def detect_image(cfgfile, weightfile, imgfile, namesfile, conf_thresh=0.4, nms_thresh=0.6, output_path=None):
    """偵測單張圖片"""
    # 強制檢查 GPU 是否可用
    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用！此腳本需要 NVIDIA GPU。")
        print("  請安裝 CUDA 版本的 PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    print(f"GPU 已檢測: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 載入模型
    print(f"\n載入模型設定: {cfgfile}")
    model = Darknet(cfgfile)

    print(f"載入權重: {weightfile}")
    if weightfile.endswith('.pth'):
        model.load_state_dict(torch.load(weightfile, map_location='cuda'))
    else:
        model.load_weights(weightfile)

    use_cuda = True
    model.cuda()
    print(f"模型已移到 GPU")

    model.eval()

    # 載入類別名稱
    class_names = load_class_names(namesfile)
    print(f"類別: {class_names}")

    # 讀取圖片
    img = cv2.imread(imgfile)
    if img is None:
        print(f"無法讀取圖片: {imgfile}")
        return

    print(f"圖片尺寸: {img.shape[1]}x{img.shape[0]}")

    # 前處理
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    # 偵測
    print("正在偵測...")
    boxes = do_detect(model, sized, conf_thresh, nms_thresh, use_cuda)

    # 畫框
    result_img = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255)]
    n_detections = 0

    if len(boxes) > 0 and len(boxes[0]) > 0:
        width = img.shape[1]
        height = img.shape[0]

        for box in boxes[0]:
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            conf = box[5]
            cls_id = int(box[6])

            color = colors[cls_id % len(colors)]
            label = f"{class_names[cls_id]}: {conf:.2f}"

            # 畫框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

            # 畫標籤背景
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_img, (x1, y1 - th - 10),
                          (x1 + tw, y1), color, -1)
            cv2.putText(result_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            n_detections += 1
            print(f"  偵測到: {label} at ({x1},{y1})-({x2},{y2})")

    print(f"\n共偵測到 {n_detections} 個目標")

    # 儲存結果
    if output_path is None:
        base, ext = os.path.splitext(imgfile)
        output_path = base + "_predictions" + ext

    cv2.imwrite(output_path, result_img)
    print(f"結果已儲存至: {output_path}")

    # 顯示結果
    pass

    return result_img


def main():
    parser = argparse.ArgumentParser(description='YOLOv4-tiny 圖片偵測')
    parser.add_argument('--image', '-i', type=str,
                        required=True, help='要偵測的圖片路徑')
    parser.add_argument('--cfg', type=str,
                        default=os.path.join(
                            PROJECT_DIR, "cfg", "yolov4-tiny-custom.cfg"),
                        help='模型設定檔')
    parser.add_argument('--weights', '-w', type=str,
                        default=os.path.join(
                            PROJECT_DIR, "backup", "yolov4-tiny-custom_best.pth"),
                        help='權重檔路徑')
    parser.add_argument('--names', type=str,
                        default=os.path.join(PROJECT_DIR, "cfg", "obj.names"),
                        help='類別名稱檔')
    parser.add_argument('--conf', type=float, default=0.4, help='信心度門檻')
    parser.add_argument('--nms', type=float, default=0.6, help='NMS 門檻')
    parser.add_argument('--output', '-o', type=str,
                        default=None, help='輸出圖片路徑')
    args = parser.parse_args()

    detect_image(args.cfg, args.weights, args.image, args.names,
                 args.conf, args.nms, args.output)


if __name__ == "__main__":
    main()
