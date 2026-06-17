# -*- coding: utf-8 -*-
"""
detect_video.py
===============
使用訓練好的 YOLOv4-tiny 模型偵測影片中的車輛，並輸出標記結果影片

等效於 Colab 上的:
  !./darknet detector demo obj.data yolov4-custom.cfg backup/yolov4_custom_best.weights
    input.mp4 -dont_show -out_filename output.mp4

使用方式:
  python detect_video_yoloV4.py --video path/to/video.mp4
  python detect_video_yoloV4.py --video path/to/video.mp4 --output result.mp4
  python detect_video_yoloV4.py --video path/to/video.mp4 --weights backup/yolov4-tiny-custom_best.pth
"""


from tool.torch_utils import do_detect
from tool.utils import load_class_names
from tool.darknet2pytorch import Darknet
import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch

# 將 darknet 加入路徑
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DARKNET_DIR = os.path.join(PROJECT_DIR, "darknet")
sys.path.insert(0, DARKNET_DIR)


def detect_video(cfgfile, weightfile, videofile, namesfile,
                 conf_thresh=0.4, nms_thresh=0.6, output_path=None):
    """偵測影片中的車輛"""
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

    # 開啟影片
    cap = cv2.VideoCapture(videofile)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {videofile}")
        return

    # 影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\n影片資訊:")
    print(f"  檔案: {videofile}")
    print(f"  解析度: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  總幀數: {total_frames}")
    print(f"  時長: {duration:.1f} 秒")

    # 設定輸出
    if output_path is None:
        base, ext = os.path.splitext(videofile)
        output_path = base + "_detected" + ext

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"❌ 無法建立輸出影片: {output_path}")
        cap.release()
        return

    # 顏色設定
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    # 開始處理
    print(f"\n開始處理影片...")
    print(f"輸出至: {output_path}\n")

    frame_count = 0
    total_detections = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 前處理
        sized = cv2.resize(frame, (model.width, model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        # 偵測
        with torch.no_grad():
            boxes = do_detect(model, sized, conf_thresh, nms_thresh, use_cuda)

        # 畫框
        n_detections = 0
        if len(boxes) > 0 and len(boxes[0]) > 0:
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 畫標籤
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10),
                              (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                n_detections += 1

        total_detections += n_detections

        # 在左上角顯示資訊
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {n_detections}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 寫入輸出影片
        out.write(frame)

        # 進度顯示
        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            eta = (total_frames - frame_count) / \
                fps_actual if fps_actual > 0 else 0
            print(f"  進度: {progress:.1f}% | 幀: {frame_count}/{total_frames} | "
                  f"FPS: {fps_actual:.1f} | 預計剩餘: {eta:.0f}秒")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ 影片處理完成！")
    print(f"  處理幀數: {frame_count}")
    print(f"  總偵測數: {total_detections}")
    print(f"  平均每幀偵測: {total_detections/max(frame_count,1):.1f}")
    print(f"  處理時間: {elapsed:.1f} 秒")
    print(f"  平均 FPS: {frame_count/max(elapsed,1):.1f}")
    print(f"  輸出檔案: {output_path}")
    print(f"{'='*60}")
    print(f"\n📤 將 {output_path} 上傳至 YouTube 即可完成專案！")


def main():
    parser = argparse.ArgumentParser(description='YOLOv4-tiny 影片偵測')
    parser.add_argument('--video', '-v', type=str,
                        required=True, help='要偵測的影片路徑')
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
                        default=None, help='輸出影片路徑')
    args = parser.parse_args()

    detect_video(args.cfg, args.weights, args.video, args.names,
                 args.conf, args.nms, args.output)


if __name__ == "__main__":
    main()
