# -*- coding: utf-8 -*-
"""
main.py — Detect_random_imgs
=============================
1. 從三支 YouTube 影片中隨機擷取圖片，存入 imgs_original/
2. 用 YOLO26 對每張圖片做物件偵測，把標記結果存入 imgs_detected/

YouTube 影片：
  - https://www.youtube.com/watch?v=5LmEo7RmojM  (晴天 / 高速公路+一般道路)
  - https://www.youtube.com/watch?v=BTmncLlPGvk  (大雨天)
  - https://www.youtube.com/watch?v=N-P79qS3EIY  (夜晚)

使用方式：
  python main.py
  python main.py --frames 8          # 每支影片擷取 8 張 (預設 5)
  python main.py --conf 0.3          # YOLO 信心門檻 (預設 0.4)
  python main.py --weights path/to/best.pt
"""

import os
import sys
import json
import random
import argparse
import subprocess
from pathlib import Path

import cv2

# ─────────────────────────────────────────────
# 影片清單  (URL, 標籤)
# ─────────────────────────────────────────────

# ("https://www.youtube.com/watch?v=5LmEo7RmojM", "sunny"),
VIDEOS = [
    ("https://www.youtube.com/watch?v=CcTK5nQGhmg", "highway"),
    ("https://www.youtube.com/watch?v=BTmncLlPGvk", "rainy"),
    ("https://www.youtube.com/watch?v=N-P79qS3EIY", "night"),
]

# ─────────────────────────────────────────────
# 預設路徑
# ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

IMGS_ORIGINAL = SCRIPT_DIR / "imgs_original"
IMGS_DETECTED = SCRIPT_DIR / "imgs_detected"

DEFAULT_WEIGHTS = PROJECT_DIR / "runs_yolo26" / \
    "car_detect_merged" / "weights" / "best.pt"


# ═══════════════════════════════════════════════════════════════
# Part 1 — 從 YouTube 影片隨機擷取圖片
# ═══════════════════════════════════════════════════════════════

def get_video_info(url: str) -> dict | None:
    """用 yt-dlp 取得影片的 duration（秒）與最佳串流 URL。"""
    try:
        cmd = ["yt-dlp", "--dump-json", "--no-playlist", url]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  [yt-dlp] 取得 metadata 失敗: {result.stderr[:200]}")
            return None
        info = json.loads(result.stdout)
        return info
    except FileNotFoundError:
        print("錯誤：找不到 yt-dlp。請先安裝：pip install yt-dlp")
        sys.exit(1)
    except Exception as e:
        print(f"  [yt-dlp] 例外: {e}")
        return None


def get_stream_url(url: str) -> str | None:
    """用 yt-dlp 取得 mp4 串流直連 URL（不下載完整影片）。"""
    try:
        cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]/bestvideo/best",
               "-g", "--no-playlist", url]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  [yt-dlp] 取得串流 URL 失敗: {result.stderr[:200]}")
            return None
        stream_url = result.stdout.strip().splitlines()[0]
        return stream_url
    except Exception as e:
        print(f"  [yt-dlp] 例外: {e}")
        return None


def capture_random_frames(url: str, label: str, n_frames: int, out_dir: Path, max_retries: int = 2) -> list[Path]:
    """
    從指定 YouTube 影片隨機擷取 n_frames 張圖片。
    回傳成功儲存的圖片路徑清單。
    支援失敗重試機制。
    """
    print(f"\n{'─'*50}")
    print(f"[{label}] 取得影片資訊中…")
    print(f"  URL: {url}")

    # 1. 取得影片時長
    info = get_video_info(url)
    if info is None:
        print(f"  ✗ 無法取得 metadata，跳過 {label}")
        return []

    duration: float = info.get("duration", 0)
    title: str = info.get("title", label)[:60]
    print(f"  ✓ 標題: {title}")
    print(f"  ✓ 時長: {duration:.0f} 秒 ({duration/60:.1f} 分鐘)")

    if duration < 10:
        print(f"  ✗ 影片太短（< 10 秒），跳過")
        return []

    # 2. 取得串流 URL（含重試）
    print(f"  正在取得串流 URL…")
    stream_url = None
    for attempt in range(max_retries + 1):
        stream_url = get_stream_url(url)
        if stream_url:
            print(f"  ✓ 串流 URL 取得成功")
            break
        elif attempt < max_retries:
            print(f"  ⚠ 取得失敗，{attempt + 1}/{max_retries} 重試中…")
        else:
            print(f"  ✗ 無法取得串流 URL，跳過 {label}")
            return []

    # 3. 開啟串流（含重試）
    cap = None
    for attempt in range(max_retries + 1):
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  ✓ 開啟成功 | FPS: {fps:.1f} | 解析度: {width}x{height}")
            break
        elif attempt < max_retries:
            print(f"  ⚠ 開啟失敗，{attempt + 1}/{max_retries} 重試中…")
        else:
            print(f"  ✗ 無法開啟串流，跳過 {label}")
            return []

    # 4. 隨機選取時間點（避開開頭/結尾 5% 避免廣告或黑畫面）
    margin = duration * 0.05
    safe_start = max(0, margin)
    safe_end = min(duration, duration - margin)
    if safe_end <= safe_start:
        safe_start, safe_end = 0, duration

    safe_range = int(safe_end - safe_start)
    n_to_capture = min(n_frames, safe_range)
    if n_to_capture == 0:
        print(f"  ✗ 安全範圍太小，無法選取時間點")
        cap.release()
        return []

    timestamps_sec = sorted(random.sample(
        range(int(safe_start), int(safe_end)),
        n_to_capture
    ))
    print(f"  隨機時間點（秒）: {timestamps_sec}")

    # 5. 逐一 seek 並截圖
    saved: list[Path] = []
    for i, t_sec in enumerate(timestamps_sec):
        ms = t_sec * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, ms)

        ret, frame = cap.read()
        if not ret or frame is None:
            # 重試一次（略微調整時間）
            for offset_ms in [500, -500, 1000, -1000]:
                cap.set(cv2.CAP_PROP_POS_MSEC, ms + offset_ms)
                ret, frame = cap.read()
                if ret and frame is not None:
                    break

        if not ret or frame is None:
            print(f"  ⚠ 第 {i+1:2d} 張: t={t_sec}s 擷取失敗，略過")
            continue

        filename = f"{label}_{i+1:02d}_t{t_sec:05d}s.jpg"
        filepath = out_dir / filename
        cv2.imwrite(str(filepath), frame)
        print(
            f"  ✓ 第 {i+1:2d} 張: {filename}  ({frame.shape[1]}x{frame.shape[0]})")
        saved.append(filepath)

    cap.release()
    print(f"  [{label}] 共擷取 {len(saved)} / {n_frames} 張")
    return saved


# ═══════════════════════════════════════════════════════════════
# Part 2 — YOLO26 物件偵測
# ═══════════════════════════════════════════════════════════════

def detect_images_yolo26(
    image_paths: list[Path],
    weights: Path,
    out_dir: Path,
    conf: float = 0.25,
    iou: float = 0.6,
):
    """
    對所有圖片跑 YOLO26 推論，將標記結果存入 out_dir。
    """
    if not image_paths:
        print("\n沒有圖片可以偵測，略過 YOLO26。")
        return

    print(f"\n{'═'*50}")
    print("YOLO26 物件偵測")
    print(f"{'═'*50}")
    print(f"權重: {weights}")
    print(f"圖片數量: {len(image_paths)}")
    print(f"信心門檻: {conf}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 找不到 ultralytics。請先安裝：pip install ultralytics")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA 不可用，使用 CPU（速度較慢）")

    print(f"\n載入模型中…")
    model = YOLO(str(weights))

    out_dir.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        result = results[0]
        n = len(result.boxes)
        total_detections += n

        # 取得標記後的圖片（ultralytics 已繪製 BBox + 標籤）
        annotated = result.plot()

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"  ✓ {img_path.name}  →  {n} 個偵測結果  →  {out_path.name}")

    print(f"\n完成！共偵測 {len(image_paths)} 張圖片，總計 {total_detections} 個物件。")
    print(f"結果儲存於: {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Random YouTube frame capture + YOLO26 detection")
    parser.add_argument("--frames", "-n", type=int, default=20,
                        help="每支影片隨機擷取的圖片數量 (預設: 20)")
    parser.add_argument("--weights", "-w", type=str, default=str(DEFAULT_WEIGHTS),
                        help=f"YOLO26 權重路徑 (預設: {DEFAULT_WEIGHTS})")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="YOLO 信心門檻 (預設: 0.4)")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="YOLO IoU 門檻 (預設: 0.6)")
    parser.add_argument("--seed", type=int, default=None,
                        help="亂數種子（設定後每次結果相同）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] 找不到 YOLO26 權重: {weights}")
        print("請確認訓練已完成，或用 --weights 指定正確路徑")
        sys.exit(1)

    # 建立輸出資料夾
    IMGS_ORIGINAL.mkdir(parents=True, exist_ok=True)
    IMGS_DETECTED.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("YouTube 隨機擷圖 + YOLO26 車輛偵測")
    print("=" * 50)
    print(f"每部影片擷取: {args.frames} 張")
    print(f"輸出（原始）: {IMGS_ORIGINAL}")
    print(f"輸出（偵測）: {IMGS_DETECTED}")

    # ── Part 1: 擷取圖片 ──────────────────────────
    all_images: list[Path] = []
    for url, label in VIDEOS:
        saved = capture_random_frames(url, label, args.frames, IMGS_ORIGINAL)
        all_images.extend(saved)

    print(f"\n{'═'*50}")
    print(f"圖片擷取完成，共 {len(all_images)} 張")

    # ── Part 2: YOLO26 偵測 ───────────────────────
    detect_images_yolo26(
        image_paths=all_images,
        weights=weights,
        out_dir=IMGS_DETECTED,
        conf=args.conf,
        iou=args.iou,
    )

    print("\n全部完成！")


if __name__ == "__main__":
    main()
