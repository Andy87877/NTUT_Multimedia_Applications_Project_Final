# -*- coding: utf-8 -*-
"""
train_yolov4tiny_darknet.py
============================
Local replacement for Google Colab YOLOv4-tiny training.

Dataset  : _SignDetection.yolov4pytorch
Output   : jetbot_deploy/yolov4-tiny-416.cfg
           jetbot_deploy/yolov4-tiny-416.weights
           jetbot_deploy/obj.names

JetBot deployment (Section 7 in lecture notes):
  1. Copy jetbot_deploy/* -> trt_yolv4-tiny-master/yolo/
  2. python3 yolo_to_onnx.py -c 4 -m yolov4-tiny-416
  3. python3 onnx_to_tensorrt.py -c 4 -m yolov4-tiny-416

Class ID order (matches Project06.ipynb):
  0 = stop        -> stop 2 sec
  1 = rail        -> stop 5 sec
  2 = pedestrian  -> slow down x0.7
  3 = blocked     -> stop immediately
"""

import urllib.request
import subprocess
import argparse
import shutil
import cv2
import os
from pathlib import Path
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding='utf-8', errors='replace')


# ================================================================
#  Constants
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATASET_DIR = PROJECT_ROOT / "_SignDetection.yolov4pytorch" / "train"
ANN_FILE = DATASET_DIR / "_annotations.txt"

OBJ_DIR = PROJECT_ROOT / "obj"
BACKUP_DIR = PROJECT_ROOT / "backup"
OUTPUT_DIR = PROJECT_ROOT / "jetbot_deploy"

# Class order that matches Project06.ipynb
# sign[1]==0 stop, sign[1]==1 rail, sign[1]==2 pedestrian, sign[1]==3 blocked
CLASS_NAMES = ["stop", "rail", "pedestrian", "blocked"]
NUM_CLASSES = 4

# Dataset _classes.txt order: 0=blocked 1=pedestrian 2=rail 3=stop
# Remap: dataset_id -> training_id
# blocked->3, pedestrian->2, rail->1, stop->0
REMAP = {0: 3, 1: 2, 2: 1, 3: 0}

# Darknet hyper-params for 4 classes
MAX_BATCHES = NUM_CLASSES * 2000     # 8000
STEPS = f"{int(MAX_BATCHES*0.8)},{int(MAX_BATCHES*0.9)}"  # 6400,7200
FILTERS = (NUM_CLASSES + 5) * 3  # 27


# ================================================================
#  STEP 1  Convert annotations -> obj/ flat folder
# ================================================================
def prepare_dataset():
    print("\n[STEP 1] Preparing obj/ dataset folder ...")
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    ok = skip = 0
    with open(ANN_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            fname = parts[0]
            src = DATASET_DIR / fname
            if not src.exists():
                print(f"  [WARN] not found: {fname}")
                skip += 1
                continue

            img = cv2.imread(str(src))
            if img is None:
                skip += 1
                continue
            H, W = img.shape[:2]

            shutil.copy2(src, OBJ_DIR / fname)

            label_lines = []
            for ann in parts[1:]:
                try:
                    x1, y1, x2, y2, raw_id = map(int, ann.split(","))
                except ValueError:
                    continue
                cls_id = REMAP.get(raw_id, raw_id)   # remap to ipynb order
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                if bw > 0 and bh > 0:
                    label_lines.append(
                        f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (OBJ_DIR / (Path(fname).stem + ".txt")
             ).write_text("\n".join(label_lines))
            ok += 1

    print(f"  Images converted: {ok}  (skipped: {skip})")

    # obj.names  (order = ipynb class IDs)
    obj_names = CONFIG_DIR / "obj.names"
    obj_names.write_text("\n".join(CLASS_NAMES) + "\n")
    print(f"  obj.names  -> {obj_names}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"               {i} = {n}")

    # train.txt
    imgs = sorted(OBJ_DIR.glob("*.jpg"))
    train_txt = CONFIG_DIR / "train.txt"
    train_txt.write_text("\n".join(str(p) for p in imgs) + "\n")
    print(f"  train.txt  -> {len(imgs)} images")

    # obj.data
    obj_data = CONFIG_DIR / "obj.data"
    obj_data.write_text(
        f"classes = {NUM_CLASSES}\n"
        f"train   = {train_txt}\n"
        f"valid   = {train_txt}\n"
        f"names   = {obj_names}\n"
        f"backup  = {BACKUP_DIR}\n"
    )
    print(f"  obj.data   -> {obj_data}")


# ================================================================
#  STEP 2  Generate yolov4-tiny-custom.cfg
# ================================================================
def make_cfg():
    print(f"\n[STEP 2] Generating yolov4-tiny-custom.cfg ...")
    cfg_path = CONFIG_DIR / "yolov4-tiny-custom.cfg"

    cfg = f"""[net]
batch=64
subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation=1.5
exposure=1.5
hue=.1
learning_rate=0.00261
burn_in=1000
max_batches={MAX_BATCHES}
policy=steps
steps={STEPS}
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1,-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1,-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1,-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

##################################

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={FILTERS}
activation=linear

[yolo]
mask=3,4,5
anchors=10,14, 23,27, 37,58, 81,82, 135,169, 344,319
classes={NUM_CLASSES}
num=6
jitter=.3
scale_x_y=1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh=.7
truth_thresh=1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6

[route]
layers=-4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers=-1,23

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={FILTERS}
activation=linear

[yolo]
mask=0,1,2
anchors=10,14, 23,27, 37,58, 81,82, 135,169, 344,319
classes={NUM_CLASSES}
num=6
jitter=.3
scale_x_y=1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh=.7
truth_thresh=1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6
"""

    cfg_path.write_text(cfg)
    print(
        f"  classes={NUM_CLASSES}  filters={FILTERS}  max_batches={MAX_BATCHES}  steps={STEPS}")
    print(f"  Saved -> {cfg_path}")
    return cfg_path


# ================================================================
#  STEP 3  Download pretrained conv weights
# ================================================================
def download_weights():
    print("\n[STEP 3] Downloading yolov4-tiny.conv.29 ...")
    dst = PROJECT_ROOT / "yolov4-tiny.conv.29"
    if dst.exists():
        print(f"  Already exists: {dst.name}  ({dst.stat().st_size//1024} KB)")
        return dst
    url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29"
    print(f"  Downloading from GitHub ...")
    urllib.request.urlretrieve(url, dst)
    print(f"  Saved -> {dst}  ({dst.stat().st_size//1024} KB)")
    return dst


# ================================================================
#  STEP 4  Detect darknet binary
# ================================================================
def find_darknet():
    candidates = [
        PROJECT_ROOT / "darknet.exe",
        PROJECT_ROOT / "darknet" / "darknet.exe",
        Path(r"C:\darknet\darknet.exe"),
    ]
    for c in candidates:
        if c.exists():
            print(f"  [OK] darknet.exe found: {c}")
            return ("win", str(c))

    # WSL check
    if shutil.which("wsl"):
        result = subprocess.run(
            ["wsl", "bash", "-c",
                "which darknet 2>/dev/null || ls /tmp/darknet/darknet 2>/dev/null"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            wsl_path = result.stdout.strip()
            print(f"  [OK] darknet in WSL: {wsl_path}")
            return ("wsl", wsl_path)

    return None


# ================================================================
#  STEP 5  Build darknet via WSL
# ================================================================
def build_via_wsl():
    print("\n[STEP 4b] Building darknet in WSL ...")
    script = (
        "sudo apt-get install -y build-essential git 2>/dev/null | tail -1; "
        "if [ ! -d /tmp/darknet ]; then git clone https://github.com/AlexeyAB/darknet /tmp/darknet 2>&1 | tail -3; fi; "
        "cd /tmp/darknet && make -j$(nproc) 2>&1 | tail -5; "
        "ls /tmp/darknet/darknet && echo BUILD_SUCCESS"
    )
    result = subprocess.run(
        ["wsl", "bash", "-c", script],
        capture_output=False, text=True
    )
    if result.returncode == 0:
        return ("wsl", "/tmp/darknet/darknet")
    return None


# ================================================================
#  STEP 5  Train
# ================================================================
def w2wsl(p):
    """Convert Windows path to WSL /mnt/... path."""
    p = str(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def do_train(darknet_info, resume=False):
    kind, bin_path = darknet_info

    obj_data = PROJECT_ROOT / "obj.data"
    cfg_path = PROJECT_ROOT / "yolov4-tiny-custom.cfg"
    conv_w = PROJECT_ROOT / "yolov4-tiny.conv.29"
    last_w = BACKUP_DIR / "yolov4-tiny-custom_last.weights"

    weights_arg = str(last_w) if (resume and last_w.exists()) else str(conv_w)

    print(f"\n[STEP 5] Starting training ...")
    print(f"  Binary    : {bin_path} ({kind})")
    print(f"  Weights   : {weights_arg}")
    print(f"  Max iter  : {MAX_BATCHES}")

    if kind == "wsl":
        # Write a WSL-path version of obj.data so darknet inside WSL can read it
        obj_names = PROJECT_ROOT / "obj.names"
        train_txt = PROJECT_ROOT / "train.txt"

        # WSL train.txt: convert each Windows path in train.txt to /mnt/... format
        wsl_train_txt = PROJECT_ROOT / "train_wsl.txt"
        with open(train_txt, encoding="utf-8") as f:
            lines = [w2wsl(l.strip()) for l in f if l.strip()]
        wsl_train_txt.write_text("\n".join(lines) + "\n")

        wsl_backup = w2wsl(BACKUP_DIR)
        wsl_data_path = PROJECT_ROOT / "obj_wsl.data"
        wsl_data_path.write_text(
            f"classes = {NUM_CLASSES}\n"
            f"train   = {w2wsl(wsl_train_txt)}\n"
            f"valid   = {w2wsl(wsl_train_txt)}\n"
            f"names   = {w2wsl(obj_names)}\n"
            f"backup  = {wsl_backup}\n"
        )
        print(f"  WSL data  : {wsl_data_path.name}")

        cmd = (
            f"{bin_path} detector train "
            f"{w2wsl(wsl_data_path)} {w2wsl(cfg_path)} {w2wsl(weights_arg)} -dont_show"
        )
        print(f"  Command   : {cmd}\n")
        subprocess.run(["wsl", "bash", "-c", cmd])

    elif kind == "win":
        cmd = f'"{bin_path}" detector train "{obj_data}" "{cfg_path}" "{weights_arg}" -dont_show'
        print(f"  Command   : {cmd}\n")
        subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))


# ================================================================
#  STEP 6  Package for JetBot
# ================================================================
def package_for_jetbot():
    print("\n[STEP 6] Packaging for JetBot ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_w = BACKUP_DIR / "yolov4-tiny-custom_best.weights"
    last_w = BACKUP_DIR / "yolov4-tiny-custom_last.weights"
    cfg_s = PROJECT_ROOT / "yolov4-tiny-custom.cfg"

    # Use best, fallback to last
    src_w = None
    for w in [best_w, last_w]:
        if w.exists():
            src_w = w
            break

    if src_w is None:
        print("  [ERROR] No weights found in backup/. Run training first.")
        return

    dst_cfg = OUTPUT_DIR / "yolov4-tiny-416.cfg"
    dst_w = OUTPUT_DIR / "yolov4-tiny-416.weights"
    dst_n = OUTPUT_DIR / "obj.names"

    shutil.copy2(cfg_s, dst_cfg)
    shutil.copy2(src_w, dst_w)
    shutil.copy2(PROJECT_ROOT / "obj.names", dst_n)

    print(f"  Source weights : {src_w.name}")
    print(f"  {dst_cfg.name}")
    print(f"  {dst_w.name}")
    print(f"  {dst_n.name}")

    # Write deployment README
    readme = OUTPUT_DIR / "DEPLOY.txt"
    readme.write_text(
        "== JetBot Deployment (Section 7 of lecture notes) ==\n\n"
        "1. Copy all files to JetBot:  trt_yolv4-tiny-master/yolo/\n"
        "   - yolov4-tiny-416.cfg\n"
        "   - yolov4-tiny-416.weights\n\n"
        "2. Open Terminal in trt_yolv4-tiny-master/yolo/\n\n"
        "3. Convert to ONNX:\n"
        "   python3 yolo_to_onnx.py -c 4 -m yolov4-tiny-416\n"
        "   -> generates: yolov4-tiny-416.onnx\n\n"
        "4. Convert to TensorRT:\n"
        "   python3 onnx_to_tensorrt.py -c 4 -m yolov4-tiny-416\n"
        "   -> generates: yolov4-tiny-416.trt\n\n"
        "5. In Project06.ipynb Cell 1:\n"
        "   trt_yolo = TRT_YOLO('yolov4-tiny-416', (416, 416), 4)\n\n"
        "Class IDs (match ipynb):\n"
        "  0 = stop        -> robot stops 2 sec\n"
        "  1 = rail        -> robot stops 5 sec\n"
        "  2 = pedestrian  -> robot slows down x0.7\n"
        "  3 = blocked     -> robot stops immediately\n"
    )

    print(f"\n  Output folder: {OUTPUT_DIR}")
    print(f"\n  Copy to JetBot:")
    print(
        f"    scp {OUTPUT_DIR}\\*.cfg jetbot@<IP>:~/trt_yolv4-tiny-master/yolo/")
    print(
        f"    scp {OUTPUT_DIR}\\*.weights jetbot@<IP>:~/trt_yolv4-tiny-master/yolo/")
    print(f"\n  On JetBot:")
    print(f"    python3 yolo_to_onnx.py -c 4 -m yolov4-tiny-416")
    print(f"    python3 onnx_to_tensorrt.py -c 4 -m yolov4-tiny-416")


# ================================================================
#  Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv4-tiny Local Training (replaces Google Colab)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mode", "-m", default="prepare",
                        choices=["prepare", "train",
                                 "resume", "package", "all"],
                        help=(
                            "prepare  - Convert dataset + generate all config files\n"
                            "train    - Run darknet training (auto-detects WSL or darknet.exe)\n"
                            "resume   - Resume from last checkpoint\n"
                            "package  - Rename outputs -> yolov4-tiny-416.* for JetBot\n"
                            "all      - Full pipeline (prepare + train + package)\n"
                        )
                        )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  YOLOv4-tiny Training  (NTUT Project 6)")
    print("=" * 60)
    print(f"  Mode       : {args.mode.upper()}")
    print(f"  Classes    : {NUM_CLASSES} -> {CLASS_NAMES}")
    print(f"  max_batches: {MAX_BATCHES}")
    print(f"  filters    : {FILTERS}")
    print("=" * 60)

    if args.mode in ("prepare", "all"):
        prepare_dataset()
        make_cfg()
        download_weights()

    if args.mode in ("train", "resume", "all"):
        print("\n[STEP 4] Detecting darknet binary ...")
        darknet = find_darknet()

        if darknet is None and shutil.which("wsl"):
            print("  darknet not found, trying WSL build ...")
            darknet = build_via_wsl()

        if darknet is None:
            print("\n" + "=" * 60)
            print("  [!] darknet binary not found.")
            print()
            print("  Choose one option:")
            print()
            print("  A) Download darknet.exe (easiest):")
            print("     https://github.com/AlexeyAB/darknet/releases")
            print("     -> place darknet.exe in this folder")
            print("     -> run: python train_yolov4tiny_darknet.py --mode train")
            print()
            print("  B) WSL2 (if WSL not installed):")
            print("     1. PowerShell (Admin): wsl --install -d Ubuntu")
            print("     2. Restart, open Ubuntu, run:")
            print("        sudo apt-get install -y build-essential git")
            print("        git clone https://github.com/AlexeyAB/darknet /tmp/darknet")
            print("        cd /tmp/darknet && make -j4")
            print(
                "     3. run_training.bat  or  python train_yolov4tiny_darknet.py --mode train")
            print("=" * 60)
            return

        # Make sure dataset/cfg/weights are ready
        if not (PROJECT_ROOT / "obj.data").exists():
            prepare_dataset()
            make_cfg()
        download_weights()

        do_train(darknet, resume=(args.mode == "resume"))

    if args.mode in ("package", "all"):
        package_for_jetbot()

    if args.mode == "prepare":
        print("\n" + "=" * 60)
        print("  Preparation complete!")
        print()
        print("  Next step -> start training:")
        print("    python train_yolov4tiny_darknet.py --mode train")
        print()
        print("  Or double-click: run_training.bat")
        print("=" * 60)


if __name__ == "__main__":
    main()
