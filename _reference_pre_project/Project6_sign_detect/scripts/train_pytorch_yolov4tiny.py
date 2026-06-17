# -*- coding: utf-8 -*-
"""
train_pytorch_yolov4tiny.py
===========================
GPU-Accelerated PyTorch Implementation of YOLOv4-tiny Training for local Windows GPU.
Natively outputs compliant Darknet .weights and .cfg files for direct JetBot deployment.

Steps:
1. Load dataset from obj/ directory.
2. Initialize native YoloV4Tiny model.
3. Load pre-trained backbone yolov4-tiny.conv.29 to fast-track training.
4. Train on GTX 1650 local GPU for 60 epochs (takes ~15 seconds!).
5. Save final weights to compliant Darknet binary yolov4-tiny-416.weights.
6. Verify by loading exported weights and predicting on a sample image.
"""

import sys
import io
import os
import time
import math
import shutil
import urllib.request
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set standard UTF-8 encoding for safety
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OBJ_DIR = PROJECT_ROOT / "obj"
OUTPUT_DIR = PROJECT_ROOT / "jetbot_deploy"
BACKUP_DIR = PROJECT_ROOT / "backup"

# Class names in Project06.ipynb order:
#   0 = stop, 1 = rail, 2 = pedestrian, 3 = blocked
CLASS_NAMES = ["stop", "rail", "pedestrian", "blocked"]
NUM_CLASSES = 4
IMG_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 65
LEARNING_RATE = 0.001

# YOLOv4-tiny Anchors
ANCHORS = [
    [(81, 82), (135, 169), (344, 319)],  # Head 1 (stride 32)
    [(10, 14), (23, 27), (37, 58)]      # Head 2 (stride 16)
]

# ══════════════════════════════════════════════════════════════════
#  1. PyTorch YOLOv4-tiny Architecture
# ══════════════════════════════════════════════════════════════════


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn=True, leaky=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=not bn)
        self.bn = nn.BatchNorm2d(
            out_channels, eps=1e-5, momentum=0.1) if bn else None
        self.leaky = nn.LeakyReLU(0.1, inplace=True) if leaky else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.leaky:
            x = self.leaky(x)
        return x


class YoloV4Tiny(nn.Module):
    def __init__(self, num_classes=4):
        super(YoloV4Tiny, self).__init__()
        self.num_classes = num_classes

        # backbone
        self.conv1 = ConvBlock(3, 32, 3, 2, 1)
        self.conv2 = ConvBlock(32, 64, 3, 2, 1)
        self.conv3 = ConvBlock(64, 64, 3, 1, 1)
        self.conv4 = ConvBlock(32, 32, 3, 1, 1)
        self.conv5 = ConvBlock(32, 32, 3, 1, 1)
        self.conv6 = ConvBlock(64, 64, 1, 1, 0)
        self.maxpool9 = nn.MaxPool2d(2, 2)

        self.conv7 = ConvBlock(128, 128, 3, 1, 1)
        self.conv8 = ConvBlock(64, 64, 3, 1, 1)
        self.conv9 = ConvBlock(64, 64, 3, 1, 1)
        self.conv10 = ConvBlock(128, 128, 1, 1, 0)
        self.maxpool17 = nn.MaxPool2d(2, 2)

        self.conv11 = ConvBlock(256, 256, 3, 1, 1)
        self.conv12 = ConvBlock(128, 128, 3, 1, 1)
        self.conv13 = ConvBlock(128, 128, 3, 1, 1)
        self.conv14 = ConvBlock(256, 256, 1, 1, 0)
        self.maxpool25 = nn.MaxPool2d(2, 2)

        self.conv15 = ConvBlock(512, 512, 3, 1, 1)

        # head 1
        self.conv16 = ConvBlock(512, 256, 1, 1, 0)
        self.conv17 = ConvBlock(256, 512, 3, 1, 1)
        self.conv18 = ConvBlock(
            512, 3 * (5 + num_classes), 1, 1, 0, bn=False, leaky=False)

        # head 2
        self.conv19 = ConvBlock(256, 128, 1, 1, 0)
        self.upsample33 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv20 = ConvBlock(384, 256, 3, 1, 1)
        self.conv21 = ConvBlock(
            256, 3 * (5 + num_classes), 1, 1, 0, bn=False, leaky=False)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        # split
        c3_1 = torch.chunk(c3, 2, dim=1)[1]
        c4 = self.conv4(c3_1)
        c5 = self.conv5(c4)

        r6 = torch.cat([c5, c4], dim=1)
        c6 = self.conv6(r6)
        r8 = torch.cat([c3, c6], dim=1)
        p9 = self.maxpool9(r8)

        c7 = self.conv7(p9)
        c7_1 = torch.chunk(c7, 2, dim=1)[1]
        c8 = self.conv8(c7_1)
        c9 = self.conv9(c8)
        r14 = torch.cat([c9, c8], dim=1)
        c10 = self.conv10(r14)
        r16 = torch.cat([c7, c10], dim=1)
        p17 = self.maxpool17(r16)

        c11 = self.conv11(p17)
        c11_1 = torch.chunk(c11, 2, dim=1)[1]
        c12 = self.conv12(c11_1)
        c13 = self.conv13(c12)
        r22 = torch.cat([c13, c12], dim=1)
        c14 = self.conv14(r22)
        r24 = torch.cat([c11, c14], dim=1)
        p25 = self.maxpool25(r24)

        c15 = self.conv15(p25)
        c16 = self.conv16(c15)
        c17 = self.conv17(c16)
        out1 = self.conv18(c17)  # head 1 (larger anchors)

        c19 = self.conv19(c16)
        up33 = self.upsample33(c19)
        r34 = torch.cat([up33, c14], dim=1)
        c20 = self.conv20(r34)
        out2 = self.conv21(c20)  # head 2 (smaller anchors)

        return out1, out2

# ══════════════════════════════════════════════════════════════════
#  2. Darknet Weight Exporter & Parser
# ══════════════════════════════════════════════════════════════════


def load_darknet_weights(model, weights_path):
    if not os.path.exists(weights_path):
        print(f"  [WARN] Backbone weights not found at: {weights_path}")
        return False

    with open(weights_path, 'rb') as f:
        # Read header
        header = np.fromfile(f, dtype=np.int32, count=5)

        conv_layers = [
            model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.conv6,
            model.conv7, model.conv8, model.conv9, model.conv10, model.conv11, model.conv12,
            model.conv13, model.conv14, model.conv15, model.conv16, model.conv17, model.conv18,
            model.conv19, model.conv20, model.conv21
        ]

        loaded_layers = 0
        for layer in conv_layers:
            has_bn = hasattr(layer, 'bn') and layer.bn is not None

            # 1. Load bias
            num_b = layer.bn.bias.numel() if has_bn else layer.conv.bias.numel()
            bias_data = np.fromfile(f, dtype=np.float32, count=num_b)
            if len(bias_data) < num_b:
                break

            if has_bn:
                layer.bn.bias.data.copy_(torch.from_numpy(bias_data))

                # 2. Load weights (scale/gamma)
                scale_data = np.fromfile(f, dtype=np.float32, count=num_b)
                layer.bn.weight.data.copy_(torch.from_numpy(scale_data))

                # 3. Load mean
                mean_data = np.fromfile(f, dtype=np.float32, count=num_b)
                layer.bn.running_mean.data.copy_(torch.from_numpy(mean_data))

                # 4. Load variance
                var_data = np.fromfile(f, dtype=np.float32, count=num_b)
                layer.bn.running_var.data.copy_(torch.from_numpy(var_data))
            else:
                layer.conv.bias.data.copy_(torch.from_numpy(bias_data))

            # 5. Load conv weights
            num_w = layer.conv.weight.numel()
            weights_data = np.fromfile(f, dtype=np.float32, count=num_w)
            if len(weights_data) < num_w:
                break

            weights_data = weights_data.reshape(layer.conv.weight.shape)
            layer.conv.weight.data.copy_(torch.from_numpy(weights_data))
            loaded_layers += 1

        print(
            f"  [OK] Successfully loaded {loaded_layers}/21 layers from {Path(weights_path).name}")
        return True


def save_darknet_weights(model, filename):
    with open(filename, 'wb') as f:
        # Header: major, minor, revision, seen (seen is 64-bit, so represented as two 32-bit ints)
        header = np.array([0, 2, 5, 320000, 0], dtype=np.int32)
        header.tofile(f)

        conv_layers = [
            model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.conv6,
            model.conv7, model.conv8, model.conv9, model.conv10, model.conv11, model.conv12,
            model.conv13, model.conv14, model.conv15, model.conv16, model.conv17, model.conv18,
            model.conv19, model.conv20, model.conv21
        ]

        for layer in conv_layers:
            has_bn = hasattr(layer, 'bn') and layer.bn is not None

            if has_bn:
                bias = layer.bn.bias.detach().cpu().numpy()
                weight = layer.bn.weight.detach().cpu().numpy()
                mean = layer.bn.running_mean.detach().cpu().numpy()
                var = layer.bn.running_var.detach().cpu().numpy()

                bias.tofile(f)
                weight.tofile(f)
                mean.tofile(f)
                var.tofile(f)
            else:
                bias = layer.conv.bias.detach().cpu().numpy()
                bias.tofile(f)

            w = layer.conv.weight.detach().cpu().numpy()
            w.tofile(f)

    print(
        f"  [OK] Successfully exported weights to compliant Darknet format -> {filename}")

# ══════════════════════════════════════════════════════════════════
#  3. PyTorch Dataset Loader
# ══════════════════════════════════════════════════════════════════


class YoloDataset(Dataset):
    def __init__(self, obj_dir, img_size=416):
        self.obj_dir = Path(obj_dir)
        self.img_size = img_size
        self.image_paths = sorted(list(self.obj_dir.glob(
            "*.jpg")) + list(self.obj_dir.glob("*.png")))
        print(
            f"  Found {len(self.image_paths)} images in {self.obj_dir.name}/")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        txt_path = img_path.with_suffix(".txt")
        boxes = []
        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    boxes.append([cls_id, cx, cy, bw, bh])

        if len(boxes) == 0:
            targets = torch.zeros((0, 5))
        else:
            targets = torch.tensor(boxes, dtype=torch.float32)

        return img_t, targets


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)

    packed_targets = []
    for i, target in enumerate(targets):
        if target.shape[0] > 0:
            num_boxes = target.shape[0]
            batch_idx = torch.full((num_boxes, 1), i, dtype=torch.float32)
            packed_targets.append(torch.cat([batch_idx, target], 1))

    if packed_targets:
        packed_targets = torch.cat(packed_targets, 0)
    else:
        packed_targets = torch.zeros((0, 6))

    return images, packed_targets

# ══════════════════════════════════════════════════════════════════
#  4. Custom YOLO Loss Module
# ══════════════════════════════════════════════════════════════════


class YoloLoss(nn.Module):
    def __init__(self, num_classes=4, img_size=416):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.anchors = ANCHORS

        self.bce_conf = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_class = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse_box = nn.MSELoss(reduction="mean")

    def forward(self, preds, targets, device):
        loss_box = torch.tensor(0.0, device=device)
        loss_conf = torch.tensor(0.0, device=device)
        loss_class = torch.tensor(0.0, device=device)

        for scale_idx, pred in enumerate(preds):
            B, _, H, W = pred.shape
            pred = pred.view(B, 3, 5 + self.num_classes,
                             H, W).permute(0, 1, 3, 4, 2)

            pred_xy = torch.sigmoid(pred[..., 0:2])
            pred_wh = pred[..., 2:4]
            pred_conf = pred[..., 4]
            pred_class = pred[..., 5:]

            tconf = torch.zeros((B, 3, H, W), device=device)
            txy = torch.zeros((B, 3, H, W, 2), device=device)
            twh = torch.zeros((B, 3, H, W, 2), device=device)
            tclass = torch.zeros((B, 3, H, W, self.num_classes), device=device)
            loss_mask = torch.zeros((B, 3, H, W), device=device)

            scale_anchors = self.anchors[scale_idx]

            for t in targets:
                b_idx = int(t[0])
                cls_id = int(t[1])
                cx, cy, bw, bh = t[2], t[3], t[4], t[5]

                gx = cx * W
                gy = cy * H
                g_col = int(gx)
                g_row = int(gy)

                if g_col < 0 or g_col >= W or g_row < 0 or g_row >= H:
                    continue

                best_iou = -1
                best_anchor = -1
                for a_idx, anchor in enumerate(scale_anchors):
                    aw, ah = anchor[0] / \
                        self.img_size, anchor[1] / self.img_size
                    inter = min(bw, aw) * min(bh, ah)
                    union = bw * bh + aw * ah - inter
                    iou = inter / (union + 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = a_idx

                if best_iou > 0.05:
                    tconf[b_idx, best_anchor, g_row, g_col] = 1.0
                    txy[b_idx, best_anchor, g_row, g_col, 0] = gx - g_col
                    txy[b_idx, best_anchor, g_row, g_col, 1] = gy - g_row

                    aw, ah = scale_anchors[best_anchor]
                    twh[b_idx, best_anchor, g_row, g_col, 0] = torch.log(
                        bw * self.img_size / aw + 1e-8)
                    twh[b_idx, best_anchor, g_row, g_col, 1] = torch.log(
                        bh * self.img_size / ah + 1e-8)

                    tclass[b_idx, best_anchor, g_row, g_col, cls_id] = 1.0
                    loss_mask[b_idx, best_anchor, g_row, g_col] = 1.0

            if loss_mask.sum() > 0:
                loss_box += self.mse_box(pred_xy[loss_mask == 1.0],
                                         txy[loss_mask == 1.0]) * 4.0
                loss_box += self.mse_box(pred_wh[loss_mask == 1.0],
                                         twh[loss_mask == 1.0]) * 2.0
                loss_class += self.bce_class(
                    pred_class[loss_mask == 1.0], tclass[loss_mask == 1.0]) * 1.0

            loss_conf += self.bce_conf(pred_conf, tconf)

        total_loss = loss_box + loss_conf + loss_class
        return total_loss, loss_box.item(), loss_conf.item(), loss_class.item()

# ══════════════════════════════════════════════════════════════════
#  5. Predict Verification (loads exported weights)
# ══════════════════════════════════════════════════════════════════


def verify_and_predict(weights_path):
    print("\n" + "=" * 60)
    print("  VERIFYING EXPORTED WEIGHTS VIA PREDICTION")
    print("=" * 60)

    # Create validation model
    verify_model = YoloV4Tiny(num_classes=NUM_CLASSES)
    # Load exported weights
    success = load_darknet_weights(verify_model, weights_path)
    if not success:
        print("  [ERROR] Failed to load exported weights for verification.")
        return

    verify_model.eval()

    # Get a sample image
    image_paths = list(OBJ_DIR.glob("*.jpg")) + list(OBJ_DIR.glob("*.png"))
    if not image_paths:
        print("  [WARN] No validation image found.")
        return
    img_path = image_paths[0]
    print(f"  Validating on image: {img_path.name}")

    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    # Preprocess
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = cv2.resize(img_in, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(img_in).permute(
        2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        out1, out2 = verify_model(img_t)

    # Standard YOLO Decode logic
    detections = []
    for scale_idx, out in enumerate([out1, out2]):
        _, _, H_out, W_out = out.shape
        out = out.view(1, 3, 5 + NUM_CLASSES, H_out,
                       W_out).permute(0, 1, 3, 4, 2)

        xy = torch.sigmoid(out[..., 0:2])
        wh = torch.exp(out[..., 2:4])
        conf = torch.sigmoid(out[..., 4])
        cls_probs = torch.sigmoid(out[..., 5:])

        scale_anchors = ANCHORS[scale_idx]

        for a in range(3):
            for y in range(H_out):
                for x in range(W_out):
                    score = conf[0, a, y, x].item()
                    if score > 0.35:  # confidence threshold
                        aw, ah = scale_anchors[a]

                        # grid normalized coordinates
                        gx = (xy[0, a, y, x, 0].item() + x) / W_out
                        gy = (xy[0, a, y, x, 1].item() + y) / H_out
                        gw = (wh[0, a, y, x, 0].item() * aw) / IMG_SIZE
                        gh = (wh[0, a, y, x, 1].item() * ah) / IMG_SIZE

                        # map back to original image coordinates
                        x1 = int((gx - gw/2) * W)
                        y1 = int((gy - gh/2) * H)
                        x2 = int((gx + gw/2) * W)
                        y2 = int((gy + gh/2) * H)

                        class_id = torch.argmax(cls_probs[0, a, y, x]).item()
                        class_score = cls_probs[0, a, y, x, class_id].item()

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "score": score * class_score,
                            "class": class_id
                        })

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class"]
        score = det["score"]

        label = f"{CLASS_NAMES[class_id]} {score:.2f}"
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(
            f"  [DETECT] Detected: {CLASS_NAMES[class_id]} at [{x1},{y1},{x2},{y2}] score={score:.4f}")

    out_vis = PROJECT_ROOT / "inference_pytorch.jpg"
    cv2.imwrite(str(out_vis), img)
    print(f"  [OK] Saved prediction verification result -> {out_vis}")

# ══════════════════════════════════════════════════════════════════
#  6. High-Speed Training Pipeline
# ══════════════════════════════════════════════════════════════════


def main():
    print("\n" + "=" * 60)
    print("  YOLOv4-tiny PyTorch Local GPU Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Target Device : {device}")

    # Initialize Model
    model = YoloV4Tiny(num_classes=NUM_CLASSES).to(device)

    # Download yolov4-tiny.conv.29 backbone weights if not already present
    backbone_path = PROJECT_ROOT / "yolov4-tiny.conv.29"
    if not backbone_path.exists():
        print("  Downloading yolov4-tiny.conv.29 from GitHub releases ...")
        url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29"
        urllib.request.urlretrieve(url, backbone_path)
        print(f"  Saved -> {backbone_path}")

    # Load backbone weights
    load_darknet_weights(model, str(backbone_path))

    # Load Dataset
    dataset = YoloDataset(OBJ_DIR, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = YoloLoss(num_classes=NUM_CLASSES, img_size=IMG_SIZE)

    model.train()
    print("\n  [START] Starting Training Loop ...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        epoch_box = 0.0
        epoch_conf = 0.0
        epoch_class = 0.0

        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss, l_box, l_conf, l_class = criterion(preds, targets, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_box += l_box * images.size(0)
            epoch_conf += l_conf * images.size(0)
            epoch_class += l_class * images.size(0)

        epoch_loss /= len(dataset)
        epoch_box /= len(dataset)
        epoch_conf /= len(dataset)
        epoch_class /= len(dataset)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {epoch_loss:.4f} (box:{epoch_box:.3f}, conf:{epoch_conf:.3f}, cls:{epoch_class:.3f})")

    training_duration = time.time() - start_time
    print(f"\n  [OK] Training completed in {training_duration:.2f} seconds!")

    # Save weights to backup/
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    last_weights = BACKUP_DIR / "yolov4-tiny-custom_last.weights"
    save_darknet_weights(model, str(last_weights))

    # Copy best weights to deploy/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dst_weights = OUTPUT_DIR / "yolov4-tiny-416.weights"
    shutil.copy2(last_weights, dst_weights)

    # Copy custom config file to deploy/
    src_cfg = CONFIG_DIR / "yolov4-tiny-custom.cfg"
    dst_cfg = OUTPUT_DIR / "yolov4-tiny-416.cfg"
    if src_cfg.exists():
        shutil.copy2(src_cfg, dst_cfg)
        print(f"  [OK] Copied config -> {dst_cfg}")

    # Copy obj.names
    src_names = CONFIG_DIR / "obj.names"
    dst_names = OUTPUT_DIR / "obj.names"
    if src_names.exists():
        shutil.copy2(src_names, dst_names)
        print(f"  [OK] Copied names -> {dst_names}")

    # Generate DEPLOY.txt
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
    print(f"  [OK] Generated JetBot DEPLOY guide -> {readme}")

    # Run prediction verification using the newly exported weights
    verify_and_predict(str(dst_weights))


if __name__ == "__main__":
    main()
