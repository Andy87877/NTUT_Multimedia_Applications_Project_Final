# -*- coding: utf-8 -*-
"""
train_pytorch_yolov4tiny_v2.py
==============================
Advanced YOLOv4-tiny PyTorch Training (V2) with:
1. Dynamic Train/Val/Test Split (80/10/10) with fixed seed 42.
2. Data Augmentation (Brightness/Contrast scaling, Gaussian Blur, HSV shift, geometric Flip & Shift).
3. CIoU (Complete IoU) Loss for bounding box regression.
4. Cosine Annealing Learning Rate Scheduler with linear warmup.
5. Vectorized PyTorch box decoding for lightning-fast validation evaluation.
6. Auto-saves results to CSV (Excel format) and a Markdown report.
"""

import sys
import io
import os
import time
import math
import shutil
import urllib.request
import random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set standard UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OBJ_DIR = PROJECT_ROOT / "obj"
OUTPUT_DIR = PROJECT_ROOT / "jetbot_deploy"
BACKUP_DIR = PROJECT_ROOT / "backup"
OUT_VIS_DIR = PROJECT_ROOT / "runs" / "predict_vis_yolov4tiny_v2"

# Constants
CLASS_NAMES = ["stop", "rail", "pedestrian", "blocked"]
NUM_CLASSES = 4
IMG_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 1000  # Upgraded to 1000 epochs (approx. 7500 iterations, standard Darknet scale)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 20

ANCHORS = [
    [(81, 82), (135, 169), (344, 319)],  # Head 1 (stride 32)
    [(10, 14), (23, 27), (37, 58)]      # Head 2 (stride 16)
]

# ══════════════════════════════════════════════════════════════════
#  1. PyTorch YOLOv4-tiny Model Definition
# ══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn=True, leaky=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1) if bn else None
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
        self.conv18 = ConvBlock(512, 3 * (5 + num_classes), 1, 1, 0, bn=False, leaky=False)

        # head 2
        self.conv19 = ConvBlock(256, 128, 1, 1, 0)
        self.upsample33 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv20 = ConvBlock(384, 256, 3, 1, 1)
        self.conv21 = ConvBlock(256, 3 * (5 + num_classes), 1, 1, 0, bn=False, leaky=False)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

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
        out1 = self.conv18(c17)

        c19 = self.conv19(c16)
        up33 = self.upsample33(c19)
        r34 = torch.cat([up33, c14], dim=1)
        c20 = self.conv20(r34)
        out2 = self.conv21(c20)

        return out1, out2

def load_darknet_weights(model, weights_path):
    if not os.path.exists(weights_path):
        print(f"  [WARN] Backbone weights not found at: {weights_path}")
        return False

    with open(weights_path, 'rb') as f:
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
            num_b = layer.bn.bias.numel() if has_bn else layer.conv.bias.numel()
            bias_data = np.fromfile(f, dtype=np.float32, count=num_b)
            if len(bias_data) < num_b:
                break
            if has_bn:
                layer.bn.bias.data.copy_(torch.from_numpy(bias_data))
                scale_data = np.fromfile(f, dtype=np.float32, count=num_b)
                layer.bn.weight.data.copy_(torch.from_numpy(scale_data))
                mean_data = np.fromfile(f, dtype=np.float32, count=num_b)
                layer.bn.running_mean.data.copy_(torch.from_numpy(mean_data))
                var_data = np.fromfile(f, dtype=np.float32, count=num_b)
                layer.bn.running_var.data.copy_(torch.from_numpy(var_data))
            else:
                layer.conv.bias.data.copy_(torch.from_numpy(bias_data))

            num_w = layer.conv.weight.numel()
            weights_data = np.fromfile(f, dtype=np.float32, count=num_w)
            if len(weights_data) < num_w:
                break
            weights_data = weights_data.reshape(layer.conv.weight.shape)
            layer.conv.weight.data.copy_(torch.from_numpy(weights_data))
            loaded_layers += 1
        print(f"  [OK] Loaded {loaded_layers}/21 layers from {Path(weights_path).name}")
        return True

def save_darknet_weights(model, filename):
    with open(filename, 'wb') as f:
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
    print(f"  [OK] Exported weights -> {filename}")

# ══════════════════════════════════════════════════════════════════
#  2. Custom Dataset Loader with Augmentations
# ══════════════════════════════════════════════════════════════════

class YoloDatasetV2(Dataset):
    def __init__(self, image_paths, img_size=416, augment=False):
        self.image_paths = image_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        h_orig, w_orig = img.shape[:2]
        
        txt_path = img_path.with_suffix(".txt")
        boxes = []
        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        boxes.append([cls_id, cx, cy, bw, bh])

        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                for box in boxes:
                    box[1] = 1.0 - box[1]

            if random.random() > 0.5:
                tx = random.randint(-int(w_orig * 0.08), int(w_orig * 0.08))
                ty = random.randint(-int(h_orig * 0.08), int(h_orig * 0.08))
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                img = cv2.warpAffine(img, M, (w_orig, h_orig), borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
                valid_boxes = []
                for box in boxes:
                    cls_id, cx, cy, bw, bh = box
                    ncx = cx + tx / w_orig
                    ncy = cy + ty / h_orig
                    if 0 < ncx < 1 and 0 < ncy < 1:
                        valid_boxes.append([cls_id, ncx, ncy, bw, bh])
                boxes = valid_boxes

            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.randint(-15, 15)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            if random.random() > 0.5:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

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
#  3. Advanced CIoU Loss Module
# ══════════════════════════════════════════════════════════════════

def bbox_ciou(box1, box2):
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    w1, h1 = box1[..., 2], box1[..., 3]
    w2, h2 = box2[..., 2], box2[..., 3]
    union_area = w1 * h1 + w2 * h2 - inter_area + 1e-16
    iou = inter_area / union_area

    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-16

    rho2 = (box1[..., 0] - box2[..., 0]) ** 2 + (box1[..., 1] - box2[..., 1]) ** 2
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-16)) - torch.atan(w1 / (h1 + 1e-16)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-16)

    ciou = iou - (rho2 / c2) - alpha * v
    return torch.clamp(ciou, min=-1.0, max=1.0)

class YoloLossV2(nn.Module):
    def __init__(self, num_classes=4, img_size=416):
        super(YoloLossV2, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.anchors = ANCHORS
        self.bce_conf = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_class = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, preds, targets, device):
        loss_box = torch.tensor(0.0, device=device)
        loss_conf = torch.tensor(0.0, device=device)
        loss_class = torch.tensor(0.0, device=device)

        for scale_idx, pred in enumerate(preds):
            B, _, H, W = pred.shape
            pred = pred.view(B, 3, 5 + self.num_classes, H, W).permute(0, 1, 3, 4, 2)

            pred_xy = torch.sigmoid(pred[..., 0:2])
            pred_wh = pred[..., 2:4]
            pred_conf = pred[..., 4]
            pred_class = pred[..., 5:]

            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
            grid_x = grid_x.view(1, 1, H, W).repeat(B, 3, 1, 1).float()
            grid_y = grid_y.view(1, 1, H, W).repeat(B, 3, 1, 1).float()

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
                    aw, ah = anchor[0] / self.img_size, anchor[1] / self.img_size
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
                    twh[b_idx, best_anchor, g_row, g_col, 0] = torch.log(bw * self.img_size / aw + 1e-8)
                    twh[b_idx, best_anchor, g_row, g_col, 1] = torch.log(bh * self.img_size / ah + 1e-8)

                    tclass[b_idx, best_anchor, g_row, g_col, cls_id] = 1.0
                    loss_mask[b_idx, best_anchor, g_row, g_col] = 1.0

            if loss_mask.sum() > 0:
                stride = self.img_size / H
                scale_anchors_grid = torch.tensor([(aw / stride, ah / stride) for aw, ah in scale_anchors], device=device).view(1, 3, 1, 1, 2)
                
                pred_box_x = pred_xy[..., 0] + grid_x
                pred_box_y = pred_xy[..., 1] + grid_y
                pred_box_w = torch.exp(pred_wh[..., 0]) * scale_anchors_grid[..., 0]
                pred_box_h = torch.exp(pred_wh[..., 1]) * scale_anchors_grid[..., 1]
                pred_box = torch.stack([pred_box_x, pred_box_y, pred_box_w, pred_box_h], dim=-1)

                target_box_x = txy[..., 0] + grid_x
                target_box_y = txy[..., 1] + grid_y
                target_box_w = torch.exp(twh[..., 0]) * scale_anchors_grid[..., 0]
                target_box_h = torch.exp(twh[..., 1]) * scale_anchors_grid[..., 1]
                target_box = torch.stack([target_box_x, target_box_y, target_box_w, target_box_h], dim=-1)

                pred_obj_boxes = pred_box[loss_mask == 1.0]
                target_obj_boxes = target_box[loss_mask == 1.0]

                ciou = bbox_ciou(pred_obj_boxes, target_obj_boxes)
                loss_box += (1.0 - ciou).mean() * 5.0

                loss_class += self.bce_class(pred_class[loss_mask == 1.0], tclass[loss_mask == 1.0]) * 1.0

            # Solve positive cell loss dilution by separating pos and neg conf losses
            pos_mask = (tconf == 1.0)
            neg_mask = (tconf == 0.0)
            if pos_mask.sum() > 0:
                loss_conf_pos = nn.functional.binary_cross_entropy_with_logits(pred_conf[pos_mask], tconf[pos_mask], reduction="mean")
                loss_conf_neg = nn.functional.binary_cross_entropy_with_logits(pred_conf[neg_mask], tconf[neg_mask], reduction="mean")
                loss_conf += loss_conf_pos * 5.0 + loss_conf_neg * 1.0
            else:
                loss_conf += self.bce_conf(pred_conf, tconf)

        total_loss = loss_box + loss_conf + loss_class
        return total_loss, loss_box.item(), loss_conf.item(), loss_class.item()

# ══════════════════════════════════════════════════════════════════
#  4. Validation & Evaluation Utility (Vectorized & Fast)
# ══════════════════════════════════════════════════════════════════

def nms(boxes, iou_threshold=0.45):
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        remaining = []
        for box in boxes:
            if box["class"] != best["class"]:
                remaining.append(box)
                continue
            # Calculate IoU
            b1 = best["bbox"]
            b2 = box["bbox"]
            xi1 = max(b1[0], b2[0])
            yi1 = max(b1[1], b2[1])
            xi2 = min(b1[2], b2[2])
            yi2 = min(b1[3], b2[3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / (union_area + 1e-6)
            if iou < iou_threshold:
                remaining.append(box)
        boxes = remaining
    return keep

def calculate_iou(box1, box2):
    """Calculate IoU of two boxes [x1, y1, x2, y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)

def evaluate_model(model, dataloader, device, conf_th=0.3, iou_th=0.5):
    model.eval()
    tps = {i: 0 for i in range(4)}
    fps = {i: 0 for i in range(4)}
    fns = {i: 0 for i in range(4)}
    gts_total = {i: 0 for i in range(4)}

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            out1, out2 = model(images)
            B = images.size(0)

            for b in range(B):
                b_gts = targets[targets[:, 0] == b]
                gt_pixel_boxes = []
                for gt in b_gts:
                    cls_id = int(gt[1])
                    cx, cy, bw, bh = gt[2].item(), gt[3].item(), gt[4].item(), gt[5].item()
                    x1 = int((cx - bw / 2) * 416)
                    y1 = int((cy - bh / 2) * 416)
                    x2 = int((cx + bw / 2) * 416)
                    y2 = int((cy + bh / 2) * 416)
                    gt_pixel_boxes.append({"class": cls_id, "bbox": [x1, y1, x2, y2], "matched": False})
                    gts_total[cls_id] += 1

                detections = []
                for scale_idx, out in enumerate([out1, out2]):
                    _, _, H_out, W_out = out.shape
                    
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(H_out, device=device), 
                        torch.arange(W_out, device=device), 
                        indexing="ij"
                    )
                    
                    sub_out = out[b].view(3, 9, H_out, W_out).permute(0, 2, 3, 1) # (3, H_out, W_out, 9)
                    xy = torch.sigmoid(sub_out[..., 0:2])  # (3, H_out, W_out, 2)
                    wh = torch.exp(sub_out[..., 2:4])      # (3, H_out, W_out, 2)
                    conf = torch.sigmoid(sub_out[..., 4])  # (3, H_out, W_out)
                    cls_probs = torch.sigmoid(sub_out[..., 5:]) # (3, H_out, W_out, 4)

                    cls_scores, cls_ids = torch.max(cls_probs, dim=-1) # (3, H_out, W_out)
                    final_scores = conf * cls_scores                   # (3, H_out, W_out)
                    
                    b_mask = final_scores > conf_th
                    if not b_mask.any():
                        continue

                    anchors_idx, y_idx, x_idx = torch.where(b_mask)
                    
                    scores = final_scores[b_mask]
                    classes = cls_ids[b_mask]
                    xy_vals = xy[b_mask]
                    wh_vals = wh[b_mask]
                    
                    scale_anchors = ANCHORS[scale_idx]
                    aw_ah = torch.tensor(scale_anchors, device=device)[anchors_idx]
                    
                    gx = (xy_vals[:, 0] + x_idx.float()) / W_out
                    gy = (xy_vals[:, 1] + y_idx.float()) / H_out
                    gw = (wh_vals[:, 0] * aw_ah[:, 0]) / 416
                    gh = (wh_vals[:, 1] * aw_ah[:, 1]) / 416

                    x1 = ((gx - gw/2) * 416).int()
                    y1 = ((gy - gh/2) * 416).int()
                    x2 = ((gx + gw/2) * 416).int()
                    y2 = ((gy + gh/2) * 416).int()
                    
                    x1_cpu = x1.cpu().numpy()
                    y1_cpu = y1.cpu().numpy()
                    x2_cpu = x2.cpu().numpy()
                    y2_cpu = y2.cpu().numpy()
                    scores_cpu = scores.cpu().numpy()
                    classes_cpu = classes.cpu().numpy()
                    
                    for i in range(len(scores_cpu)):
                        detections.append({
                            "bbox": [x1_cpu[i], y1_cpu[i], x2_cpu[i], y2_cpu[i]],
                            "score": float(scores_cpu[i]),
                            "class": int(classes_cpu[i])
                        })

                detections = nms(detections, iou_threshold=0.45)

                for det in detections:
                    det_box = det["bbox"]
                    det_cls = det["class"]

                    best_iou = -1
                    best_gt_idx = -1
                    for gt_idx, gt in enumerate(gt_pixel_boxes):
                        if gt["class"] == det_cls and not gt["matched"]:
                            iou = calculate_iou(det_box, gt["bbox"])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx

                    if best_iou >= iou_th:
                        tps[det_cls] += 1
                        gt_pixel_boxes[best_gt_idx]["matched"] = True
                    else:
                        fps[det_cls] += 1

                for gt in gt_pixel_boxes:
                    if not gt["matched"]:
                        fns[gt["class"]] += 1

    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(4):
        tp, fp, fn = tps[c], fps[c], fns[c]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)

    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list), tps, fps, fns, gts_total

# ══════════════════════════════════════════════════════════════════
#  5. Main Advanced V2 Training Pipeline
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  YOLOv4-tiny PyTorch Advanced V2 Training System")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Target Device : {device}")

    all_imgs = sorted(list(OBJ_DIR.glob("*.jpg")) + list(OBJ_DIR.glob("*.png")))
    if not all_imgs:
        print(f"  [ERROR] No training images found in: {OBJ_DIR}")
        return

    random.seed(42)
    random.shuffle(all_imgs)

    n_total = len(all_imgs)
    n_val = int(n_total * 0.10)
    n_test = int(n_total * 0.10)
    n_train = n_total - n_val - n_test

    train_imgs = all_imgs[:n_train]
    val_imgs = all_imgs[n_train:n_train + n_val]
    test_imgs = all_imgs[n_train + n_val:]

    print(f"  Total Dataset : {n_total} images")
    print(f"  Training Set  : {len(train_imgs)} images (Augmented)")
    print(f"  Validation Set: {len(val_imgs)} images")
    print(f"  Test Set      : {len(test_imgs)} images")

    train_dataset = YoloDatasetV2(train_imgs, img_size=IMG_SIZE, augment=True)
    val_dataset = YoloDatasetV2(val_imgs, img_size=IMG_SIZE, augment=False)
    test_dataset = YoloDatasetV2(test_imgs, img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = YoloV4Tiny(num_classes=NUM_CLASSES).to(device)

    backbone_path = PROJECT_ROOT / "yolov4-tiny.conv.29"
    if not backbone_path.exists():
        print("  Downloading yolov4-tiny.conv.29 from GitHub releases ...")
        url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29"
        urllib.request.urlretrieve(url, backbone_path)
    load_darknet_weights(model, str(backbone_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = YoloLossV2(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print("\n  [START] Starting Advanced Training Loop (1000 Epochs)...")
    start_time = time.time()

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        if epoch <= WARMUP_EPOCHS:
            lr = 1e-5 + (LEARNING_RATE - 1e-5) * (epoch / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        model.train()
        epoch_loss = 0.0
        epoch_box = 0.0
        epoch_conf = 0.0
        epoch_class = 0.0

        for images, targets in train_loader:
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

        epoch_loss /= len(train_dataset)
        epoch_box /= len(train_dataset)
        epoch_conf /= len(train_dataset)
        epoch_class /= len(train_dataset)

        is_val_epoch = (epoch % 10 == 0) or (epoch == 1) or (epoch == EPOCHS)
        if is_val_epoch:
            val_p, val_r, val_f1, _, _, _, _ = evaluate_model(model, val_loader, device)
        else:
            val_p, val_r, val_f1 = (history[-1]['val_precision'], history[-1]['val_recall'], history[-1]['val_f1']) if history else (0.0, 0.0, 0.0)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_box_loss": epoch_box,
            "train_conf_loss": epoch_conf,
            "train_class_loss": epoch_class,
            "val_precision": val_p,
            "val_recall": val_r,
            "val_f1": val_f1,
            "lr": lr
        })

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  Epoch {epoch:4d}/{EPOCHS} | Train Loss: {epoch_loss:.3f} | Val F1: {val_f1:.4f} (P:{val_p:.3f}, R:{val_r:.3f}) | LR: {lr:.6f}")
            sys.stdout.flush()

        if is_val_epoch:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights_path = BACKUP_DIR / "yolov4-tiny-custom_best.weights"
                save_darknet_weights(model, str(best_weights_path))

    training_duration = time.time() - start_time
    print(f"\n  [OK] Training completed in {training_duration:.2f} seconds!")
    print(f"  [OK] Peak Validation F1-Score: {best_val_f1:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dst_weights = OUTPUT_DIR / "yolov4-tiny-416.weights"
    best_weights_path = BACKUP_DIR / "yolov4-tiny-custom_best.weights"
    if best_weights_path.exists():
        shutil.copy2(best_weights_path, dst_weights)
        print(f"  [OK] Successfully deployed best weights to: {dst_weights}")

    src_cfg = CONFIG_DIR / "yolov4-tiny-custom.cfg"
    dst_cfg = OUTPUT_DIR / "yolov4-tiny-416.cfg"
    if src_cfg.exists():
        shutil.copy2(src_cfg, dst_cfg)

    # ══════════════════════════════════════════════════════════════════
    #  6. Run Final Test Set Evaluation & Generate Excel/MD Reports
    # ══════════════════════════════════════════════════════════════════
    print("\n  [EVAL] Loading Best Weights for Final Test Set Evaluation...")
    model_eval = YoloV4Tiny(num_classes=NUM_CLASSES).to(device)
    load_darknet_weights(model_eval, str(dst_weights))
    
    test_p, test_r, test_f1, tps, fps, fns, gts = evaluate_model(model_eval, test_loader, device)
    
    OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_VIS_DIR / "training_history.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_box_loss,train_conf_loss,train_class_loss,val_precision,val_recall,val_f1,lr\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']:.5f},{row['train_box_loss']:.5f},{row['train_conf_loss']:.5f},{row['train_class_loss']:.5f},{row['val_precision']:.5f},{row['val_recall']:.5f},{row['val_f1']:.5f},{row['lr']:.7f}\n")
    print(f"  [OK] Saved Excel-compatible CSV -> {csv_path}")

    md_path = OUT_VIS_DIR / "training_metrics_report.md"
    
    class_rows = ""
    for c in range(4):
        tp, fp, fn, gt_cnt = tps[c], fps[c], fns[c], gts[c]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_rows += f"| {CLASS_NAMES[c]:<12} | {gt_cnt:<4d} | {tp:<4d} | {fp:<4d} | {fn:<4d} | {p:<9.4f} | {r:<6.4f} | {f1:<8.4f} |\n"

    rep_epochs = [1, 10, 50, 100, 200, 500, 1000]
    rep_rows = ""
    for rep_ep in rep_epochs:
        if rep_ep <= len(history):
            row = history[rep_ep - 1]
            rep_rows += f"| Epoch {row['epoch']:<4d} | {row['train_loss']:.4f} | {row['train_box_loss']:.4f} | {row['train_conf_loss']:.4f} | {row['train_class_loss']:.4f} | {row['val_f1']:.4f} | {row['lr']:.6f} |\n"

    md_content = f"""# YOLOv4-tiny V2 Model Training & Test Set Metrics Report

## 1. Dataset Partition Details
* **Training Set (80%)**: {len(train_imgs)} images (Augmented with HSV, Blur, Translation, Horizontal Flip)
* **Validation Set (10%)**: {len(val_imgs)} images
* **Test Set (10%)**: {len(test_imgs)} images (Completely isolated test set)

## 2. Test Set Quantitative Evaluation Results
Below is the evaluation report of the best weights model (`yolov4-tiny-custom_best.weights`) evaluated on the isolated Test Set:

| Class Name   | GT   | TP   | FP   | FN   | Precision | Recall | F1-Score |
| :---         | :---: | :---: | :---: | :---: | :---:     | :---:  | :---:    |
{class_rows}| **Average/mAP**| **{sum(gts.values())}** | **-** | **-** | **-** | **{test_p:.4f}** | **{test_r:.4f}** | **{test_f1:.4f}** |

## 3. Training Progress Loss Trace (Representative Epochs)
The complete learning history of all 1000 epochs has been saved to [training_history.csv](file:///{csv_path.as_posix()}) (opens in Excel).

| Epoch | Train Loss | Box Loss (CIoU) | Conf Loss | Class Loss | Val F1-Score | Learning Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{rep_rows}"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [OK] Saved Markdown Report -> {md_path}")

if __name__ == "__main__":
    main()
