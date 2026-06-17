# -*- coding: utf-8 -*-
"""
train_local.py
==============
YOLOv4-tiny 本地端訓練腳本

等效於 Colab 上的:
  !./darknet detector train obj.data yolov4-tiny-custom.cfg yolov4-tiny.conv.29

使用方式:
  python train_local.py

或自訂參數:
  python train_local.py --epochs 100 --batch 64 --subdivisions 32
"""

from easydict import EasyDict as edict
from dataset import Yolo_dataset
from tool.darknet2pytorch import Darknet
import os
import sys
import time
import logging
import argparse
import datetime
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

# 將 darknet 加入路徑
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DARKNET_DIR = os.path.join(PROJECT_DIR, "darknet")
sys.path.insert(0, DARKNET_DIR)


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


class YoloLoss(nn.Module):
    """Simplified YOLO loss for yolov4-tiny (2 output scales)"""

    def __init__(self, n_classes=1, n_anchors=3, device=None, batch=2):
        super(YoloLoss, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        # yolov4-tiny anchors (pixel values at input resolution)
        self.anchors = [[10, 14], [23, 27], [37, 58],
                        [81, 82], [135, 169], [344, 319]]
        # mask 0 = large feature map (small stride) = small anchors
        # mask 1 = small feature map (large stride) = large anchors
        self.anch_masks = [[0, 1, 2], [3, 4, 5]]
        self.ignore_thre = 0.5
        self.batch = batch

    def _get_output_id(self, fsize):
        """Determine which anchor mask to use based on feature map size.
        Larger fsize (e.g. 26) -> smaller stride -> output_id=0 (smaller anchors)
        Smaller fsize (e.g. 13) -> larger stride -> output_id=1 (larger anchors)
        """
        # We'll store the mapping when we first see fsize values
        return 0 if fsize >= 20 else 1  # 26 vs 13 for 416 input

    def _make_grid(self, fsize, batchsize):
        grid_x = torch.arange(fsize, dtype=torch.float).repeat(
            batchsize, 3, fsize, 1).to(self.device)
        grid_y = torch.arange(fsize, dtype=torch.float).repeat(
            batchsize, 3, fsize, 1).permute(0, 1, 3, 2).to(self.device)
        return grid_x, grid_y

    def _make_anchors(self, fsize, batchsize, output_id, stride):
        all_anchors_grid = [(w / stride, h / stride) for w, h in self.anchors]
        masked_anchors = np.array(
            [all_anchors_grid[j] for j in self.anch_masks[output_id]], dtype=np.float32)
        ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
        ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
        ref_anchors = torch.from_numpy(ref_anchors)
        anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(
            batchsize, fsize, fsize, 1).permute(0, 3, 1, 2).to(self.device)
        anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(
            batchsize, fsize, fsize, 1).permute(0, 3, 1, 2).to(self.device)
        return masked_anchors, ref_anchors, anchor_w, anchor_h

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id, stride, masked_anchors, ref_anchors):
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize,
                               fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors,
                              fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).to(self.device)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (stride * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (stride * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / stride
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / stride
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            anchor_ious_all = self._bboxes_iou(truth_box.cpu(), ref_anchors)
            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = self._bboxes_iou(
                pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            obj_mask[b] = ~pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                        truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                        truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti,
                                                  4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    @staticmethod
    def _bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
            br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        area_u = area_a[:, None] + area_b - area_i
        iou = area_i / area_u
        return iou

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0

        # Sort outputs by fsize descending (large fsize = small stride first)
        sorted_xin = sorted(
            enumerate(xin), key=lambda x: x[1].shape[2], reverse=True)

        for idx, (orig_idx, output) in enumerate(sorted_xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            # Determine stride and output_id from fsize
            # 0=large fmap (small anchors), 1=small fmap (large anchors)
            output_id = idx
            stride = output.shape[3]  # input_size / fsize (approx)
            # Calculate stride from cfg image size
            # For 416 input: fsize=26 -> stride=16, fsize=13 -> stride=32
            stride = 416 // fsize  # Use the cfg image size

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)

            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
                output[..., np.r_[:2, 4:n_ch]])

            # Compute grids dynamically
            grid_x, grid_y = self._make_grid(fsize, batchsize)
            masked_anchors, ref_anchors, anchor_w, anchor_h = self._make_anchors(
                fsize, batchsize, output_id, stride)

            pred = output[..., :4].clone()
            pred[..., 0] += grid_x
            pred[..., 1] += grid_y
            pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w
            pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(
                pred, labels, batchsize, fsize, n_ch, output_id, stride, masked_anchors, ref_anchors)

            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4],
                                  target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(
                input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(
                input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls
        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def get_config():
    """建立訓練設定"""
    parser = argparse.ArgumentParser(description='YOLOv4-tiny 本地端訓練')
    parser.add_argument('--epochs', type=int, default=600, help='訓練 epoch 數')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--subdivisions', type=int,
                        default=32, help='Subdivisions (降低 VRAM 用量)')
    parser.add_argument('--lr', type=float,
                        default=0.00261, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=416, help='輸入圖片尺寸')
    parser.add_argument('--weights', type=str, default=None,
                        help='預訓練權重路徑 (.weights 或 .pth)')
    parser.add_argument('--cfg-file', type=str, default=None,
                        help='模型設定檔路徑 (.cfg)')
    parser.add_argument('--resume', type=str, default=None,
                        help='繼續訓練的 checkpoint 路徑 (.pth)')
    args = parser.parse_args()

    cfg = edict()
    cfg.cfgfile = args.cfg_file or os.path.join(
        PROJECT_DIR, "cfg", "yolov4-tiny-custom.cfg")
    cfg.pretrained = args.weights or os.path.join(
        PROJECT_DIR, "weights", "yolov4-tiny.conv.29")
    cfg.batch = args.batch
    cfg.subdivisions = args.subdivisions
    cfg.width = args.img_size
    cfg.height = args.img_size
    cfg.w = args.img_size
    cfg.h = args.img_size
    cfg.channels = 3
    cfg.momentum = 0.9
    cfg.decay = 0.0005
    cfg.angle = 0
    cfg.saturation = 1.5
    cfg.exposure = 1.5
    cfg.hue = 0.1
    cfg.learning_rate = args.lr
    cfg.burn_in = 1000
    cfg.max_batches = 2000
    cfg.steps = [1600, 1800]
    cfg.classes = 1
    cfg.boxes = 60
    cfg.jitter = 0.2
    cfg.flip = 1
    cfg.blur = 0
    cfg.gaussian = 0
    cfg.cutmix = 0
    cfg.mosaic = 0
    cfg.mixup = 0  # Roboflow 已做 augmentation，關閉 mosaic 避免維度不符
    cfg.letter_box = 0
    cfg.TRAIN_EPOCHS = args.epochs
    cfg.train_label = os.path.join(PROJECT_DIR, "cfg", "train.txt")
    cfg.val_label = os.path.join(PROJECT_DIR, "cfg", "test.txt")
    cfg.dataset_dir = ""  # 因為 train.txt 裡用絕對路徑
    cfg.TRAIN_OPTIMIZER = 'adam'
    cfg.checkpoints = os.path.join(PROJECT_DIR, "backup")
    cfg.TRAIN_TENSORBOARD_DIR = os.path.join(PROJECT_DIR, "log")
    cfg.keep_checkpoint_max = 10
    cfg.iou_type = 'iou'
    cfg.use_darknet_cfg = True
    cfg.resume = args.resume

    return cfg


def train(model, device, cfg):
    """主訓練迴圈"""
    epochs = cfg.TRAIN_EPOCHS
    mini_batch = cfg.batch // cfg.subdivisions

    # 建立資料集
    train_dataset = Yolo_dataset(cfg.train_label, cfg, train=True)
    n_train = len(train_dataset)

    if n_train == 0:
        print("\n[ERROR] train data is 0! Run prepare_dataset.py first")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, batch_size=mini_batch, shuffle=True,
        num_workers=0,  # Windows 建議用 0
        pin_memory=True, drop_last=True, collate_fn=collate
    )

    print(f"\n{'='*60}")
    print(f"  開始訓練 YOLOv4-tiny")
    print(f"{'='*60}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {cfg.batch}")
    print(f"  Subdivisions:  {cfg.subdivisions}")
    print(f"  Mini-batch:    {mini_batch}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Train images:  {n_train}")
    print(f"  Image size:    {cfg.width}x{cfg.height}")
    print(f"  Device:        {device}")
    print(f"  Checkpoints:   {cfg.checkpoints}")
    print(f"{'='*60}\n")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,  # Fix: Adam does not need LR divided by batch size.
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    # Learning rate scheduler
    def burnin_schedule(i):
        if i < cfg.burn_in:
            factor = pow(i / cfg.burn_in, 4)
        elif i < cfg.steps[0]:
            factor = 1.0
        elif i < cfg.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    # Loss
    criterion = YoloLoss(device=device, batch=mini_batch,
                         n_classes=cfg.classes)

    # Checkpoint 管理
    os.makedirs(cfg.checkpoints, exist_ok=True)
    saved_models = deque()
    best_loss = float('inf')

    model.train()
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100)
        for i, batch_data in enumerate(pbar):
            global_step += 1
            epoch_step += 1

            images = batch_data[0].to(device=device, dtype=torch.float32)
            bboxes = batch_data[1].to(device=device)

            bboxes_pred = model(images)
            loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(
                bboxes_pred, bboxes)

            loss.backward()
            epoch_loss += loss.item()

            if global_step % cfg.subdivisions == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0] * cfg.batch:.6f}'
            })

        avg_loss = epoch_loss / max(epoch_step, 1)
        print(f"  Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")

        # 存檔
        save_path = os.path.join(
            cfg.checkpoints, f'yolov4-tiny-custom_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        saved_models.append(save_path)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(
                cfg.checkpoints, 'yolov4-tiny-custom_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  [BEST] model saved (loss={best_loss:.4f})")

        # 保存最後模型
        last_path = os.path.join(
            cfg.checkpoints, 'yolov4-tiny-custom_last.pth')
        torch.save(model.state_dict(), last_path)

        # 限制 checkpoint 數量
        if len(saved_models) > cfg.keep_checkpoint_max > 0:
            model_to_remove = saved_models.popleft()
            try:
                os.remove(model_to_remove)
            except:
                pass

    print(f"\n{'='*60}")
    print(f"  [DONE] Training complete!")
    print(
        f"  最佳模型: {os.path.join(cfg.checkpoints, 'yolov4-tiny-custom_best.pth')}")
    print(
        f"  最後模型: {os.path.join(cfg.checkpoints, 'yolov4-tiny-custom_last.pth')}")
    print(f"{'='*60}")


def main():
    cfg = get_config()

    # 強制檢查 GPU 是否可用
    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用！此腳本需要 NVIDIA GPU。")
        print("  請安裝 CUDA 版本的 PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    # 設定 GPU
    device = torch.device('cuda')
    print(f"使用裝置: {device}")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_mem:.1f} GB")

    # 載入模型
    print(f"\n載入模型設定: {cfg.cfgfile}")
    model = Darknet(cfg.cfgfile)

    # 載入預訓練權重
    if cfg.resume:
        print(f"繼續訓練: {cfg.resume}")
        model.load_state_dict(torch.load(cfg.resume, map_location=device))
    elif cfg.pretrained and os.path.exists(cfg.pretrained):
        print(f"載入預訓練權重: {cfg.pretrained}")
        try:
            model.load_weights(cfg.pretrained)
        except Exception as e:
            print(f"  [警告] 無法載入 .weights 檔: {e}")
            print(f"  將使用隨機初始化權重")

    model.to(device=device)
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 檢查訓練資料
    if not os.path.exists(cfg.train_label):
        print(f"\n[ERROR] Cannot find {cfg.train_label}")
        print("請先執行: python prepare_dataset.py")
        sys.exit(1)

    with open(cfg.train_label, 'r') as f:
        n_lines = len(f.readlines())
    if n_lines == 0:
        print(f"\n[ERROR] {cfg.train_label} is empty!")
        print("請先執行: python prepare_dataset.py")
        sys.exit(1)

    print(f"訓練資料: {n_lines} 筆")

    # 開始訓練
    try:
        train(model=model, config=cfg, device=device)
    except KeyboardInterrupt:
        print('\n\n[WARN] Training interrupted!')
        interrupt_path = os.path.join(
            cfg.checkpoints, 'yolov4-tiny-custom_interrupted.pth')
        torch.save(model.state_dict(), interrupt_path)
        print(f'已存檔至: {interrupt_path}')


# 修正: 讓 train 函數正確接收參數
def _main():
    cfg = get_config()
    device = torch.device('cuda')
    print(f"使用裝置: {device}")
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass

    print(f"\n載入模型設定: {cfg.cfgfile}")
    model = Darknet(cfg.cfgfile)

    if cfg.resume:
        print(f"繼續訓練: {cfg.resume}")
        model.load_state_dict(torch.load(cfg.resume, map_location=device))
    elif cfg.pretrained and os.path.exists(cfg.pretrained):
        print(f"載入預訓練權重: {cfg.pretrained}")
        try:
            model.load_weights(cfg.pretrained)
        except Exception as e:
            print(f"  [警告] 無法載入 .weights 檔: {e}")

    model.to(device=device)
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    if not os.path.exists(cfg.train_label):
        print(f"\n[ERROR] Cannot find {cfg.train_label}")
        print("請先執行: python prepare_dataset.py")
        sys.exit(1)

    try:
        train(model=model, device=device, cfg=cfg)
    except KeyboardInterrupt:
        print('\n\n[WARN] Training interrupted!')
        interrupt_path = os.path.join(
            cfg.checkpoints, 'yolov4-tiny-custom_interrupted.pth')
        torch.save(model.state_dict(), interrupt_path)
        print(f'已存檔至: {interrupt_path}')


if __name__ == "__main__":
    _main()
