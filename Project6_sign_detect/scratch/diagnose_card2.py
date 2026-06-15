# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import cv2
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.append(str(SCRIPTS_DIR))

from train_pytorch_yolov4tiny_v2 import YoloV4Tiny, load_darknet_weights, ANCHORS

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV4Tiny(num_classes=4).to(device)
    weights_path = Path(__file__).resolve().parent.parent / "jetbot_deploy" / "yolov4-tiny-416.weights"
    load_darknet_weights(model, str(weights_path))
    model.eval()

    # Find the image xy_092_057_d4ae3f52-5a9d-11f1-816a-7404f1c2a475_jpg
    obj_dir = Path(__file__).resolve().parent.parent / "obj"
    img_path = next(obj_dir.glob("*d4ae3f52*.jpg"))
    print(f"Found target image: {img_path.name}")

    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = cv2.resize(img_in, (416, 416))
    img_t = torch.from_numpy(img_in).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        out1, out2 = model(img_t)

    print("\n--- Raw Detections (conf > 0.05) ---")
    CLASS_NAMES = ["stop", "rail", "pedestrian", "blocked"]
    
    for scale_idx, out in enumerate([out1, out2]):
        _, _, H_out, W_out = out.shape
        out = out.view(1, 3, 9, H_out, W_out).permute(0, 1, 3, 4, 2)
        
        xy = torch.sigmoid(out[..., 0:2])
        wh = torch.exp(out[..., 2:4])
        conf = torch.sigmoid(out[..., 4])
        cls_probs = torch.sigmoid(out[..., 5:])

        scale_anchors = ANCHORS[scale_idx]

        for a in range(3):
            for y in range(H_out):
                for x in range(W_out):
                    score = conf[0, a, y, x].item()
                    class_probs = cls_probs[0, a, y, x]
                    class_id = torch.argmax(class_probs).item()
                    class_score = class_probs[class_id].item()
                    final_score = score * class_score

                    if final_score > 0.05:
                        aw, ah = scale_anchors[a]
                        gx = (xy[0, a, y, x, 0].item() + x) / W_out
                        gy = (xy[0, a, y, x, 1].item() + y) / H_out
                        gw = (wh[0, a, y, x, 0].item() * aw) / 416
                        gh = (wh[0, a, y, x, 1].item() * ah) / 416

                        x1 = int((gx - gw/2) * W)
                        y1 = int((gy - gh/2) * H)
                        x2 = int((gx + gw/2) * W)
                        y2 = int((gy + gh/2) * H)

                        print(f"Scale {scale_idx} | Anchor {a} | Cell ({y},{x}) | Class: {CLASS_NAMES[class_id]} | Score: {final_score:.4f} (conf: {score:.4f}, cls: {class_score:.4f}) | Box: [{x1}, {y1}, {x2}, {y2}]")

if __name__ == "__main__":
    main()
