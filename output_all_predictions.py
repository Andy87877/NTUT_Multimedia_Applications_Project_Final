import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split

# ========== 設定與路徑 ==========
DATASET_DIR = './road_following_dataset_xy_2026-06-17_08-40-29/dataset_xy'
MODEL_PATH = './road_following_model/best_steering_model_xy.pth'
OUTPUT_DIR = './road_following_model/test_predictions'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置: {device}")

class XYDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(directory, 'xy_*.jpg')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        basename = os.path.basename(image_path)
        parts = basename.split('_')
        x = int(parts[1])
        y = int(parts[2])

        # 正規化
        x = (x - 112.0) / 112.0
        y = (y - 112.0) / 112.0

        if self.transform:
            image = self.transform(image)

        target = torch.tensor([x, y], dtype=torch.float32)
        return image, target, image_path

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATASET_DIR):
        print("❌ 錯誤：模型或資料集不存在！")
        return

    # 建立輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 載入資料集
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = XYDataset(DATASET_DIR, transform=test_transform)
    
    # 劃分測試集 (與訓練時相同 seed)
    total = len(full_dataset)
    test_size = int(total * 0.1)
    train_size = total - test_size
    generator = torch.Generator().manual_seed(42)
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

    # 載入模型
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    print(f"正在輸出所有 {len(test_dataset)} 張測試影像預測結果...")

    for idx in range(len(test_dataset)):
        img_tensor, target, img_path = test_dataset[idx]
        
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device)).cpu().squeeze()

        # 反正規化座標 (0~224 區間)
        tx = target[0].item() * 112.0 + 112.0
        ty = target[1].item() * 112.0 + 112.0
        px = pred[0].item() * 112.0 + 112.0
        py = pred[1].item() * 112.0 + 112.0

        # 計算像素誤差
        error = np.sqrt((tx - px)**2 + (ty - py)**2)

        # 讀取原始影像並進行繪圖
        raw_img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(raw_img)

        # 畫綠點 (真實中心)
        r = 6
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill='#00FF00', outline='black')
        
        # 畫紅點 (預測轉向點)
        draw.ellipse([px - r, py - r, px + r, py + r], fill='#FF0000', outline='black')

        # 畫一條黃色線連接兩點，表示偏差
        draw.line([tx, ty, px, py], fill='yellow', width=2)

        # 繪製文字標籤 (誤差)
        # 繪製文字背景框以提高辨識度
        draw.rectangle([5, 5, 120, 25], fill='black')
        draw.text((10, 8), f"Error: {error:.1f} px", fill='yellow')

        # 儲存圖片
        filename = os.path.basename(img_path)
        output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
        raw_img.save(output_path)

    print(f"✓ 成功將所有 80 張預測圖片輸出至：{os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()
