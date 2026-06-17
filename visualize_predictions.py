import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode for terminal execution
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# ========== 設定與路徑 ==========
DATASET_DIR = './road_following_dataset_xy_2026-06-17_08-40-29/dataset_xy'
MODEL_PATH = './road_following_model/best_steering_model_xy.pth'
OUTPUT_IMAGE = './road_following_model/prediction_visual.png'

# 支援中文顯示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置: {device}")

# ========== 自定義資料集讀取 ==========
class XYDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(directory, 'xy_*.jpg')))
        print(f"從 {directory} 載入 {len(self.image_paths)} 張影像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # 從檔名解析座標: xy_XXX_YYY_uuid.jpg
        basename = os.path.basename(image_path)
        parts = basename.split('_')
        x = int(parts[1])
        y = int(parts[2])

        # 正規化到 [-1, 1] (原始解析度 224x224，中心點在 112)
        x = (x - 112.0) / 112.0
        y = (y - 112.0) / 112.0

        if self.transform:
            image = self.transform(image)

        target = torch.tensor([x, y], dtype=torch.float32)
        return image, target, image_path

# ========== 主程式 ==========
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 錯誤：找不到模型權重檔：{MODEL_PATH}，請先確認是否已完成訓練。")
        return
    if not os.path.exists(DATASET_DIR):
        print(f"❌ 錯誤：找不到資料集路徑：{DATASET_DIR}")
        return

    # 測試集 transform (與訓練時相同)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = XYDataset(DATASET_DIR, transform=test_transform)
    total = len(full_dataset)
    
    # 採用與訓練時相同的 90%/10% 分割，並設定固定的 random seed 以獲取一致的測試集
    test_size = int(total * 0.1)
    train_size = total - test_size
    
    # 使用固定隨機種子以分割出相同的測試集
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    print(f"成功劃分資料集！測試集共有 {len(test_dataset)} 張影像。")

    # 載入 ResNet-18 模型並載入權重
    print("正在載入模型與權重...")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 隨機挑選 8 張測試影像進行預測與視覺化
    random.seed(42)  # 固定隨機種子以確保圖表可重複生成
    sample_indices = random.sample(range(len(test_dataset)), 8)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle('新道路模型測試集預測結果 (綠點=真實路線中心, 紅點=模型預測轉向點)', fontsize=16, fontweight='bold')

    for i, idx in enumerate(sample_indices):
        img_tensor, target, img_path = test_dataset[idx]
        
        # 進行預測
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device)).cpu().squeeze()

        # 反正規化座標到像素 (0~224 區間)
        tx = int(target[0].item() * 112 + 112)
        ty = int(target[1].item() * 112 + 112)
        px = int(pred[0].item() * 112 + 112)
        py = int(pred[1].item() * 112 + 112)

        # 讀取原始影像以進行乾淨的繪圖 (避免顯示經 Normalization 的怪異顏色)
        raw_img = Image.open(img_path)

        ax = axes[i // 4][i % 4]
        ax.imshow(raw_img)
        # 標記真實值 (綠點) 與預測值 (紅點)
        ax.plot(tx, ty, 'go', markersize=12, label='真實中心' if i == 0 else "")
        ax.plot(px, py, 'ro', markersize=12, label='預測中心' if i == 0 else "")
        
        # 計算預測誤差 (像素距離)
        error = np.sqrt((tx - px)**2 + (ty - py)**2)
        ax.set_title(f"樣本 {i+1}\n真實({tx}, {ty}) | 預測({px}, {py})\n誤差: {error:.1f} 像素", fontsize=11)
        ax.axis('off')

    # 加入圖例
    fig.legend(loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # 確保儲存目錄存在
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close()
    
    print(f"✓ 視覺化測試完成！預測結果圖表已儲存至：{os.path.abspath(OUTPUT_IMAGE)}")

if __name__ == '__main__':
    main()
