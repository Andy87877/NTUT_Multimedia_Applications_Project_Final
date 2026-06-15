import os
import random
import shutil
import csv
from pathlib import Path


def random_select_and_copy(source_folders, num_per_folder=25, output_folder="train_img"):
    """
    從多個資料夾中隨機選擇照片，複製到輸出資料夾，並記錄來源

    Args:
        source_folders: 源資料夾列表 (e.g., ['screenshots_1', 'screenshots_2', ...])
        num_per_folder: 每個資料夾要選擇的照片數量
        output_folder: 輸出資料夾名稱
    """

    # 創建輸出資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✓ 已創建輸出資料夾: {output_folder}\n")

    # 記錄來源的 CSV 文件
    source_log_file = os.path.join(output_folder, "source_log.csv")

    with open(source_log_file, 'w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['新檔名', '原檔名', '來源資料夾'])  # 表頭

        total_copied = 0

        # 遍歷每個源資料夾
        for source_folder in source_folders:
            if not os.path.exists(source_folder):
                print(f"⚠ 資料夾不存在: {source_folder}")
                continue

            # 獲取該資料夾中的所有圖片
            image_files = [
                f for f in os.listdir(source_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
            ]

            if len(image_files) == 0:
                print(f"⚠ {source_folder} 中沒有找到圖片")
                continue

            print(f"📁 {source_folder}: 找到 {len(image_files)} 張圖片")

            # 檢查是否有足夠的圖片
            if len(image_files) < num_per_folder:
                print(f"   ⚠ 只有 {len(image_files)} 張，無法取得 {num_per_folder} 張")
                selected = image_files
            else:
                # 隨機選擇指定數量的圖片
                selected = random.sample(image_files, num_per_folder)

            # 複製選中的圖片
            for idx, filename in enumerate(selected, 1):
                source_path = os.path.join(source_folder, filename)

                # 生成新檔名：保留原檔案副檔名
                file_extension = os.path.splitext(filename)[1]
                new_filename = f"{source_folder}_{idx:02d}{file_extension}"
                output_path = os.path.join(output_folder, new_filename)

                # 複製檔案
                shutil.copy2(source_path, output_path)

                # 記錄到 CSV
                writer.writerow([new_filename, filename, source_folder])

                total_copied += 1

            print(f"   ✓ 已複製 {len(selected)} 張圖片\n")

        print("=" * 60)
        print(f"✓ 完成！共複製 {total_copied} 張圖片到 '{output_folder}'")
        print(f"✓ 來源紀錄已保存到: {source_log_file}")
        print("=" * 60)


def main():
    # 定義源資料夾
    source_folders = [
        'screenshots_1',
        'screenshots_3',
        'screenshots_4'
    ]

    print("\n" + "=" * 60)
    print("隨機圖片選擇和複製工具")
    print("=" * 60 + "\n")

    # 檢查源資料夾是否存在
    missing_folders = [f for f in source_folders if not os.path.exists(f)]
    if missing_folders:
        print(f"⚠ 警告：以下資料夾不存在:")
        for f in missing_folders:
            print(f"   - {f}")
        print()

    # 執行隨機選擇和複製
    random_select_and_copy(
        source_folders=source_folders,
        num_per_folder=35,
        output_folder="train_img"
    )

    # 顯示結果
    if os.path.exists("train_img"):
        image_count = len([
            f for f in os.listdir("train_img")
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
        ])
        print(f"\n📊 train_img 資料夾中現有 {image_count} 張圖片")


if __name__ == "__main__":
    main()
