import cv2
import os
import subprocess
from datetime import timedelta
import time

# url = "https://youtu.be/P8Uh9f0EaU8" #------ 1 (Driving Downtown - Taipei Taiwan 4K HDR)
# url = "https://youtu.be/Rynnth1qeVk" #------ 2 (4K 台北駕駛 新北市大漢橋→台北市中心)
# url = "https://youtu.be/N-P79qS3EIY" #------ 3 (Taipei Taiwan 4K - Night Drive)
# ------ 4 (國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1)
url = "https://youtu.be/0crwED4yhBA"


def get_youtube_stream_url(url):
    """
    使用 yt-dlp 獲取 YouTube 視頻的直播流 URL（不下載完整視頻）
    優先選擇最高畫質
    """
    try:
        # 只獲取視頻流（不包含音頻），優先最高分辨率
        # bestvideo = 最高分辨率的視頻，因為我們只需要截圖
        cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]/bestvideo/best", "-g", url]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            stream_url = result.stdout.strip()
            # 處理多個 URL 的情況（視頻和音頻分開時）
            if '\n' in stream_url:
                stream_url = stream_url.split('\n')[0]  # 只取第一個 URL（視頻）
            return stream_url
        else:
            print(f"獲取串流 URL 失敗: {result.stderr}")
            return None
    except FileNotFoundError:
        print("錯誤：未找到 yt-dlp。請先安裝：pip install yt-dlp")
        return None


def get_latest_screenshot_number(output_folder="screenshots"):
    """
    檢測已有的最高截圖序號（用於恢復）
    """
    if not os.path.exists(output_folder):
        return -1

    max_num = -1
    for filename in os.listdir(output_folder):
        if filename.startswith("screenshot_") and filename.endswith(".jpg"):
            try:
                # 從檔案名提取序號 (e.g., screenshot_134_00-02-00.jpg)
                num = int(filename.split("_")[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass

    return max_num


def screenshot_every_minute(stream_url, output_folder="screenshots", start_screenshot_num=None):
    """
    從視頻流每分鐘截一張圖（不需要下載完整視頻）
    可以從指定的截圖序號繼續，包含錯誤處理
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 自動檢測起始序號
    if start_screenshot_num is None:
        latest_num = get_latest_screenshot_number(output_folder)
        start_screenshot_num = latest_num + 1
        if start_screenshot_num > 0:
            print(
                f"偵測到已有 {latest_num + 1} 張截圖，將從第 {start_screenshot_num + 1} 張開始繼續...")

    print("正在連接視頻流...")

    # 打開視頻流
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("錯誤：無法連接視頻流")
        return False

    # 獲取視頻信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # 默認 30 FPS

    print(f"✓ 連接成功！")
    print(f"  FPS: {fps}")

    # 每分鐘（60秒）截一張圖
    screenshot_interval = 60  # 秒
    frame_interval = int(fps * screenshot_interval)  # 對應的幀數

    # 根據已有的截圖數計算起始幀數
    frame_count = start_screenshot_num * frame_interval
    screenshot_count = start_screenshot_num

    # 如果需要續接，跳轉到對應的時間位置
    if start_screenshot_num > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        start_time = frame_count / fps
        print(f"正在跳轉到時間點: {str(timedelta(seconds=int(start_time)))}")

    print(f"\n開始截圖（間隔：{screenshot_interval}秒）...")

    consecutive_errors = 0  # 連續錯誤計數
    max_consecutive_errors = 10  # 允許的最大連續錯誤次數

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n✗ 視頻流連續出現 {consecutive_errors} 次讀取失敗，視頻結束或連接中斷")
                    break
                else:
                    print(f"⚠ 讀取失敗，嘗試恢復... ({consecutive_errors}/{max_consecutive_errors})")
                    time.sleep(1)
                    continue
            else:
                consecutive_errors = 0  # 成功讀取，重置錯誤計數

            # 如果當前幀是我們要截的
            if frame_count % frame_interval == 0:
                current_time = frame_count / fps
                time_str = str(timedelta(seconds=int(current_time)))

                # 保存截圖
                filename = f"{output_folder}/screenshot_{screenshot_count:03d}_{time_str.replace(':', '-')}.jpg"
                cv2.imwrite(filename, frame)
                print(
                    f"  ✓ 第 {screenshot_count + 1} 張: {filename} (時間: {time_str})")
                screenshot_count += 1

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\n用戶中止截圖")
    finally:
        cap.release()
        print(f"\n完成！共截 {screenshot_count} 張圖，保存在 '{output_folder}' 資料夾")

    return True


def main():
    # YouTube 連結
    # url = "https://youtu.be/P8Uh9f0EaU8"

    print("=" * 50)
    print("YouTube 視頻自動截圖工具（帶重試機制）")
    print("=" * 50)

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        print(f"\n正在獲取: {url}")

        # 獲取視頻流 URL（不下載完整視頻）
        stream_url = get_youtube_stream_url(url)

        if stream_url:
            print(f"✓ 成功獲取視頻流\n")

            # 直接從流截圖
            success = screenshot_every_minute(stream_url)
            
            if success:
                print("\n✓ 截圖成功完成！")
                break
            else:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\n⚠ 本次嘗試失敗，{retry_count}/{max_retries}，等待 5 秒後重試...")
                    time.sleep(5)
                else:
                    print("\n✗ 達到最大重試次數，請檢查網絡連接後重新執行")
        else:
            retry_count += 1
            if retry_count < max_retries:
                print(f"\n⚠ 無法獲取視頻流，{retry_count}/{max_retries}，等待 5 秒後重試...")
                time.sleep(5)
            else:
                print("\n✗ 多次嘗試後仍無法獲取視頻流")


if __name__ == "__main__":
    main()
