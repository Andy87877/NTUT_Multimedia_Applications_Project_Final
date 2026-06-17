<!-- <https://www.youtube.com/watch?v=5LmEo7RmojM>
這是晴天的，有高速公路也有一般道路

<https://www.youtube.com/watch?v=BTmncLlPGvk>
這是大雨天的

<https://www.youtube.com/watch?v=N-P79qS3EIY>
這是夜晚的

請參考 *\Project4\Imgs\YT_to_Img.py 的程式碼
把上述的影片的圖片抓下來(用隨機的方式(?)) ，然後放到*\Project4\Detect_random_imgs\imgs_original 裡面
然後再寫一個程式 用yolo26 把這些圖片做物件偵測，並把結果放到 *\Project4\Detect_random_imgs\imgs_detected 裡面

就是這樣，請幫我完成這個任務，謝謝！ -->

# YOLO26 物件偵測

這個專案的目標是從 YouTube 影片中隨機擷取圖片，並使用 YOLO26 模型對這些圖片進行物件偵測。以下是專案的詳細說明：

## 目標

1. 從指定的 YouTube 影片中隨機擷取圖片。
2. 使用 YOLO26 模型對擷取的圖片進行物件偵測。
3. 將偵測結果保存到指定的資料夾中。
