# 爬取交通部 CCTV 即時影像
# https://steam.oxxostudio.tw/category/python/spider/mjpeg.html

# import cv2
# # 來源串流網址
# url = 'https://cctvn5.freeway.gov.tw/abs2mjpg/bmjpg?camera=fffe8f1f-fac6-4fef-8a71-01462fc8354d&0.3955987206798097&t1968=0.5135032914195135'
# cap = cv2.VideoCapture(url)             # 讀取來源

# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     ret, frame = cap.read()             # 讀取影片的每一幀
#     if not ret:
#         print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
#         cap = cv2.VideoCapture(url)     # 有時候串流間隔時間較久會中斷，中斷時重新讀取
#         continue
#     cv2.imshow('camera', frame)     # 如果讀取成功，顯示該幀的畫面
#     if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
#         break
# cap.release()                           # 所有作業都完成後，釋放資源
# cv2.destroyAllWindows()                 # 結束所有視窗


import cv2
import time

# 隧道
tunnel_url = 'https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=10000&0.93428325644914'

url = tunnel_url

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()             # 讀取影片的每一幀
    if ret:
        print('ok')
    else:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        cap = cv2.VideoCapture(url)
        continue
    key = cv2.waitKey(100)              # 每 0.1 秒更新一次
    if key == ord('q'):                 # 按下 q 結束
        break
    elif key == ord('a'):               # 按下 a 儲存當下影格
        # 存成 jpg，取得當下時間作為檔名
        cv2.imwrite(f'tunnel_img/{time.time_ns()}.jpg', frame)
    cv2.imshow('camera', frame)     # 顯示畫面

cap.release()                           # 所有作業都完成後，釋放資源
cv2.destroyAllWindows()                 # 結束所有視窗
