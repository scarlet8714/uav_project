import cv2
import sys

def main():
    # 0 通常是內建或第一個接上的 USB Webcam
    # 如果畫面沒出來，可以嘗試換成 1, 2 等數字
    camera_id = 0 
    
    # 初始化攝影機
    cap = cv2.VideoCapture(camera_id)
    
    # 檢查攝影機是否成功開啟
    if not cap.isOpened():
        print(f"錯誤：無法開啟編號為 {camera_id} 的攝影機。")
        print("請檢查 USB 是否插緊，或嘗試更換 camera_id。")
        sys.exit()
        
    # 設定影像解析度（可依你的 Webcam 規格調整，Jetson Nano 建議不要調太高以維持流暢度）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("視訊串流已開啟！")
    print("提示：在畫面上按下 'q' 鍵可以關閉視窗並離開程式。")

    while True:
        # 讀取下一幀畫面
        ret, frame = cap.read()
        
        # 如果讀取失敗，跳出迴圈
        if not ret:
            print("錯誤：無法接收影像畫面（Stream end?）。")
            break
            
        # 顯示畫面，視窗名稱為 'Webcam Stream'
        cv2.imshow('Webcam Stream', frame)
        
        # 偵測鍵盤事件，等待 1 毫秒；如果按下 'q' 鍵則跳出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("使用者按下 'q'，正在關閉...")
            break
            
    # 釋放攝影機資源並關閉所有 OpenCV 視窗
    cap.release()
    cv2.destroyAllWindows()
    print("程式已安全結束。")

if __name__ == '__main__':
    main()
