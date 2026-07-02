from flask import Flask, render_template_string, Response
import cv2

app = Flask(__name__)

# 初始化攝影機
camera = cv2.VideoCapture(0)

# 【修改點 1】強制設定攝影機硬體解析度（釋放最廣角視野）
# 很多攝影機在預設 640x480 下會縮減視角，改成 1920x1080 或 1280x720 才能看到最廣角
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 預留的 YOLO 載入位置
# TODO: model = yolov8.load(...)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # ---------------------------------------------------
            # 【預留位置】未來你的 YOLO 辨識程式碼寫在這裡！
            # ---------------------------------------------------

            # 將影像編碼為 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # 使用 multipart/x-mixed-replace 格式推送影像流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # 【修改點 2】修改 CSS 樣式，將 width 設為 95vw，讓視訊畫面在網頁上撐到最大
    return render_template_string('''
        <html>
          <head>
            <title>UAV Video Stream</title>
            <style>
                body { 
                    background-color: #333; 
                    color: white; 
                    text-align: center; 
                    font-family: Arial; 
                    margin: 0; 
                    padding: 0; 
                }
                h1 { 
                    margin-top: 15px; 
                    font-size: 24px; 
                }
                img { 
                    width: 95vw;       /* 寬度佔滿螢幕寬度的 95% */
                    max-height: 80vh;  /* 高度最高佔螢幕高度的 80%，防止爆出螢幕 */
                    object-fit: contain;/* 保持原有比例，不拉伸變形 */
                    border: 3px solid #00FF00; /* 改成綠框，確認畫面放大成功 */
                    box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
                }
            </style>
          </head>
          <body>
            <h1>無人機視訊流 (最高解析度廣角模式)</h1>
            <img src="{{ url_for('video_feed') }}">
          </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
