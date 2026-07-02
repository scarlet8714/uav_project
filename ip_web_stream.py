from flask import Flask, render_template_string, Response
import cv2

app = Flask(__name__)
# 初始化攝影機
camera = cv2.VideoCapture(0)

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
            # 例如：
            # results = model(frame)
            # frame = results[0].plot() # 把框框畫回 frame 上
            # ---------------------------------------------------

            # 將影像編碼為 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # 使用 multipart/x-mixed-replace 格式推送影像流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # 簡單的 HTML 網頁，直接嵌入視訊流
    return render_template_string('''
        <html>
          <head>
            <title>UAV Video Stream</title>
            <style>
                body { background-color: #333; color: white; text-align: center; font-family: Arial; }
                img { max-width: 100%; height: auto; border: 2px solid #fff; margin-top: 20px; }
            </style>
          </head>
          <body>
            <h1>無人機視訊流</h1>
            <img src="{{ url_for('video_feed') }}">
          </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    # 串流路由
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # host='0.0.0.0' 代表允許本機以外的所有設備（如手機）連進來
    # port=5000 是網頁的通訊埠
    app.run(host='0.0.0.0', port=5000, threaded=True)
