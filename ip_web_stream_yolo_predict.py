from flask import Flask, render_template_string, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# # 關掉自動曝光
# # 有些 UVC webcam：0.25 = 手動曝光，0.75 = 自動曝光
# camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# # 設定手動曝光值
# # 數值依相機不同，常見是負數，例如 -4, -5, -6
# camera.set(cv2.CAP_PROP_EXPOSURE, -5)

# # 關掉自動白平衡
# camera.set(cv2.CAP_PROP_AUTO_WB, 0)

# # 設定手動白平衡
# camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

print("Camera width:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Camera height:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera FPS:", camera.get(cv2.CAP_PROP_FPS))

# 直接使用 Ultralytics 載入模型
# 可以是 .pt，也可以是 Ultralytics 支援的 TensorRT .engine
model = YOLO("model/last.engine")

HTML = """
<!doctype html>
<html>
<head>
    <title>Ultralytics YOLO Stream</title>
</head>
<body>
    <h1>Ultralytics YOLO Stream</h1>
    <img src="/video_feed" width="800">
</body>
</html>
"""

# 如果之後想改成每 3 幀推論一次，可取消下面兩行註解
# frame_count = 0
# last_frame = None


@app.route("/")
def index():
    return render_template_string(HTML)


def gen_frames():
    # 如果要每 3 幀推論一次，取消下一行註解
    # global frame_count, last_frame

    while True:
        success, frame = camera.read()

        if not success:
            break

        # Ultralytics YOLO predict
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        # results[0].plot() 會回傳已畫好框的 BGR ndarray
        annotated_frame = results[0].plot()

        # ===== 每 3 幀推論一次的版本（需要時再使用） =====
        # frame_count += 1
        #
        # if frame_count % 3 == 0 or last_frame is None:
        #     results = model.predict(
        #         source=frame,
        #         imgsz=640,
        #         conf=0.25,
        #         iou=0.45,
        #         verbose=False
        #     )
        #     last_frame = results[0].plot()
        #
        # annotated_frame = last_frame
        # ==============================================

        ret, buffer = cv2.imencode(".jpg", annotated_frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=False,
            use_reloader=False
        )
    finally:
        camera.release()
