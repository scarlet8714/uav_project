from flask import Flask, render_template_string, Response
import cv2
from trt_detector import TRTDetector

app = Flask(__name__)

camera = cv2.VideoCapture(0)

print("Camera width:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Camera height:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera FPS:", camera.get(cv2.CAP_PROP_FPS))

model = TRTDetector(
    engine_path="model/best.engine",
    input_size=640,
    conf_thres=0.25,
    iou_thres=0.45
)

HTML = """
<!doctype html>
<html>
<head>
    <title>TensorRT YOLOv8 Stream</title>
</head>
<body>
    <h1>TensorRT YOLOv8 Stream</h1>
    <img src="/video_feed" width="800">
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


def gen_frames():
    ############
    global frame_count, last_detections
    ############
    while True:
        success, frame = camera.read()

        if not success:
            break

        # detections = model(frame)
        # frame = model.draw(frame, detections, class_names=["car"])

        ##############
        frame_count += 1

        # 每 3 幀才推論一次
        if frame_count % 3 == 0:
            last_detections = model(frame)

        # 其他幀沿用上一次結果
        frame = model.draw(frame, last_detections, class_names=["car"])
        ##############

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False, use_reloader=False)
