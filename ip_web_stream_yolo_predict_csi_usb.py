from flask import Flask, render_template_string, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# ==================================================
# Camera settings
# ==================================================
# 可選：
#   "usb" -> USB webcam
#   "csi" -> Jetson CSI camera
CAMERA_TYPE = "csi"

# USB camera index
USB_CAMERA_INDEX = 0

# CSI camera settings
CSI_SENSOR_ID = 0
CSI_CAPTURE_WIDTH = 1920
CSI_CAPTURE_HEIGHT = 1080
CSI_DISPLAY_WIDTH = 1280
CSI_DISPLAY_HEIGHT = 720
CSI_FRAMERATE = 30
CSI_FLIP_METHOD = 0


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id={} ! "
        "video/x-raw(memory:NVMM), "
        "width=(int){}, height=(int){}, "
        "format=(string)NV12, framerate=(fraction){}/1 ! "
        "nvvidconv flip-method={} ! "
        "video/x-raw, width=(int){}, height=(int){}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    ).format(
        sensor_id,
        capture_width,
        capture_height,
        framerate,
        flip_method,
        display_width,
        display_height,
    )


def open_camera():
    if CAMERA_TYPE.lower() == "csi":
        pipeline = gstreamer_pipeline(
            sensor_id=CSI_SENSOR_ID,
            capture_width=CSI_CAPTURE_WIDTH,
            capture_height=CSI_CAPTURE_HEIGHT,
            display_width=CSI_DISPLAY_WIDTH,
            display_height=CSI_DISPLAY_HEIGHT,
            framerate=CSI_FRAMERATE,
            flip_method=CSI_FLIP_METHOD,
        )

        print("[Camera] Opening CSI camera...")
        print("[Camera] Pipeline:")
        print(pipeline)

        camera = cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER,
        )

    elif CAMERA_TYPE.lower() == "usb":
        print(
            "[Camera] Opening USB camera index:",
            USB_CAMERA_INDEX,
        )

        camera = cv2.VideoCapture(
            USB_CAMERA_INDEX
        )

    else:
        raise ValueError(
            "CAMERA_TYPE must be 'usb' or 'csi'"
        )

    if not camera.isOpened():
        raise RuntimeError(
            "Failed to open {} camera".format(
                CAMERA_TYPE
            )
        )

    return camera


camera = open_camera()

print(
    "Camera width:",
    camera.get(cv2.CAP_PROP_FRAME_WIDTH),
)

print(
    "Camera height:",
    camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
)

print(
    "Camera FPS:",
    camera.get(cv2.CAP_PROP_FPS),
)


# ==================================================
# Load YOLO model
# ==================================================
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


@app.route("/")
def index():
    return render_template_string(HTML)


def gen_frames():
    while True:
        success, frame = camera.read()

        if not success:
            print("[Camera] Failed to read frame")
            break

        # ==================================================
        # YOLO Predict
        # ==================================================
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            verbose=False,
        )

        annotated_frame = results[0].plot()

        # ==================================================
        # JPEG encode
        # ==================================================
        ret, buffer = cv2.imencode(
            ".jpg",
            annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

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
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


if __name__ == "__main__":
    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=False,
            use_reloader=False,
        )

    finally:
        camera.release()
