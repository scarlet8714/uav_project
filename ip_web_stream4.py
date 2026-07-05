from flask import Flask, render_template_string, Response
import cv2

from trt_detector import TRTDetector


app = Flask(__name__)


# ============================================================
# Camera 設定
# ============================================================

CAMERA_TYPE = "csi"
# CAMERA_TYPE = "usb"


# ------------------------------------------------------------
# CSI Camera GStreamer Pipeline
# ------------------------------------------------------------
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0
):
    return (
        "nvarguscamerasrc sensor-id={} ! "
        "video/x-raw(memory:NVMM), "
        "width=(int){}, "
        "height=(int){}, "
        "framerate=(fraction){}/1 ! "
        "nvvidconv flip-method={} ! "
        "video/x-raw, "
        "width=(int){}, "
        "height=(int){}, "
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
        display_height
    )


# ============================================================
# Camera 初始化
# ============================================================

if CAMERA_TYPE == "csi":

    pipeline = gstreamer_pipeline(
        sensor_id=0,

        # CSI 實際擷取解析度
        capture_width=1920,
        capture_height=1080,

        # OpenCV 收到的解析度
        display_width=1280,
        display_height=720,

        framerate=30,
        flip_method=0
    )

    print("Using CSI Camera")
    print(pipeline)

    camera = cv2.VideoCapture(
        pipeline,
        cv2.CAP_GSTREAMER
    )


elif CAMERA_TYPE == "usb":

    print("Using USB Camera")

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_V4L2
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )


else:

    raise ValueError(
        "CAMERA_TYPE must be 'csi' or 'usb'"
    )


# ============================================================
# Camera 檢查
# ============================================================

if not camera.isOpened():

    raise RuntimeError(
        "Failed to open camera"
    )


print(
    "Camera width:",
    camera.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

print(
    "Camera height:",
    camera.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

print(
    "Camera FPS:",
    camera.get(
        cv2.CAP_PROP_FPS
    )
)


# ============================================================
# TensorRT YOLO Detector
# ============================================================

model = TRTDetector(
    engine_path="model/last.engine",

    # 如果你的 TRTDetector 已改成
    # 自動讀 engine input shape，
    # 這個 input_size 就可以移除。

    input_size=640,

    conf_thres=0.25,
    iou_thres=0.45
)


# ============================================================
# HTML
# ============================================================

HTML = """
<!doctype html>

<html>

<head>

    <title>
        TensorRT YOLO Stream
    </title>

</head>

<body>

    <h1>
        TensorRT YOLO Stream
    </h1>

    <img
        src="/video_feed"
        width="800"
    >

</body>

</html>
"""


# ============================================================
# Flask
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


# ============================================================
# Stream Generator
# ============================================================

def gen_frames():

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Camera read failed"
            )

            break


        # ==============================================
        # TensorRT YOLO inference
        # ==============================================

        detections = model(
            frame
        )


        # ==============================================
        # Draw detections
        # ==============================================

        frame = model.draw(
            frame,
            detections,
            class_names=["car"]
        )


        # ==============================================
        # JPEG Encode
        # ==============================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
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


# ============================================================
# Video Feed
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        gen_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# Main
# ============================================================

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