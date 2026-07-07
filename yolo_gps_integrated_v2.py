from flask import Flask, render_template_string, Response
import cv2
from ultralytics import YOLO

# ============================================================
# GPS 模組
# 請把 gps_geolocation.py 放在和這支主程式相同的資料夾
# ============================================================
from gps_geolocation import (
    GPSReader,
    estimate_target_gps_from_reader,
)


app = Flask(__name__)


# ============================================================
# 【需要你確認 / 手動修改的參數】
# ============================================================

# ---------- Camera ----------
# USB camera 通常是 0 或 1
CAMERA_SOURCE = 0

# ---------- YOLO ----------
# 你的 TensorRT engine 或 .pt 模型路徑
MODEL_PATH = "model/last.engine"

# YOLO 推論參數
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
YOLO_IOU = 0.45

# ---------- GPS ----------
# SAM-M8Q 接在 Orin NX 的 serial device
GPS_PORT = "/dev/ttyUSB0"

# SAM-M8Q / u-blox NMEA 常見 baudrate。
# 若你的模組實際設定不同，請改成實際 baudrate。
GPS_BAUDRATE = 9600

# ---------- Flight altitude ----------
# 【目前需要你手動填】
# 相機距離地面的高度 AGL，單位：公尺。
# 你的情況大約 60~80 m，例如先用 70。
ALTITUDE_AGL_M = 70.0

# ---------- Camera FOV ----------
# 【目前需要你手動填】
# 相機水平與垂直 FOV，單位：degree。
# 下列只是暫時測試值，不代表你的白牌 USB camera 真實規格。
HFOV_DEG = 70.0
VFOV_DEG = 43.0

# ---------- Camera direction offset ----------
# 【需要你確認】
# 畫面「上方」相對於 GPS COG 的固定偏移角。
#
# 0   = 畫面上方視為無人機移動方向
# 90  = 畫面上方比移動方向順時針偏 90°
# 180 = 相反方向
# 270 = 逆時針偏 90°
#
# 第一版若相機安裝方向與飛行方向一致，先設 0。
CAMERA_YAW_OFFSET_DEG = 0.0

# 是否在 Orin 終端機印出每個目標的 GPS
PRINT_TARGET_GPS = True


# ============================================================
# Camera 初始化
# ============================================================

camera = cv2.VideoCapture(CAMERA_SOURCE)

# # 關掉自動曝光
# # 有些 UVC webcam：0.25 = 手動曝光，0.75 = 自動曝光
# camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# # 設定手動曝光值
# camera.set(cv2.CAP_PROP_EXPOSURE, -5)

# # 關掉自動白平衡
# camera.set(cv2.CAP_PROP_AUTO_WB, 0)

# # 設定手動白平衡
# camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

print("Camera width:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Camera height:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera FPS:", camera.get(cv2.CAP_PROP_FPS))


# ============================================================
# YOLO 初始化
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# GPS Reader 初始化
#
# 注意：
# import 不會自動監聽 GPS。
# 真正開始監聽是在 main 裡呼叫 gps.start()。
# ============================================================

gps = GPSReader(
    port=GPS_PORT,
    baudrate=GPS_BAUDRATE,
)


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

        # ----------------------------------------------------
        # 【相機畫面若上下顛倒 / 倒轉 180°】
        #
        # 需要時取消下一行註解。
        # 一定要在 YOLO predict 之前旋轉，
        # 這樣 YOLO bbox 座標與後續 GPS 換算座標才會一致。
        # ----------------------------------------------------
        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        # ----------------------------------------------------
        # 原始 frame 尺寸
        #
        # 注意：
        # GPS 換算使用的是原始 frame size，
        # 不是 YOLO_IMGSZ。
        # ----------------------------------------------------
        image_height, image_width = frame.shape[:2]

        # ====================================================
        # YOLO predict
        # ====================================================

        results = model.predict(
            source=frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False
        )

        result = results[0]

        # 已經畫好 YOLO bbox 的 BGR ndarray
        annotated_frame = result.plot()


        # ====================================================
        # GPS 定位整合區
        #
        # 每個 YOLO bbox：
        # 1. 抓 xyxy
        # 2. 算 bbox center
        # 3. 呼叫 GPS geolocation function
        # ====================================================

        for box_index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # YOLO bbox center
            target_x = (x1 + x2) / 2.0
            target_y = (y1 + y2) / 2.0

            # 畫 bbox center，方便你確認抓到的中心位置
            cv2.circle(
                annotated_frame,
                (int(target_x), int(target_y)),
                5,
                (0, 0, 255),
                -1
            )

            # ------------------------------------------------
            # 呼叫座標換算函式
            #
            # GPS 正常：
            #   回傳 target GPS
            #
            # GPS 無效 / 沒資料：
            #   回傳 None
            #   YOLO 與 Flask 串流仍繼續運作
            #
            # 依照你之前的要求：
            # 目前沒有加入「使用最後一筆 stale GPS」功能。
            # ------------------------------------------------

            target_gps = estimate_target_gps_from_reader(
                gps_reader=gps,

                # YOLO 自動提供
                target_x=target_x,
                target_y=target_y,

                # frame 自動提供
                image_width=image_width,
                image_height=image_height,

                # 【手動設定】
                altitude_agl_m=ALTITUDE_AGL_M,

                # 【手動設定 / 確認】
                hfov_deg=HFOV_DEG,
                vfov_deg=VFOV_DEG,
                camera_yaw_offset_deg=CAMERA_YAW_OFFSET_DEG,
            )

            # ------------------------------------------------
            # GPS 有效
            # ------------------------------------------------

            if target_gps is not None:
                target_lat = target_gps["target_lat"]
                target_lon = target_gps["target_lon"]

                gps_text = (
                    f"{target_lat:.7f}, {target_lon:.7f}"
                )

                # 在畫面上顯示目標 GPS
                cv2.putText(
                    annotated_frame,
                    gps_text,
                    (
                        int(x1),
                        min(image_height - 10, int(y2) + 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

                if PRINT_TARGET_GPS:
                    print(
                        f"[Target {box_index}] "
                        f"pixel=({target_x:.1f}, {target_y:.1f}) "
                        f"GPS=({target_lat:.7f}, {target_lon:.7f})"
                    )

            # ------------------------------------------------
            # GPS 無效
            # ------------------------------------------------

            else:
                cv2.putText(
                    annotated_frame,
                    "GPS unavailable",
                    (
                        int(x1),
                        min(image_height - 10, int(y2) + 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )


        # ===== 每 3 幀推論一次的原本版本（目前仍未啟用） =====
        # frame_count += 1
        #
        # if frame_count % 3 == 0 or last_frame is None:
        #     results = model.predict(
        #         source=frame,
        #         imgsz=YOLO_IMGSZ,
        #         conf=YOLO_CONF,
        #         iou=YOLO_IOU,
        #         verbose=False
        #     )
        #     last_frame = results[0].plot()
        #
        # annotated_frame = last_frame
        # ====================================================


        # ====================================================
        # Flask MJPEG stream
        # ====================================================

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
        # ====================================================
        # GPS 背景監聽只啟動一次
        # ====================================================
        gps.start()

        print("GPS background reader started.")
        print("Starting Flask YOLO stream...")

        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=False,
            use_reloader=False
        )

    finally:
        # 程式離開時停止 GPS thread 並釋放 camera
        gps.stop()
        camera.release()
