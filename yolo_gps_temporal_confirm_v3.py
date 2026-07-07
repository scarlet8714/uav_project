from flask import Flask, render_template_string, Response
import cv2
import math
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
# 【連續幀確認設定】
# ============================================================

# True：
#   同一目標連續出現 CONFIRM_FRAMES 幀後，
#   才開始計算與顯示 GPS。
#
# False：
#   關閉連續幀確認，YOLO 偵測到就直接計算 GPS。
#
# 想反悔時只需要把 True 改成 False。
ENABLE_TEMPORAL_CONFIRMATION = True

# 【可調參數】連續幾幀視為確認成功，建議先從 3 開始。
CONFIRM_FRAMES = 3

# 【可調參數】中心點距離門檻，單位 pixel。
# 新 detection 與舊 candidate 中心距離小於此值，視為同一目標。
MATCH_DISTANCE_PX = 60.0

# 【可調參數】candidate 最多允許連續消失幾幀。
# 超過後刪除，之後再出現會重新從 count=1 開始。
MAX_MISSING_FRAMES = 5


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


# ============================================================
# 連續幀確認狀態
# ============================================================

candidates = []
next_candidate_id = 0


def update_candidates(detections):
    global candidates, next_candidate_id

    # 關閉連續幀確認時，所有 detection 直接視為 confirmed。
    if not ENABLE_TEMPORAL_CONFIRMATION:
        return [
            (
                detection,
                {
                    "id": -1,
                    "center_x": detection["center_x"],
                    "center_y": detection["center_y"],
                    "count": CONFIRM_FRAMES,
                    "missing": 0,
                    "confirmed": True,
                },
            )
            for detection in detections
        ]

    for candidate in candidates:
        candidate["missing"] += 1

    matched_candidate_ids = set()
    frame_matches = []

    for detection in detections:
        cx = detection["center_x"]
        cy = detection["center_y"]

        best_candidate = None
        best_distance = None

        for candidate in candidates:
            if candidate["id"] in matched_candidate_ids:
                continue

            distance = math.hypot(
                cx - candidate["center_x"],
                cy - candidate["center_y"],
            )

            if distance <= MATCH_DISTANCE_PX:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate

        if best_candidate is not None:
            best_candidate["center_x"] = cx
            best_candidate["center_y"] = cy
            best_candidate["count"] += 1
            best_candidate["missing"] = 0

            if best_candidate["count"] >= CONFIRM_FRAMES:
                best_candidate["confirmed"] = True

            matched_candidate_ids.add(best_candidate["id"])
            frame_matches.append((detection, best_candidate))

        else:
            new_candidate = {
                "id": next_candidate_id,
                "center_x": cx,
                "center_y": cy,
                "count": 1,
                "missing": 0,
                "confirmed": CONFIRM_FRAMES <= 1,
            }

            candidates.append(new_candidate)
            matched_candidate_ids.add(next_candidate_id)
            next_candidate_id += 1

            frame_matches.append((detection, new_candidate))

    candidates = [
        candidate
        for candidate in candidates
        if candidate["missing"] <= MAX_MISSING_FRAMES
    ]

    return frame_matches


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

        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            target_x = (x1 + x2) / 2.0
            target_y = (y1 + y2) / 2.0

            detections.append(
                {
                    "box": box,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": target_x,
                    "center_y": target_y,
                }
            )

        matched_detections = update_candidates(detections)

        for box_index, (detection, candidate) in enumerate(matched_detections):
            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

            target_x = detection["center_x"]
            target_y = detection["center_y"]

            cv2.circle(
                annotated_frame,
                (int(target_x), int(target_y)),
                5,
                (0, 0, 255),
                -1,
            )

            # 尚未 confirmed：不 call GPS，只顯示確認進度。
            if not candidate["confirmed"]:
                progress_text = (
                    f"confirm {candidate['count']}/{CONFIRM_FRAMES}"
                )

                cv2.putText(
                    annotated_frame,
                    progress_text,
                    (
                        int(x1),
                        min(image_height - 10, int(y2) + 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                continue

            # confirmed 後才 call GPS。
            target_gps = estimate_target_gps_from_reader(
                gps_reader=gps,
                target_x=target_x,
                target_y=target_y,
                image_width=image_width,
                image_height=image_height,
                altitude_agl_m=ALTITUDE_AGL_M,
                hfov_deg=HFOV_DEG,
                vfov_deg=VFOV_DEG,
                camera_yaw_offset_deg=CAMERA_YAW_OFFSET_DEG,
            )

            if target_gps is not None:
                target_lat = target_gps["target_lat"]
                target_lon = target_gps["target_lon"]

                gps_text = (
                    f"{target_lat:.7f}, {target_lon:.7f}"
                )

                cv2.putText(
                    annotated_frame,
                    gps_text,
                    (
                        int(x1),
                        min(image_height - 10, int(y2) + 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                if PRINT_TARGET_GPS:
                    print(
                        f"[Target {box_index}] "
                        f"candidate_id={candidate['id']} "
                        f"count={candidate['count']} "
                        f"pixel=({target_x:.1f}, {target_y:.1f}) "
                        f"GPS=({target_lat:.7f}, {target_lon:.7f})"
                    )

            else:
                cv2.putText(
                    annotated_frame,
                    "GPS unavailable",
                    (
                        int(x1),
                        min(image_height - 10, int(y2) + 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
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
