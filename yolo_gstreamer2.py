from flask import Flask, render_template_string, Response
import cv2
import math
import numpy as np
import threading
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
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
# ============================================================
# 【相機擷取方式切換】
# ============================================================
#
# "opencv"      = 原本 cv2.VideoCapture()
# "gstreamer_gi" = 使用 gi.repository.Gst + appsink
#
# 只需要改這一個參數。
CAMERA_BACKEND = "gstreamer_gi"

# OpenCV 模式使用的 camera index
CAMERA_SOURCE = 0

# GStreamer 模式使用的裝置
GSTREAMER_DEVICE = "/dev/video0"

# 【需要你依相機支援模式設定】
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# GStreamer JPEG decoder
#
# Jetson 建議先用 nvjpegdec。
# 若要比較軟體解碼，可改成 jpegdec。
# 你目前已確認 jpegdec 可以正常工作。
# 若之後要再測硬體 decoder，可改成 "nvjpegdec"。
GSTREAMER_JPEG_DECODER = "jpegdec"

# True：appsink 只保留最新 frame，避免舊 frame 累積
# False：不主動丟 frame
GSTREAMER_DROP_OLD_FRAMES = True

# pull-sample 最長等待時間，單位秒
GSTREAMER_READ_TIMEOUT_SEC = 2.0

# ============================================================
# 【Camera Thread 開關】
# ============================================================
#
# True：
#   背景 thread 持續讀相機，只保留最新 frame。
#   YOLO 每次只取得「比上次更新」的新 frame。
#
# False：
#   維持原本同步 camera.read() 流程。
#
# 建議先用 True 做 A/B 測試。
ENABLE_CAMERA_THREAD = True

# 主處理流程等待新 frame 的最長時間，單位秒。
CAMERA_THREAD_READ_TIMEOUT_SEC = 2.0

# ---------- YOLO ----------
# 你的 TensorRT engine 或 .pt 模型路徑
MODEL_PATH = "model/11s_car_960.engine"

# YOLO 推論參數
# 你的 TensorRT engine 以 imgsz=960 export
YOLO_IMGSZ = (544, 960)
YOLO_CONF = 0.4
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
ALTITUDE_AGL_M = 75.0

# ---------- Camera FOV ----------
# 【目前需要你手動填】
# 相機水平與垂直 FOV，單位：degree。
# 下列只是暫時測試值，不代表你的白牌 USB camera 真實規格。
HFOV_DEG = 52.0
VFOV_DEG = 31.0

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
CAMERA_YAW_OFFSET_DEG = 180.0

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


class OpenCVCamera:
    """
    原本的 OpenCV VideoCapture 包裝。
    對外保留 read() / release() 介面。
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)

        # 明確要求 MJPG
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError("OpenCV camera 開啟失敗")

        print("[Camera] Backend: OpenCV")
        print("[Camera] Width:", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("[Camera] Height:", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("[Camera] FPS:", self.cap.get(cv2.CAP_PROP_FPS))

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class GStreamerGICamera:
    """
    使用 gi.repository.Gst 直接從 appsink 取得 BGR frame。

    這個方法不依賴 OpenCV 的 GStreamer backend，
    所以即使 cv2.getBuildInformation() 顯示：
        GStreamer: NO
    仍然可以使用系統 GStreamer。
    """

    def __init__(self):
        Gst.init(None)

        drop_value = "true" if GSTREAMER_DROP_OLD_FRAMES else "false"

        self.pipeline_string = (
            f"v4l2src device={GSTREAMER_DEVICE} ! "
            f"image/jpeg,"
            f"width={CAMERA_WIDTH},"
            f"height={CAMERA_HEIGHT},"
            f"framerate={CAMERA_FPS}/1 ! "
            f"{GSTREAMER_JPEG_DECODER} ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink name=appsink "
            f"emit-signals=false "
            f"sync=false "
            f"async=false "
            f"drop={drop_value} "
            f"max-buffers=1"
        )

        print("[Camera] Backend: GStreamer GI")
        print("[Camera] Pipeline:")
        print(self.pipeline_string)

        try:
            self.pipeline = Gst.parse_launch(self.pipeline_string)
        except Exception as exc:
            raise RuntimeError(
                f"GStreamer pipeline 建立失敗: {exc}"
            ) from exc

        self.appsink = self.pipeline.get_by_name("appsink")

        if self.appsink is None:
            raise RuntimeError("找不到 appsink")

        result = self.pipeline.set_state(Gst.State.PLAYING)

        if result == Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer pipeline 無法進入 PLAYING 狀態")

        # 不強制要求 get_state() 立即回報 PLAYING。
        # appsink / live source 可能暫時處於 PAUSED -> PLAYING 過渡狀態。
        # 真正的 runtime 問題由 read() 的 timeout 與 bus ERROR 判斷。

        self.bus = self.pipeline.get_bus()
        self.closed = False

    def _check_bus_error(self):
        """
        非阻塞檢查 GStreamer error / EOS。
        """

        if self.bus is None:
            return None

        message = self.bus.pop_filtered(
            Gst.MessageType.ERROR | Gst.MessageType.EOS
        )

        if message is None:
            return None

        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            return f"GStreamer ERROR: {err}; debug={debug}"

        if message.type == Gst.MessageType.EOS:
            return "GStreamer EOS"

        return None

    def read(self):
        if self.closed:
            return False, None

        bus_error = self._check_bus_error()
        if bus_error:
            print("[Camera]", bus_error)
            return False, None

        timeout_ns = int(GSTREAMER_READ_TIMEOUT_SEC * Gst.SECOND)

        sample = self.appsink.emit(
            "try-pull-sample",
            timeout_ns,
        )

        if sample is None:
            bus_error = self._check_bus_error()

            if bus_error:
                print("[Camera]", bus_error)
            else:
                print("[Camera] GStreamer frame timeout")

            return False, None

        caps = sample.get_caps()
        structure = caps.get_structure(0)

        width = structure.get_value("width")
        height = structure.get_value("height")

        buffer = sample.get_buffer()

        ok, map_info = buffer.map(Gst.MapFlags.READ)

        if not ok:
            print("[Camera] Gst.Buffer map 失敗")
            return False, None

        try:
            frame = np.frombuffer(
                map_info.data,
                dtype=np.uint8,
            ).reshape(
                (height, width, 3)
            ).copy()
        except Exception as exc:
            print(f"[Camera] frame 轉換失敗: {exc}")
            return False, None
        finally:
            buffer.unmap(map_info)

        return True, frame

    def release(self):
        if self.closed:
            return

        self.closed = True
        self.pipeline.set_state(Gst.State.NULL)



class ThreadedCamera:
    """
    Camera thread 包裝層。

    背景 thread：
        持續 base_camera.read()
        只保留最新 frame

    主處理流程：
        read() 只回傳「新的 frame」
        不會把同一張 frame 重複交給 YOLO，
        避免 temporal confirmation 把同一張影像算多次。
    """

    def __init__(self, base_camera):
        self.base_camera = base_camera

        self.condition = threading.Condition()

        self.latest_frame = None
        self.latest_frame_id = 0

        # 單一 processing consumer 的最後已交付 frame id
        self.last_delivered_frame_id = 0

        self.running = True
        self.capture_failed = False
        self.capture_error = None

        self.thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True,
        )
        self.thread.start()

        print("[CameraThread] started")

    def _capture_loop(self):
        while self.running:
            try:
                success, frame = self.base_camera.read()
            except Exception as exc:
                with self.condition:
                    self.capture_failed = True
                    self.capture_error = str(exc)
                    self.running = False
                    self.condition.notify_all()
                return

            if not success or frame is None:
                with self.condition:
                    self.capture_failed = True
                    self.capture_error = "base camera read failed"
                    self.running = False
                    self.condition.notify_all()
                return

            # 用 Condition 保護 frame 與 frame_id。
            # frame 本身直接替換成最新一張，舊 frame 自動被丟棄。
            with self.condition:
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.condition.notify_all()

    def read(self):
        """
        等待一張比上次更新的 frame。

        回傳：
            (True, frame)  成功取得新 frame
            (False, None) timeout 或 capture thread 失敗
        """

        deadline = time.monotonic() + CAMERA_THREAD_READ_TIMEOUT_SEC

        with self.condition:
            while (
                self.running
                and self.latest_frame_id <= self.last_delivered_frame_id
            ):
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    print("[CameraThread] wait for new frame timeout")
                    return False, None

                self.condition.wait(timeout=remaining)

            if self.capture_failed:
                print(
                    "[CameraThread] capture failed:",
                    self.capture_error,
                )
                return False, None

            if self.latest_frame is None:
                return False, None

            self.last_delivered_frame_id = self.latest_frame_id

            # copy() 避免 capture thread 更新引用時影響 processing
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False

        with self.condition:
            self.condition.notify_all()

        # 先釋放底層 camera，讓可能阻塞中的 read() 有機會返回。
        try:
            self.base_camera.release()
        except Exception as exc:
            print("[CameraThread] base camera release error:", exc)

        if self.thread.is_alive():
            self.thread.join(timeout=3.0)

        print("[CameraThread] stopped")



def create_base_camera():
    backend = CAMERA_BACKEND.strip().lower()

    if backend == "opencv":
        return OpenCVCamera()

    if backend == "gstreamer_gi":
        return GStreamerGICamera()

    raise ValueError(
        "CAMERA_BACKEND 只能是 "
        "'opencv' 或 'gstreamer_gi'，"
        f"目前值：{CAMERA_BACKEND!r}"
    )


base_camera = create_base_camera()

if ENABLE_CAMERA_THREAD:
    camera = ThreadedCamera(base_camera)
else:
    camera = base_camera
    print("[CameraThread] disabled")


# ============================================================
# YOLO 初始化
# ============================================================

model = YOLO(MODEL_PATH, task="detect")


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

    first_frame = True

    while True:
        success, frame = camera.read()

        if not success:
            print("[Camera] frame read failed.")
            break

        if first_frame:
            actual_h, actual_w = frame.shape[:2]
            print(
                f"[Camera] Actual frame: "
                f"{actual_w}x{actual_h}"
            )
            first_frame = False

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
