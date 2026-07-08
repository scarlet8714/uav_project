from flask import Flask, render_template_string, Response
import cv2
import math
import numpy as np
import threading
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

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

# Camera Thread 等待新 frame 的最長時間，單位：秒。
CAMERA_THREAD_READ_TIMEOUT_SEC = 2.0

# ---------- YOLO ----------
# 你的 TensorRT engine 或 .pt 模型路徑
MODEL_PATH = "model/11s_car_rec.engine"

# YOLO 推論參數
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
ALTITUDE_AGL_M = 80.0

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
# 【HSV 紅色判斷與 Annotator 框線設定】
# ============================================================

# True：啟用 HSV 紅色檢測
# False：關閉 HSV 判斷，全部使用藍色框
ENABLE_HSV_RED_CHECK = True

# 【需要你確認 / 可調】
# 以 bbox 中心為基準，向上下左右各取幾個 pixel。
# 9 代表 ROI 大約是 19 x 19。
ROI_HALF_SIZE = 11

# 【需要你實測調整】
# ROI 內紅色 pixel 數大於此值時，框改成紅色。
# 目前依你的設計先用 100。
RED_PIXEL_THRESHOLD = 60

# OpenCV HSV 的 Hue 範圍是 0~179。
# 紅色跨越 Hue 頭尾，因此使用兩段範圍。
#
# 【可調參數】
# 若戶外實測時紅色抓不到或誤判過多，主要調整這裡。
RED_LOWER_1 = np.array([0, 70, 50], dtype=np.uint8)
RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)

RED_LOWER_2 = np.array([170, 70, 50], dtype=np.uint8)
RED_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

# Annotator 使用 BGR 顏色順序
RED_BOX_COLOR = (0, 0, 255)
BLUE_BOX_COLOR = (255, 0, 0)

# 【可調參數】框線粗細
BOX_LINE_WIDTH = 2

# True：終端機印出 red_count / red_ratio，方便找 threshold
# False：正常使用時關閉，避免終端機一直刷
PRINT_HSV_DEBUG = False


# ============================================================
# Camera Thread
#
# 只使用 OpenCV cv2.VideoCapture，不使用 GStreamer。
#
# 背景 thread：
#   持續讀取 camera，只保留最新 frame。
#
# 主處理 thread：
#   每次 read() 只會取得比上一次更新的 frame。
#   內部 frame_id 不會顯示在畫面上，只用來避免同一張影像
#   被 temporal confirmation 重複計數。
# ============================================================


class ThreadedCamera:
    def __init__(self, base_camera):
        self.base_camera = base_camera
        self.condition = threading.Condition()

        self.latest_frame = None
        self.latest_frame_id = 0
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
                    self.capture_error = "OpenCV camera read failed"
                    self.running = False
                    self.condition.notify_all()
                return

            # latest-frame only：
            # 直接覆蓋舊 frame，不建立 queue。
            with self.condition:
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.condition.notify_all()

    def read(self):
        deadline = (
            time.monotonic()
            + CAMERA_THREAD_READ_TIMEOUT_SEC
        )

        with self.condition:
            while (
                self.running
                and self.latest_frame_id
                <= self.last_delivered_frame_id
            ):
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    print(
                        "[CameraThread] wait for new frame timeout"
                    )
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

            self.last_delivered_frame_id = (
                self.latest_frame_id
            )

            return True, self.latest_frame.copy()

    def release(self):
        self.running = False

        with self.condition:
            self.condition.notify_all()

        try:
            self.base_camera.release()
        except Exception as exc:
            print(
                "[CameraThread] base camera release error:",
                exc,
            )

        if self.thread.is_alive():
            self.thread.join(timeout=3.0)

        print("[CameraThread] stopped")


# ============================================================
# Camera 初始化
# ============================================================

base_camera = cv2.VideoCapture(CAMERA_SOURCE)
base_camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)
base_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
base_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
base_camera.set(cv2.CAP_PROP_FPS, 15)

if not base_camera.isOpened():
    raise RuntimeError(
        f"Cannot open camera source: {CAMERA_SOURCE}"
    )

print(
    "Camera width:",
    base_camera.get(cv2.CAP_PROP_FRAME_WIDTH),
)
print(
    "Camera height:",
    base_camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
)
print(
    "Camera FPS:",
    base_camera.get(cv2.CAP_PROP_FPS),
)

camera = ThreadedCamera(base_camera)


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
<body style="width:100%; display:flex justify-content:center">
    <h1>Ultralytics YOLO Stream</h1>
    <div style="width:80%">
    <img src="/video_feed" width="800">
    </div>
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


def count_red_pixels(frame, center_x, center_y):
    """
    以 bbox 中心為基準，切出固定大小 ROI，
    轉 HSV 後計算紅色 pixel 數。

    回傳：
        red_count
        red_ratio
        roi_bounds
    """

    image_height, image_width = frame.shape[:2]

    cx = int(round(center_x))
    cy = int(round(center_y))

    x_start = max(0, cx - ROI_HALF_SIZE)
    x_end = min(image_width, cx + ROI_HALF_SIZE + 1)

    y_start = max(0, cy - ROI_HALF_SIZE)
    y_end = min(image_height, cy + ROI_HALF_SIZE + 1)

    roi = frame[y_start:y_end, x_start:x_end]

    if roi.size == 0:
        return 0, 0.0, (x_start, y_start, x_end, y_end)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # mask_1 = cv2.inRange(
    #     hsv_roi,
    #     RED_LOWER_1,
    #     RED_UPPER_1,
    # )

    # mask_2 = cv2.inRange(
    #     hsv_roi,
    #     RED_LOWER_2,
    #     RED_UPPER_2,
    # )

    # red_mask = cv2.bitwise_or(mask_1, mask_2)

    # red_count = int(cv2.countNonZero(red_mask))
    ######################## 第二版##################
    mask_red1 = cv2.inRange(
        hsv_roi,
        np.array([0, 30, 15]),
        np.array([20, 255, 255]),
    )

    mask_purple = cv2.inRange(
        hsv_roi,
        np.array([105, 35, 15]),
        np.array([155, 255, 255]),
    )

    mask_red2 = cv2.inRange(
        hsv_roi,
        np.array([165, 30, 15]),
        np.array([179, 255, 255]),
    )

    red_mask = cv2.bitwise_or(mask_red1, mask_purple)
    red_mask = cv2.bitwise_or(red_mask, mask_red2)

    red_count = cv2.countNonZero(red_mask)
    #############################################
    roi_pixel_count = roi.shape[0] * roi.shape[1]

    red_ratio = (
        red_count / roi_pixel_count
        if roi_pixel_count > 0
        else 0.0
    )

    return (
        red_count,
        red_ratio,
        (x_start, y_start, x_end, y_end),
    )


# ============================================================
# GPS 文字避碰與畫面邊界限制
# ============================================================

GPS_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
GPS_TEXT_SCALE = 0.65
GPS_TEXT_THICKNESS = 1
GPS_TEXT_MARGIN = 5


def rectangles_overlap(rect_a, rect_b, margin=GPS_TEXT_MARGIN):
    """
    rect 格式：(left, top, right, bottom)
    margin 讓不同 GPS 文字之間保留一點空隙。
    """

    return not (
        rect_a[2] + margin < rect_b[0]
        or rect_b[2] + margin < rect_a[0]
        or rect_a[3] + margin < rect_b[1]
        or rect_b[3] + margin < rect_a[1]
    )


def clamp_text_position(
    text_x,
    text_y,
    text_width,
    text_height,
    baseline,
    image_width,
    image_height,
):
    """
    OpenCV putText 的座標是文字 baseline 左下角。
    將文字完整限制在畫面內。
    """

    max_x = max(0, image_width - text_width - 1)
    min_y = text_height + 1
    max_y = max(
        min_y,
        image_height - baseline - 1,
    )

    clamped_x = max(0, min(int(text_x), max_x))
    clamped_y = max(
        min_y,
        min(int(text_y), max_y),
    )

    return clamped_x, clamped_y


def choose_gps_text_position(
    text,
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    used_text_rects,
):
    """
    依序嘗試：
      1. 框下方
      2. 框上方
      3. 下方第二行
      4. 上方第二行
      5. 左右偏移位置

    每個位置都先 clamp 到畫面內，再檢查是否與其他 GPS 文字重疊。
    """

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        GPS_TEXT_FONT,
        GPS_TEXT_SCALE,
        GPS_TEXT_THICKNESS,
    )

    center_x = (x1 + x2) / 2.0

    candidate_positions = [
        # 下方
        (x1, y2 + text_height + 6),

        # 上方
        (x1, y1 - 6),

        # 下方第二行
        (x1, y2 + 2 * text_height + 14),

        # 上方第二行
        (x1, y1 - text_height - 14),

        # 以 bbox center 對齊文字中心
        (
            center_x - text_width / 2,
            y2 + text_height + 6,
        ),
        (
            center_x - text_width / 2,
            y1 - 6,
        ),

        # 往 bbox 右側靠
        (
            x2 + 6,
            y2 + text_height + 6,
        ),
        (
            x2 + 6,
            y1 - 6,
        ),

        # 往 bbox 左側靠
        (
            x1 - text_width - 6,
            y2 + text_height + 6,
        ),
        (
            x1 - text_width - 6,
            y1 - 6,
        ),
    ]

    fallback = None

    for candidate_x, candidate_y in candidate_positions:
        text_x, text_y = clamp_text_position(
            candidate_x,
            candidate_y,
            text_width,
            text_height,
            baseline,
            image_width,
            image_height,
        )

        text_rect = (
            text_x,
            text_y - text_height,
            text_x + text_width,
            text_y + baseline,
        )

        if fallback is None:
            fallback = (text_x, text_y, text_rect)

        has_overlap = any(
            rectangles_overlap(text_rect, used_rect)
            for used_rect in used_text_rects
        )

        if not has_overlap:
            used_text_rects.append(text_rect)
            return text_x, text_y

    # 所有候選位置都衝突時，至少保證文字仍在畫面內。
    text_x, text_y, text_rect = fallback
    used_text_rects.append(text_rect)

    return text_x, text_y


def draw_gps_text(
    image,
    text,
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    used_text_rects,
    color,
):
    text_x, text_y = choose_gps_text_position(
        text=text,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_width=image_width,
        image_height=image_height,
        used_text_rects=used_text_rects,
    )

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        GPS_TEXT_FONT,
        GPS_TEXT_SCALE,
        color,
        GPS_TEXT_THICKNESS,
        cv2.LINE_AA,
    )


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

        # ====================================================
        # Annotator
        #
        # 不再使用 result.plot()，改成自己控制 bbox 樣式。
        # ====================================================

        annotator = Annotator(
            frame.copy(),
            line_width=BOX_LINE_WIDTH,
        )

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

        # 本幀已使用的 GPS 文字區域。
        # 只避免座標文字彼此重疊，不影響 bbox / HSV 變色。
        used_gps_text_rects = []

        for box_index, (detection, candidate) in enumerate(matched_detections):
            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

            target_x = detection["center_x"]
            target_y = detection["center_y"]

            # =================================================
            # HSV 紅色判定
            # =================================================

            if ENABLE_HSV_RED_CHECK:
                red_count, red_ratio, roi_bounds = count_red_pixels(
                    frame,
                    target_x,
                    target_y,
                )

                is_red_target = red_count > RED_PIXEL_THRESHOLD
            else:
                red_count = 0
                red_ratio = 0.0
                roi_bounds = None
                is_red_target = False

            # 紅色像素超過 threshold -> 紅框
            # 其餘 -> 藍框
            box_color = (
                RED_BOX_COLOR
                if is_red_target
                else BLUE_BOX_COLOR
            )

            # -------------------------------------------------
            # 不顯示 YOLO confidence。
            #
            # 未 confirmed：只顯示確認進度
            # confirmed：只顯示 car
            #
            # HSV 判斷與框變色邏輯完全保留。
            # -------------------------------------------------

            if not candidate["confirmed"]:
                box_label = (
                    f"confirm {candidate['count']}/{CONFIRM_FRAMES}"
                )
            else:
                box_label = "car"

            annotator.box_label(
                [x1, y1, x2, y2],
                label=box_label,
                color=box_color,
            )

            # 畫 bbox 中心點
            # cv2.circle(
            #     annotator.im,
            #     (int(target_x), int(target_y)),
            #     5,
            #     (0, 255, 255),
            #     -1,
            # )

            if PRINT_HSV_DEBUG:
                print(
                    f"[HSV] candidate_id={candidate['id']} "
                    f"red_count={red_count} "
                    f"red_ratio={red_ratio:.3f} "
                    f"is_red={is_red_target}"
                )

            # 尚未 confirmed：不 call GPS
            if not candidate["confirmed"]:
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

                draw_gps_text(
                    image=annotator.im,
                    text=gps_text,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    image_width=image_width,
                    image_height=image_height,
                    used_text_rects=used_gps_text_rects,
                    color=(238, 43, 57),
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
                draw_gps_text(
                    image=annotator.im,
                    text="GPS unavailable",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    image_width=image_width,
                    image_height=image_height,
                    used_text_rects=used_gps_text_rects,
                    color=(0, 0, 255),
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
        # 取得 Annotator 畫好的最終影像
        # ====================================================

        annotated_frame = annotator.result()

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
