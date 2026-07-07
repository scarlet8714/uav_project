from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import serial
import pynmea2


# ============================================================
# 你最常需要手動修改的固定參數
# ============================================================

GPS_PORT = "/dev/ttyUSB0"     # GPS 裝置位置
GPS_BAUDRATE = 9600           # SAM-M8Q 常見 NMEA baudrate；若你的裝置不同可改
GPS_TIMEOUT_SEC = 1.0         # serial readline timeout
GPS_STALE_SEC = 3.0           # 超過幾秒沒更新就視為資料過期

# 相機固定參數：之後測到真實值再修改
DEFAULT_HFOV_DEG = 70.0       # 水平 FOV，必須自行確認或量測
DEFAULT_VFOV_DEG = 43.0       # 垂直 FOV，必須自行確認或量測

# 相機畫面上方相對於 COG 的偏移角
# 0 = 畫面上方視為飛行方向
# 90 = 畫面上方相對飛行方向順時針偏 90 度
DEFAULT_CAMERA_YAW_OFFSET_DEG = 0.0

# 低速時 COG 可能不穩定。
# 若速度低於此值，會保留上一個有效 course。
MIN_SPEED_FOR_COG_MPS = 1.0


@dataclass
class GPSState:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    course_deg: Optional[float] = None
    speed_mps: Optional[float] = None

    fix_valid: bool = False
    connected: bool = False

    last_position_update: Optional[float] = None
    last_course_update: Optional[float] = None

    error: Optional[str] = None


class GPSReader:
    """
    在同一個 Python 程式內，用背景 thread 持續監聽 GPS。

    主程式不需要一直自己讀 /dev/ttyUSB0。
    需要 GPS 時只要呼叫：
        gps.get_latest()

    如果 GPS 故障或沒有資料：
        不丟出致命錯誤，
        GPSState.fix_valid 會是 False，
        error 會保留原因。
    """

    def __init__(
        self,
        port: str = GPS_PORT,
        baudrate: int = GPS_BAUDRATE,
        timeout_sec: float = GPS_TIMEOUT_SEC,
        stale_sec: float = GPS_STALE_SEC,
        min_speed_for_cog_mps: float = MIN_SPEED_FOR_COG_MPS,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout_sec = timeout_sec
        self.stale_sec = stale_sec
        self.min_speed_for_cog_mps = min_speed_for_cog_mps

        self._state = GPSState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # 保留最後一次可信 course，低速或暫時缺資料時使用
        self._last_valid_course: Optional[float] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="GPSReader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> GPSState:
        """
        取得目前最新 GPS snapshot。

        不會直接回傳內部共享物件，避免其他 thread 同時修改。
        """
        with self._lock:
            state = GPSState(**self._state.__dict__)

        now = time.monotonic()

        # 若位置太久沒有更新，視為 stale
        if (
            state.last_position_update is None
            or now - state.last_position_update > self.stale_sec
        ):
            state.fix_valid = False

            if state.error is None:
                state.error = "GPS position is missing or stale."

        return state

    def _update_state(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_serial_session()

            except serial.SerialException as exc:
                self._update_state(
                    connected=False,
                    fix_valid=False,
                    error=f"Serial error: {exc}",
                )

                # 裝置暫時拔掉時，不讓程式死亡；稍後重新嘗試
                time.sleep(2.0)

            except Exception as exc:
                self._update_state(
                    connected=False,
                    fix_valid=False,
                    error=f"Unexpected GPS error: {exc}",
                )
                time.sleep(2.0)

    def _run_serial_session(self) -> None:
        with serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout_sec,
        ) as ser:
            self._update_state(
                connected=True,
                error=None,
            )

            while not self._stop_event.is_set():
                raw = ser.readline()

                if not raw:
                    continue

                line = raw.decode("ascii", errors="ignore").strip()

                if not line.startswith("$"):
                    continue

                try:
                    msg = pynmea2.parse(line)
                except pynmea2.ParseError:
                    # 單筆資料壞掉直接跳過，不影響後續
                    continue

                self._handle_nmea(msg)

    def _handle_nmea(self, msg) -> None:
        now = time.monotonic()

        # ----------------------------------------------------
        # RMC：
        # 主要來源：lat, lon, speed, course
        # ----------------------------------------------------
        if isinstance(msg, pynmea2.types.talker.RMC):
            if msg.status != "A":
                self._update_state(
                    fix_valid=False,
                    error="RMC received but fix is invalid.",
                )
                return

            latitude = float(msg.latitude)
            longitude = float(msg.longitude)

            speed_knots = msg.spd_over_grnd
            speed_mps = (
                float(speed_knots) * 0.514444
                if speed_knots is not None
                else None
            )

            course = msg.true_course

            if (
                course is not None
                and speed_mps is not None
                and speed_mps >= self.min_speed_for_cog_mps
            ):
                self._last_valid_course = float(course)

            self._update_state(
                latitude=latitude,
                longitude=longitude,
                speed_mps=speed_mps,
                course_deg=self._last_valid_course,
                fix_valid=True,
                connected=True,
                last_position_update=now,
                last_course_update=(
                    now if self._last_valid_course is not None else None
                ),
                error=None,
            )

        # ----------------------------------------------------
        # VTG：
        # 補充 speed 與 course
        # ----------------------------------------------------
        elif isinstance(msg, pynmea2.types.talker.VTG):
            speed_kmph = msg.spd_over_grnd_kmph
            speed_mps = (
                float(speed_kmph) / 3.6
                if speed_kmph is not None
                else None
            )

            course = msg.true_track

            if (
                course is not None
                and speed_mps is not None
                and speed_mps >= self.min_speed_for_cog_mps
            ):
                self._last_valid_course = float(course)

                self._update_state(
                    course_deg=self._last_valid_course,
                    speed_mps=speed_mps,
                    last_course_update=now,
                    error=None,
                )


# ============================================================
# Pixel -> Ground offset
# ============================================================

def pixel_to_ground_offset(
    target_x: float,
    target_y: float,
    image_width: int,
    image_height: int,
    altitude_agl_m: float,
    hfov_deg: float = DEFAULT_HFOV_DEG,
    vfov_deg: float = DEFAULT_VFOV_DEG,
) -> tuple[float, float]:
    """
    將 YOLO bbox center 的 pixel 座標轉換成影像座標系下的地面位移。

    輸入：
        target_x
            [動態 / YOLO 提供]
            bbox center x

        target_y
            [動態 / YOLO 提供]
            bbox center y

        image_width
            [動態 / frame shape]
            原始影像寬度，不是 YOLO imgsz

        image_height
            [動態 / frame shape]
            原始影像高度，不是 YOLO imgsz

        altitude_agl_m
            [目前建議手動輸入]
            相機距離地面的高度，單位 m
            你的情況約 60 ~ 80

        hfov_deg
            [手動設定]
            相機水平 FOV

        vfov_deg
            [手動設定]
            相機垂直 FOV

    回傳：
        image_right_m
            + = 畫面右方

        image_forward_m
            + = 畫面上方

    假設：
        1. 相機垂直向下
        2. 地面近似平坦
        3. 暫不補償 roll / pitch
        4. 暫不做鏡頭畸變校正
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")

    if altitude_agl_m <= 0:
        raise ValueError("altitude_agl_m must be positive.")

    if not (0.0 < hfov_deg < 180.0):
        raise ValueError("hfov_deg must be between 0 and 180.")

    if not (0.0 < vfov_deg < 180.0):
        raise ValueError("vfov_deg must be between 0 and 180.")

    # 正規化到 [-1, +1]
    nx = (target_x - image_width / 2.0) / (image_width / 2.0)
    ny = (target_y - image_height / 2.0) / (image_height / 2.0)

    # 避免 bbox 座標意外跑到影像外太遠
    nx = max(-1.0, min(1.0, nx))
    ny = max(-1.0, min(1.0, ny))

    # 對應視線角
    horizontal_angle_rad = math.radians(nx * hfov_deg / 2.0)
    vertical_angle_rad = math.radians(ny * vfov_deg / 2.0)

    # 高度投影到地面
    image_right_m = altitude_agl_m * math.tan(horizontal_angle_rad)

    # 影像 y 向下增加，所以畫面上方取正值
    image_forward_m = -altitude_agl_m * math.tan(vertical_angle_rad)

    return image_right_m, image_forward_m


# ============================================================
# Ground offset -> North/East -> GPS
# ============================================================

def rotate_image_offset_to_ne(
    image_right_m: float,
    image_forward_m: float,
    course_deg: float,
    camera_yaw_offset_deg: float = DEFAULT_CAMERA_YAW_OFFSET_DEG,
) -> tuple[float, float]:
    """
    將影像座標系地面位移轉成 North/East 位移。

    course_deg:
        [GPS 自動取得，或手動傳入]
        0 = North
        90 = East
        180 = South
        270 = West

    camera_yaw_offset_deg:
        [手動設定]
        相機畫面上方相對於 COG 的固定偏移。
    """

    heading_deg = (course_deg + camera_yaw_offset_deg) % 360.0
    heading_rad = math.radians(heading_deg)

    north_m = (
        image_forward_m * math.cos(heading_rad)
        - image_right_m * math.sin(heading_rad)
    )

    east_m = (
        image_forward_m * math.sin(heading_rad)
        + image_right_m * math.cos(heading_rad)
    )

    return north_m, east_m


def offset_to_gps(
    drone_lat: float,
    drone_lon: float,
    north_m: float,
    east_m: float,
) -> tuple[float, float]:
    """
    將局部 North/East 位移近似轉換成 GPS 經緯度。

    對你 60~80 m 高度、局部百米級偏移的情境，
    使用局部小距離近似即可作為 V1。
    """

    earth_radius_m = 6_378_137.0

    d_lat = north_m / earth_radius_m
    d_lon = east_m / (
        earth_radius_m * math.cos(math.radians(drone_lat))
    )

    target_lat = drone_lat + math.degrees(d_lat)
    target_lon = drone_lon + math.degrees(d_lon)

    return target_lat, target_lon


# ============================================================
# 你最後主要呼叫的總函式
# ============================================================

def estimate_target_gps(
    *,
    # ---------- YOLO 自動提供 ----------
    target_x: float,
    target_y: float,

    # ---------- Camera frame 自動提供 ----------
    image_width: int,
    image_height: int,

    # ---------- 目前建議手動傳入 ----------
    altitude_agl_m: float,

    # ---------- GPS 自動提供 / 也可手動傳入 ----------
    drone_lat: float,
    drone_lon: float,
    course_deg: float,

    # ---------- 相機固定設定，可手動修改 ----------
    hfov_deg: float = DEFAULT_HFOV_DEG,
    vfov_deg: float = DEFAULT_VFOV_DEG,
    camera_yaw_offset_deg: float = DEFAULT_CAMERA_YAW_OFFSET_DEG,
) -> dict:
    """
    YOLO bbox center -> Target GPS

    回傳 dict，方便之後顯示與除錯。
    """

    image_right_m, image_forward_m = pixel_to_ground_offset(
        target_x=target_x,
        target_y=target_y,
        image_width=image_width,
        image_height=image_height,
        altitude_agl_m=altitude_agl_m,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
    )

    north_m, east_m = rotate_image_offset_to_ne(
        image_right_m=image_right_m,
        image_forward_m=image_forward_m,
        course_deg=course_deg,
        camera_yaw_offset_deg=camera_yaw_offset_deg,
    )

    target_lat, target_lon = offset_to_gps(
        drone_lat=drone_lat,
        drone_lon=drone_lon,
        north_m=north_m,
        east_m=east_m,
    )

    return {
        "target_lat": target_lat,
        "target_lon": target_lon,

        # 以下保留方便測試和除錯
        "image_right_m": image_right_m,
        "image_forward_m": image_forward_m,
        "north_m": north_m,
        "east_m": east_m,
    }


def estimate_target_gps_from_reader(
    gps_reader: GPSReader,
    *,
    # YOLO 自動提供
    target_x: float,
    target_y: float,

    # frame 自動提供
    image_width: int,
    image_height: int,

    # 目前手動提供
    altitude_agl_m: float,

    # 相機固定參數
    hfov_deg: float = DEFAULT_HFOV_DEG,
    vfov_deg: float = DEFAULT_VFOV_DEG,
    camera_yaw_offset_deg: float = DEFAULT_CAMERA_YAW_OFFSET_DEG,

    # GPS 故障時的「保留懸念」：
    # 可傳手動替代值；不傳則 GPS 無效時回傳 None
    fallback_lat: Optional[float] = None,
    fallback_lon: Optional[float] = None,
    fallback_course_deg: Optional[float] = None,
) -> Optional[dict]:
    """
    正式整合時最方便的入口。

    GPS 正常：
        使用 GPSReader 的最新資料。

    GPS 無資料：
        如果有 fallback 值，就使用 fallback。
        如果沒有，就回傳 None，不讓 YOLO 主流程崩潰。
    """

    gps = gps_reader.get_latest()

    if (
        gps.fix_valid
        and gps.latitude is not None
        and gps.longitude is not None
        and gps.course_deg is not None
    ):
        drone_lat = gps.latitude
        drone_lon = gps.longitude
        course_deg = gps.course_deg
        source = "gps"

    elif (
        fallback_lat is not None
        and fallback_lon is not None
        and fallback_course_deg is not None
    ):
        drone_lat = fallback_lat
        drone_lon = fallback_lon
        course_deg = fallback_course_deg
        source = "fallback"

    else:
        return None

    result = estimate_target_gps(
        target_x=target_x,
        target_y=target_y,
        image_width=image_width,
        image_height=image_height,
        altitude_agl_m=altitude_agl_m,
        drone_lat=drone_lat,
        drone_lon=drone_lon,
        course_deg=course_deg,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        camera_yaw_offset_deg=camera_yaw_offset_deg,
    )

    result["source"] = source
    result["gps_error"] = gps.error

    return result


if __name__ == "__main__":
    # ========================================================
    # 不接硬體也能先測試換算函式
    # ========================================================

    test_result = estimate_target_gps(
        # YOLO bbox center
        target_x=1300.0,
        target_y=600.0,

        # 原始 frame size
        image_width=1920,
        image_height=1080,

        # 你目前手動輸入的高度
        altitude_agl_m=70.0,

        # 假 GPS
        drone_lat=25.0330,
        drone_lon=121.5654,

        # 假 COG
        course_deg=45.0,

        # 相機參數，之後換成實測值
        hfov_deg=DEFAULT_HFOV_DEG,
        vfov_deg=DEFAULT_VFOV_DEG,
        camera_yaw_offset_deg=DEFAULT_CAMERA_YAW_OFFSET_DEG,
    )

    print(test_result)
