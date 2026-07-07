import time

import serial
import pynmea2


# ============================================================
# 【需要你確認的參數】
# ============================================================

GPS_PORT = "/dev/ttyUSB0"

# 若你 cat /dev/ttyUSB0 可看到正常 NMEA，
# 這裡先沿用目前設定。
GPS_BAUDRATE = 9600

# readline timeout，避免一直卡住
GPS_TIMEOUT_SEC = 1.0

# 和目前正式程式相同：
# 只有速度 >= 1.0 m/s 時，才接受 course 當有效 COG。
MIN_SPEED_FOR_COG_MPS = 1.0


# ============================================================
# Debug 狀態
# ============================================================

last_valid_course = None

rmc_count = 0
vtg_count = 0
gga_count = 0
parse_error_count = 0
other_count = 0


def safe_float(value):
    """
    pynmea2 某些欄位可能是 None 或空字串。
    這裡安全轉 float，失敗就回傳 None。
    """
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


print("=" * 70)
print("GPS minimal debug test")
print("=" * 70)
print(f"Port: {GPS_PORT}")
print(f"Baudrate: {GPS_BAUDRATE}")
print(f"MIN_SPEED_FOR_COG_MPS: {MIN_SPEED_FOR_COG_MPS}")
print()
print("注意：測試時請先停止 cat /dev/ttyUSB0")
print("按 Ctrl+C 結束")
print("=" * 70)


try:
    with serial.Serial(
        port=GPS_PORT,
        baudrate=GPS_BAUDRATE,
        timeout=GPS_TIMEOUT_SEC,
    ) as ser:

        print("[OK] Serial port opened.")
        print()

        while True:
            raw = ser.readline()

            # ------------------------------------------------
            # 沒讀到資料
            # ------------------------------------------------
            if not raw:
                print("[TIMEOUT] No serial data received within timeout.")
                continue

            # ------------------------------------------------
            # Decode 原始資料
            # ------------------------------------------------
            line = raw.decode("ascii", errors="ignore").strip()

            if not line:
                continue

            print()
            print("-" * 70)
            print("[RAW]")
            print(line)

            # 只處理 NMEA
            if not line.startswith("$"):
                print("[SKIP] Not an NMEA sentence.")
                continue

            # ------------------------------------------------
            # Parse NMEA
            # ------------------------------------------------
            try:
                msg = pynmea2.parse(line)

            except pynmea2.ParseError as exc:
                parse_error_count += 1

                print("[PARSE ERROR]")
                print(exc)
                print(f"parse_error_count = {parse_error_count}")
                continue

            print(f"[TYPE] {type(msg).__name__}")
            print(f"[TALKER] {getattr(msg, 'talker', None)}")
            print(f"[SENTENCE] {getattr(msg, 'sentence_type', None)}")

            # =================================================
            # RMC
            # =================================================
            if isinstance(msg, pynmea2.types.talker.RMC):
                rmc_count += 1

                status = getattr(msg, "status", None)
                latitude = safe_float(msg.latitude)
                longitude = safe_float(msg.longitude)

                speed_knots = safe_float(
                    getattr(msg, "spd_over_grnd", None)
                )

                speed_mps = (
                    speed_knots * 0.514444
                    if speed_knots is not None
                    else None
                )

                raw_course = safe_float(
                    getattr(msg, "true_course", None)
                )

                fix_valid = status == "A"

                course_accepted = False

                if (
                    raw_course is not None
                    and speed_mps is not None
                    and speed_mps >= MIN_SPEED_FOR_COG_MPS
                ):
                    last_valid_course = raw_course
                    course_accepted = True

                print("[RMC DEBUG]")
                print(f"  status            = {status}")
                print(f"  fix_valid         = {fix_valid}")
                print(f"  latitude          = {latitude}")
                print(f"  longitude         = {longitude}")
                print(f"  speed_knots       = {speed_knots}")
                print(f"  speed_mps         = {speed_mps}")
                print(f"  raw_course        = {raw_course}")
                print(f"  course_accepted   = {course_accepted}")
                print(f"  last_valid_course = {last_valid_course}")
                print(f"  rmc_count         = {rmc_count}")

                # 和正式程式目前條件相同的可用性判斷
                ready_for_geolocation = (
                    fix_valid
                    and latitude is not None
                    and longitude is not None
                    and last_valid_course is not None
                )

                print(
                    f"  READY_FOR_GEOLOCATION = "
                    f"{ready_for_geolocation}"
                )

                if (
                    fix_valid
                    and latitude is not None
                    and longitude is not None
                ):
                    print()
                    print(
                        f"[GPS POSITION] "
                        f"lat={latitude:.8f}, "
                        f"lon={longitude:.8f}"
                    )

                if not ready_for_geolocation:
                    print()
                    print("[WHY NOT READY?]")

                    if not fix_valid:
                        print("  - RMC status is not A (valid fix).")

                    if latitude is None or longitude is None:
                        print("  - latitude / longitude missing.")

                    if last_valid_course is None:
                        print(
                            "  - no accepted COG yet. "
                            "This is expected when static or too slow."
                        )

            # =================================================
            # VTG
            # =================================================
            elif isinstance(msg, pynmea2.types.talker.VTG):
                vtg_count += 1

                speed_kmph = safe_float(
                    getattr(msg, "spd_over_grnd_kmph", None)
                )

                speed_mps = (
                    speed_kmph / 3.6
                    if speed_kmph is not None
                    else None
                )

                raw_course = safe_float(
                    getattr(msg, "true_track", None)
                )

                course_accepted = False

                if (
                    raw_course is not None
                    and speed_mps is not None
                    and speed_mps >= MIN_SPEED_FOR_COG_MPS
                ):
                    last_valid_course = raw_course
                    course_accepted = True

                print("[VTG DEBUG]")
                print(f"  speed_kmph        = {speed_kmph}")
                print(f"  speed_mps         = {speed_mps}")
                print(f"  raw_course        = {raw_course}")
                print(f"  course_accepted   = {course_accepted}")
                print(f"  last_valid_course = {last_valid_course}")
                print(f"  vtg_count         = {vtg_count}")

            # =================================================
            # GGA
            # =================================================
            elif isinstance(msg, pynmea2.types.talker.GGA):
                gga_count += 1

                latitude = safe_float(msg.latitude)
                longitude = safe_float(msg.longitude)

                gps_qual = getattr(msg, "gps_qual", None)
                num_sats = getattr(msg, "num_sats", None)
                hdop = getattr(msg, "horizontal_dil", None)
                altitude = getattr(msg, "altitude", None)

                print("[GGA DEBUG]")
                print(f"  latitude      = {latitude}")
                print(f"  longitude     = {longitude}")
                print(f"  gps_quality   = {gps_qual}")
                print(f"  satellites    = {num_sats}")
                print(f"  hdop          = {hdop}")
                print(f"  altitude      = {altitude}")
                print(f"  gga_count     = {gga_count}")

            # =================================================
            # 其他 NMEA
            # =================================================
            else:
                other_count += 1

                print("[OTHER NMEA]")
                print(f"  other_count = {other_count}")

except KeyboardInterrupt:
    print()
    print("=" * 70)
    print("Test stopped by user.")
    print("=" * 70)

except serial.SerialException as exc:
    print()
    print("[SERIAL ERROR]")
    print(exc)

except Exception as exc:
    print()
    print("[UNEXPECTED ERROR]")
    print(type(exc).__name__, exc)
