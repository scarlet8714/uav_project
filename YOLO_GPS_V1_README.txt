# YOLO + GPS V1 使用方式

檔案：

- `gps_geolocation.py`
  - GPS 背景監聽
  - NMEA RMC / VTG 解析
  - pixel -> ground offset
  - ground offset -> GPS

- `yolo_gps_main.py`
  - YOLO 主程式
  - 相機讀取
  - bbox center 取得
  - 呼叫 GPS 定位函式
  - 顯示與列印目標 GPS

## 啟動方式

只需要啟動：

python3 yolo_gps_main.py

不需要另外啟動 gps_geolocation.py。

## 必須確認或修改的值

在 `yolo_gps_main.py`：

MODEL_PATH                         V
CAMERA_SOURCE                      V
GPS_PORT
GPS_BAUDRATE
ALTITUDE_AGL_M
HFOV_DEG
VFOV_DEG
CAMERA_YAW_OFFSET_DEG
CONF_THRESHOLD
IMGSZ

## GPS 行為

GPSReader 在主程式內以 background thread 持續監聽。

GPS 正常：
- 定位函式使用最新有效 GPS + COG。

GPS 無資料或無效：
- `estimate_target_gps_from_reader()` 回傳 `None`。
- YOLO 仍繼續推論。
- 畫面顯示 `GPS unavailable`。
- 不使用先前舊 GPS 資料。
