#!/bin/bash

# ==============================================================================
#  無人機 Webcam 參數微調腳本 - 變數集中管理版 (v3)
# ==============================================================================
# 💡 提示：大太陽下如果主角道路太亮，請直接修改下方【大太陽模式】的數值。
# ==============================================================================

DEVICE="/dev/video0"

case "$1" in
    "sun5")
        echo "☀️ 正在載入【自訂大太陽模式】..."
        
        # ─── 這裡就是你可以自由修改的變數區 ───
        EXPOSURE=5         # 曝光時間 (0 或 1)
        GAIN=0             # 增益 (大太陽建議歸 0)
        BRIGHTNESS=128     # 亮度 (預設 128，太亮就往下降到 100 或 80)
        CONTRAST=60        # 對比度 (拉高到 50 左右能讓矮草與道路界線更明顯)
        SATURATION=60      # 飽和度
        WB_TEMP=5400       # 白平衡色溫
        # ──────────────────────────────────────
        ;;
        
    "sun4")
        echo "⛅ 正在載入【自訂多雲模式】..."
        
        # ─── 多雲環境的變數區 ───
        EXPOSURE=4
        GAIN=0
        BRIGHTNESS=128
        CONTRAST=60
        SATURATION=60
        WB_TEMP=5400
        ;;

    "sun3")
        echo "⛅ 正在載入【自訂多雲模式】..."
        
        # ─── 多雲環境的變數區 ───
        EXPOSURE=3
        GAIN=0
        BRIGHTNESS=128
        CONTRAST=60
        SATURATION=60
        WB_TEMP=5400
        ;;
    
    "sun2")
        echo "⛅ 正在載入【自訂多雲模式】..."
        
        # ─── 多雲環境的變數區 ───
        EXPOSURE=2
        GAIN=0
        BRIGHTNESS=128
        CONTRAST=60
        SATURATION=60
        WB_TEMP=5400
        ;;
    
    "sun1")
        echo "⛅ 正在載入【自訂多雲模式】..."
        
        # ─── 多雲環境的變數區 ───
        EXPOSURE=1
        GAIN=0
        BRIGHTNESS=128
        CONTRAST=60
        SATURATION=60
        WB_TEMP=5400
        ;;
        
    "status")
        v4l2-ctl -d $DEVICE --list-ctrls
        exit 0
        ;;
        
    *)
        echo "用法: ./set_camera_v3.sh [sun | cloud | status]"
        exit 1
        ;;
esac

# ==============================================================================
#  核心執行區：這裡會自動讀取上面帶有 $ 的變數並寫入相機
# ==============================================================================

# 關閉所有自動化大腦
v4l2-ctl -d $DEVICE -c exposure_dynamic_framerate=0
v4l2-ctl -d $DEVICE -c white_balance_automatic=0
v4l2-ctl -d $DEVICE -c auto_exposure=1
v4l2-ctl -d $DEVICE -c backlight_compensation=0

# 套用色溫變數
v4l2-ctl -d $DEVICE -c white_balance_temperature=$WB_TEMP

# 一鍵套用畫質變數（注意這裡的變數前面都有加 $ 喔！）
v4l2-ctl -d $DEVICE -c brightness=$BRIGHTNESS,contrast=$CONTRAST,saturation=$SATURATION,exposure_time_absolute=$EXPOSURE,gain=$GAIN

echo "----------------------------------------"
echo " ✅ 成功套用變數值："
echo "    曝光 (Exposure) : $EXPOSURE"
echo "    增益 (Gain)     : $GAIN"
echo "    亮度 (Brightness) : $BRIGHTNESS"
echo "    對比 (Contrast) : $CONTRAST"
echo "----------------------------------------"