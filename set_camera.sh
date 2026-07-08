#!/bin/bash

# ==============================================================================
# 無人機 Webcam 參數設定腳本 (V4L2)
# ==============================================================================
# 說明：
# 本腳本用於無人機起飛前或影像串流中，一鍵關閉自動曝光、自動白平衡與動態影格，
# 並根據天氣載入預設的硬體參數，防止大白天拍矮草道路時過曝（白成一片）或產生殘影。
#
# 使用方式：
#   1. 載入強烈陽光模式 (大太陽/無雲)：  ./set_camera.sh sun
#   2. 載入中等光線模式 (多雲/微陰天)：  ./set_camera.sh cloud
#   3. 查看相機目前參數狀態：            ./set_camera.sh status
# ==============================================================================

# 設定相機節點，若你的相機不是 video0 請自行修改
DEVICE="/dev/video0"

# 檢查 v4l2-ctl 是否安裝
if ! command -v v4l2-ctl &> /dev/null; then
    echo "❌ 錯誤: 系統未安裝 v4l-utils。請先執行: sudo apt install v4l-utils"
    exit 1
fi

# 檢查相機裝置是否存在
if [ ! -e "$DEVICE" ]; then
    echo "❌ 錯誤: 找不到相機裝置 $DEVICE，請確認 Webcam 是否連接。"
    exit 1
fi

case "$1" in
    "sun")
        echo "☀️ 正在載入【萬里無雲大太陽 - 最強光模式】配置..."
        
        # 1. 關閉影響 FPS 流暢度的動態影格
        v4l2-ctl -d $DEVICE -c exposure_dynamic_framerate=0
        
        # 2. 關閉自動白平衡，固定色溫在 5200K (標準戶本陽光色溫)
        v4l2-ctl -d $DEVICE -c white_balance_automatic=0
        v4l2-ctl -d $DEVICE -c white_balance_temperature=5200
        
        # 3. 關閉自動曝光 (1 代表手動模式)，並將曝光時間壓低防止過曝
        v4l2-ctl -d $DEVICE -c auto_exposure=1
        v4l2-ctl -d $DEVICE -c exposure_time_absolute=45
        
        # 4. 調低增益以減少強光下的噪點
        v4l2-ctl -d $DEVICE -c gain=32
        
        echo "✅ 最強光模式設定完成！(曝光已壓低，適合大太陽空拍)"
        ;;
        
    "cloud")
        echo "⛅ 正在載入【一般多雲晴天/微陰 - 中等光模式】配置..."
        
        # 1. 關閉影響 FPS 流暢度的動態影格
        v4l2-ctl -d $DEVICE -c exposure_dynamic_framerate=0
        
        # 2. 關閉自動白平衡，固定色溫在 5500K
        v4l2-ctl -d $DEVICE -c white_balance_automatic=0
        v4l2-ctl -d $DEVICE -c white_balance_temperature=5500
        
        # 3. 關閉自動曝光，給予適中的進光時間
        v4l2-ctl -d $DEVICE -c auto_exposure=1
        v4l2-ctl -d $DEVICE -c exposure_time_absolute=85
        
        # 4. 使用預設增益
        v4l2-ctl -d $DEVICE -c gain=64
        
        echo "✅ 中等光模式設定完成！(參數較均衡，適合多雲環境)"
        ;;
        
    "status")
        echo "📊 當前相機控制參數狀態："
        v4l2-ctl -d $DEVICE --list-ctrls
        ;;
        
    *)
        echo "⚠️  請輸入正確的參數！"
        echo "用法:"
        echo "  ./set_camera.sh sun    -> 載入大太陽最強光模式"
        echo "  ./set_camera.sh cloud  -> 載入多雲中等光模式"
        echo "  ./set_camera.sh status -> 查看目前相機參數"
        exit 1
        ;;
esac