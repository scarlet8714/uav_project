#!/usr/bin/env python3
import subprocess
import time
import sys

# 📝 設定已知網路的優先順序（由上往下，越上面越優先）
# 格式：{"Wi-Fi 名稱 (SSID)": "連線密碼"}
KNOWN_NETWORKS = {
    "ASUS_C8": "history_2254",       # 🌟 第一優先：實驗室/地面站分享器
    "李鯊魚": "10101010",              # 📱 第二優先：手機熱點（出外試飛備用）
    "lab508_WiFi": "實驗室密碼"        # 🏫 第三優先：校園固定網路
}

def get_available_ssids():
    """ 叫系統重新掃描並抓取附近的 Wi-Fi 名稱 """
    try:
        # 強制 Linux 網卡重新掃描周圍訊號
        subprocess.run(["sudo", "nmcli", "dev", "wifi", "rescan"], capture_output=True, text=True)
        
        # 抓取掃描到的 SSID 列表
        result = subprocess.run(
            ["nmcli", "-t", "-f", "SSID", "dev", "wifi"], 
            capture_output=True, text=True, check=True
        )
        # 過濾掉隱藏或空白的 Wi-Fi 名稱並去重複
        ssids = set([line.strip() for line in result.stdout.split("\n") if line.strip()])
        return ssids
    except subprocess.CalledProcessError as e:
        print(f"❌ 掃描失敗，錯誤訊息: {e}", file=sys.stderr)
        return set()

def connect_to_network(ssid, password):
    """ 執行 nmcli 連線指令 """
    print(f"📡 偵測到已知網路，正在嘗試無線連線: 【{ssid}】...")
    cmd = ["sudo", "nmcli", "dev", "wifi", "connect", ssid, "password", password]
    
    # 設定 15 秒連線逾時，避免死卡
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if "successfully activated" in result.stdout:
            print(f"✅ 【{ssid}】 連線成功！")
            return True
        else:
            print(f"❌ 【{ssid}】 連線失敗。")
            return False
    except subprocess.TimeoutExpired:
        print(f"⚠️ 連線 【{ssid}】 逾時（超過 15 秒）。")
        return False

def main():
    print(f"\n🚀 [{time.strftime('%Y-%m-%d %H:%M:%S')}] 啟動無人機伴隨電腦網路對接腳本...")
    
    # 1. 抓取附近所有看得到的 Wi-Fi
    print("🔍 正在掃描周圍無線網路訊號...")
    detected_ssids = get_available_ssids()
    print(f"📊 附近偵測到的所有 Wi-Fi: {list(detected_ssids)}")
    
    # 2. 嚴格按照 KNOWN_NETWORKS 的順序進行比對與連線
    for ssid, password in KNOWN_NETWORKS.items():
        if ssid in detected_ssids:
            print(f"💡 發現目標網路：{ssid}（符合優先清單）")
            # 嘗試連線，成功就直接結束腳本，不繼續往下連
            if connect_to_network(ssid, password):
                print("🏁 網路已確立，腳本安全退出。")
                return
            print("⏳ 嘗試下一組備用網路...")
            
    print("⚠️ 警告：附近沒有發現任何已知的 Wi-Fi 訊號，進入離線模式。")

if __name__ == "__main__":
    main()