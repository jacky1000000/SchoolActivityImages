import os
import sys
from huggingface_hub import snapshot_download

def main():
    # 強制指定一個絕對路徑，避免 Windows 亂塞檔案
    # 我們直接在當前目錄下建立一個名為 my_checkpoints 的資料夾
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "my_checkpoints")
    
    vicuna_target = os.path.join(target_dir, "vicuna-7b-v1.5")
    sd_target = os.path.join(target_dir, "stable-diffusion-v1-5")
    
    print(f"====== 🔍 開始強制下載流程 ======")
    print(f"🎯 預期目標下載絕對路徑:\n  1. {vicuna_target}\n  2. {sd_target}\n")

    try:
        print("⏳ 1/2 正在下載 Vicuna-7B-v1.5 結構與權重 (這需要一些時間，請觀察網路流量)...")
        snapshot_download(
            repo_id="lmsys/vicuna-7b-v1.5",
            local_dir=vicuna_target,
            local_dir_use_symlinks=False,  # 停用 Windows 捷徑符號連結，強制寫入實體檔案
            ignore_patterns=["*.safetensors"] # 僅下載 .bin，省一半時間
        )
        print("✅ Vicuna 下載成功！")
    except Exception as e:
        print(f"❌ Vicuna 下載失敗。錯誤原因: {e}", file=sys.stderr)

    try:
        print("\n⏳ 2/2 正在下載 Stable Diffusion v1-5 結構與權重...")
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir=sd_target,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.safetensors", "*.onnx", "*.fp16.*"]
        )
        print("✅ Stable Diffusion 下載成功！")
    except Exception as e:
        print(f"❌ Stable Diffusion 下載失敗。錯誤原因: {e}", file=sys.stderr)

    print("\n====== 🎉 流程結束 ======")
    if os.path.exists(vicuna_target):
        print(f"實際目錄確認: {vicuna_target} 內含檔案: {os.listdir(vicuna_target)}")

if __name__ == "__main__":
    main()
