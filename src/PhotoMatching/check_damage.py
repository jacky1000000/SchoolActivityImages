import os
import shutil
from PIL import Image, ImageFile

# 強制嚴格模式：不允許載入截斷或不完整的圖片
ImageFile.LOAD_TRUNCATED_IMAGES = False

def clean_corrupted_images(photo_dir, quarantine_dir="../../data/corrupted_activity_photos", delete_permanently=False):
    """
    掃描指定資料夾內的所有圖片，將無法讀取、0 KB 或像素損壞的檔案移至隔離區（或直接刪除）。
    """
    if not os.path.exists(photo_dir):
        print(f"❌ 找不到相片資料夾: {photo_dir}")
        return

    if not delete_permanently:
        os.makedirs(quarantine_dir, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_files = [f for f in os.listdir(photo_dir) if f.lower().endswith(valid_exts)]
    
    print(f"🔍 開始掃描 '{photo_dir}'，共發現 {len(all_files)} 個圖片檔案...\n")

    valid_count = 0
    corrupted_files = []

    for filename in all_files:
        print(f"正在檢查圖片: {filename} ({all_files.index(filename)+1}/{len(all_files)})")
        filepath = os.path.join(photo_dir, filename)
        is_corrupted = False
        error_reason = ""

        # 1. 檢查是否為 0 KB 空檔案
        if os.path.getsize(filepath) == 0:
            is_corrupted = True
            error_reason = "檔案大小為 0 KB"
        else:
            try:
                # 2. 驗證圖片檔案結構與標頭完整性
                with Image.open(filepath) as img:
                    img.verify()
                
                # 3. 重新開啟並強制解碼所有像素資料（verify 之後必須重新 open 才能 load）
                with Image.open(filepath) as img:
                    img.load()
                    # 測試轉換為 RGB（確保 ImageBind 預處理時不會因為色彩模式異常報錯）
                    img.convert("RGB")
                    
            except Exception as e:
                is_corrupted = True
                error_reason = str(e)

        if is_corrupted:
            corrupted_files.append((filename, error_reason))
            if delete_permanently:
                os.remove(filepath)
                print(f"🗑️ [已永久刪除] {filename} | 原因: {error_reason}")
            else:
                dest_path = os.path.join(quarantine_dir, filename)
                shutil.move(filepath, dest_path)
                print(f"⚠️ [已移至隔離區] {filename} | 原因: {error_reason}")
        else:
            valid_count += 1

    print("\n" + "=" * 60)
    print(f"✅ 掃描完成！正常可用圖片: {valid_count} 張 | 損壞異常圖片: {len(corrupted_files)} 張")
    if corrupted_files and not delete_permanently:
        print(f"📂 所有損壞圖片已安全移至: {quarantine_dir}")
    print("=" * 60)

if __name__ == "__main__":
    # 請將路徑指向您的 1,330 張校園相片資料夾
    target_photo_folder = "../../data/activity_photos"
    
    # 預設為安全模式 (delete_permanently=False)，先將壞圖移至 corrupted_photos 資料夾
    # 若確認要直接刪除，可改為 delete_permanently=True
    clean_corrupted_images(target_photo_folder, delete_permanently=False)
