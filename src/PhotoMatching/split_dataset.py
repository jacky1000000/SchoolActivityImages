import os
import random
import shutil

def split_campus_photos(
    source_dir="../../data/activity_photos",
    train_dir="../../data/train_photos",
    test_dir="../../data/test_photos",
    train_ratio=0.8,
    seed=42
):
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    all_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)])
    
    if not all_files:
        raise FileNotFoundError(f"❌ 找不到相片，請確認 '{source_dir}' 內有圖片檔案！")

    # 固定隨機種子打亂順序，確保實驗可重現
    random.seed(seed)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))

    print("=" * 60)
    print(f"✅ 資料集物理切分完成！(總計 {len(all_files)} 張)")
    print(f"  • 訓練集 (80%): {len(train_files)} 張 -> 已存至 {train_dir}")
    print(f"  • 測試集 (20%): {len(test_files)} 張 -> 已存至 {test_dir}")
    print("=" * 60)

if __name__ == "__main__":
    split_campus_photos()
