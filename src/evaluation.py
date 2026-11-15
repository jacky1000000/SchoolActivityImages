import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance

def load_image_paths(folder_path):
    """取得資料夾內所有圖片檔案路徑（依檔名排序）"""
    if not os.path.exists(folder_path):
        return []
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    # list must be sorted by creation time in ascending order
    return [
        os.path.join(folder_path, fname)
        for fname in sorted(os.listdir(folder_path), key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
        if fname.lower().endswith(valid_exts)
    ]

def evaluate_folder(gen_dir, real_dir, prompt_list, device="cuda"):
    """針對指定生成圖片資料夾，計算平均 CLIP Score 與 FID"""
    gen_paths = load_image_paths(gen_dir)
    if not gen_paths:
        print(f"❌ 找不到生成圖片：請確認 '{gen_dir}' 資料夾存在且內含圖片。")
        return None, None

    # print(f"\n🔍 正在評估資料夾: {gen_dir} (共 {len(gen_paths)} 張圖片)...")
    # print('contents inside gen_path:', gen_paths)

    # ==========================================
    # 1. 計算 Text-to-Image Faithfulness (CLIP Score)
    # ==========================================
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.uint8)
    ])

    # # 若資料夾內有先前儲存的 generation_log.json，優先直接讀取對應的提示詞
    # log_path = os.path.join(gen_dir, "generation_log.json")
    # log_prompts = None
    # if os.path.exists(log_path):
    #     with open(log_path, "r", encoding="utf-8") as f:
    #         logs = json.load(f)
    #         if len(logs) == len(gen_paths):
    #             log_prompts = [item.get("expanded_description", item.get("original_prompt", "")) for item in logs]

    clip_scores = []
    # 計算每組提示詞對應幾張圖（例如 50 張圖對應 10 個提示詞，則每 5 張換一題）
    imgs_per_prompt = max(1, len(gen_paths) // len(prompt_list))
    # print(f"每個提示詞對應的圖片數量: {imgs_per_prompt}")

    for idx, img_path in enumerate(gen_paths):
        print(f"正在處理圖片 {idx+1}/{len(gen_paths)}: {img_path}")
        prompt_idx = min(idx // imgs_per_prompt, len(prompt_list) - 1)
        text_prompt = prompt_list[prompt_idx]
        print(f"使用的提示詞: {text_prompt}")

        img = Image.open(img_path).convert("RGB")
        tensor = clip_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            score = clip_metric(tensor, text_prompt).item()
        clip_scores.append(score)

    # print(f"CLIP Scores: {clip_scores}")
    # print('length of CLIP Scores:', len(clip_scores))
    avg_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0
    print('Average CLIP Score:', avg_clip)

    del clip_metric
    if device == "cuda":
        torch.cuda.empty_cache()

    # ==========================================
    # 2. 計算 Fréchet Inception Distance (FID)
    # ==========================================
    fid_score = None
    real_paths = load_image_paths(real_dir)
    # print("contents of real_paths:", real_paths)

    if len(real_paths) >= 2 and len(gen_paths) >= 2:
        fid_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.uint8)
        ])

        real_tensors = torch.stack([fid_transform(Image.open(p).convert("RGB")) for p in real_paths])
        # print('length of real_tensors:', len(real_tensors))
        fake_tensors = torch.stack([fid_transform(Image.open(p).convert("RGB")) for p in gen_paths])
        # print('length of fake_tensors:', len(fake_tensors))

        fid = FrechetInceptionDistance(feature=2048).to(device)
        # # 分批送入以避免單次顯存過高
        batch_size = 10
        for i in range(0, len(real_tensors), batch_size):
            print(f"Processing real tensors batch {i // batch_size + 1}")
            fid.update(real_tensors[i:i+batch_size].to(device), real=True)
        for i in range(0, len(fake_tensors), batch_size):
            print(f"Processing fake tensors batch {i // batch_size + 1}")
            fid.update(fake_tensors[i:i+batch_size].to(device), real=False)

        fid_score = fid.compute().item()
        print('FID Score:', fid_score)

        del fid
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"⚠️ 真實圖片資料夾 '{real_dir}' 圖片不足（目前 {len(real_paths)} 張，至少需 2 張），跳過 FID 計算。")

    # return avg_clip, fid_score
    return avg_clip, fid_score

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device:', device)

    # 1. 設定各資料夾路徑（請依您的實際目錄名稱調整）
    real_facilities_dir = "../data/facility_images"
    not_trained_dir = "../output/untrained"
    trained_dir = "../output/trained"

    # 2. 對應的英文測試主題清單（若資料夾內無 generation_log.json，則依序自動配對）
    test_prompts = [
        # 1. 電腦室 / 資訊中心 (Computer Lab)
        "blue futuristic school computer science lab with dual-screen workstations",
        
        # 2. 科學 / 化學實驗室 (Science Laboratory)
        "modern high school chemistry laboratory with fume hoods and clean lab benches",
        
        # 3. 圖書館 / 閱讀自習區 (School Library)
        "bright modern school library with wooden bookshelves, study pods, and natural sunlight",
        
        # 4. 室內體育館 (Indoor Gymnasium)
        "school indoor gymnasium with wooden flooring, badminton courts, and fitness equipment",
        
        # 5. 標準課室 / 互動教室 (Interactive Classroom)
        "smart school classroom with modular desks and a large interactive digital whiteboard",
        
        # 6. 大禮堂 / 多功能演講廳 (Auditorium / Lecture Hall)
        "翻新後的現代化學校多功能演講廳，配有階梯式軟墊座椅與大型 LED 舞台螢幕",
        
        # 7. 學生餐廳 / 休憩中庭 (Student Canteen / Cafeteria)
        "明亮整潔的校園學生餐廳，採用木質長桌與綠色植物隔斷牆設計",
        
        # 8. 入口大廳 / 電梯與無障礙走廊 (Entrance Lobby & Elevators)
        "building new accessible elevators and a bright waiting lobby near the school entrance",
        
        # 9. 音樂教室 (Music Room)
        "明亮的校園音樂教室，配有鋼琴、結他與隔音牆",

        # 10. 音樂與藝術室 (Music & Art Studio)
        "設計一間採光良好且配有畫架與陶藝工作台的校園視覺藝術教室",
    ]

    # 3. 執行評估
    base_clip, base_fid = evaluate_folder(not_trained_dir, real_facilities_dir, test_prompts, device)
    tuned_clip, tuned_fid = evaluate_folder(trained_dir, real_facilities_dir, test_prompts, device)

    # 4. 列印最終對比成績表
    print("\n" + "="*72)
    print("📊 NExT-GPT 校園設施圖像生成：量化評估對比報告")
    print("="*72)
    print(f"{'模型配置 (Model Setup)':<32} | {'CLIP Score (越高越好)':<20} | {'FID (越低越好)':<15}")
    print("-"*72)
    
    base_clip_str = f"{base_clip:.2f}" if base_clip is not None else "N/A"
    base_fid_str = f"{base_fid:.2f}" if base_fid is not None else "N/A"
    print(f"{'Not Trained (Base Vicuna + SD)':<32} | {base_clip_str:<20} | {base_fid_str:<15}")

    tuned_clip_str = f"{tuned_clip:.2f}" if tuned_clip is not None else "N/A"
    tuned_fid_str = f"{tuned_fid:.2f}" if tuned_fid is not None else "N/A"
    print(f"{'Trained (LoRA Vicuna + SD)':<32} | {tuned_clip_str:<20} | {tuned_fid_str:<15}")
    print("="*72)
