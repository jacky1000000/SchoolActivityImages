import os
import shutil
import torch
import torch.nn.functional as F
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data

def search_photos(input_prompt, top_k=10, photo_dir="../../data/activity_photos", 
                  checkpoint_path="../../my_checkpoints/imagebind_huge.pth",
                  cache_path="../../data/photo_embeddings.pt",
                  save_dir="../../output/search_results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 載入 ImageBind 模型
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    print("✅ ImageBind 模型載入完成")

    # 2. 載入或建立 1,330 張相片的向量特徵庫
    if os.path.exists(cache_path):
        print(f"📦 載入相片特徵索引: {cache_path}")
        cache = torch.load(cache_path, map_location=device)
        all_paths = cache["paths"]
        photo_embeddings = cache["embeddings"].to(device)
        print(f"📦 相片特徵索引載入完成，共 {len(all_paths)} 張相片")
    else:
        print(f"📸 首次執行：正在掃描並提取 '{photo_dir}' 中所有相片的特徵...")
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
        all_paths = [
            os.path.join(photo_dir, f).replace("\\", "/")
            for f in sorted(os.listdir(photo_dir))
            if f.lower().endswith(valid_exts)
        ]
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(all_paths), 32):
                print(f"Processing batch {i // 32 + 1} / {(len(all_paths) + 31) // 32}")
                print('image numbers starting from ', i, ' to ', i + 31, ' (total ', len(all_paths), ')')
                batch = all_paths[i:i+32]
                inputs = {ModalityType.VISION: imagebind_data.load_and_transform_vision_data(batch, device)}
                embeds = F.normalize(model(inputs)[ModalityType.VISION], p=2, dim=-1)
                all_embeds.append(embeds)
        photo_embeddings = torch.cat(all_embeds, dim=0)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"paths": all_paths, "embeddings": photo_embeddings.cpu()}, cache_path)
        print("💾 相片特徵索引建立完成！")

    # 3. 將輸入的活動描述編碼成文字向量並計算相似度
    with torch.no_grad():
        print(f"Encoding input prompt: {input_prompt}")
        text_inputs = {ModalityType.TEXT: imagebind_data.load_and_transform_text([input_prompt], device)}
        print(f"text_inputs: {text_inputs}")
        text_embed = F.normalize(model(text_inputs)[ModalityType.TEXT], p=2, dim=-1)

    similarities = torch.matmul(photo_embeddings, text_embed.T).squeeze(-1)
    # print("Size of similarities:", similarities.size())
    print(f"similarities: {similarities}")
    top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(all_paths)))
    print(f"top_scores: {top_scores}")
    print(f"top_indices: {top_indices}")
    # list out all image titles for the top matches
    top_image_titles = [os.path.basename(all_paths[idx.item()]) for idx in top_indices]
    print(f"top_image_titles: {top_image_titles}")

    # # 4. 印出結果並自動複製匹配圖片到輸出資料夾
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)  # 清空上一次搜尋的舊圖
    # os.makedirs(save_dir, exist_ok=True)

    # print(f"\n🔎 活動描述: 「{input_prompt}」")
    # print(f"🎯 最匹配的前 {top_k} 張相片：")
    # matched_paths = []
    # for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
    #     img_path = all_paths[idx.item()]
    #     sim = score.item()
    #     matched_paths.append(img_path)
    #     print(f"  第 {rank} 名 | 相似度: {sim:.4f} | 檔案: {img_path}")
    #     shutil.copy(img_path, os.path.join(save_dir, f"top{rank}_{sim:.3f}_{os.path.basename(img_path)}"))

    # print(f"\n📂 已將找到的相片複製到 '{save_dir}'，可直接打開檢視！")
    # return matched_paths

if __name__ == "__main__":
    # 直接在這裡修改您想搜尋的校園活動描述
    prompt = "drawing lesson"
    search_photos(input_prompt=prompt, top_k=10)
