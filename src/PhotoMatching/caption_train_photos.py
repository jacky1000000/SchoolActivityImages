import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_real_captions_for_training(
    train_photo_dir="../../data/train_photos",
    output_json="../../data/imagebind_train.json",
    batch_size=16
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "Salesforce/blip-image-captioning-base"
    print(f"⏳ 正在載入 BLIP 看圖說話模型 ({model_id}) 至 {device}...")
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    train_paths = [
        os.path.join(train_photo_dir, f).replace("\\", "/")
        for f in sorted(os.listdir(train_photo_dir))
        if f.lower().endswith(valid_exts)
    ]

    print(f"📸 開始為 {len(train_paths)} 張校園訓練相片生成真實畫面描述...")
    train_items = []

    with torch.no_grad():
        for i in range(0, len(train_paths), batch_size):
            batch_paths = train_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            # 加上引導前綴 "a school photo of"，讓模型生成的描述自動貼合校園語境
            prompts = ["a school photo of"] * len(images)
            inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device, dtype)

            out_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                min_new_tokens=8,
                num_beams=3,
                repetition_penalty=1.2
            )
            captions = processor.batch_decode(out_ids, skip_special_tokens=True)

            for path, cap in zip(batch_paths, captions):
                clean_cap = cap.strip()
                train_items.append({
                    "image_path": path,
                    "text": clean_cap
                })

            done = min(i + batch_size, len(train_paths))
            if done % 160 == 0 or done == len(train_paths):
                print(f"  進度: {done}/{len(train_paths)} 張完成 | 最新範例: {train_items[-1]['text']}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(train_items, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print(f"🎉 完成！已為全部 {len(train_items)} 張訓練相片生成專屬真實畫面描述！")
    print(f"📁 儲存路徑: {output_json}")
    print("=" * 68)

if __name__ == "__main__":
    generate_real_captions_for_training()
