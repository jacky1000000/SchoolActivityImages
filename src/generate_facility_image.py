import torch
from models.nextgpt_arch import NextGPTImageGenModel
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os
import re

# 1. 定義專屬停止器：一旦檢測到 <IMG0> 出現，立刻強制停止生成
class StopOnImgTag(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 只解碼每條序列最後生成的 6 個 tokens 來檢查是否包含 <IMG0>
        tail_len = min(6, input_ids.shape[-1])
        tail_token_ids = input_ids[:, -tail_len:].detach().tolist()
        recent_texts = self.tokenizer.batch_decode(tail_token_ids, skip_special_tokens=False)
        return any("<IMG0>" in text for text in recent_texts)


def do_inference(user_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"Model dtype: {model_dtype}")
    vicuna_path = "../my_checkpoints/vicuna-7b-v1.5"
    sd_path = "../my_checkpoints/stable-diffusion-v1-5"
    
    # 1. 載入基礎模型架構
    tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
    print("Tokenizer loaded.")
    model = NextGPTImageGenModel(vicuna_path=vicuna_path, freeze_llm=False, torch_dtype=model_dtype)
    
    # # Try to use checkpoints before training
    # model.llm = PeftModel.from_pretrained(model.llm, "../ckpt_updated/nextgpt_facility_output/vicuna_lora")
    
    # 2. 載入微調後的自訂校園 LoRA 與連接層權重
    model.llm = PeftModel.from_pretrained(model.llm, "../ckpt_updated/nextgpt_facility_output/vicuna_lora")
    model.to_image_diffusion.load_state_dict(torch.load("../ckpt_updated/nextgpt_facility_output/image_projection.pth"))
    model.to(device)
    model.eval()
    
    # tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
    # tokenizer.add_special_tokens({'additional_special_tokens': ['<IMG0>']})
    # print("Tokenizer loaded.")
    
    # 3. 處理開放式輸入提示詞
    # system_instruction = "SYSTEM: 你是一個校園設施圖像生成系統。請用一句簡短的話描述畫面細節，並在結尾加上 <IMG0>，禁止條列式長篇大論。\n"
    formatted_prompt = f"USER: {user_prompt} ASSISTANT:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    print("Inputs tokenized.")
    print('EOS token ID:', tokenizer.eos_token_id)

    # 2. 初始化停止條件清單
    stop_criteria = StoppingCriteriaList([StopOnImgTag(tokenizer)])

    # 4. 前向傳播提取隱藏特徵
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids)
        output_tokens = model.llm.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,  # 掛載停止器：看到 <IMG0> 立刻剎車！
            do_sample=True,
            temperature=0.6,           # 降低隨機性，讓輸出更穩定
            top_p=0.9,                 # 核採樣，過濾掉低機率的奇怪詞彙
            repetition_penalty=1.25    # 關鍵參數：大於 1.0 可強力抑制「藍色裝飾...」這類鬼打牆重複
        )

        
    print('output tokens:', output_tokens)
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
    print(f"\n💬 模型生成對話描述:\n{generated_text}\n")

    # 1. 擷取 ASSISTANT: 之後的視覺描述內容
    if "ASSISTANT:" in generated_text:
        assistant_reply = generated_text.split("ASSISTANT:")[-1]
    else:
        assistant_reply = user_prompt
    # 若模型不小心生出變形的 ASSistant:，從那裡切斷
    raw_reply = re.split(r"assistant:|user:|<img0>", assistant_reply, flags=re.IGNORECASE)[0].strip()
    
    # 用正規表達式過濾掉所有奇怪的符號亂碼，只保留中英文、數字與基本標點
    clean_description = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s.,;!?，。；！？、-]", "", raw_reply).strip()
    
    print(f"\n💬 最終乾淨輸出:\n{clean_description}\n")
    # print(f"Assistant reply extracted: {assistant_reply}")

    # if "<IMG0>" in assistant_reply:
    #     final_output = assistant_reply
    # else:
    #     final_output = f"{assistant_reply} <IMG0>"
    # print(f"💬 最終輸出: {final_output}")
    
    # # 5. 調用 Stable Diffusion 畫圖
    # if "<IMG" in final_output:
    print("🎨 檢測到圖像生成訊號！正在調用 Stable Diffusion 進行潛在空間特徵解碼...")
    
    # # 提取最後一層中對齊校園設施特徵的隱藏特徵向量
    # img_embeds = outputs["image_features"][:, -1, :].unsqueeze(1)
    
    # 載入本地端的 Stable Diffusion Pipeline
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_path, 
        safety_checker=None,
        # use_safetensors=False,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    print("Stable Diffusion Pipeline loaded successfully.")

    # expected_embed_dim = sd_pipe.text_encoder.config.hidden_size
    # if img_embeds.shape[-1] != expected_embed_dim:
    #     raise ValueError(
    #         f"Embedding dimension mismatch: got {img_embeds.shape[-1]}, expected {expected_embed_dim}."
    #     )
    # if img_embeds.shape[1] == 0:
    #     raise ValueError("Image prompt embedding sequence length is 0.")
    
    # 透過對齊後的特徵引導擴散模型繪製圖像
    # os.makedirs("../output", exist_ok=True)
    # prompt_embeds = img_embeds.to(dtype=sd_pipe.dtype, device=device)
    # negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    # image = sd_pipe(
    #     prompt_embeds=prompt_embeds,
    #     negative_prompt_embeds=negative_prompt_embeds,
    #     num_inference_steps=25,
    # ).images[0]

    image = sd_pipe(    
        prompt=clean_description,
        negative_prompt="text",
        guidance_scale=7.5,
        num_inference_steps=25,
    ).images[0]
    
    output_filename = "../output/trained/generated_facility.png"
    image.save(output_filename)
    print(f"🎉 設施圖像已成功生成並儲存至: {output_filename}")
# else:
#     print("❌ 模型在此次對話中未觸發圖像生成標記。")

if __name__ == "__main__":
    # 測試一個開放式的設施生成請求
    test_prompt = "blue computer lab"
    do_inference(test_prompt)
