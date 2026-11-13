import torch
from models.nextgpt_arch import NextGPTImageGenModel
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def do_inference(user_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"Model dtype: {model_dtype}")
    vicuna_path = "../my_checkpoints/vicuna-7b-v1.5"
    sd_path = "../my_checkpoints/stable-diffusion-v1-5"
    
    # 1. 載入基礎模型架構
    model = NextGPTImageGenModel(vicuna_path=vicuna_path, freeze_llm=False, torch_dtype=model_dtype)
    
    # 2. 載入微調後的自訂校園 LoRA 與連接層權重
    model.llm = PeftModel.from_pretrained(model.llm, "../ckpt_updated/nextgpt_facility_output/vicuna_lora")
    model.to_image_diffusion.load_state_dict(torch.load("../ckpt_updated/nextgpt_facility_output/image_projection.pth"))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
    print("Tokenizer loaded.")
    
    # 3. 處理開放式輸入提示詞
    formatted_prompt = f"USER: {user_prompt} ASSISTANT:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    print("Inputs tokenized.")
    
    # 4. 前向傳播提取隱藏特徵
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids)
        # print('output:', outputs)
        output_tokens = model.llm.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,           # 降低隨機性，讓輸出更穩定
            top_p=0.9,                 # 核採樣，過濾掉低機率的奇怪詞彙
            repetition_penalty=1.25    # 關鍵參數：大於 1.0 可強力抑制「藍色裝飾...」這類鬼打牆重複
        )

        
    print('output tokens:', output_tokens)
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
    print(f"\n💬 模型生成對話描述:\n{generated_text}\n")
    
    # # 5. 檢測訊號標記，調用 Stable Diffusion 畫圖
    # if "<IMG" in generated_text:
    #     print("🎨 檢測到圖像生成訊號！正在調用 Stable Diffusion 進行潛在空間特徵解碼...")
        
    #     # 提取最後一層中對齊校園設施特徵的隱藏特徵向量
    #     img_embeds = outputs["image_features"][:, -1, :].unsqueeze(1)
        
    #     # 載入本地端的 Stable Diffusion Pipeline
    #     sd_pipe = StableDiffusionPipeline.from_pretrained(
    #         sd_path, 
    #         torch_dtype=torch.float16 if device == "cuda" else torch.float32
    #     ).to(device)
        
    #     # 透過對齊後的特徵引導擴散模型繪製圖像
    #     os.makedirs("../output", exist_ok=True)
    #     image = sd_pipe(prompt_embeds=img_embeds.to(sd_pipe.dtype), num_inference_steps=25).images[0]
        
    #     output_filename = "../output/generated_facility.png"
    #     image.save(output_filename)
    #     print(f"🎉 設施圖像已成功生成並儲存至: {output_filename}")
    # else:
    #     print("❌ 模型在此次對話中未觸發圖像生成標記。")

if __name__ == "__main__":
    # 測試一個開放式的設施生成請求
    test_prompt = "幫我設計一個富有現代感、科技風，且帶有藍色氛圍的學校電腦科學系機房中心。"
    do_inference(test_prompt)
