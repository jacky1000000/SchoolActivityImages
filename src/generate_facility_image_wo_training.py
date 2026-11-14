# use vicuna checkpoints without training

import torch
from models.nextgpt_arch import NextGPTImageGenModel
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os
import re
import random, string

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
    model.to(device)
    
    # # Try to use checkpoints before training
    # model.llm = PeftModel.from_pretrained(model.llm, "../ckpt_updated/nextgpt_facility_output/vicuna_lora")
    model.eval()

    # 使用 Vicuna 官方標準對話格式，並給予明確的建築攝影關鍵字指令
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        f"USER: Act as an architectural visualizer. Convert this school facility request into a concise, "
        f"highly detailed English image generation prompt (under 45 words, comma-separated visual keywords, "
        f"focus on lighting, materials, and layout, no conversational filler): '{user_prompt}'\n"
        "ASSISTANT:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_tokens = model.llm.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.25
        )

    print('output tokens:', output_tokens)
    full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    expanded_keywords = full_text.split("ASSISTANT:")[-1].strip()
    print(f"\n💬 原始 Vicuna 生成的視覺關鍵字:\n{expanded_keywords}\n")

    # 釋放 Vicuna 佔用的顯存，為後續 Stable Diffusion 騰出完整空間
    del model.llm
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return expanded_keywords

def generate_facility_image(expanded_keywords):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd_path = "../my_checkpoints/stable-diffusion-v1-5"

    # 載入本地端的 Stable Diffusion Pipeline
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_path, 
        safety_checker=None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    print("Stable Diffusion Pipeline loaded successfully.")

    # os.makedirs("../output", exist_ok=True)
    image = sd_pipe(    
        prompt=expanded_keywords,
        negative_prompt="text",
        guidance_scale=7.5,
        num_inference_steps=25,
    ).images[0]

    # add random characters for filename
    random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    output_filename = f"../output/untrained/generated_facility_{random_chars}.png"
    image.save(output_filename)
    print(f"🎉 設施圖像已成功生成並儲存至: {output_filename}")

if __name__ == "__main__":
    # 測試一個開放式的設施生成請求
    test_prompt = "blue computer lab"
    expanded_keywords = do_inference(test_prompt)
    generate_facility_image(expanded_keywords)
