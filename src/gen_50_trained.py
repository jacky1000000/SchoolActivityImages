# import do_inference and generate_facility_image
import torch
from models.nextgpt_arch import NextGPTImageGenModel
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os
import re
import random, string
from gen_50_untrained import SD_usage
# from generate_facility_image_wo_training import do_inference, generate_facility_image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # load model and tokenizer for untrained
    use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"Model dtype: {model_dtype}")
    vicuna_path = "../my_checkpoints/vicuna-7b-v1.5"
    sd_path = "../my_checkpoints/stable-diffusion-v1-5"

    # 1. 載入基礎模型架構
    tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
    print("Tokenizer loaded.")
    model = NextGPTImageGenModel(vicuna_path=vicuna_path, freeze_llm=False, torch_dtype=model_dtype)

    # 2. 載入微調後的自訂校園 LoRA 與連接層權重
    model.llm = PeftModel.from_pretrained(model.llm, "../ckpt_updated/nextgpt_facility_output/vicuna_lora")
    model.to_image_diffusion.load_state_dict(torch.load("../ckpt_updated/nextgpt_facility_output/image_projection.pth"))
    model.to(device)
    model.eval()

    # 載入本地端的 Stable Diffusion Pipeline once only
    sd_path = "../my_checkpoints/stable-diffusion-v1-5"
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_path, 
        safety_checker=None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    print("Stable Diffusion Pipeline loaded successfully.")

    # def SD_usage(expanded_keywords, category, index, use_trained, sd_pipe):
    #     # device = "cuda" if torch.cuda.is_available() else "cpu"
    #     # sd_path = "../my_checkpoints/stable-diffusion-v1-5"

    #     # # 載入本地端的 Stable Diffusion Pipeline
    #     # sd_pipe = StableDiffusionPipeline.from_pretrained(
    #     #     sd_path, 
    #     #     safety_checker=None,
    #     #     torch_dtype=torch.float16 if device == "cuda" else torch.float32
    #     # ).to(device)
    #     # print("Stable Diffusion Pipeline loaded successfully.")

    #     # os.makedirs("../output", exist_ok=True)
    #     image = sd_pipe(    
    #         prompt=expanded_keywords,
    #         negative_prompt="text",
    #         guidance_scale=7.5,
    #         num_inference_steps=25,
    #     ).images[0]

    #     # add random characters for filename
    #     # filename should be based on category of facility being generated and index from 1 to 5
    #     # random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    #     # index should be converted to string, like 01, 02, etc
    #     index_str = str(index).zfill(2)
    #     # if using untrained
    #     if not use_trained:
    #         # os.makedirs("../output/untrained", exist_ok=True)
    #         output_filename = f"../output/untrained/generated_facility_{category}_{index_str}.jpg"
    #     else:
    #         # os.makedirs("../output/trained", exist_ok=True)
    #         output_filename = f"../output/trained/generated_facility_{category}_{index_str}.jpg"
    #     image.save(output_filename)
    #     print(f"🎉 設施圖像已成功生成並儲存至: {output_filename}")

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
    # for file rename only
    category = [
        "computer_lab",
        "science_lab",
        "library",
        "gymnasium",
        "classroom",
        "auditorium",
        "canteen",
        "entrance_lobby",
        "music_room",
        "art_studio"
    ]


    # generate 50 images using untrained model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    # use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    # model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    # print(f"Model dtype: {model_dtype}")
    # vicuna_path = "../my_checkpoints/vicuna-7b-v1.5"
    # sd_path = "../my_checkpoints/stable-diffusion-v1-5"

    # # 1. 載入基礎模型架構
    # tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
    # print("Tokenizer loaded.")
    # model = NextGPTImageGenModel(vicuna_path=vicuna_path, freeze_llm=False, torch_dtype=model_dtype)
    # model.to(device)

    # # # Try to use checkpoints before training
    # # model.llm = PeftModel.from_pretrained(model.llm, "../ckpt_updated/nextgpt_facility_output/vicuna_lora")
    # model.eval()
    for prompt_index, prompt in enumerate(test_prompts):
        # repeat for 5 times
        for repeat_index in range(5):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_tokens = model.llm.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=200,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.25
                )
        
            print('output tokens for prompt', prompt_index, 'repeat', repeat_index, ':', output_tokens)
            full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            print(f"\n💬 模型生成的完整文本 for prompt {prompt_index} repeat {repeat_index}:\n{full_text}\n")
            expanded_keywords = full_text
            # expanded_keywords = full_text.split("ASSISTANT:")[-1].strip()
            # generate_facility_image(expanded_keywords)
            # print(f"\n💬 原始 Vicuna 生成的視覺關鍵字:\n{expanded_keywords}\n")
        
            SD_usage(expanded_keywords, category[prompt_index], repeat_index + 1, True, sd_pipe)

    # 釋放顯存資源
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

# generate 50 images using trained model

# Compare the generated images from untrained and trained models by Text-to-Image Faithfulness and Fréchet Inception Distance