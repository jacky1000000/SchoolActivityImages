# def main() -> None:
#     print("Hello from schoolactivityimages!")
import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType

from dataset_image import CampusImageDataset
from models.nextgpt_arch import NextGPTImageGenModel

# 訓練校園設施生成模型的函式
def train_facility_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"Model dtype: {model_dtype}")
    # os.makedirs("./ckpt/nextgpt_facility_output", exist_ok=True)
    
    vicuna_path = "../my_checkpoints/vicuna-7b-v1.5"
    
    # 1. 初始化模型 (不凍結 LLM 主幹，因為稍後要外掛 LoRA)
    model = NextGPTImageGenModel(
        vicuna_path=vicuna_path, freeze_llm=False, torch_dtype=model_dtype
    )
    
    # 2. 注入 LoRA 配置，專注訓練注意力投影層
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1
    )
    model.llm = get_peft_model(model.llm, peft_config)
    model.to(device)
    
    # 3. 準備開放式設施圖像數據集
    train_dataset = CampusImageDataset(
        json_path="../data/image_gen_train.json",
        tokenizer_path=vicuna_path,
        max_length=256
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print(f"訓練數據集大小: {len(train_dataset)}")
    # print('Show the content inside dataset: ')
    # for i in range(len(train_dataset)):
    #     print(train_dataset[i])
    
    # 4. 優化器設定
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 5
    num_training_steps = len(train_loader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # 5. 訓練循環
    print("🚀 開始進行開放式校園設施生成微調...")
    model.train()
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        print(f"開始 Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
            print(f'outputs returned for epoch {epoch+1}')
            loss = outputs["loss"]
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
        elapsed = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} | 平均訓練 Loss: {total_loss/len(train_loader):.4f} | {elapsed:.1f}s")
        
    # 6. 儲存微調後增量參數
    torch.save(model.to_image_diffusion.state_dict(), "../ckpt_updated/nextgpt_facility_output/image_projection.pth")
    model.llm.save_pretrained("../ckpt_updated/nextgpt_facility_output/vicuna_lora")
    print("🎉 微調成功結束！權重已儲存至 ../ckpt_updated/nextgpt_facility_output/")

def main() -> None:
    train_facility_model()

if __name__ == "__main__":
    main()